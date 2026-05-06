#!/usr/bin/env python3
"""
train_basic_point.py

Temporal point-graph training script (no AMR runtime mesh).
Optional physics-derived node inputs are supported (DEC-like or MLS backend),
mirroring the shock-ramp PARC-style conditioning pathway.

Assumed data contract per timestep (PyG Data-like or dict-like):
  - x:          [N, Fx] node features at time t
  - edge_index: [2, E] or [E, 2] point connectivity
  - pos:        [N, Dp] point coordinates (D=2 or 3 typical)
Optional:
  - y:          [N, Fy] supervised target at time t (if present and enabled)
  - time/t/sim_time or global_params.timestep_current (for metadata only)

Targets:
  - if data.use_y_as_target=true and y exists: predict y_t from x_t
  - otherwise: predict x_{t+1} from x_t

Training modes:
  - one-step pairs (legacy baseline)
  - multi-step windows (shock-ramp style): unrolled autoregressive training where
    predicted output at step k is fed as input at step k+1

This script intentionally avoids AMR runtime meshing; physics inputs are optional.

Data source modes:
  - single file: cfg.data.pt_path points to one .pt/.pth/.zip file
  - pre-split directory: cfg.data.pt_path points to a directory containing
    train/, val/, test/ subfolders with PT files for each split
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torch_geometric.data import Data

from models import FeatureNet
import utils.dec_ops as dec
import utils.mls as mls


_MLS_STATE = {
    "sig": None,
    "grad": None,
    "lapw": None,
    "adv": None,
    "diff": None,
}


def _physics_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("physics", {}) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _extract_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_edge_index(x: Any) -> torch.Tensor:
    ei = torch.as_tensor(x, dtype=torch.long)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(ei.shape)}")
    if ei.size(0) == 2:
        out = ei
    elif ei.size(1) == 2:
        out = ei.t().contiguous()
    else:
        raise ValueError(f"edge_index must be (2,E) or (E,2), got {tuple(ei.shape)}")
    return out


def _as_2d_float(x: Any, name: str) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if t.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {tuple(t.shape)}")
    return t


def _extract_time(step_obj: Any) -> Optional[float]:
    for key in ("time", "t", "sim_time"):
        v = _extract_attr(step_obj, key, None)
        if v is not None:
            try:
                return float(torch.as_tensor(v).view(-1)[0].item())
            except Exception:
                pass

    gp = _extract_attr(step_obj, "global_params", None)
    if isinstance(gp, dict):
        for key in ("timestep_current", "time", "t"):
            if key in gp:
                try:
                    return float(torch.as_tensor(gp[key]).view(-1)[0].item())
                except Exception:
                    pass
    return None


def _load_torch_object(path_or_buf: Any, map_location: str = "cpu") -> Any:
    # Try modern options first; gracefully degrade for older torch versions.
    attempts = [
        {"map_location": map_location, "weights_only": False, "mmap": True},
        {"map_location": map_location, "weights_only": False},
        {"map_location": map_location},
    ]
    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return torch.load(path_or_buf, **kwargs)
        except TypeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("torch.load failed with all compatibility options.")


def _load_pt_or_zip(path: str) -> Any:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"PT path not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        return _load_torch_object(path, map_location="cpu")

    if ext == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".pt", ".pth"))]
            if not names:
                raise RuntimeError(f"No .pt/.pth file found inside zip: {path}")
            with zf.open(names[0], "r") as f:
                buf = io.BytesIO(f.read())
        return _load_torch_object(buf, map_location="cpu")

    raise RuntimeError(f"Unsupported file extension: {ext}. Expected .pt/.pth/.zip")


def _list_pt_like_files(root_dir: str) -> List[str]:
    root_dir = os.path.expanduser(root_dir)
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Not a directory: {root_dir}")
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            low = name.lower()
            if low.endswith(".pt") or low.endswith(".pth") or low.endswith(".zip"):
                out.append(os.path.join(dirpath, name))
    out.sort()
    return out


def _resolve_pt_source(pt_path: str) -> Dict[str, Any]:
    """
    Resolve cfg.data.pt_path into either:
      - {"mode": "single_file", "path": "..."}
      - {"mode": "pre_split_dir", "root": "...", "split_files": {...}}
    """
    p = os.path.expanduser(str(pt_path).strip())
    if not p:
        raise ValueError("cfg['data']['pt_path'] is required.")
    if os.path.isfile(p):
        return {"mode": "single_file", "path": p}
    if not os.path.isdir(p):
        raise FileNotFoundError(f"pt_path not found: {p}")

    split_dirs = {k: os.path.join(p, k) for k in ("train", "val", "test")}
    if not all(os.path.isdir(d) for d in split_dirs.values()):
        missing = [k for k, d in split_dirs.items() if not os.path.isdir(d)]
        raise ValueError(
            "When cfg['data']['pt_path'] is a directory, it must contain "
            f"train/, val/, test/ subdirectories. Missing: {missing}"
        )

    split_files: Dict[str, List[str]] = {}
    for split_name, split_dir in split_dirs.items():
        files = _list_pt_like_files(split_dir)
        if len(files) == 0:
            raise ValueError(f"No .pt/.pth/.zip files found under split directory: {split_dir}")
        split_files[split_name] = files

    return {"mode": "pre_split_dir", "root": p, "split_files": split_files}


def _coerce_time_series_to_tnf(x3: torch.Tensor, n_nodes: int, name: str) -> torch.Tensor:
    """
    Coerce a 3D tensor into [T, N, F] using node-count heuristics.
    Accepts common layouts: [T,N,F], [N,T,F], [T,F,N].
    """
    if x3.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape {tuple(x3.shape)}")
    if x3.size(1) == n_nodes:
        return x3
    if x3.size(0) == n_nodes:
        return x3.permute(1, 0, 2).contiguous()
    if x3.size(2) == n_nodes:
        return x3.permute(0, 2, 1).contiguous()
    raise ValueError(
        f"Could not align {name} to [T,N,F] with N={n_nodes}; got shape {tuple(x3.shape)}."
    )


def _extract_case_level_series(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Handle case-level tensors format, e.g.:
      pos: [N,2], edge_index: [2,E], velocity: [T,N,2], time_steps: [T]
    Returns per-step dict list compatible with _extract_step_fields, or None if
    this format is not detected.
    """
    if not isinstance(obj, dict):
        return None

    pos_raw = obj.get("pos", None)
    pos_phys_raw = obj.get("pos_physical", None)
    edge_raw = obj.get("edge_index", obj.get("ei", None))
    if edge_raw is None:
        return None

    x_key = None
    x_raw = None
    for k in (
        "velocity",
        "x",
        "features",
        "feature_series",
        "x_series",
        "state_series",
        "states",
    ):
        v = obj.get(k, None)
        if v is None:
            continue
        t = torch.as_tensor(v)
        if t.ndim == 3:
            x_key = k
            x_raw = v
            break
    if x_raw is None:
        return None

    if pos_raw is None and pos_phys_raw is None:
        return None
    if pos_raw is None:
        pos_raw = pos_phys_raw

    pos = _as_2d_float(pos_raw, "pos")
    edge_index = _as_edge_index(edge_raw)
    n_nodes = int(pos.size(0))

    x_tnf = _coerce_time_series_to_tnf(torch.as_tensor(x_raw, dtype=torch.float32), n_nodes, x_key)
    T = int(x_tnf.size(0))

    y_tnf = None
    for yk in ("y", "targets", "target_series", "y_series", "labels"):
        yv = obj.get(yk, None)
        if yv is None:
            continue
        yt = torch.as_tensor(yv, dtype=torch.float32)
        if yt.ndim != 3:
            continue
        y_tnf = _coerce_time_series_to_tnf(yt, n_nodes, yk)
        if int(y_tnf.size(0)) != T:
            raise ValueError(
                f"Time length mismatch between {x_key} (T={T}) and {yk} (T={int(y_tnf.size(0))})."
            )
        break

    time_vec = None
    for tk in ("time_steps", "times", "time", "t"):
        tv = obj.get(tk, None)
        if tv is None:
            continue
        tt = torch.as_tensor(tv).view(-1)
        if int(tt.numel()) == T:
            time_vec = tt.to(torch.float32)
            break
    if time_vec is None:
        t0 = int(obj.get("time_start", 0))
        dt_idx = int(obj.get("time_stride", 1))
        time_vec = torch.arange(T, dtype=torch.float32) * float(dt_idx) + float(t0)

    gp_base = {
        "case_name": obj.get("case_name", None),
        "split_name": obj.get("split_name", None),
        "reynolds_number": obj.get("reynolds_number", None),
        "frame_dt_seconds": obj.get("frame_dt_seconds", None),
        "time_stride": obj.get("time_stride", None),
        "time_start": obj.get("time_start", None),
        "time_end": obj.get("time_end", None),
        "source_x_key": x_key,
    }

    print(
        f"[INFO] detected case-level format: case={gp_base['case_name']} split_meta={gp_base['split_name']} "
        f"x_key={x_key} steps={T} nodes={n_nodes} feat_dim={int(x_tnf.size(2))}"
    )

    steps: List[Dict[str, Any]] = []
    for i in range(T):
        steps.append(
            {
                "x": x_tnf[i],
                "y": (None if y_tnf is None else y_tnf[i]),
                "pos": pos,
                "edge_index": edge_index,
                "time": float(time_vec[i].item()),
                "global_params": gp_base,
            }
        )
    return steps


def _extract_series(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("timesteps", "snapshots", "steps", "time_steps", "sequence", "data_list"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        case_steps = _extract_case_level_series(obj)
        if case_steps is not None:
            return case_steps
    # Single timestep object is not enough for temporal training.
    raise ValueError(
        "Could not find a list of timesteps in loaded object. "
        "Expected list or dict with timesteps/snapshots/steps/sequence/data_list, "
        "or a case-level dict with (pos, edge_index, velocity[time,node,feature])."
    )


def _select_columns(x: torch.Tensor, cols: Optional[Sequence[int]]) -> torch.Tensor:
    if cols is None:
        return x
    if len(cols) == 0:
        raise ValueError("Column selection list is empty.")
    cols_t = torch.as_tensor([int(c) for c in cols], dtype=torch.long)
    if int(cols_t.max().item()) >= x.size(1) or int(cols_t.min().item()) < 0:
        raise ValueError(f"Column selection {list(cols)} out of bounds for tensor with shape {tuple(x.shape)}")
    return x[:, cols_t]


def _extract_step_fields(step: Any) -> Dict[str, Any]:
    x = _extract_attr(step, "x", None)
    y = _extract_attr(step, "y", None)
    pos = _extract_attr(step, "pos", _extract_attr(step, "xy", None))
    edge_index = _extract_attr(step, "edge_index", _extract_attr(step, "ei", None))
    if x is None:
        x = _extract_attr(step, "features", None)

    if x is None or pos is None or edge_index is None:
        missing = []
        if x is None:
            missing.append("x/features")
        if pos is None:
            missing.append("pos/xy")
        if edge_index is None:
            missing.append("edge_index/ei")
        raise KeyError(f"Timestep is missing required fields: {', '.join(missing)}")

    out = {
        "x": _as_2d_float(x, "x"),
        "y": None if y is None else _as_2d_float(y, "y"),
        "pos": _as_2d_float(pos, "pos"),
        "edge_index": _as_edge_index(edge_index),
        "time": _extract_time(step),
        "global_params": _extract_attr(step, "global_params", None),
    }
    return out


def _build_z_groups(z: torch.Tensor, z_tol: float) -> List[torch.Tensor]:
    z = z.view(-1).to(torch.float32)
    if z.numel() == 0:
        return []

    if z_tol > 0:
        z0 = z.min()
        keys = torch.round((z - z0) / float(z_tol)).to(torch.long)
    else:
        # Exact grouping (typical when slices are stored at exact z values).
        _, keys = torch.unique(z, sorted=True, return_inverse=True)

    groups: List[torch.Tensor] = []
    for g in torch.unique(keys, sorted=True):
        idx = torch.nonzero(keys == g, as_tuple=False).view(-1)
        if idx.numel() > 0:
            groups.append(idx)
    return groups


def _subgraph_by_index(
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    keep_idx: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    keep_idx = keep_idx.to(torch.long)
    n = x.size(0)
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep_idx] = torch.arange(keep_idx.numel(), dtype=torch.long)

    src = edge_index[0].long()
    dst = edge_index[1].long()
    valid = (
        (src >= 0)
        & (src < n)
        & (dst >= 0)
        & (dst < n)
        & (remap[src] >= 0)
        & (remap[dst] >= 0)
        & (src != dst)
    )
    ei_sub = torch.stack([remap[src[valid]], remap[dst[valid]]], dim=0)

    return (
        x.index_select(0, keep_idx),
        None if y is None else y.index_select(0, keep_idx),
        pos.index_select(0, keep_idx),
        ei_sub,
    )


@dataclass
class PointPair:
    x: torch.Tensor
    y: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    t_src: Optional[float] = None
    t_dst: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class PointGraphTemporalDataset(Dataset):
    """
    Temporal point-graph pairs for one-step supervision.
    """

    def __init__(self, cfg: Dict[str, Any], pt_paths: Optional[Sequence[str]] = None):
        super().__init__()
        data_cfg = cfg.get("data", {}) or {}
        train_cfg = cfg.get("train", {}) or {}
        feat_cfg = cfg.get("features", {}) or {}

        if pt_paths is None:
            pt_path = str(data_cfg.get("pt_path", "")).strip()
            if not pt_path:
                raise ValueError("cfg['data']['pt_path'] is required.")
            pt_paths = [pt_path]
        pt_paths = [os.path.expanduser(str(p)) for p in pt_paths]
        if len(pt_paths) == 0:
            raise ValueError("PointGraphTemporalDataset requires at least one PT path.")

        use_y_target = bool(data_cfg.get("use_y_as_target", True))
        reverse_time = bool(data_cfg.get("reverse_time", False))
        split_by_z = bool(data_cfg.get("split_by_z", False))
        z_index = int(data_cfg.get("z_index", 2))
        z_tol = float(data_cfg.get("z_tol", 0.0))

        x_cols = feat_cfg.get("use_columns", None)
        y_cols = feat_cfg.get("target_columns", None)
        pos_cols = feat_cfg.get("pos_columns", None)

        x_cols = None if x_cols is None else [int(c) for c in x_cols]
        y_cols = x_cols if y_cols is None else [int(c) for c in y_cols]
        pos_cols = None if pos_cols is None else [int(c) for c in pos_cols]

        pairs: List[PointPair] = []
        for src_idx, src_path in enumerate(pt_paths):
            raw_obj = _load_pt_or_zip(src_path)
            raw_steps = _extract_series(raw_obj)
            if len(raw_steps) < 2:
                raise ValueError(
                    f"Need at least 2 timesteps in source file, found {len(raw_steps)}: {src_path}"
                )
            steps = [_extract_step_fields(s) for s in raw_steps]
            if reverse_time:
                steps = list(reversed(steps))

            for t in range(len(steps) - 1):
                s0 = steps[t]
                s1 = steps[t + 1]

                x0 = s0["x"]
                pos0 = s0["pos"]
                ei0 = s0["edge_index"]

                # Target selection policy.
                if use_y_target and (s0["y"] is not None):
                    y_target = s0["y"]
                else:
                    y_target = s1["x"]

                if x0.size(0) != y_target.size(0):
                    raise ValueError(
                        f"Node count mismatch at pair t={t}: "
                        f"x_t has {x0.size(0)} nodes, target has {y_target.size(0)}. "
                        f"(source={src_path})"
                    )
                if x0.size(0) != pos0.size(0):
                    raise ValueError(
                        f"Node count mismatch at pair t={t}: "
                        f"x_t has {x0.size(0)} nodes, pos has {pos0.size(0)}. "
                        f"(source={src_path})"
                    )

                x_sel = _select_columns(x0, x_cols)
                y_sel = _select_columns(y_target, y_cols)
                pos_sel = _select_columns(pos0, pos_cols)

                if split_by_z:
                    if z_index < 0 or z_index >= pos0.size(1):
                        raise ValueError(
                            f"split_by_z requested with z_index={z_index}, but pos has shape {tuple(pos0.shape)} "
                            f"(source={src_path})"
                        )
                    z = pos0[:, z_index]
                    groups = _build_z_groups(z, z_tol=z_tol)
                    for gidx, keep in enumerate(groups):
                        xs, ys, ps, eis = _subgraph_by_index(x_sel, y_sel, pos_sel, ei0, keep)
                        if xs.size(0) == 0:
                            continue
                        pairs.append(
                            PointPair(
                                x=xs,
                                y=ys,
                                pos=ps,
                                edge_index=eis,
                                t_src=s0["time"],
                                t_dst=s1["time"],
                                meta={
                                    "pair_t": t,
                                    "z_group": gidx,
                                    "source_index": src_idx,
                                    "source_path": os.path.abspath(src_path),
                                },
                            )
                        )
                else:
                    pairs.append(
                        PointPair(
                            x=x_sel,
                            y=y_sel,
                            pos=pos_sel,
                            edge_index=ei0,
                            t_src=s0["time"],
                            t_dst=s1["time"],
                            meta={
                                "pair_t": t,
                                "source_index": src_idx,
                                "source_path": os.path.abspath(src_path),
                            },
                        )
                    )

        if len(pairs) == 0:
            raise RuntimeError("No training pairs were built from the dataset.")

        self.pairs = pairs
        self.x_dim = int(self.pairs[0].x.size(1))
        self.y_dim = int(self.pairs[0].y.size(1))
        self.pos_dim = int(self.pairs[0].pos.size(1))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pairs[idx]
        return {
            "x": p.x,
            "y": p.y,
            "pos": p.pos,
            "edge_index": p.edge_index,
            "t_src": p.t_src,
            "t_dst": p.t_dst,
            "meta": p.meta,
        }


@dataclass
class PointWindow:
    x_list: List[torch.Tensor]
    y_list: List[torch.Tensor]
    pos_list: List[torch.Tensor]
    edge_index_list: List[torch.Tensor]
    t_list: List[Optional[float]]
    meta: Optional[Dict[str, Any]] = None


class PointGraphWindowDataset(Dataset):
    """
    Contiguous windows of timesteps for multi-step autoregressive training.

    Each sample contains K timesteps and K-1 supervised transitions.
    """

    def __init__(self, cfg: Dict[str, Any], pt_paths: Optional[Sequence[str]] = None):
        super().__init__()
        data_cfg = cfg.get("data", {}) or {}
        train_cfg = cfg.get("train", {}) or {}
        feat_cfg = cfg.get("features", {}) or {}

        if pt_paths is None:
            pt_path = str(data_cfg.get("pt_path", "")).strip()
            if not pt_path:
                raise ValueError("cfg['data']['pt_path'] is required.")
            pt_paths = [pt_path]
        pt_paths = [os.path.expanduser(str(p)) for p in pt_paths]
        if len(pt_paths) == 0:
            raise ValueError("PointGraphWindowDataset requires at least one PT path.")

        self.window_size = int(train_cfg.get("window_size", 2))
        self.stride = int(train_cfg.get("stride", 1))
        if self.window_size < 2:
            raise ValueError("train.window_size must be >= 2.")
        if self.stride < 1:
            raise ValueError("train.stride must be >= 1.")

        self.use_y_target = bool(data_cfg.get("use_y_as_target", True))
        reverse_time = bool(data_cfg.get("reverse_time", False))
        split_by_z = bool(data_cfg.get("split_by_z", False))
        z_index = int(data_cfg.get("z_index", 2))
        z_tol = float(data_cfg.get("z_tol", 0.0))

        x_cols = feat_cfg.get("use_columns", None)
        y_cols = feat_cfg.get("target_columns", None)
        pos_cols = feat_cfg.get("pos_columns", None)
        x_cols = None if x_cols is None else [int(c) for c in x_cols]
        y_cols = x_cols if y_cols is None else [int(c) for c in y_cols]
        pos_cols = None if pos_cols is None else [int(c) for c in pos_cols]

        # One sequence per z-group when split_by_z=true; otherwise one global sequence.
        sequences: List[List[Dict[str, Any]]] = []
        for src_idx, src_path in enumerate(pt_paths):
            raw_obj = _load_pt_or_zip(src_path)
            raw_steps = _extract_series(raw_obj)
            if len(raw_steps) < self.window_size:
                raise ValueError(
                    f"Need at least window_size={self.window_size} timesteps, found {len(raw_steps)} "
                    f"(source={src_path})."
                )
            steps = [_extract_step_fields(s) for s in raw_steps]
            if reverse_time:
                steps = list(reversed(steps))

            src_sequences: Optional[List[List[Dict[str, Any]]]] = None
            for t, s in enumerate(steps):
                x_sel = _select_columns(s["x"], x_cols)
                y_sel = None if s["y"] is None else _select_columns(s["y"], y_cols)
                pos_sel = _select_columns(s["pos"], pos_cols)
                ei = s["edge_index"]

                if split_by_z:
                    if z_index < 0 or z_index >= s["pos"].size(1):
                        raise ValueError(
                            f"split_by_z requested with z_index={z_index}, but pos has shape {tuple(s['pos'].shape)} "
                            f"(source={src_path})"
                        )
                    z = s["pos"][:, z_index]
                    groups = _build_z_groups(z, z_tol=z_tol)
                    if len(groups) == 0:
                        raise RuntimeError(f"No z groups found at timestep t={t} (source={src_path}).")
                    if src_sequences is None:
                        src_sequences = [[] for _ in range(len(groups))]
                    if len(groups) != len(src_sequences):
                        raise RuntimeError(
                            f"split_by_z produced inconsistent group count at t={t}: "
                            f"expected {len(src_sequences)}, got {len(groups)} "
                            f"(source={src_path})."
                        )
                    for gidx, keep in enumerate(groups):
                        xs, ys, ps, eis = _subgraph_by_index(x_sel, y_sel, pos_sel, ei, keep)
                        if xs.size(0) == 0:
                            raise RuntimeError(f"Empty z-group at t={t}, group={gidx} (source={src_path}).")
                        if ys is not None and ys.size(0) != xs.size(0):
                            raise RuntimeError(
                                f"Node mismatch at t={t}, group={gidx}: x has {xs.size(0)}, y has {ys.size(0)} "
                                f"(source={src_path})."
                            )
                        src_sequences[gidx].append(
                            {
                                "x": xs,
                                "y": ys,
                                "pos": ps,
                                "edge_index": eis,
                                "time": s["time"],
                                "meta": {
                                    "t": t,
                                    "z_group": gidx,
                                    "source_index": src_idx,
                                    "source_path": os.path.abspath(src_path),
                                },
                            }
                        )
                else:
                    if src_sequences is None:
                        src_sequences = [[]]
                    if y_sel is not None and y_sel.size(0) != x_sel.size(0):
                        raise RuntimeError(
                            f"Node mismatch at t={t}: x has {x_sel.size(0)}, y has {y_sel.size(0)} "
                            f"(source={src_path})."
                        )
                    src_sequences[0].append(
                        {
                            "x": x_sel,
                            "y": y_sel,
                            "pos": pos_sel,
                            "edge_index": ei,
                            "time": s["time"],
                            "meta": {
                                "t": t,
                                "source_index": src_idx,
                                "source_path": os.path.abspath(src_path),
                            },
                        }
                    )

            if src_sequences is None:
                raise RuntimeError(f"No sequences were built from source file: {src_path}")
            sequences.extend(src_sequences)

        windows: List[Tuple[int, int]] = []
        for seq_id, seq in enumerate(sequences):
            if len(seq) < self.window_size:
                continue
            for start in range(0, len(seq) - self.window_size + 1, self.stride):
                windows.append((seq_id, start))

        if len(windows) == 0:
            raise RuntimeError(
                f"No windows built: window_size={self.window_size}, stride={self.stride}, "
                f"sequence_lengths={[len(s) for s in sequences]}"
            )

        self.sequences = sequences
        self.windows = windows

        ex0 = self[0]
        self.x_dim = int(ex0["x_list"][0].size(1))
        self.y_dim = int(ex0["y_list"][0].size(1))
        self.pos_dim = int(ex0["pos_list"][0].size(1))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_id, start = self.windows[idx]
        seq = self.sequences[seq_id]
        chunk = seq[start : start + self.window_size]

        x_list = [s["x"] for s in chunk]
        pos_list = [s["pos"] for s in chunk]
        edge_index_list = [s["edge_index"] for s in chunk]
        t_list = [s["time"] for s in chunk]

        y_list: List[torch.Tensor] = []
        for k in range(self.window_size - 1):
            s = chunk[k]
            s_next = chunk[k + 1]
            if self.use_y_target and (s["y"] is not None):
                y_tgt = s["y"]
            else:
                y_tgt = s_next["x"]
            if x_list[k].size(0) != y_tgt.size(0):
                raise RuntimeError(
                    f"Node mismatch in window idx={idx}, step={k}: "
                    f"x has {x_list[k].size(0)}, target has {y_tgt.size(0)}."
                )
            y_list.append(y_tgt)

        out = {
            "x_list": x_list,
            "y_list": y_list,
            "pos_list": pos_list,
            "edge_index_list": edge_index_list,
            "t_list": t_list,
            "meta": {
                "seq_id": seq_id,
                "start_t": start,
                "window_size": self.window_size,
                "step_meta": [s.get("meta", {}) for s in chunk],
            },
            # Back-compat convenience for one-step paths.
            "x": x_list[0],
            "y": y_list[0],
            "pos": pos_list[0],
            "edge_index": edge_index_list[0],
            "t_src": t_list[0],
            "t_dst": t_list[1] if len(t_list) > 1 else None,
        }
        return out


def _build_dataset_for_mode(
    cfg: Dict[str, Any],
    *,
    use_window_mode: bool,
    pt_paths: Optional[Sequence[str]] = None,
) -> Dataset:
    if use_window_mode:
        return PointGraphWindowDataset(cfg, pt_paths=pt_paths)
    return PointGraphTemporalDataset(cfg, pt_paths=pt_paths)


def _assert_dataset_dims_match(ref_ds: Dataset, other_ds: Dataset, label: str) -> None:
    req = ("x_dim", "y_dim", "pos_dim")
    for k in req:
        if not hasattr(ref_ds, k) or not hasattr(other_ds, k):
            raise AttributeError(f"Dataset objects must expose '{k}' for split compatibility checks.")
    ref_dims = (int(getattr(ref_ds, "x_dim")), int(getattr(ref_ds, "y_dim")), int(getattr(ref_ds, "pos_dim")))
    oth_dims = (int(getattr(other_ds, "x_dim")), int(getattr(other_ds, "y_dim")), int(getattr(other_ds, "pos_dim")))
    if ref_dims != oth_dims:
        raise ValueError(
            f"Dataset dimension mismatch for split '{label}': ref(train) dims={ref_dims}, split dims={oth_dims}."
        )


def _collate_one(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise RuntimeError(
            "This baseline trainer currently supports batch_size=1 only "
            "(variable-size graph samples)."
        )
    return batch[0]


@dataclass
class NormStats:
    x_mu: Optional[torch.Tensor]
    x_std: Optional[torch.Tensor]
    y_mu: Optional[torch.Tensor]
    y_std: Optional[torch.Tensor]
    mode: str = "zscore"


def _normalization_mode_from_cfg(cfg: Dict[str, Any]) -> str:
    feats = cfg.get("features", {}) or {}
    norm_blk = feats.get("normalization", None)
    raw = None
    if isinstance(norm_blk, dict):
        raw = norm_blk.get("normalization_mode", None)
    if raw is None:
        raw = feats.get("normalization_mode", feats.get("norm_mode", "zscore"))
    mode = str(raw).strip().lower()
    if mode in ("zscore", "z_score", "standard", "standardize", "std"):
        return "zscore"
    if mode in (
        "minmax_11",
        "minmax",
        "min_max",
        "minus1_1",
        "neg1_1",
        "-1_1",
        "-1to1",
        "-1_to_1",
    ):
        return "minmax_11"
    raise ValueError(
        "Unsupported normalization mode. "
        "Use features.normalization.normalization_mode (or legacy "
        "features.normalization_mode) with value 'zscore' or 'minmax_11'."
    )


def _component_scale_mode_from_cfg(cfg: Dict[str, Any]) -> str:
    feats = cfg.get("features", {}) or {}
    raw = feats.get("component_scale_mode", feats.get("sigma_mode", "independent"))
    norm_blk = feats.get("normalization", None)
    if isinstance(norm_blk, dict):
        raw = norm_blk.get("velocity_sigma_mode", norm_blk.get("momentum_sigma_mode", raw))

    mode = str(raw).strip().lower()
    if mode in ("independent", "per_channel", "per-channel", "channelwise"):
        return "independent"
    if mode in ("shared", "shared_rms", "shared-rms", "tied"):
        return "shared"
    raise ValueError(
        "Unsupported component scale mode. "
        "Use 'independent' or 'shared' (e.g., features.normalization.velocity_sigma_mode)."
    )


@torch.no_grad()
def _compute_norm_stats(
    dataset: Dataset,
    indices: Sequence[int],
    device: torch.device,
    *,
    mode: str = "zscore",
    component_mode: str = "independent",
    shared_channels: Optional[Sequence[int]] = None,
    rollout_steps: Optional[int] = None,
) -> NormStats:
    mode = str(mode).strip().lower()
    if mode not in ("zscore", "minmax_11"):
        raise ValueError(f"Unsupported normalization mode: {mode}")
    component_mode = str(component_mode).strip().lower()
    if component_mode not in ("independent", "shared"):
        raise ValueError(f"Unsupported component scale mode: {component_mode}")

    x_sum = None
    x_sq_sum = None
    y_sum = None
    y_sq_sum = None
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    n_x = 0
    n_y = 0

    def _accum_xy(x_raw: torch.Tensor, y_raw: torch.Tensor) -> None:
        nonlocal x_sum, x_sq_sum, y_sum, y_sq_sum, x_min, x_max, y_min, y_max, n_x, n_y
        x = x_raw.to(device=device, dtype=torch.float32)
        y = y_raw.to(device=device, dtype=torch.float32)

        if mode == "zscore":
            if x_sum is None:
                x_sum = x.sum(dim=0)
                x_sq_sum = (x * x).sum(dim=0)
            else:
                x_sum += x.sum(dim=0)
                x_sq_sum += (x * x).sum(dim=0)

            if y_sum is None:
                y_sum = y.sum(dim=0)
                y_sq_sum = (y * y).sum(dim=0)
            else:
                y_sum += y.sum(dim=0)
                y_sq_sum += (y * y).sum(dim=0)
        else:
            x_lo = x.min(dim=0).values
            x_hi = x.max(dim=0).values
            y_lo = y.min(dim=0).values
            y_hi = y.max(dim=0).values
            if x_min is None:
                x_min = x_lo
                x_max = x_hi
                y_min = y_lo
                y_max = y_hi
            else:
                x_min = torch.minimum(x_min, x_lo)
                x_max = torch.maximum(x_max, x_hi)
                y_min = torch.minimum(y_min, y_lo)
                y_max = torch.maximum(y_max, y_hi)
        n_x += int(x.size(0))
        n_y += int(y.size(0))

    for idx in indices:
        ex = dataset[int(idx)]
        if ("x" in ex) and ("y" in ex) and torch.is_tensor(ex["x"]) and torch.is_tensor(ex["y"]):
            _accum_xy(ex["x"], ex["y"])
            continue

        x_list = ex.get("x_list", None)
        y_list = ex.get("y_list", None)
        if isinstance(x_list, list) and isinstance(y_list, list) and len(y_list) > 0:
            use_steps = len(y_list)
            if rollout_steps is not None:
                use_steps = min(use_steps, max(1, int(rollout_steps)))
            for k in range(use_steps):
                _accum_xy(x_list[k], y_list[k])
            continue

        raise KeyError("Dataset example must provide x/y tensors or x_list/y_list tensors.")

    if n_x == 0 or n_y == 0:
        return NormStats(None, None, None, None, mode=mode)

    if mode == "zscore":
        if x_sum is None or y_sum is None:
            return NormStats(None, None, None, None, mode=mode)
        x_mu = x_sum / float(n_x)
        y_mu = y_sum / float(n_y)
        x_var = (x_sq_sum / float(n_x)) - (x_mu * x_mu)
        y_var = (y_sq_sum / float(n_y)) - (y_mu * y_mu)
        x_std = torch.sqrt(torch.clamp(x_var, min=1e-12))
        y_std = torch.sqrt(torch.clamp(y_var, min=1e-12))
    else:
        if x_min is None or x_max is None or y_min is None or y_max is None:
            return NormStats(None, None, None, None, mode=mode)
        x_mu = 0.5 * (x_min + x_max)
        y_mu = 0.5 * (y_min + y_max)
        x_std = (0.5 * (x_max - x_min)).clamp_min(1e-12)
        y_std = (0.5 * (y_max - y_min)).clamp_min(1e-12)

    if component_mode == "shared" and shared_channels is not None:
        idx: List[int] = []
        for c in shared_channels:
            ci = int(c)
            if 0 <= ci < int(x_std.numel()) and ci not in idx:
                idx.append(ci)
        if len(idx) >= 2:
            idx_t = torch.as_tensor(idx, device=x_std.device, dtype=torch.long)
            x_sel = x_std.index_select(0, idx_t)
            y_sel = y_std.index_select(0, idx_t)

            if mode == "zscore":
                # Tie selected components by their RMS scale.
                x_shared = torch.sqrt(torch.mean(x_sel * x_sel)).clamp_min(1e-12)
                y_shared = torch.sqrt(torch.mean(y_sel * y_sel)).clamp_min(1e-12)
            else:
                # Keep minmax_11 bounded for all tied channels by using max half-range.
                x_shared = torch.max(x_sel).clamp_min(1e-12)
                y_shared = torch.max(y_sel).clamp_min(1e-12)

            x_std = x_std.clone()
            y_std = y_std.clone()
            x_std[idx_t] = x_shared
            y_std[idx_t] = y_shared

    return NormStats(x_mu, x_std, y_mu, y_std, mode=mode)


def _maybe_norm(x: torch.Tensor, mu: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mu is None or std is None:
        return x
    return (x - mu.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)


def _maybe_denorm(x: torch.Tensor, mu: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mu is None or std is None:
        return x
    mu_ = mu.to(device=x.device, dtype=x.dtype)
    std_ = std.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)
    return x * std_ + mu_


def _physics_inputs_enabled(cfg: Dict[str, Any]) -> bool:
    phys = _physics_cfg(cfg)
    include_adv = bool(phys.get("parc_include_adv", False))
    include_diff = bool(phys.get("parc_include_diff", False))
    return bool(include_adv or include_diff)


def _physics_backend(cfg: Dict[str, Any]) -> str:
    phys = _physics_cfg(cfg)
    backend = str(phys.get("physics_backend", "dec")).lower().strip()
    if backend in ("moving_least_squares", "moving-least-squares"):
        backend = "mls"
    if backend not in ("dec", "mls"):
        backend = "dec"
    return backend


def _feature_names_for_dim(cfg: Dict[str, Any], fdim: int) -> List[str]:
    feats = cfg.get("features", {}) or {}
    raw = feats.get("names", None)
    if not isinstance(raw, list) or len(raw) == 0:
        return [f"feat_{i}" for i in range(fdim)]

    names = [str(x) for x in raw]
    use_cols = feats.get("use_columns", None)
    if isinstance(use_cols, list) and len(use_cols) == fdim:
        mapped: List[str] = []
        ok = True
        for c in use_cols:
            ci = int(c)
            if ci < 0 or ci >= len(names):
                ok = False
                break
            mapped.append(names[ci])
        if ok:
            names = mapped

    if len(names) < fdim:
        names = names + [f"feat_{i}" for i in range(len(names), fdim)]
    return names[:fdim]


def _parse_channel_list(
    spec: Any,
    *,
    fdim: int,
    names: List[str],
    default: List[int],
) -> List[int]:
    if spec is None:
        return list(default)

    if not isinstance(spec, list):
        return list(default)

    out: List[int] = []
    name_map = {n.lower(): i for i, n in enumerate(names)}
    for item in spec:
        if isinstance(item, (int, np.integer)):
            j = int(item)
            if 0 <= j < fdim:
                out.append(j)
            continue
        s = str(item).strip().lower()
        if not s:
            continue
        if s in ("all", "*"):
            return list(range(fdim))
        if s in name_map:
            out.append(int(name_map[s]))
            continue
        # permissive fallback: exact token contained in channel name
        for i, nm in enumerate(names):
            if s == nm.lower() or s in nm.lower():
                out.append(i)
                break

    if len(out) == 0:
        return list(default)

    # de-dup preserve order
    seen = set()
    dedup: List[int] = []
    for j in out:
        if j not in seen:
            seen.add(j)
            dedup.append(j)
    return dedup


def _physics_channel_indices(cfg: Dict[str, Any], fdim: int, kind: str) -> List[int]:
    phys = _physics_cfg(cfg)
    names = _feature_names_for_dim(cfg, fdim)
    default = list(range(fdim))
    if kind == "adv":
        spec = (
            phys.get("parc_input_channels_adv", None)
            or phys.get("adv_channels", None)
            or phys.get("dec_adv_channels", None)
            or phys.get("parc_input_channels", None)
            or phys.get("channels", None)
        )
    else:
        spec = (
            phys.get("parc_input_channels_diff", None)
            or phys.get("diff_channels", None)
            or phys.get("dec_diff_channels", None)
            or phys.get("parc_input_channels", None)
            or phys.get("channels", None)
        )
    return _parse_channel_list(spec, fdim=fdim, names=names, default=default)


def _physics_extra_in_channels(cfg: Dict[str, Any], fdim: int) -> int:
    if not _physics_inputs_enabled(cfg):
        return 0
    phys = _physics_cfg(cfg)
    include_adv = bool(phys.get("parc_include_adv", False))
    include_diff = bool(phys.get("parc_include_diff", False))
    n = 0
    if include_adv:
        n += len(_physics_channel_indices(cfg, fdim, "adv"))
    if include_diff:
        n += len(_physics_channel_indices(cfg, fdim, "diff"))
    return int(n)


def _infer_velocity_columns(cfg: Dict[str, Any], fdim: int) -> Tuple[int, int]:
    phys = _physics_cfg(cfg)
    names = [n.lower() for n in _feature_names_for_dim(cfg, fdim)]

    vc = phys.get("velocity_channels", None)
    if isinstance(vc, list) and len(vc) >= 2:
        if all(isinstance(v, (int, np.integer)) for v in vc[:2]):
            i0 = int(vc[0]); i1 = int(vc[1])
            if 0 <= i0 < fdim and 0 <= i1 < fdim:
                return i0, i1
        # name-based velocity channels
        idx = []
        for v in vc[:2]:
            s = str(v).strip().lower()
            if s in names:
                idx.append(names.index(s))
            else:
                idx.append(-1)
        if idx[0] >= 0 and idx[1] >= 0:
            return int(idx[0]), int(idx[1])

    def _find(keys: Sequence[str], fallback: int) -> int:
        for k in keys:
            if k in names:
                return names.index(k)
        for i, nm in enumerate(names):
            if any(k in nm for k in keys):
                return i
        return fallback

    ix = _find(("velocity_x", "vel_x", "x_velocity", "ux"), 0)
    iy = _find(("velocity_y", "vel_y", "y_velocity", "uy"), 1 if fdim > 1 else 0)
    ix = max(0, min(ix, fdim - 1))
    iy = max(0, min(iy, fdim - 1))
    return int(ix), int(iy)


def _shared_component_channels_from_cfg(cfg: Dict[str, Any], fdim: int) -> List[int]:
    if fdim <= 0:
        return []

    feats = cfg.get("features", {}) or {}
    names = _feature_names_for_dim(cfg, fdim)
    norm_blk = feats.get("normalization", None)
    spec = feats.get("shared_channels", None)
    if spec is None and isinstance(norm_blk, dict):
        spec = norm_blk.get("shared_channels", None)

    if spec is not None:
        chans = _parse_channel_list(spec, fdim=fdim, names=names, default=[])
        out: List[int] = []
        for c in chans:
            ci = int(c)
            if 0 <= ci < fdim and ci not in out:
                out.append(ci)
        if len(out) >= 2:
            return out

    # Default shared pair: inferred velocity components.
    try:
        i0, i1 = _infer_velocity_columns(cfg, fdim)
        out = []
        for c in (i0, i1):
            ci = int(c)
            if 0 <= ci < fdim and ci not in out:
                out.append(ci)
        if len(out) >= 2:
            return out
    except Exception:
        pass

    if fdim >= 2:
        return [0, 1]
    return [0]


def _mls_sig_from_cfg(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    phys = _physics_cfg(cfg)
    return (
        bool(phys.get("mls_cache_by_geometry", False)),
        bool(phys.get("mls_use_2hop_extension", True)),
        bool(phys.get("mls_use_neighbor_damping", True)),
        float(phys.get("mls_damping_alpha", 0.5)),
        int(phys.get("mls_poly_order", 2)),
        int(phys.get("mls_min_neighbors", 6)),
    )


def _get_mls_ops(cfg: Dict[str, Any]):
    sig = _mls_sig_from_cfg(cfg)
    if _MLS_STATE["sig"] == sig and _MLS_STATE["adv"] is not None and _MLS_STATE["diff"] is not None:
        return _MLS_STATE["adv"], _MLS_STATE["diff"]

    cache_by_geometry, use_2hop, use_damp, alpha, poly_order, min_nbrs = sig
    grad = mls.SolveGradientsLST(
        cache_by_geometry=cache_by_geometry,
        use_2hop_extension=use_2hop,
        use_neighbor_damping=use_damp,
        damping_alpha=alpha,
    )
    lapw = mls.SolveWeightLST2d(
        polynomial_order=poly_order,
        min_neighbors=min_nbrs,
        cache_by_geometry=cache_by_geometry,
        use_2hop_extension=use_2hop,
        use_neighbor_damping=use_damp,
        damping_alpha=alpha,
    )
    adv = mls.AdvectionMLS(grad)
    diff = mls.DiffusionMLS(lapw)
    _MLS_STATE.update({"sig": sig, "grad": grad, "lapw": lapw, "adv": adv, "diff": diff})
    return adv, diff


def _geometry_from_pos_edge(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    # Do not cache geometry tensors here.
    #
    # Prior versions cached by (data_ptr, sizes, device). In CUDA training, each
    # batch/step does `.to(device)` and typically gets fresh device allocations,
    # so pointer-based keys almost never hit and the cache grows without bound.
    # That manifests as epoch-to-epoch GPU memory growth and eventual OOM.
    #
    # Geometry assembly is inexpensive relative to GNN forward/backward, so
    # recomputing per-step is safer and avoids retaining large CUDA tensors.

    if pos.size(1) < 2:
        raise ValueError(f"Physics operators require at least 2D positions; got pos shape {tuple(pos.shape)}")

    src = edge_index[0].long()
    dst = edge_index[1].long()
    pxy = pos[:, :2].to(dtype=torch.float32)
    dxy = pxy[dst] - pxy[src]
    dist = torch.linalg.norm(dxy, dim=1).clamp_min(1e-12)

    nx = dxy[:, 0] / dist
    ny = dxy[:, 1] / dist
    face_len = dist
    dual_len = dist
    tau = (face_len / dual_len).clamp_min(1e-12)

    # Pseudo-cell area from local edge geometry:
    # A_i ~= 0.5 * sum_j (face_len_ij * dual_len_ij) over outgoing edges.
    contrib = 0.5 * (face_len * dual_len)
    area = torch.zeros((pos.size(0),), device=pos.device, dtype=torch.float32)
    area.index_add_(0, src, contrib)
    pos_area = area > 0
    if bool(pos_area.any()):
        fallback = torch.median(area[pos_area]).clamp_min(1e-12)
    else:
        fallback = torch.tensor(1.0, device=pos.device, dtype=torch.float32)
    area = torch.where(pos_area, area, fallback)
    area = area.clamp_min(1e-12)

    return {
        "nx": nx,
        "ny": ny,
        "face_len": face_len,
        "dual_len": dual_len,
        "tau": tau,
        "area": area,
    }


def _velocity_from_features_or_state(x_abs: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    phys = _physics_cfg(cfg)
    adv_type = str(phys.get("advection_type", "scalar")).lower()
    if adv_type == "euler":
        fdim = int(x_abs.size(1))
        idx = dec.infer_feature_indices(cfg, fdim)
        rho = x_abs[:, idx["rho"]]
        mx = x_abs[:, idx["mx"]]
        my = x_abs[:, idx["my"]]
        rho_floor = float(phys.get("rho_floor", 1e-6))
        rho_eps = float(phys.get("rho_eps", 1e-8))
        rho_safe = rho.clamp_min(max(rho_floor, rho_eps))
        vel = torch.stack([mx / rho_safe, my / rho_safe], dim=1)
        u_clip = float(phys.get("u_clip", 1e3))
        if u_clip > 0:
            vel = torch.clamp(vel, min=-u_clip, max=u_clip)
        return vel

    fdim = int(x_abs.size(1))
    ix, iy = _infer_velocity_columns(cfg, fdim)
    vel = torch.stack([x_abs[:, ix], x_abs[:, iy]], dim=1)
    u_clip = float(phys.get("u_clip", 1e3))
    if u_clip > 0:
        vel = torch.clamp(vel, min=-u_clip, max=u_clip)
    return vel


def _physics_terms_dec_abs_point(
    *,
    x_abs: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: Dict[str, Any],
    compute_adv: bool,
    compute_diff: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    geom = _geometry_from_pos_edge(pos, edge_index)
    nx = geom["nx"].to(device=x_abs.device, dtype=x_abs.dtype)
    ny = geom["ny"].to(device=x_abs.device, dtype=x_abs.dtype)
    face_len = geom["face_len"].to(device=x_abs.device, dtype=x_abs.dtype)
    tau = geom["tau"].to(device=x_abs.device, dtype=x_abs.dtype)
    area = geom["area"].to(device=x_abs.device, dtype=x_abs.dtype)

    N, fdim = x_abs.shape
    phys = _physics_cfg(cfg)
    sel_adv = _physics_channel_indices(cfg, fdim, "adv")
    sel_diff = _physics_channel_indices(cfg, fdim, "diff")
    adv_type = str(phys.get("advection_type", "scalar")).lower()

    r_adv = x_abs.new_zeros((N, fdim))
    r_diff = x_abs.new_zeros((N, fdim))

    if compute_adv and len(sel_adv) > 0:
        if adv_type == "euler":
            scheme = str(phys.get("euler_flux_scheme", "rusanov")).lower()
            rho_eps = float(phys.get("rho_eps", 1e-8))
            cfg_dec = dict(cfg)
            cfg_dec["loss"] = dict(phys)
            div_full = dec.dec_divergence_euler_flux(
                x_abs=x_abs,
                edge_index=edge_index,
                nx=nx,
                ny=ny,
                face_len=face_len,
                area=area,
                cfg=cfg_dec,
                scheme=scheme,
                eps=rho_eps,
            )  # [N,4] in [rho,mx,my,E]
            idx = dec.infer_feature_indices(cfg, fdim)
            euler_idx = [idx["rho"], idx["mx"], idx["my"], idx["E"]]
            col_map = {j: k for k, j in enumerate(euler_idx)}
            sel = [j for j in sel_adv if j in col_map]
            if len(sel) > 0:
                cols = [col_map[j] for j in sel]
                r_adv[:, sel] = -div_full[:, cols]
        else:
            vel = _velocity_from_features_or_state(x_abs, cfg)
            phi_adv = x_abs[:, sel_adv]
            scheme = str(phys.get("advection_scheme", "upwind")).lower()
            div_adv = dec.dec_divergence_advective_flux(
                phi=phi_adv,
                vel=vel,
                edge_index=edge_index,
                nx=nx,
                ny=ny,
                face_len=face_len,
                area=area,
                scheme=scheme,
            )
            r_adv[:, sel_adv] = -div_adv

    if compute_diff and len(sel_diff) > 0:
        nu_full = dec.as_nu_tensor(phys.get("nu", 0.0), fdim, device=x_abs.device, dtype=x_abs.dtype)
        nu_sel = nu_full[torch.as_tensor(sel_diff, device=x_abs.device)].view(1, -1)
        phi_diff = x_abs[:, sel_diff]
        lap = dec.dec_laplacian(phi_diff, edge_index=edge_index, tau=tau, area=area)
        r_diff[:, sel_diff] = lap * nu_sel

    out_adv = r_adv if compute_adv else None
    out_diff = r_diff if compute_diff else None
    return out_adv, out_diff, area


@torch.no_grad()
def _physics_terms_mls_abs_point(
    *,
    x_abs: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: Dict[str, Any],
    compute_adv: bool,
    compute_diff: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    phys = _physics_cfg(cfg)
    ops_dev = torch.device(str(phys.get("mls_ops_device", "cpu")))
    x_ops = x_abs.to(device=ops_dev, dtype=torch.float32)
    pos_ops = pos[:, :2].to(device=ops_dev, dtype=torch.float32)
    ei_ops = edge_index.to(device=ops_dev, dtype=torch.long)
    data = Data(pos=pos_ops, edge_index=ei_ops)

    vel = _velocity_from_features_or_state(x_ops, cfg).to(device=ops_dev, dtype=torch.float32)
    adv_op, diff_op = _get_mls_ops(cfg)

    r_adv = None
    if compute_adv:
        r_adv = adv_op(x_ops, vel, data)
    r_diff = None
    if compute_diff:
        r_diff = diff_op(x_ops, data)
        nu = phys.get("nu", 0.0)
        nu_full = dec.as_nu_tensor(nu, x_ops.size(1), device=x_ops.device, dtype=x_ops.dtype)
        r_diff = r_diff * nu_full.view(1, -1)

    # restrict to selected channels for parity with DEC path
    fdim = int(x_abs.size(1))
    sel_adv = _physics_channel_indices(cfg, fdim, "adv")
    sel_diff = _physics_channel_indices(cfg, fdim, "diff")
    if r_adv is not None:
        keep = torch.zeros_like(r_adv)
        keep[:, sel_adv] = r_adv[:, sel_adv]
        r_adv = keep
    if r_diff is not None:
        keep = torch.zeros_like(r_diff)
        keep[:, sel_diff] = r_diff[:, sel_diff]
        r_diff = keep

    geom = _geometry_from_pos_edge(pos, edge_index)
    area = geom["area"].to(device=x_abs.device, dtype=torch.float32)
    if r_adv is not None:
        r_adv = r_adv.to(device=x_abs.device, dtype=torch.float32)
    if r_diff is not None:
        r_diff = r_diff.to(device=x_abs.device, dtype=torch.float32)
    return r_adv, r_diff, area


def _safe_dt_scalar(t_src: Any, t_dst: Any, default_dt: float = 1.0) -> float:
    try:
        if t_src is None or t_dst is None:
            return float(default_dt)
        dt = float(t_dst) - float(t_src)
        if not np.isfinite(dt):
            return float(default_dt)
        dt = abs(dt)
        if dt <= 0.0:
            return float(default_dt)
        return float(dt)
    except Exception:
        return float(default_dt)


@torch.no_grad()
def _build_physics_extra_features(
    *,
    x_abs: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    dt_phys_scalar: float,
    cfg: Dict[str, Any],
    norm: NormStats,
    out_dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not _physics_inputs_enabled(cfg):
        return None

    phys = _physics_cfg(cfg)
    include_adv = bool(phys.get("parc_include_adv", False))
    include_diff = bool(phys.get("parc_include_diff", False))
    adv_w = float(phys.get("adv_weight", 1.0))
    diff_w = float(phys.get("diff_weight", 1.0))
    need_adv = include_adv and (adv_w != 0.0)
    need_diff = include_diff and (diff_w != 0.0)
    if not need_adv and not need_diff:
        return None

    x_abs_f = x_abs.to(device=device, dtype=torch.float32)
    pos_f = pos.to(device=device, dtype=torch.float32)
    ei_f = edge_index.to(device=device, dtype=torch.long)

    backend = _physics_backend(cfg)
    if backend == "mls":
        r_adv_abs, r_diff_abs, _area = _physics_terms_mls_abs_point(
            x_abs=x_abs_f,
            pos=pos_f,
            edge_index=ei_f,
            cfg=cfg,
            compute_adv=need_adv,
            compute_diff=need_diff,
        )
    else:
        r_adv_abs, r_diff_abs, _area = _physics_terms_dec_abs_point(
            x_abs=x_abs_f,
            pos=pos_f,
            edge_index=ei_f,
            cfg=cfg,
            compute_adv=need_adv,
            compute_diff=need_diff,
        )

    # Optional weighting before conversion, matching shock-ramp option.
    weighted = bool(phys.get("parc_input_weighted", False))
    if weighted:
        if r_adv_abs is not None:
            r_adv_abs = adv_w * r_adv_abs
        if r_diff_abs is not None:
            r_diff_abs = diff_w * r_diff_abs

    sigma = None if norm.y_std is None else norm.y_std.to(device=device, dtype=torch.float32)
    dt_phys = torch.tensor(max(1e-12, abs(float(dt_phys_scalar))), device=device, dtype=torch.float32)
    dt_ref_cfg = phys.get("dt_ref", None)
    dt_ref = None
    if dt_ref_cfg is not None:
        try:
            dt_ref = torch.tensor(float(dt_ref_cfg), device=device, dtype=torch.float32)
        except Exception:
            dt_ref = None
    form = str(phys.get("parc_input_form", "rate")).lower()
    predict_type = str(phys.get("parc_predict_type", "rate")).lower()

    fdim = int(x_abs.size(1))
    sel_adv = _physics_channel_indices(cfg, fdim, "adv")
    sel_diff = _physics_channel_indices(cfg, fdim, "diff")
    blocks: List[torch.Tensor] = []

    def _to_units(r_abs: torch.Tensor) -> torch.Tensor:
        if form == "delta":
            if sigma is None:
                return dt_phys * r_abs
            return (dt_phys * r_abs) / sigma.view(1, -1).clamp_min(1e-12)
        return dec.physics_to_model_units(
            r_abs,
            dt_phys=dt_phys,
            dt_ref=dt_ref,
            sigma=sigma,
            predict_type=predict_type,
        )

    if include_adv and (r_adv_abs is not None) and len(sel_adv) > 0:
        adv_u = _to_units(r_adv_abs.to(dtype=torch.float32))
        blocks.append(adv_u[:, sel_adv])
    if include_diff and (r_diff_abs is not None) and len(sel_diff) > 0:
        diff_u = _to_units(r_diff_abs.to(dtype=torch.float32))
        blocks.append(diff_u[:, sel_diff])

    if len(blocks) == 0:
        return None

    out = torch.cat(blocks, dim=1).to(device=device, dtype=out_dtype)
    if bool(phys.get("parc_detach_inputs", True)):
        out = out.detach()
    return out


def _build_model(cfg: Dict[str, Any], in_dim: int, out_dim: int, device: torch.device) -> FeatureNet:
    mcfg = cfg.get("model", {}) or {}
    model = FeatureNet(
        in_channels=in_dim,
        out_channels=out_dim,
        hidden=int(mcfg.get("hidden", 128)),
        layers=int(mcfg.get("layers", 3)),
        dropout=float(mcfg.get("dropout", 0.1)),
        make_score_head=False,
    ).to(device)
    return model


def _run_epoch(
    model: FeatureNet,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    *,
    cfg: Dict[str, Any],
    device: torch.device,
    include_pos: bool,
    norm: NormStats,
    use_huber: bool,
    huber_delta: float,
    grad_clip: float,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    mae_sum = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        pos = batch["pos"].to(device=device, dtype=torch.float32)
        ei = batch["edge_index"].to(device=device, dtype=torch.long)
        dt_phys = _safe_dt_scalar(batch.get("t_src", None), batch.get("t_dst", None), default_dt=1.0)

        x_in = _maybe_norm(x, norm.x_mu, norm.x_std)
        y_tgt = _maybe_norm(y, norm.y_mu, norm.y_std)

        x_parts = [x_in]
        if include_pos:
            x_parts.append(pos)

        phy_extra = _build_physics_extra_features(
            x_abs=x,
            pos=pos,
            edge_index=ei,
            dt_phys_scalar=dt_phys,
            cfg=cfg,
            norm=norm,
            out_dtype=x_in.dtype,
            device=device,
        )
        if phy_extra is not None and phy_extra.numel() > 0:
            x_parts.append(phy_extra)
        x_model = torch.cat(x_parts, dim=1)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        y_pred_norm, _score, _h = model(x_model, ei)
        if use_huber:
            loss = F.huber_loss(y_pred_norm, y_tgt, delta=float(huber_delta))
        else:
            loss = F.mse_loss(y_pred_norm, y_tgt)

        if train_mode:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

        with torch.no_grad():
            y_pred = _maybe_denorm(y_pred_norm, norm.y_mu, norm.y_std)
            mae = (y_pred - y).abs().mean()

        loss_sum += float(loss.detach().cpu().item())
        mae_sum += float(mae.detach().cpu().item())
        n += 1

    if n == 0:
        return float("nan"), float("nan")
    return loss_sum / n, mae_sum / n


def _run_epoch_multi_step(
    model: FeatureNet,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    *,
    cfg: Dict[str, Any],
    device: torch.device,
    include_pos: bool,
    norm: NormStats,
    use_huber: bool,
    huber_delta: float,
    grad_clip: float,
    rollout_steps: int,
    autoregressive: bool,
) -> Tuple[float, float, Dict[str, int]]:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0
    n_steps = 0
    n_windows = 0

    warned_input_shape = False
    warned_chain_shape = False

    for batch in loader:
        x_list = batch.get("x_list", None)
        y_list = batch.get("y_list", None)
        pos_list = batch.get("pos_list", None)
        edge_index_list = batch.get("edge_index_list", None)
        t_list = batch.get("t_list", None)
        if not isinstance(x_list, list) or not isinstance(y_list, list):
            raise RuntimeError("Multi-step mode requires batch keys: x_list, y_list, pos_list, edge_index_list.")

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        max_roll = min(int(rollout_steps), len(y_list), max(0, len(x_list) - 1))
        if max_roll < 1:
            raise RuntimeError("Window has no transitions to train on; check window_size and rollout_steps.")

        window_loss_graph = None
        x_roll_abs: Optional[torch.Tensor] = None

        for k in range(max_roll):
            x_teacher = x_list[k].to(device=device, dtype=torch.float32)
            y_tgt_abs = y_list[k].to(device=device, dtype=torch.float32)
            pos = pos_list[k].to(device=device, dtype=torch.float32)
            ei = edge_index_list[k].to(device=device, dtype=torch.long)

            # Shock-ramp style chaining: model output at k feeds input at k+1.
            if autoregressive and (k > 0) and (x_roll_abs is not None):
                x_in_abs = x_roll_abs
            else:
                x_in_abs = x_teacher

            # Fallback to teacher if an autoregressive shape mismatch appears.
            if x_in_abs.shape != x_teacher.shape:
                if autoregressive and (not warned_input_shape):
                    print(
                        "[WARN] Autoregressive input shape mismatch; falling back to teacher input for that step. "
                        "This usually means node ordering/count differs across timesteps."
                    )
                    warned_input_shape = True
                x_in_abs = x_teacher

            if x_in_abs.size(0) != pos.size(0) or x_in_abs.size(0) != y_tgt_abs.size(0):
                if autoregressive and (x_teacher.size(0) == pos.size(0) == y_tgt_abs.size(0)):
                    if not warned_input_shape:
                        print(
                            "[WARN] Autoregressive node-count mismatch; using teacher input for that step."
                        )
                        warned_input_shape = True
                    x_in_abs = x_teacher
                else:
                    raise RuntimeError(
                        f"Node mismatch at step k={k}: input={x_in_abs.size(0)} "
                        f"pos={pos.size(0)} target={y_tgt_abs.size(0)}"
                    )

            x_in = _maybe_norm(x_in_abs, norm.x_mu, norm.x_std)
            y_tgt = _maybe_norm(y_tgt_abs, norm.y_mu, norm.y_std)
            if isinstance(t_list, list) and (k + 1) < len(t_list):
                dt_phys = _safe_dt_scalar(t_list[k], t_list[k + 1], default_dt=1.0)
            else:
                dt_phys = 1.0

            x_parts = [x_in]
            if include_pos:
                x_parts.append(pos)
            phy_extra = _build_physics_extra_features(
                x_abs=x_in_abs,
                pos=pos,
                edge_index=ei,
                dt_phys_scalar=dt_phys,
                cfg=cfg,
                norm=norm,
                out_dtype=x_in.dtype,
                device=device,
            )
            if phy_extra is not None and phy_extra.numel() > 0:
                x_parts.append(phy_extra)
            x_model = torch.cat(x_parts, dim=1)

            with torch.set_grad_enabled(train_mode):
                y_pred_norm, _score, _h = model(x_model, ei)
                if use_huber:
                    loss_k = F.huber_loss(y_pred_norm, y_tgt, delta=float(huber_delta))
                else:
                    loss_k = F.mse_loss(y_pred_norm, y_tgt)

            if train_mode:
                window_loss_graph = loss_k if window_loss_graph is None else (window_loss_graph + loss_k)

            with torch.no_grad():
                y_pred_abs = _maybe_denorm(y_pred_norm.detach(), norm.y_mu, norm.y_std)
                mae_k = float((y_pred_abs - y_tgt_abs).abs().mean().cpu().item())

            total_loss += float(loss_k.detach().cpu().item())
            total_mae += mae_k
            n_steps += 1

            # Chain predicted state forward when shapes allow it.
            if autoregressive and (k + 1 < max_roll):
                next_teacher = x_list[k + 1]
                if y_pred_abs.shape == next_teacher.shape:
                    x_roll_abs = y_pred_abs.detach()
                else:
                    if not warned_chain_shape:
                        print(
                            f"[WARN] Cannot chain prediction at step k={k}: "
                            f"pred shape {tuple(y_pred_abs.shape)} != next input shape {tuple(next_teacher.shape)}. "
                            "Using teacher input for following step."
                        )
                        warned_chain_shape = True
                    x_roll_abs = None
            else:
                x_roll_abs = None

        if train_mode:
            if window_loss_graph is not None:
                window_loss_graph.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

        n_windows += 1

    denom = max(1, n_steps)
    return total_loss / denom, total_mae / denom, {"num_windows": n_windows, "num_steps": n_steps}


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg_raw = copy.deepcopy(cfg)

    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    device = torch.device(str(cfg.get("device", "cpu")))

    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    split_cfg = cfg.get("split", {}) or {}
    feat_cfg = cfg.get("features", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    physics_cfg = _physics_cfg(cfg)

    window_size = int(train_cfg.get("window_size", 2))
    stride = int(train_cfg.get("stride", 1))
    reverse_time = bool(data_cfg.get("reverse_time", False))
    if "multi_step_K" in train_cfg:
        raise ValueError(
            "train.multi_step_K is no longer supported. "
            "Control multi-step rollout length via train.window_size (rollout_steps=window_size-1)."
        )
    if ("window_size" in data_cfg) or ("stride" in data_cfg):
        print(
            "[WARN] data.window_size/data.stride are deprecated and ignored. "
            "Use train.window_size/train.stride."
        )
    legacy_phy_keys = {
        "physics_backend", "parc_include_adv", "parc_include_diff",
        "advection_type", "velocity_channels", "advection_scheme", "euler_flux_scheme",
        "adv_weight", "diff_weight", "nu", "parc_input_form", "parc_predict_type",
        "parc_detach_inputs", "parc_input_weighted", "mls_ops_device", "mls_use_2hop_extension",
        "mls_use_neighbor_damping", "mls_damping_alpha", "mls_poly_order", "mls_min_neighbors",
        "mls_cache_by_geometry", "rho_floor", "rho_eps", "u_clip",
    }
    if any(k in loss_cfg for k in legacy_phy_keys):
        print(
            "[WARN] physics keys found under cfg.loss; they are ignored. "
            "Move them under cfg.physics."
        )
    multi_step_autoreg = bool(train_cfg.get("autoregressive", True))
    use_window_mode = (window_size > 2)
    f_train = float(split_cfg.get("train", 0.8))
    f_val = float(split_cfg.get("val", 0.1))

    pt_source = _resolve_pt_source(str(data_cfg.get("pt_path", "")).strip())
    source_mode = str(pt_source.get("mode", "single_file"))
    use_pre_split_dirs = (source_mode == "pre_split_dir")
    if use_pre_split_dirs:
        print("[INFO] detected pre-split data directory; cfg.split train/val fractions will be ignored.")
    if reverse_time:
        print("[INFO] data.reverse_time=true; temporal order will be reversed before pair/window construction.")

    if use_window_mode:
        rollout_steps = max(1, window_size - 1)
        if bool(data_cfg.get("use_y_as_target", True)):
            print(
                "[WARN] data.use_y_as_target=true with autoregressive multi-step training. "
                "This is only semantically correct if y_t is the next-state target."
            )
    else:
        rollout_steps = 1
        multi_step_autoreg = False

    include_pos = bool(feat_cfg.get("include_pos", True))
    normalize = bool(feat_cfg.get("normalize", True))
    norm_mode = _normalization_mode_from_cfg(cfg)
    component_mode = _component_scale_mode_from_cfg(cfg)
    use_huber = bool(loss_cfg.get("use_huber", False))
    huber_delta = float(loss_cfg.get("huber_delta", 0.05))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    save_dir = os.path.expanduser(str(train_cfg.get("save_dir", "./runs_basic_point")))
    os.makedirs(save_dir, exist_ok=True)

    split_files: Optional[Dict[str, List[str]]] = None
    source_file: Optional[str] = None
    if use_pre_split_dirs:
        split_files = pt_source.get("split_files", None)
        if not isinstance(split_files, dict):
            raise RuntimeError("pt source resolution failed for pre-split directory mode.")
        train_ds = _build_dataset_for_mode(cfg, use_window_mode=use_window_mode, pt_paths=split_files["train"])
        val_ds = _build_dataset_for_mode(cfg, use_window_mode=use_window_mode, pt_paths=split_files["val"])
        test_ds = _build_dataset_for_mode(cfg, use_window_mode=use_window_mode, pt_paths=split_files["test"])
        _assert_dataset_dims_match(train_ds, val_ds, label="val")
        _assert_dataset_dims_match(train_ds, test_ds, label="test")
        dataset_for_dims = train_ds
        n_total = len(train_ds) + len(val_ds) + len(test_ds)
        norm_dataset = train_ds
        norm_indices = np.arange(len(train_ds), dtype=np.int64)
    else:
        source_file = str(pt_source.get("path", ""))
        dataset = _build_dataset_for_mode(cfg, use_window_mode=use_window_mode)
        dataset_for_dims = dataset
        n_total = len(dataset)
        idxs = np.arange(n_total)
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)

        n_train = int(round(f_train * n_total))
        n_val = int(round(f_val * n_total))
        n_train = max(1, min(n_train, n_total - 1))
        n_val = max(1, min(n_val, n_total - n_train))

        train_idx = idxs[:n_train]
        val_idx = idxs[n_train : n_train + n_val]
        test_idx = idxs[n_train + n_val :]
        if len(test_idx) == 0:
            test_idx = val_idx[:1]

        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())
        test_ds = Subset(dataset, test_idx.tolist())
        norm_dataset = dataset
        norm_indices = train_idx

    batch_size = int(train_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("train_basic_point.py currently requires train.batch_size=1.")

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=RandomSampler(train_ds),
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=False,
        collate_fn=_collate_one,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=RandomSampler(val_ds),
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=False,
        collate_fn=_collate_one,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        sampler=RandomSampler(test_ds),
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=False,
        collate_fn=_collate_one,
    )

    shared_channels = _shared_component_channels_from_cfg(cfg, int(dataset_for_dims.x_dim))

    # Normalization stats from train split.
    if normalize:
        stats = _compute_norm_stats(
            norm_dataset,
            norm_indices,
            device=device,
            mode=norm_mode,
            component_mode=component_mode,
            shared_channels=shared_channels,
            rollout_steps=(rollout_steps if use_window_mode else 1),
        )
    else:
        stats = NormStats(None, None, None, None, mode=norm_mode)

    physics_extra_dim = _physics_extra_in_channels(cfg, dataset_for_dims.x_dim)
    in_dim = (
        dataset_for_dims.x_dim
        + (dataset_for_dims.pos_dim if include_pos else 0)
        + int(physics_extra_dim)
    )
    out_dim = dataset_for_dims.y_dim
    model = _build_model(cfg, in_dim=in_dim, out_dim=out_dim, device=device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 50))
    val_every = int(train_cfg.get("validation_every_epochs", 1))
    val_every = max(1, val_every)

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_model.pt")
    final_path = os.path.join(save_dir, "final_model.pt")
    log_path = os.path.join(save_dir, "train_log.csv")

    with open(log_path, "w") as f:
        f.write("epoch,split,loss,mae\n")

    if use_pre_split_dirs:
        n_train_files = len(split_files["train"]) if split_files is not None else 0
        n_val_files = len(split_files["val"]) if split_files is not None else 0
        n_test_files = len(split_files["test"]) if split_files is not None else 0
        if use_window_mode:
            print(
                f"[INFO] dataset windows={n_total} "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) "
                f"files(train={n_train_files}, val={n_val_files}, test={n_test_files}) "
                f"x_dim={dataset_for_dims.x_dim} y_dim={dataset_for_dims.y_dim} pos_dim={dataset_for_dims.pos_dim} "
                f"window_size={window_size} stride={stride} rollout_steps={rollout_steps} "
                f"autoregressive={multi_step_autoreg} reverse_time={reverse_time}"
            )
        else:
            print(
                f"[INFO] dataset pairs={n_total} "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) "
                f"files(train={n_train_files}, val={n_val_files}, test={n_test_files}) "
                f"x_dim={dataset_for_dims.x_dim} y_dim={dataset_for_dims.y_dim} pos_dim={dataset_for_dims.pos_dim} "
                f"reverse_time={reverse_time}"
            )
    else:
        if use_window_mode:
            print(
                f"[INFO] dataset windows={n_total} "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) "
                f"x_dim={dataset_for_dims.x_dim} y_dim={dataset_for_dims.y_dim} pos_dim={dataset_for_dims.pos_dim} "
                f"window_size={window_size} stride={stride} rollout_steps={rollout_steps} "
                f"autoregressive={multi_step_autoreg} reverse_time={reverse_time}"
            )
        else:
            print(
                f"[INFO] dataset pairs={n_total} "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) "
                f"x_dim={dataset_for_dims.x_dim} y_dim={dataset_for_dims.y_dim} pos_dim={dataset_for_dims.pos_dim} "
                f"reverse_time={reverse_time}"
            )
    if normalize and stats.x_mu is not None:
        print(
            f"[INFO] normalization enabled "
            f"(mode={stats.mode}, component_mode={component_mode}; train split stats computed)."
        )
        if component_mode == "shared":
            print(f"[INFO] shared component scaling channels={shared_channels}")
    else:
        print(
            f"[INFO] normalization disabled "
            f"(configured mode={norm_mode}, component_mode={component_mode})."
        )
    if _physics_inputs_enabled(cfg):
        print(
            f"[INFO] physics inputs enabled: backend={_physics_backend(cfg)} "
            f"extra_in_channels={physics_extra_dim}"
        )

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        if use_window_mode:
            tr_loss, tr_mae, _tr_stats = _run_epoch_multi_step(
                model,
                train_loader,
                optimizer,
                cfg=cfg,
                device=device,
                include_pos=include_pos,
                norm=stats,
                use_huber=use_huber,
                huber_delta=huber_delta,
                grad_clip=grad_clip,
                rollout_steps=rollout_steps,
                autoregressive=multi_step_autoreg,
            )
        else:
            tr_loss, tr_mae = _run_epoch(
                model,
                train_loader,
                optimizer,
                cfg=cfg,
                device=device,
                include_pos=include_pos,
                norm=stats,
                use_huber=use_huber,
                huber_delta=huber_delta,
                grad_clip=grad_clip,
            )

        run_val = (ep % val_every == 0) or (ep == epochs)
        if run_val:
            if use_window_mode:
                va_loss, va_mae, _va_stats = _run_epoch_multi_step(
                    model,
                    val_loader,
                    None,
                    cfg=cfg,
                    device=device,
                    include_pos=include_pos,
                    norm=stats,
                    use_huber=use_huber,
                    huber_delta=huber_delta,
                    grad_clip=0.0,
                    rollout_steps=rollout_steps,
                    autoregressive=multi_step_autoreg,
                )
            else:
                va_loss, va_mae = _run_epoch(
                    model,
                    val_loader,
                    None,
                    cfg=cfg,
                    device=device,
                    include_pos=include_pos,
                    norm=stats,
                    use_huber=use_huber,
                    huber_delta=huber_delta,
                    grad_clip=0.0,
                )
            if va_loss < best_val:
                best_val = va_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg,
                        "norm": {
                            "mode": str(stats.mode),
                            "component_mode": str(component_mode),
                            "x_mu": None if stats.x_mu is None else stats.x_mu.detach().cpu(),
                            "x_std": None if stats.x_std is None else stats.x_std.detach().cpu(),
                            "y_mu": None if stats.y_mu is None else stats.y_mu.detach().cpu(),
                            "y_std": None if stats.y_std is None else stats.y_std.detach().cpu(),
                        },
                        "dims": {"in_dim": in_dim, "out_dim": out_dim},
                    },
                    best_path,
                )
        else:
            va_loss, va_mae = float("nan"), float("nan")

        with open(log_path, "a") as f:
            f.write(f"{ep},train,{tr_loss:.10f},{tr_mae:.10f}\n")
            if run_val:
                f.write(f"{ep},val,{va_loss:.10f},{va_mae:.10f}\n")

        ep_s = time.perf_counter() - ep_t0
        if run_val:
            print(
                f"[E{ep:04d}] "
                f"train loss={tr_loss:.6f} mae={tr_mae:.6f} | "
                f"val loss={va_loss:.6f} mae={va_mae:.6f} | "
                f"epoch_time={ep_s:.2f}s"
            )
        else:
            print(
                f"[E{ep:04d}] train loss={tr_loss:.6f} mae={tr_mae:.6f} | "
                f"epoch_time={ep_s:.2f}s"
            )

    # Final + test.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "norm": {
                "mode": str(stats.mode),
                "component_mode": str(component_mode),
                "x_mu": None if stats.x_mu is None else stats.x_mu.detach().cpu(),
                "x_std": None if stats.x_std is None else stats.x_std.detach().cpu(),
                "y_mu": None if stats.y_mu is None else stats.y_mu.detach().cpu(),
                "y_std": None if stats.y_std is None else stats.y_std.detach().cpu(),
            },
            "dims": {"in_dim": in_dim, "out_dim": out_dim},
        },
        final_path,
    )

    # Evaluate best checkpoint if available.
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    if use_window_mode:
        te_loss, te_mae, _te_stats = _run_epoch_multi_step(
            model,
            test_loader,
            None,
            cfg=cfg,
            device=device,
            include_pos=include_pos,
            norm=stats,
            use_huber=use_huber,
            huber_delta=huber_delta,
            grad_clip=0.0,
            rollout_steps=rollout_steps,
            autoregressive=multi_step_autoreg,
        )
    else:
        te_loss, te_mae = _run_epoch(
            model,
            test_loader,
            None,
            cfg=cfg,
            device=device,
            include_pos=include_pos,
            norm=stats,
            use_huber=use_huber,
            huber_delta=huber_delta,
            grad_clip=0.0,
        )

    split_file_paths = (
        {k: [os.path.abspath(p) for p in v] for k, v in split_files.items()}
        if split_files is not None
        else None
    )
    summary = {
        "config_path": os.path.abspath(os.path.expanduser(config_path)),
        "config_raw": cfg_raw,
        "config_effective": {
            "seed": int(seed),
            "device": str(device),
            "data": {
                "pt_path": str(data_cfg.get("pt_path", "")),
                "source_mode": source_mode,
                "source_file": (None if source_file is None else os.path.abspath(source_file)),
                "source_root": (
                    None
                    if not use_pre_split_dirs
                    else os.path.abspath(str(pt_source.get("root", data_cfg.get("pt_path", ""))))
                ),
                "split_file_counts": (
                    None if split_files is None else {k: len(v) for k, v in split_files.items()}
                ),
                "num_workers": int(data_cfg.get("num_workers", 0)),
                "reverse_time": bool(reverse_time),
                "use_y_as_target": bool(data_cfg.get("use_y_as_target", True)),
                "use_y_for_input": bool(data_cfg.get("use_y_for_input", False)),
                "x_input_columns": data_cfg.get("x_input_columns", None),
                "target_columns": data_cfg.get("target_columns", None),
                "split_by_z": bool(data_cfg.get("split_by_z", False)),
                "z_index": data_cfg.get("z_index", None),
                "z_tol": float(data_cfg.get("z_tol", 0.0)),
            },
            "split": {
                "train": float(f_train),
                "val": float(f_val),
            },
            "features": {
                "include_pos": bool(include_pos),
                "normalize": bool(normalize),
                "normalization_mode": str(norm_mode),
                "component_scale_mode": str(component_mode),
                "shared_channels": [int(c) for c in shared_channels],
            },
            "loss": {
                "use_huber": bool(use_huber),
                "huber_delta": float(huber_delta),
            },
            "physics": {
                "physics_backend": str(physics_cfg.get("physics_backend", "dec")),
                "parc_include_adv": bool(physics_cfg.get("parc_include_adv", False)),
                "parc_include_diff": bool(physics_cfg.get("parc_include_diff", False)),
                "advection_type": str(physics_cfg.get("advection_type", "scalar")),
                "velocity_channels": physics_cfg.get("velocity_channels", None),
                "advection_scheme": str(physics_cfg.get("advection_scheme", "upwind")),
                "euler_flux_scheme": str(physics_cfg.get("euler_flux_scheme", "rusanov")),
                "adv_weight": float(physics_cfg.get("adv_weight", 1.0)),
                "diff_weight": float(physics_cfg.get("diff_weight", 1.0)),
                "nu": physics_cfg.get("nu", 0.0),
                "parc_input_form": str(physics_cfg.get("parc_input_form", "rate")),
                "parc_predict_type": str(physics_cfg.get("parc_predict_type", "rate")),
                "parc_detach_inputs": bool(physics_cfg.get("parc_detach_inputs", True)),
                "parc_input_weighted": bool(physics_cfg.get("parc_input_weighted", False)),
                "mls_ops_device": str(physics_cfg.get("mls_ops_device", "cpu")),
                "mls_use_2hop_extension": bool(physics_cfg.get("mls_use_2hop_extension", True)),
                "mls_use_neighbor_damping": bool(physics_cfg.get("mls_use_neighbor_damping", True)),
                "mls_damping_alpha": float(physics_cfg.get("mls_damping_alpha", 0.5)),
                "mls_poly_order": int(physics_cfg.get("mls_poly_order", 2)),
                "mls_min_neighbors": int(physics_cfg.get("mls_min_neighbors", 6)),
                "mls_cache_by_geometry": bool(physics_cfg.get("mls_cache_by_geometry", False)),
                "rho_floor": float(physics_cfg.get("rho_floor", 1e-6)),
                "rho_eps": float(physics_cfg.get("rho_eps", 1e-8)),
                "u_clip": float(physics_cfg.get("u_clip", 1e3)),
            },
            "train": {
                "save_dir": str(save_dir),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "validation_every_epochs": int(val_every),
                "lr": float(train_cfg.get("lr", 1e-3)),
                "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
                "grad_clip": float(grad_clip),
                "window_size": int(window_size),
                "stride": int(stride),
                "autoregressive": bool(multi_step_autoreg),
            },
            "model": {
                "hidden": int((cfg.get("model", {}) or {}).get("hidden", 128)),
                "layers": int((cfg.get("model", {}) or {}).get("layers", 3)),
                "dropout": float((cfg.get("model", {}) or {}).get("dropout", 0.1)),
            },
            "derived": {
                "dataset_mode": "window" if use_window_mode else "pair",
                "rollout_steps": int(rollout_steps),
                "split_mode": ("pre_split_dir" if use_pre_split_dirs else "random_split"),
                "reverse_time": bool(reverse_time),
                "physics_inputs_enabled": bool(_physics_inputs_enabled(cfg)),
                "physics_backend": _physics_backend(cfg),
                "physics_extra_in_channels": int(physics_extra_dim),
                "model_in_dim": int(in_dim),
            },
        },
        "data_source_mode": source_mode,
        "data_source_file": (None if source_file is None else os.path.abspath(source_file)),
        "data_source_root": (
            None if not use_pre_split_dirs else os.path.abspath(str(pt_source.get("root", data_cfg.get("pt_path", ""))))
        ),
        "data_source_split_files": split_file_paths,
        "dataset_mode": "window" if use_window_mode else "pair",
        "dataset_samples": int(n_total),
        "dataset_pairs": (None if use_window_mode else int(n_total)),
        "dataset_windows": (int(n_total) if use_window_mode else None),
        "window": {
            "window_size": int(window_size),
            "stride": int(stride),
            "rollout_steps": int(rollout_steps),
            "autoregressive": bool(multi_step_autoreg),
            "reverse_time": bool(reverse_time),
        },
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "dims": {
            "x_dim": dataset_for_dims.x_dim,
            "y_dim": dataset_for_dims.y_dim,
            "pos_dim": dataset_for_dims.pos_dim,
            "physics_extra_in_channels": int(physics_extra_dim),
            "model_in_dim": int(in_dim),
            "model_out_dim": int(out_dim),
        },
        "best_val_loss": best_val,
        "test_loss": te_loss,
        "test_mae": te_mae,
        "best_checkpoint": best_path,
        "final_checkpoint": final_path,
        "log_csv": log_path,
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(save_dir, "config_raw.json"), "w") as f:
        json.dump(cfg_raw, f, indent=2)
    with open(os.path.join(save_dir, "config_effective.json"), "w") as f:
        json.dump(summary["config_effective"], f, indent=2)

    print(f"[DONE] test loss={te_loss:.6f} mae={te_mae:.6f}")
    print(f"[DONE] artifacts saved in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config_karman_basic.json"),
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    main(args.config)

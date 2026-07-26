#!/usr/bin/env python3
"""
rollout_viz_basic_point.py

Rollout visualization for the point-graph baseline (train_basic_point.py).

Writes one GIF per feature.
Default frame layout is 3x1:
  [ GT(t+1)   ]
  [ Pred(t+1) ]
  [ Pred-GT   ]
Optional frame layout (`--pred-gt-only`) is 2x1:
  [ Pred(t+1) ]
  [ GT(t+1)   ]

Supports:
  - autoregressive rollout by default (only when output channels align with input channels)
  - optional teacher-forced one-step rollout (always valid)
  - optional z-slice extraction for "stored-as-3D but physically-2D" datasets
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from types import SimpleNamespace
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from train_basic_point import (
    _build_boundary_mask_node_feature,
    _build_model,
    _build_physics_extra_features,
    _build_relative_geometry_node_features,
    _build_reynolds_node_feature,
    _dt_hat_tensor,
    _dt_ref_scalar_from_cfg,
    _forward_model,
    _infer_reynolds_number,
    _physics_inputs_enabled,
    _predict_type_from_cfg,
    _safe_dt_scalar,
    _state_from_model_output,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _extract_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _load_torch_object(path_or_buf: Any, map_location: str = "cpu") -> Any:
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
        raise FileNotFoundError(f"Data path not found: {path}")

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
    raise RuntimeError(f"Unsupported extension: {ext}. Expected .pt/.pth/.zip")


def _coerce_time_series_to_tnf(x3: torch.Tensor, n_nodes: int, name: str) -> torch.Tensor:
    """
    Coerce a 3D tensor into [T, N, F] with node-count heuristics.
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
    dual_volume_key = None
    dual_volume = None
    for vk in ("dual_volume", "dual_volumes", "control_volume", "control_volumes", "cell_area", "cell_areas", "node_area", "node_areas"):
        vv = obj.get(vk, None)
        if vv is not None:
            dual_volume_key = vk
            dual_volume = _as_optional_node_scalar(vv, vk, n_nodes)
            break

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
        "dual_volume_key": dual_volume_key,
    }

    print(
        f"[INFO] detected case-level format: case={gp_base['case_name']} split={gp_base['split_name']} "
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
                "dual_volume": dual_volume,
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
    raise ValueError(
        "Could not find a list of timesteps in loaded data object. "
        "Expected list/dict timesteps or case-level dict with pos/edge_index/velocity[time,node,feature]."
    )


def _as_2d_float(x: Any, name: str) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if t.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {tuple(t.shape)}")
    return t


def _as_optional_node_scalar(x: Any, name: str, n_nodes: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    t = _as_2d_float(x, name)
    if t.size(0) != n_nodes and t.size(0) == 1 and t.size(1) == n_nodes:
        t = t.t().contiguous()
    if t.size(0) != n_nodes:
        raise ValueError(f"{name} must have one row per node ({n_nodes}), got shape {tuple(t.shape)}")
    return t[:, :1].contiguous()


def _as_edge_index(x: Any) -> torch.Tensor:
    ei = torch.as_tensor(x, dtype=torch.long)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(ei.shape)}")
    if ei.size(0) == 2:
        return ei
    if ei.size(1) == 2:
        return ei.t().contiguous()
    raise ValueError(f"edge_index must be (2,E) or (E,2), got {tuple(ei.shape)}")


def _select_columns(x: torch.Tensor, cols: Optional[Sequence[int]]) -> torch.Tensor:
    if cols is None:
        return x
    idx = torch.as_tensor([int(c) for c in cols], dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("Column list is empty.")
    if int(idx.min().item()) < 0 or int(idx.max().item()) >= x.size(1):
        raise ValueError(f"Requested columns {list(cols)} out of bounds for tensor shape {tuple(x.shape)}")
    return x[:, idx]


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


def _extract_step_fields(step: Any) -> Dict[str, Any]:
    x = _extract_attr(step, "x", _extract_attr(step, "features", None))
    y = _extract_attr(step, "y", None)
    pos = _extract_attr(step, "pos", _extract_attr(step, "xy", None))
    ei = _extract_attr(step, "edge_index", _extract_attr(step, "ei", None))
    dual_volume_raw = None
    dual_volume_name = None
    for vk in ("dual_volume", "dual_volumes", "control_volume", "control_volumes", "cell_area", "cell_areas", "node_area", "node_areas"):
        vv = _extract_attr(step, vk, None)
        if vv is not None:
            dual_volume_raw = vv
            dual_volume_name = vk
            break
    if x is None or pos is None or ei is None:
        missing = []
        if x is None:
            missing.append("x/features")
        if pos is None:
            missing.append("pos/xy")
        if ei is None:
            missing.append("edge_index/ei")
        raise KeyError(f"Timestep missing required fields: {', '.join(missing)}")
    pos_t = _as_2d_float(pos, "pos")
    return {
        "x": _as_2d_float(x, "x"),
        "y": None if y is None else _as_2d_float(y, "y"),
        "pos": pos_t,
        "edge_index": _as_edge_index(ei),
        "dual_volume": (
            None
            if dual_volume_raw is None
            else _as_optional_node_scalar(dual_volume_raw, str(dual_volume_name), int(pos_t.size(0)))
        ),
        "time": _extract_time(step),
        "global_params": _extract_attr(step, "global_params", None),
    }


def _build_z_groups(z: torch.Tensor, z_tol: float) -> List[torch.Tensor]:
    z = z.view(-1).to(torch.float32)
    if z_tol > 0:
        z0 = z.min()
        keys = torch.round((z - z0) / float(z_tol)).to(torch.long)
    else:
        _, keys = torch.unique(z, sorted=True, return_inverse=True)

    groups: List[torch.Tensor] = []
    for g in torch.unique(keys, sorted=True):
        idx = torch.nonzero(keys == g, as_tuple=False).view(-1)
        if idx.numel() > 0:
            groups.append(idx)
    return groups


def _induce_subgraph(
    x: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    keep_idx: torch.Tensor,
    dual_volume: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
        & (src != dst)
        & (remap[src] >= 0)
        & (remap[dst] >= 0)
    )
    ei = torch.stack([remap[src[valid]], remap[dst[valid]]], dim=0)
    return (
        x.index_select(0, keep_idx),
        pos.index_select(0, keep_idx),
        ei,
        None if dual_volume is None else dual_volume.index_select(0, keep_idx),
    )


def _maybe_norm(x: torch.Tensor, mu: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mu is None or std is None:
        return x
    mu_ = mu.to(device=x.device, dtype=x.dtype)
    std_ = std.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)
    return (x - mu_) / std_


def _maybe_denorm(x: torch.Tensor, mu: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    if mu is None or std is None:
        return x
    mu_ = mu.to(device=x.device, dtype=x.dtype)
    std_ = std.to(device=x.device, dtype=x.dtype).clamp_min(1e-12)
    return x * std_ + mu_


def _to_opt_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def _build_model_from_ckpt(ckpt: Dict[str, Any], cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    dims = ckpt.get("dims", {}) or {}
    in_dim = int(dims.get("in_dim"))
    out_dim = int(dims.get("out_dim"))
    model = _build_model(cfg, in_dim=in_dim, out_dim=out_dim, device=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def _choose_feature_names(cfg: Dict[str, Any], used_cols: Optional[Sequence[int]], n_feat: int) -> List[str]:
    names = cfg.get("features", {}).get("names", None)
    if not isinstance(names, list) or len(names) == 0:
        return [f"feat_{i}" for i in range(n_feat)]

    names = [str(n) for n in names]
    if used_cols is None:
        if len(names) >= n_feat:
            return names[:n_feat]
        return names + [f"feat_{i}" for i in range(len(names), n_feat)]

    out = []
    for c in used_cols:
        c = int(c)
        if 0 <= c < len(names):
            out.append(names[c])
        else:
            out.append(f"feat_{c}")
    if len(out) < n_feat:
        out += [f"feat_{i}" for i in range(len(out), n_feat)]
    return out[:n_feat]


def _parse_zoom_bbox(text: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--zoom-bbox must be xmin,xmax,ymin,ymax")
    xmin, xmax, ymin, ymax = [float(v) for v in parts]
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("--zoom-bbox must satisfy xmax>xmin and ymax>ymin")
    return xmin, xmax, ymin, ymax


def _sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _scatter_panel(
    ax,
    pos_np: np.ndarray,
    val_np: np.ndarray,
    *,
    tri: Optional[mtri.Triangulation] = None,
    point_idx: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    point_size: float,
    zoom_bbox: Optional[Tuple[float, float, float, float]],
):
    _ = tri  # kept for call-signature parity with _tri_panel
    p = pos_np[point_idx]
    v = val_np[point_idx]
    sc = ax.scatter(
        p[:, 0],
        p[:, 1],
        c=v,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker="s",
        linewidths=0,
    )
    if zoom_bbox is not None:
        xmin, xmax, ymin, ymax = zoom_bbox
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def _build_triangulation(
    pos_np: np.ndarray,
    *,
    point_idx: np.ndarray,
    edge_quantile: float,
    edge_factor: float,
) -> Optional[mtri.Triangulation]:
    p = pos_np[point_idx]
    if p.ndim != 2 or p.shape[0] < 3:
        return None
    try:
        tri = mtri.Triangulation(p[:, 0], p[:, 1])
    except Exception:
        return None

    tris = getattr(tri, "triangles", None)
    if tris is None or len(tris) == 0:
        return tri

    # Mask long-edge triangles to avoid bridging across geometric voids
    # (e.g., around the cylinder boundary) when using Delaunay interpolation.
    pts = p[tris]  # [ntri, 3, 2]
    e01 = np.linalg.norm(pts[:, 0, :] - pts[:, 1, :], axis=1)
    e12 = np.linalg.norm(pts[:, 1, :] - pts[:, 2, :], axis=1)
    e20 = np.linalg.norm(pts[:, 2, :] - pts[:, 0, :], axis=1)
    max_edge = np.maximum(e01, np.maximum(e12, e20))

    finite = np.isfinite(max_edge)
    if np.any(finite):
        q = float(np.clip(edge_quantile, 0.5, 0.999))
        base = float(np.quantile(max_edge[finite], q))
        thr = base * float(max(1.0, edge_factor))
        mask = max_edge > thr
    else:
        mask = np.zeros((len(tris),), dtype=bool)

    try:
        analyzer = mtri.TriAnalyzer(tri)
        flat_mask = analyzer.get_flat_tri_mask(min_circle_ratio=0.01)
        if flat_mask is not None and len(flat_mask) == len(mask):
            mask = np.logical_or(mask, flat_mask)
    except Exception:
        pass

    if np.any(mask):
        tri.set_mask(mask)
    return tri


def _tri_panel(
    ax,
    pos_np: np.ndarray,
    val_np: np.ndarray,
    *,
    tri: Optional[mtri.Triangulation] = None,
    point_idx: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    point_size: float,
    zoom_bbox: Optional[Tuple[float, float, float, float]],
):
    if tri is None:
        return _scatter_panel(
            ax,
            pos_np,
            val_np,
            point_idx=point_idx,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            point_size=point_size,
            zoom_bbox=zoom_bbox,
        )

    v = val_np[point_idx]
    # Gouraud shading gives smooth interpolation from point values.
    m = ax.tripcolor(
        tri,
        v,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if zoom_bbox is not None:
        xmin, xmax, ymin, ymax = zoom_bbox
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return m


def _safe_name(s: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s.strip())
    return out if out else "feature"


def _fmt_abs_time(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        vv = float(v)
        if not np.isfinite(vv):
            return "NA"
        return f"{vv:.6g}"
    except Exception:
        return "NA"


def _enforce_gif_duration_ms(gif_path: str, duration_ms: int) -> bool:
    """
    Force per-frame GIF duration with Pillow.
    This is a Safari-compatible fallback when upstream encoder metadata is ignored.
    """
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return False

    try:
        with Image.open(gif_path) as im:
            frames = [frm.copy() for frm in ImageSequence.Iterator(im)]
        if len(frames) == 0:
            return False

        tmp_path = gif_path + ".tmp.gif"
        frames[0].save(
            tmp_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(max(1, duration_ms)),
            loop=0,
            optimize=False,
            disposal=2,
        )
        os.replace(tmp_path, gif_path)
        return True
    except Exception:
        return False


def _write_rgb_gif_fixed_palette(gif_path: str, frames: List[np.ndarray], duration_ms: int) -> bool:
    """
    Save RGB frames using one global GIF palette.
    This avoids frame-to-frame colorbar palette shimmer in animated GIFs.
    """
    if len(frames) == 0:
        return False

    try:
        from PIL import Image
    except Exception:
        imageio.mimsave(gif_path, frames, duration=float(duration_ms) / 1000.0, loop=0)
        return False

    try:
        palette_enum = getattr(Image, "Palette", None)
        dither_enum = getattr(Image, "Dither", None)
        adaptive = getattr(palette_enum, "ADAPTIVE", None)
        if adaptive is None:
            adaptive = getattr(Image, "ADAPTIVE", 1)
        no_dither = getattr(dither_enum, "NONE", None)
        if no_dither is None:
            no_dither = getattr(Image, "NONE", 0)

        rgb_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
        # Each frame contains full colorbars, so the first frame is a good global palette source.
        palette_frame = rgb_frames[0].convert("P", palette=adaptive, colors=256, dither=no_dither)
        paletted = [
            im.quantize(palette=palette_frame, dither=no_dither)
            for im in rgb_frames
        ]
        paletted[0].save(
            gif_path,
            save_all=True,
            append_images=paletted[1:],
            duration=int(max(1, duration_ms)),
            loop=0,
            optimize=False,
            disposal=2,
        )
        return True
    except Exception:
        imageio.mimsave(gif_path, frames, duration=float(duration_ms) / 1000.0, loop=0)
        return False


def run_rollout(
    *,
    model: torch.nn.Module,
    steps: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    norm_x_mu: Optional[torch.Tensor],
    norm_x_std: Optional[torch.Tensor],
    norm_y_mu: Optional[torch.Tensor],
    norm_y_std: Optional[torch.Tensor],
    norm_pos_mu: Optional[torch.Tensor],
    norm_pos_std: Optional[torch.Tensor],
    device: torch.device,
    start_t: int,
    horizon: int,
    seed: int,
    force_teacher: bool,
    z_slice: Optional[int],
    source_reynolds: Optional[float],
    expected_in_dim: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    data_cfg = cfg.get("data", {}) or {}
    feat_cfg = cfg.get("features", {}) or {}

    use_y_target = bool(data_cfg.get("use_y_as_target", True))
    split_by_z = bool(data_cfg.get("split_by_z", False))
    z_index = int(data_cfg.get("z_index", 2))
    z_tol = float(data_cfg.get("z_tol", 0.0))
    if split_by_z and len(steps) > 0 and int(steps[0]["pos"].size(1)) <= z_index:
        print(
            f"[WARN] split_by_z=true but z_index={z_index} is out of bounds for pos dim "
            f"{int(steps[0]['pos'].size(1))}; disabling split_by_z for this rollout."
        )
        split_by_z = False

    x_cols = feat_cfg.get("use_columns", None)
    y_cols = feat_cfg.get("target_columns", None)
    pos_cols = feat_cfg.get("pos_columns", None)
    x_cols = None if x_cols is None else [int(c) for c in x_cols]
    y_cols = x_cols if y_cols is None else [int(c) for c in y_cols]
    pos_cols = None if pos_cols is None else [int(c) for c in pos_cols]

    t0 = max(0, int(start_t))
    t1 = min(len(steps) - 1, t0 + int(horizon))
    if t1 <= t0:
        raise ValueError("Requested rollout window is empty. Check start_t/horizon.")

    examples: List[Dict[str, Any]] = []
    prev_pred_x_abs: Optional[torch.Tensor] = None
    used_autoreg = False
    norm_obj = SimpleNamespace(
        x_mu=norm_x_mu,
        x_std=norm_x_std,
        y_mu=norm_y_mu,
        y_std=norm_y_std,
        pos_mu=norm_pos_mu,
        pos_std=norm_pos_std,
    )
    physics_inputs_enabled = bool(_physics_inputs_enabled(cfg))
    predict_type = _predict_type_from_cfg(cfg)

    for t in range(t0, t1):
        s_t = steps[t]
        s_tp1 = steps[t + 1]

        x_t = _select_columns(s_t["x"], x_cols)
        pos_t = _select_columns(s_t["pos"], pos_cols)
        ei_t = s_t["edge_index"]
        dual_volume_t = s_t.get("dual_volume", None)

        if use_y_target and (s_t["y"] is not None):
            y_target = _select_columns(s_t["y"], y_cols)
        else:
            y_target = _select_columns(s_tp1["x"], y_cols)

        # Optional per-step z-slice extraction.
        if split_by_z:
            if z_index < 0 or z_index >= s_t["pos"].size(1):
                raise ValueError(f"z_index={z_index} is out of bounds for pos shape {tuple(s_t['pos'].shape)}")
            z = s_t["pos"][:, z_index]
            groups = _build_z_groups(z, z_tol=z_tol)
            if len(groups) == 0:
                raise RuntimeError(f"No z-slice groups found at t={t}.")
            gid = int(z_slice) if z_slice is not None else 0
            if gid < 0 or gid >= len(groups):
                raise ValueError(f"Requested z_slice={gid}, but only {len(groups)} groups available at t={t}.")
            keep = groups[gid]
            x_t, pos_t, ei_t, dual_volume_t = _induce_subgraph(
                x_t,
                pos_t,
                ei_t,
                keep,
                dual_volume=dual_volume_t,
            )
            y_target = y_target.index_select(0, keep)

        if x_t.size(0) != y_target.size(0):
            raise RuntimeError(
                f"Node count mismatch at t={t}: x_t has {x_t.size(0)} nodes, "
                f"target has {y_target.size(0)}."
            )

        # The first step starts from GT; later steps use the previous prediction
        # when autoregressive rollout is enabled and the channel shapes align.
        x_in_abs = x_t
        can_autoreg = (
            (not force_teacher)
            and (prev_pred_x_abs is not None)
            and (prev_pred_x_abs.shape == x_t.shape)
        )
        if can_autoreg:
            x_in_abs = prev_pred_x_abs
            used_autoreg = True

        dt_phys = _safe_dt_scalar(s_t.get("time", None), s_tp1.get("time", None), default_dt=1.0)
        x_abs_dev = x_in_abs.to(device=device, dtype=torch.float32)
        x_in = _maybe_norm(x_abs_dev, norm_x_mu, norm_x_std)
        dt_hat = _dt_hat_tensor(dt_phys, cfg, device=x_in.device, dtype=x_in.dtype)
        pos_dev = pos_t.to(device=device, dtype=torch.float32)
        ei_dev = ei_t.to(device=device, dtype=torch.long)
        dual_volume_dev = (
            None
            if dual_volume_t is None
            else dual_volume_t.to(device=device, dtype=torch.float32)
        )

        include_pos = bool(feat_cfg.get("include_pos", True))
        pos_in = _maybe_norm(pos_dev, norm_pos_mu, norm_pos_std) if include_pos else None
        x_parts = [x_in]
        re_extra = _build_reynolds_node_feature(
            n_nodes=int(x_in.size(0)),
            cfg=cfg,
            meta=s_t.get("global_params", None),
            source_reynolds=source_reynolds,
            device=device,
            dtype=x_in.dtype,
        )
        if re_extra is not None and re_extra.numel() > 0:
            x_parts.append(re_extra)
        rel_geo = _build_relative_geometry_node_features(
            pos=pos_dev,
            edge_index=ei_dev,
            cfg=cfg,
            device=device,
            dtype=x_in.dtype,
        )
        if rel_geo is not None and rel_geo.numel() > 0:
            x_parts.append(rel_geo)
        bnd_extra = _build_boundary_mask_node_feature(
            pos=pos_dev,
            edge_index=ei_dev,
            cfg=cfg,
            device=device,
            dtype=x_in.dtype,
        )
        if bnd_extra is not None and bnd_extra.numel() > 0:
            x_parts.append(bnd_extra)
        if include_pos:
            assert pos_in is not None
            x_parts.append(pos_in)
        if physics_inputs_enabled:
            phy_extra = _build_physics_extra_features(
                x_abs=x_abs_dev,
                pos=pos_dev,
                edge_index=ei_dev,
                dual_volume=dual_volume_dev,
                dt_phys_scalar=dt_phys,
                cfg=cfg,
                norm=norm_obj,
                out_dtype=x_in.dtype,
                device=device,
            )
            if phy_extra is not None and phy_extra.numel() > 0:
                x_parts.append(phy_extra)
        x_model = torch.cat(x_parts, dim=1)
        if expected_in_dim is not None and expected_in_dim > 0 and int(x_model.size(1)) != int(expected_in_dim):
            bnd_ch = 0 if bnd_extra is None else int(bnd_extra.size(1))
            rel_ch = 0 if rel_geo is None else int(rel_geo.size(1))
            raise RuntimeError(
                "Rollout input dim mismatch: "
                f"built {int(x_model.size(1))} channels, but checkpoint expects {int(expected_in_dim)}. "
                f"(include_pos={include_pos}, physics_inputs_enabled={physics_inputs_enabled}, "
                f"reynolds_input={re_extra is not None}, relative_geometry_channels={rel_ch}, "
                f"boundary_geom_channels={bnd_ch})"
            )

        with torch.no_grad():
            y_model_out, _score, _h = _forward_model(
                model,
                x_model,
                ei_dev,
                pos_dev,
                dual_volume=dual_volume_dev,
            )
            y_pred_norm = _state_from_model_output(x_in, y_model_out, dt_hat, predict_type)
            y_pred_abs = _maybe_denorm(y_pred_norm, norm_y_mu, norm_y_std).detach().cpu()

        # Update autoregressive state only when prediction matches x channels.
        prev_pred_x_abs = y_pred_abs if (y_pred_abs.shape == x_t.shape) else None

        gt_t_ref = _select_columns(x_t, y_cols) if x_t.size(1) != y_target.size(1) else x_t
        input_ref = (
            _select_columns(x_in_abs, y_cols)
            if x_in_abs.size(1) != gt_t_ref.size(1)
            else x_in_abs
        )
        input_gt_t_mae = float((input_ref.detach().cpu() - gt_t_ref.detach().cpu()).abs().mean().item())
        mae = float((y_pred_abs - y_target).abs().mean().item())
        examples.append(
            {
                "t": t,
                "t_next": t + 1,
                "time_t": s_t["time"],
                "time_tp1": s_tp1["time"],
                "pos": pos_t.detach().cpu(),
                "gt_t": gt_t_ref.detach().cpu(),
                "gt_tp1": y_target.detach().cpu(),
                "pred_tp1": y_pred_abs,
                "mae": mae,
                "input_gt_t_mae": input_gt_t_mae,
            }
        )

    return examples, used_autoreg


def make_rollout_gifs(
    *,
    examples: List[Dict[str, Any]],
    feature_names: List[str],
    out_dir: str,
    fps: int,
    max_points: int,
    point_size: float,
    cmap_top: str,
    cmap_delta: str,
    zoom_bbox: Optional[Tuple[float, float, float, float]],
    clim_sample: int,
    sample_seed: int,
    target_label: str,
    input_label: str,
    pred_gt_only: bool = False,
    render_mode: str = "scatter",
    tri_edge_quantile: float = 0.95,
    tri_edge_factor: float = 1.75,
    reverse_time: bool = False,
) -> List[str]:
    if len(examples) == 0:
        raise ValueError("No rollout examples to visualize.")
    if render_mode not in ("scatter", "tri"):
        raise ValueError(f"Unsupported render_mode={render_mode}; use 'scatter' or 'tri'.")

    os.makedirs(out_dir, exist_ok=True)
    F = int(examples[0]["gt_t"].shape[1])
    if len(feature_names) < F:
        feature_names = feature_names + [f"feat_{i}" for i in range(len(feature_names), F)]

    # Compute fixed color limits across rollout per feature.
    top_min = np.full((F,), np.inf, dtype=np.float64)
    top_max = np.full((F,), -np.inf, dtype=np.float64)
    err_abs = np.zeros((F,), dtype=np.float64)

    for ex in examples:
        gt_t = ex["gt_t"].numpy()
        gt_tp1 = ex["gt_tp1"].numpy()
        pred = ex["pred_tp1"].numpy()
        for f in range(F):
            # Anchor top-row color limits to the target only.
            # This avoids Pred outliers flattening GT/target contrast.
            for a in (gt_tp1[:, f],):
                if a.size == 0:
                    continue
                if clim_sample > 0 and a.size > clim_sample:
                    step = max(1, a.size // clim_sample)
                    aa = a[::step]
                else:
                    aa = a
                top_min[f] = min(top_min[f], float(np.nanmin(aa)))
                top_max[f] = max(top_max[f], float(np.nanmax(aa)))

            if not pred_gt_only:
                # Anchor residual color limits to Pred-GT over the full rollout.
                err = pred[:, f] - gt_tp1[:, f]
                for d in (err,):
                    if d.size == 0:
                        continue
                    if clim_sample > 0 and d.size > clim_sample:
                        step = max(1, d.size // clim_sample)
                        dd = d[::step]
                    else:
                        dd = d
                    err_abs[f] = max(err_abs[f], float(np.nanmax(np.abs(dd))))

    top_pad = 1e-12
    top_equal = (top_max - top_min) < top_pad
    top_max[top_equal] = top_min[top_equal] + 1.0
    if not pred_gt_only:
        err_abs = np.maximum(err_abs, 1e-12)

    gif_paths: List[str] = []
    fps_i = max(1, int(fps))
    frame_ms = int(round(1000.0 / float(fps_i)))
    for f in range(F):
        nm = _safe_name(feature_names[f])
        p = os.path.join(out_dir, f"rollout_{f:02d}_{nm}.gif")
        gif_paths.append(p)
    frames_by_feature: List[List[np.ndarray]] = [[] for _ in range(F)]

    # Keep a stable plotted subset for each encountered node count.
    # This avoids per-frame "TV static" flicker when max_points < N.
    pick_cache: Dict[int, np.ndarray] = {}
    tri_cache: Dict[int, Optional[mtri.Triangulation]] = {}
    frame_examples = list(reversed(examples)) if bool(reverse_time) else examples
    for k, ex in enumerate(frame_examples):
        pos = ex["pos"].numpy()
        gt_tp1 = ex["gt_tp1"].numpy()
        pred = ex["pred_tp1"].numpy()
        frame_target_label = str(ex.get("frame_target_label", target_label))
        frame_pred_label = str(ex.get("frame_pred_label", f"Pred({frame_target_label})"))
        frame_error_label = str(ex.get("frame_error_label", f"Pred-{frame_target_label}"))

        d_pg = pred - gt_tp1
        n = pos.shape[0]
        if n not in pick_cache:
            pick_cache[n] = _sample_indices(n, max_points=max_points, seed=int(sample_seed))
        if (render_mode == "tri") and (n not in tri_cache):
            tri_cache[n] = _build_triangulation(
                pos,
                point_idx=pick_cache[n],
                edge_quantile=float(tri_edge_quantile),
                edge_factor=float(tri_edge_factor),
            )
        pick = pick_cache[n]
        tri = tri_cache.get(n, None)

        for f in range(F):
            if pred_gt_only:
                fig, ax = plt.subplots(2, 1, figsize=(7.0, 8.0), dpi=130, squeeze=False)

                sc_pred = (
                    _tri_panel if render_mode == "tri" else _scatter_panel
                )(
                    ax[0, 0], pos, pred[:, f],
                    tri=tri,
                    point_idx=pick,
                    vmin=float(top_min[f]), vmax=float(top_max[f]),
                    cmap=cmap_top, point_size=point_size, zoom_bbox=zoom_bbox,
                )
                sc_gt = (
                    _tri_panel if render_mode == "tri" else _scatter_panel
                )(
                    ax[1, 0], pos, gt_tp1[:, f],
                    tri=tri,
                    point_idx=pick,
                    vmin=float(top_min[f]), vmax=float(top_max[f]),
                    cmap=cmap_top, point_size=point_size, zoom_bbox=zoom_bbox,
                )

                ax[0, 0].set_title(frame_pred_label)
                ax[1, 0].set_title(frame_target_label)

                t_abs = ex.get("t", k)
                t_next = ex.get("t_next", t_abs + 1)
                abs_time = _fmt_abs_time(ex.get("time_tp1", None))
                mae = ex.get("mae", float("nan"))
                time_label = f"initial t={t_abs}" if bool(ex.get("is_initial_frame", False)) else f"t={t_abs}->{t_next}"
                fig.suptitle(
                    f"{time_label}  abs_time={abs_time}  "
                    f"feat={feature_names[f]}  mae={mae:.3e}  "
                    f"points={n} (plot {len(pick)})",
                    fontsize=11,
                )
                fig.subplots_adjust(left=0.07, right=0.86, bottom=0.06, top=0.91, hspace=0.24)
                ticks = np.linspace(float(top_min[f]), float(top_max[f]), 5)
                for sc, y0 in ((sc_pred, 0.55), (sc_gt, 0.12)):
                    cax = fig.add_axes([0.88, y0, 0.025, 0.32])
                    fig.colorbar(sc, cax=cax, ticks=ticks)
            else:
                fig, ax = plt.subplots(3, 1, figsize=(7.4, 11.2), dpi=130, squeeze=False)

                sc_gt = (
                    _tri_panel if render_mode == "tri" else _scatter_panel
                )(
                    ax[0, 0], pos, gt_tp1[:, f],
                    tri=tri,
                    point_idx=pick,
                    vmin=float(top_min[f]), vmax=float(top_max[f]),
                    cmap=cmap_top, point_size=point_size, zoom_bbox=zoom_bbox,
                )
                sc_pred = (
                    _tri_panel if render_mode == "tri" else _scatter_panel
                )(
                    ax[1, 0], pos, pred[:, f],
                    tri=tri,
                    point_idx=pick,
                    vmin=float(top_min[f]), vmax=float(top_max[f]),
                    cmap=cmap_top, point_size=point_size, zoom_bbox=zoom_bbox,
                )

                elim = float(err_abs[f])
                sc_err = (
                    _tri_panel if render_mode == "tri" else _scatter_panel
                )(
                    ax[2, 0], pos, d_pg[:, f],
                    tri=tri,
                    point_idx=pick,
                    vmin=-elim, vmax=elim,
                    cmap=cmap_delta, point_size=point_size, zoom_bbox=zoom_bbox,
                )

                ax[0, 0].set_title(frame_target_label)
                ax[1, 0].set_title(frame_pred_label)
                ax[2, 0].set_title(frame_error_label)

                t_abs = ex.get("t", k)
                t_next = ex.get("t_next", t_abs + 1)
                abs_time = _fmt_abs_time(ex.get("time_tp1", None))
                mae = ex.get("mae", float("nan"))
                time_label = f"initial t={t_abs}" if bool(ex.get("is_initial_frame", False)) else f"t={t_abs}->{t_next}"
                fig.suptitle(
                    f"{time_label}  abs_time={abs_time}  "
                    f"feat={feature_names[f]}  mae={mae:.3e}  "
                    f"points={n} (plot {len(pick)})",
                    fontsize=11,
                )
                fig.subplots_adjust(left=0.07, right=0.86, bottom=0.045, top=0.925, hspace=0.32)
                top_ticks = np.linspace(float(top_min[f]), float(top_max[f]), 5)
                err_ticks = np.linspace(-elim, elim, 5)
                for sc, y0, ticks in (
                    (sc_gt, 0.675, top_ticks),
                    (sc_pred, 0.375, top_ticks),
                    (sc_err, 0.075, err_ticks),
                ):
                    cax = fig.add_axes([0.88, y0, 0.025, 0.215])
                    fig.colorbar(sc, cax=cax, ticks=ticks)

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frames_by_feature[f].append(buf[:, :, :3].copy())
            plt.close(fig)

        print(f"[GIF] frame {k + 1}/{len(frame_examples)} complete")

    fixed_palette = 0
    for f, p in enumerate(gif_paths):
        if _write_rgb_gif_fixed_palette(p, frames_by_feature[f], frame_ms):
            fixed_palette += 1
    if fixed_palette > 0:
        print(
            f"[GIF] wrote {fixed_palette}/{len(gif_paths)} files with a fixed global palette "
            f"({frame_ms} ms/frame)"
        )

    return gif_paths


def _with_initial_condition_frame(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(examples) == 0:
        return examples
    first = examples[0]
    init = dict(first)
    init["t_next"] = first.get("t", 0)
    init["time_tp1"] = first.get("time_t", None)
    init["gt_tp1"] = first["gt_t"].clone()
    init["pred_tp1"] = first["gt_t"].clone()
    init["mae"] = 0.0
    init["is_initial_frame"] = True
    init["frame_target_label"] = "GT(t)"
    init["frame_pred_label"] = "Pred(t) initialized from GT(t)"
    init["frame_error_label"] = "Pred-GT(t)"
    return [init] + examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Rollout GIF visualization for basic point-graph model.")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint (e.g. runs_karman_basic/best_model.pt)")
    ap.add_argument("--config", default=None, help="Optional config override (defaults to checkpoint cfg)")
    ap.add_argument("--pt-path", default=None, help="Optional data path override (defaults to cfg.data.pt_path)")
    ap.add_argument("--out-dir", default=None, help="Output directory for GIFs")
    ap.add_argument("--start-t", type=int, default=0, help="Start timestep index")
    ap.add_argument("--horizon", type=int, default=50, help="Number of transitions to roll out")
    ap.add_argument("--fps", type=int, default=5, help="GIF frame rate")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--autoregressive",
        dest="autoregressive",
        action="store_true",
        default=True,
        help="Use Pred(t+1) as input at the next step when shapes match (default).",
    )
    mode_group.add_argument(
        "--teacher-forced",
        "--no-autoregressive",
        dest="autoregressive",
        action="store_false",
        help="Disable autoregressive chaining; evaluate each transition from GT Input x(t).",
    )
    ap.add_argument("--force-teacher", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--z-slice", type=int, default=None, help="When split_by_z=true, which z-slice group to use")
    ap.add_argument("--max-points", type=int, default=0, help="Max plotted points per frame (<=0 disables)")
    ap.add_argument("--point-size", type=float, default=2.0, help="Scatter marker size")
    ap.add_argument("--zoom-bbox", default=None, help="Optional zoom: xmin,xmax,ymin,ymax")
    ap.add_argument("--cmap-top", default="viridis", help="Colormap for state panels")
    ap.add_argument("--cmap-delta", default="coolwarm", help="Colormap for delta panels")
    ap.add_argument(
        "--render-mode",
        default="scatter",
        choices=("scatter", "tri"),
        help="Panel rendering mode: point scatter or triangulated interpolation.",
    )
    ap.add_argument(
        "--tri-edge-quantile",
        type=float,
        default=0.95,
        help="Quantile of triangle max-edge length used as baseline for masking long triangles.",
    )
    ap.add_argument(
        "--tri-edge-factor",
        type=float,
        default=1.75,
        help="Multiply long-edge baseline by this factor to set triangle mask threshold.",
    )
    ap.add_argument("--clim-sample", type=int, default=250000, help="Sample size for percentile/clim stats")
    ap.add_argument(
        "--pred-gt-only",
        action="store_true",
        help="If set, render two panels per frame: Pred(t+1) and GT(t+1).",
    )
    ap.add_argument(
        "--reverse-time",
        action="store_true",
        help="If set, write GIF frames in reverse time order.",
    )
    ap.add_argument(
        "--include-initial-frame",
        action="store_true",
        help="Prepend a GIF frame at start_t with Pred(t)=GT(t) to verify the visual starting state.",
    )
    args = ap.parse_args()

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = _load_torch_object(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint format not supported; expected dict with model_state_dict/cfg.")

    cfg = ckpt.get("cfg", None)
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)
    if cfg is None:
        raise RuntimeError("No cfg found in checkpoint and no --config provided.")

    if args.pt_path is not None:
        cfg.setdefault("data", {})["pt_path"] = args.pt_path
    pt_path = str(cfg.get("data", {}).get("pt_path", "")).strip()
    if not pt_path:
        raise RuntimeError("cfg.data.pt_path is not set; provide --pt-path.")
    predict_type = _predict_type_from_cfg(cfg)
    model_dt_ref = _dt_ref_scalar_from_cfg(cfg)

    if args.out_dir is None:
        base = os.path.dirname(os.path.abspath(args.checkpoint))
        out_dir = os.path.join(base, f"rollout_t{int(args.start_t)}_H{int(args.horizon)}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    model = _build_model_from_ckpt(ckpt, cfg, device=device)
    dims = ckpt.get("dims", {}) or {}
    expected_in_dim = int(dims.get("in_dim", -1))
    norm = ckpt.get("norm", {}) or {}
    x_mu = _to_opt_tensor(norm.get("x_mu", None))
    x_std = _to_opt_tensor(norm.get("x_std", None))
    y_mu = _to_opt_tensor(norm.get("y_mu", None))
    y_std = _to_opt_tensor(norm.get("y_std", None))
    pos_mu = _to_opt_tensor(norm.get("pos_mu", None))
    pos_std = _to_opt_tensor(norm.get("pos_std", None))

    data_obj = _load_pt_or_zip(pt_path)
    raw_steps = _extract_series(data_obj)
    source_reynolds = _infer_reynolds_number(data_obj, raw_steps, pt_path)
    steps = [_extract_step_fields(s) for s in raw_steps]
    use_y_target = bool(cfg.get("data", {}).get("use_y_as_target", True))
    has_y_targets = any((s.get("y", None) is not None) for s in steps[: min(5, len(steps))])
    has_dual_volume = any((s.get("dual_volume", None) is not None) for s in steps[: min(5, len(steps))])
    if use_y_target and has_y_targets:
        target_label = "Target y(t)"
    else:
        target_label = "GT(t+1)"
    input_label = "Input x(t)"
    print(
        f"[INFO] target semantics: {target_label} "
        f"(cfg.data.use_y_as_target={use_y_target}, y_present={has_y_targets})"
    )

    t_start = time.perf_counter()
    if bool(args.autoregressive) and bool(args.force_teacher):
        print("[WARN] --force-teacher overrides autoregressive mode; use --teacher-forced or --no-autoregressive instead.")
    force_teacher = (not bool(args.autoregressive)) or bool(args.force_teacher)
    rollout_mode = "teacher-forced" if force_teacher else "autoregressive"
    print(f"[INFO] rollout mode: {rollout_mode}")
    dt_ref_msg = "none (using raw dt)" if model_dt_ref is None else f"{model_dt_ref:g}"
    print(f"[INFO] model output predict_type: {predict_type} dt_ref={dt_ref_msg}")

    examples, used_autoreg = run_rollout(
        model=model,
        steps=steps,
        cfg=cfg,
        norm_x_mu=x_mu,
        norm_x_std=x_std,
        norm_y_mu=y_mu,
        norm_y_std=y_std,
        norm_pos_mu=pos_mu,
        norm_pos_std=pos_std,
        device=device,
        start_t=int(args.start_t),
        horizon=int(args.horizon),
        seed=int(args.seed),
        force_teacher=force_teacher,
        z_slice=args.z_slice,
        source_reynolds=source_reynolds,
        expected_in_dim=(expected_in_dim if expected_in_dim > 0 else None),
    )
    t_roll = time.perf_counter() - t_start
    initial_input_gt_t_mae = float(examples[0].get("input_gt_t_mae", float("nan")))
    model_uses_dual_volume = bool(getattr(model, "uses_dual_volume", False))
    dual_volume_passed_to_model = bool(model_uses_dual_volume and has_dual_volume)
    if model_uses_dual_volume:
        if has_dual_volume:
            print("[INFO] model uses dual_volume; passing per-node volumes during rollout.")
        else:
            print("[WARN] model uses dual_volume, but this rollout data did not expose dual_volume.")

    feat_cols = cfg.get("features", {}).get("target_columns", None)
    feat_names = _choose_feature_names(cfg, feat_cols, int(examples[0]["pred_tp1"].shape[1]))
    zoom_bbox = _parse_zoom_bbox(args.zoom_bbox)
    gif_examples = (
        _with_initial_condition_frame(examples)
        if bool(args.include_initial_frame)
        else examples
    )

    t_gif0 = time.perf_counter()
    gif_paths = make_rollout_gifs(
        examples=gif_examples,
        feature_names=feat_names,
        out_dir=out_dir,
        fps=int(args.fps),
        max_points=int(args.max_points),
        point_size=float(args.point_size),
        cmap_top=str(args.cmap_top),
        cmap_delta=str(args.cmap_delta),
        zoom_bbox=zoom_bbox,
        clim_sample=int(args.clim_sample),
        sample_seed=int(args.seed),
        target_label=target_label,
        input_label=input_label,
        pred_gt_only=bool(args.pred_gt_only),
        render_mode=str(args.render_mode),
        tri_edge_quantile=float(args.tri_edge_quantile),
        tri_edge_factor=float(args.tri_edge_factor),
        reverse_time=bool(args.reverse_time),
    )
    t_gif = time.perf_counter() - t_gif0

    mean_mae = float(np.mean([float(ex["mae"]) for ex in examples]))
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "pt_path": os.path.abspath(os.path.expanduser(pt_path)),
        "n_steps": len(examples),
        "n_gif_frames": len(gif_examples),
        "start_t": int(args.start_t),
        "horizon": int(args.horizon),
        "rollout_mode": rollout_mode,
        "predict_type": predict_type,
        "model_dt_ref": model_dt_ref,
        "force_teacher": bool(force_teacher),
        "autoregressive_requested": bool(args.autoregressive),
        "used_autoregressive": bool(used_autoreg),
        "z_slice": args.z_slice,
        "source_reynolds": (None if source_reynolds is None else float(source_reynolds)),
        "has_dual_volume": bool(has_dual_volume),
        "model_uses_dual_volume": bool(model_uses_dual_volume),
        "dual_volume_passed_to_model": bool(dual_volume_passed_to_model),
        "initial_input_gt_t_mae": initial_input_gt_t_mae,
        "mean_mae": mean_mae,
        "target_label": target_label,
        "input_label": input_label,
        "pred_gt_only": bool(args.pred_gt_only),
        "include_initial_frame": bool(args.include_initial_frame),
        "reverse_time": bool(args.reverse_time),
        "render_mode": str(args.render_mode),
        "tri_edge_quantile": float(args.tri_edge_quantile),
        "tri_edge_factor": float(args.tri_edge_factor),
        "rollout_seconds": t_roll,
        "gif_seconds": t_gif,
        "gif_paths": [os.path.abspath(p) for p in gif_paths],
    }
    with open(os.path.join(out_dir, "rollout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] frames={len(examples)} mean_mae={mean_mae:.6e}")
    print(f"[DONE] initial_input_gt_t_mae={initial_input_gt_t_mae:.6e}")
    print(f"[DONE] rollout_mode={rollout_mode} used_autoregressive={used_autoreg} force_teacher={bool(force_teacher)}")
    print(f"[DONE] rollout_time={t_roll:.2f}s gif_time={t_gif:.2f}s")
    for p in gif_paths:
        print(f"[DONE] wrote {p}")


if __name__ == "__main__":
    main()

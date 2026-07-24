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
import re
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

from models import FeatureNet, FluxGraphNet, MeshGraphNet
import utils.dec_ops as dec
import utils.mls as mls


_MLS_STATE = {
    "sig": None,
    "grad": None,
    "lapw": None,
    "adv": None,
    "diff": None,
}
_BOUNDARY_INFER_CACHE: Dict[Tuple[Any, ...], Dict[str, float]] = {}


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


def _as_optional_node_scalar(x: Any, name: str, n_nodes: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    t = _as_2d_float(x, name)
    if t.size(0) != n_nodes and t.size(0) == 1 and t.size(1) == n_nodes:
        t = t.t().contiguous()
    if t.size(0) != n_nodes:
        raise ValueError(f"{name} must have one row per node ({n_nodes}), got shape {tuple(t.shape)}")
    return t[:, :1].contiguous()


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


def _as_optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        t = torch.as_tensor(v).view(-1)
        if int(t.numel()) < 1:
            return None
        out = float(t[0].item())
    except Exception:
        try:
            out = float(v)
        except Exception:
            return None
    if not np.isfinite(out):
        return None
    return out


def _parse_reynolds_from_text(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    pats = (
        r"reynolds[_\-\s]*([0-9]+(?:\.[0-9]+)?)",
        r"\bre[_\-\s]*([0-9]+(?:\.[0-9]+)?)\b",
    )
    for pat in pats:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m is None:
            continue
        try:
            val = float(m.group(1))
        except Exception:
            continue
        if np.isfinite(val) and (val > 0.0):
            return val
    return None


def _extract_reynolds_from_mapping(d: Dict[str, Any]) -> Optional[float]:
    for key in ("reynolds_number", "reynolds", "Re", "re"):
        if key in d:
            v = _as_optional_float(d.get(key, None))
            if v is not None and v > 0.0:
                return v
    for key in ("case_name", "name", "source_name", "source_path"):
        if key in d:
            v = _parse_reynolds_from_text(d.get(key, None))
            if v is not None:
                return v
    gp = d.get("global_params", None)
    if isinstance(gp, dict):
        v = _extract_reynolds_from_mapping(gp)
        if v is not None:
            return v
    return None


def _infer_reynolds_number(raw_obj: Any, raw_steps: Optional[Sequence[Any]], src_path: str) -> Optional[float]:
    if isinstance(raw_obj, dict):
        v = _extract_reynolds_from_mapping(raw_obj)
        if v is not None:
            return v
        case_info = raw_obj.get("case_info", None)
        if isinstance(case_info, dict):
            for cmeta in case_info.values():
                if isinstance(cmeta, dict):
                    v = _extract_reynolds_from_mapping(cmeta)
                    if v is not None:
                        return v

    if isinstance(raw_steps, Sequence) and len(raw_steps) > 0:
        s0 = raw_steps[0]
        gp = _extract_attr(s0, "global_params", None)
        if isinstance(gp, dict):
            v = _extract_reynolds_from_mapping(gp)
            if v is not None:
                return v

    v = _parse_reynolds_from_text(os.path.basename(str(src_path)))
    if v is not None:
        return v
    return _parse_reynolds_from_text(str(src_path))


def _reynolds_input_cfg(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    feats = cfg.get("features", {}) or {}
    blk = feats.get("reynolds_input", None)

    enabled_raw = feats.get("include_reynolds", feats.get("use_reynolds_input", False))
    mode_raw = feats.get("reynolds_mode", "log")
    if isinstance(blk, dict):
        enabled_raw = blk.get("enabled", enabled_raw)
        mode_raw = blk.get("mode", mode_raw)
    elif isinstance(blk, (bool, int)):
        enabled_raw = bool(blk)

    enabled = bool(enabled_raw)
    mode = str(mode_raw).strip().lower()
    if mode in ("log", "log10", "log_re", "re_log"):
        mode = "log"
    elif mode in ("nu", "inv_re", "inverse_re", "one_over_re", "1/re"):
        mode = "nu"
    else:
        if enabled:
            raise ValueError(
                "Unsupported Reynolds input mode. Use features.reynolds_input.mode="
                "'log' (log10(Re)) or 'nu' (1/Re)."
            )
        mode = "log"
    return enabled, mode


def _reynolds_to_conditioning_value(reynolds_number: float, mode: str) -> float:
    re_val = float(reynolds_number)
    if (not np.isfinite(re_val)) or (re_val <= 0.0):
        raise ValueError(f"Invalid Reynolds number for conditioning: {re_val}")
    if mode == "log":
        return float(np.log10(re_val))
    if mode == "nu":
        return float(1.0 / re_val)
    raise ValueError(f"Unsupported Reynolds conditioning mode: {mode}")


def _reynolds_from_meta(meta: Any) -> Optional[float]:
    if isinstance(meta, dict):
        v = _as_optional_float(meta.get("reynolds_number", None))
        if v is not None and v > 0.0:
            return v

        gp = meta.get("global_params", None)
        if isinstance(gp, dict):
            v = _extract_reynolds_from_mapping(gp)
            if v is not None:
                return v

        step_meta = meta.get("step_meta", None)
        if isinstance(step_meta, list):
            for sm in step_meta:
                v = _reynolds_from_meta(sm)
                if v is not None:
                    return v
    return None


def _build_reynolds_node_feature(
    *,
    n_nodes: int,
    cfg: Dict[str, Any],
    meta: Any,
    source_reynolds: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    enabled, mode = _reynolds_input_cfg(cfg)
    if not enabled:
        return None

    re_val = _reynolds_from_meta(meta)
    if re_val is None:
        re_val = source_reynolds
    if re_val is None:
        src_hint = None
        if isinstance(meta, dict):
            src_hint = meta.get("source_path", None)
            if src_hint is None and isinstance(meta.get("step_meta", None), list) and len(meta["step_meta"]) > 0:
                sm0 = meta["step_meta"][0]
                if isinstance(sm0, dict):
                    src_hint = sm0.get("source_path", None)
        raise RuntimeError(
            "features.reynolds_input.enabled=true, but Reynolds number was not found in sample metadata. "
            "Provide per-case metadata key 'reynolds_number' (or include it in case_name/path). "
            f"source={src_hint}"
        )

    cond_val = _reynolds_to_conditioning_value(float(re_val), mode=mode)
    return torch.full((int(n_nodes), 1), float(cond_val), device=device, dtype=dtype)


def _boundary_mask_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    feats = cfg.get("features", {}) or {}
    blk = feats.get("boundary_mask_input", None)

    enabled_raw = feats.get("include_boundary_mask", False)
    btype_raw: Any = "cylinder"
    feature_mode_raw: Any = "mask"
    distance_scale_raw: Any = "band"
    distance_clip_raw: Any = None
    center_raw: Any = [0.0, 0.0]
    radius_raw: Any = 0.5
    band_raw: Any = 0.08
    infer_raw: Any = False
    infer_band_scale_raw: Any = 1.5
    infer_min_nodes_raw: Any = 16

    if isinstance(blk, dict):
        enabled_raw = blk.get("enabled", enabled_raw)
        btype_raw = blk.get("type", blk.get("boundary_type", btype_raw))
        feature_mode_raw = blk.get("feature_mode", blk.get("mode", feature_mode_raw))
        distance_scale_raw = blk.get("distance_scale", blk.get("distance_norm", distance_scale_raw))
        distance_clip_raw = blk.get("distance_clip", distance_clip_raw)
        center_raw = blk.get("center_xy", blk.get("center", center_raw))
        radius_raw = blk.get("radius", radius_raw)
        band_raw = blk.get("band", blk.get("tolerance", band_raw))
        infer_raw = blk.get("infer_from_data", blk.get("infer", infer_raw))
        infer_band_scale_raw = blk.get("infer_band_scale", infer_band_scale_raw)
        infer_min_nodes_raw = blk.get("infer_min_nodes", infer_min_nodes_raw)
    elif isinstance(blk, (bool, int)):
        enabled_raw = bool(blk)

    if "boundary_mask_type" in feats:
        btype_raw = feats.get("boundary_mask_type", btype_raw)
    if "boundary_mask_feature_mode" in feats:
        feature_mode_raw = feats.get("boundary_mask_feature_mode", feature_mode_raw)
    if "boundary_mask_distance_scale" in feats:
        distance_scale_raw = feats.get("boundary_mask_distance_scale", distance_scale_raw)
    if "boundary_mask_distance_clip" in feats:
        distance_clip_raw = feats.get("boundary_mask_distance_clip", distance_clip_raw)
    if "boundary_mask_center" in feats:
        center_raw = feats.get("boundary_mask_center", center_raw)
    if "boundary_mask_radius" in feats:
        radius_raw = feats.get("boundary_mask_radius", radius_raw)
    if "boundary_mask_band" in feats:
        band_raw = feats.get("boundary_mask_band", band_raw)
    if "boundary_mask_infer_from_data" in feats:
        infer_raw = feats.get("boundary_mask_infer_from_data", infer_raw)

    enabled = bool(enabled_raw)
    btype = str(btype_raw).strip().lower()
    feature_mode = str(feature_mode_raw).strip().lower().replace("-", "_").replace("+", "_")
    if feature_mode in ("distance", "signed", "signed_dist", "signed_distance"):
        feature_mode = "signed_distance"
    elif feature_mode in ("unsigned", "unsigned_dist", "unsigned_distance"):
        feature_mode = "unsigned_distance"
    elif feature_mode in ("mask_signed_distance", "signed_distance_mask", "mask_signed"):
        feature_mode = "mask_signed_distance"
    elif feature_mode in ("mask_unsigned_distance", "unsigned_distance_mask", "mask_unsigned"):
        feature_mode = "mask_unsigned_distance"
    elif feature_mode in ("mask_only",):
        feature_mode = "mask"

    distance_scale = str(distance_scale_raw).strip().lower().replace("-", "_")
    if distance_scale in ("", "off"):
        distance_scale = "none"
    if distance_scale in ("tol", "tolerance"):
        distance_scale = "band"

    distance_clip = _as_optional_float(distance_clip_raw)
    infer_from_data = bool(infer_raw)

    if isinstance(center_raw, (list, tuple)) and len(center_raw) >= 2:
        cx = float(center_raw[0])
        cy = float(center_raw[1])
    else:
        raise ValueError(
            "features.boundary_mask_input.center_xy must be a list/tuple with two values [cx, cy]."
        )

    radius = float(radius_raw)
    band = float(band_raw)
    infer_band_scale = float(infer_band_scale_raw)
    infer_min_nodes = int(infer_min_nodes_raw)
    if enabled:
        if btype != "cylinder":
            raise ValueError("features.boundary_mask_input.type currently supports only 'cylinder'.")
        valid_modes = {
            "mask",
            "signed_distance",
            "unsigned_distance",
            "mask_signed_distance",
            "mask_unsigned_distance",
        }
        if feature_mode not in valid_modes:
            raise ValueError(
                "features.boundary_mask_input.feature_mode must be one of "
                f"{sorted(valid_modes)}, got '{feature_mode_raw}'."
            )
        if distance_scale not in ("none", "band", "radius"):
            raise ValueError(
                "features.boundary_mask_input.distance_scale must be one of "
                "['none','band','radius']."
            )
        if distance_clip is not None and distance_clip <= 0.0:
            raise ValueError("features.boundary_mask_input.distance_clip must be > 0 when provided.")
        if infer_from_data:
            if infer_band_scale <= 0.0:
                raise ValueError("features.boundary_mask_input.infer_band_scale must be > 0.")
            if infer_min_nodes < 8:
                raise ValueError("features.boundary_mask_input.infer_min_nodes must be >= 8.")
        else:
            if radius <= 0.0:
                raise ValueError("features.boundary_mask_input.radius must be > 0.")
            if band <= 0.0:
                raise ValueError("features.boundary_mask_input.band must be > 0.")

    return {
        "enabled": enabled,
        "type": btype,
        "feature_mode": feature_mode,
        "distance_scale": distance_scale,
        "distance_clip": distance_clip,
        "center_xy": [cx, cy],
        "radius": radius,
        "band": band,
        "infer_from_data": infer_from_data,
        "infer_band_scale": infer_band_scale,
        "infer_min_nodes": infer_min_nodes,
    }


def _boundary_extra_in_channels(cfg: Dict[str, Any]) -> int:
    bcfg = _boundary_mask_cfg(cfg)
    if not bool(bcfg.get("enabled", False)):
        return 0
    mode = str(bcfg.get("feature_mode", "mask"))
    if mode in ("mask", "signed_distance", "unsigned_distance"):
        return 1
    if mode in ("mask_signed_distance", "mask_unsigned_distance"):
        return 2
    return 1


def _parse_relative_geometry_channels(raw: Any) -> List[str]:
    if raw is None:
        raw_items: List[Any] = ["domain_distances", "cylinder_signed_distance"]
    elif isinstance(raw, str):
        if raw.strip().lower() in ("default", "all"):
            raw_items = ["domain_distances", "cylinder_signed_distance"]
        else:
            raw_items = [p for p in re.split(r"[,+\s]+", raw) if p]
    elif isinstance(raw, (list, tuple)):
        raw_items = list(raw)
    else:
        raise ValueError("features.relative_geometry_input.channels must be a string or list.")

    out: List[str] = []
    for item in raw_items:
        name = str(item).strip().lower().replace("-", "_")
        if name in (
            "domain",
            "domain_distance",
            "domain_distances",
            "outer_boundary_distances",
            "boundary_distances",
            "wall_distances",
        ):
            name = "domain_distances"
        elif name in (
            "cylinder_signed",
            "cylinder_signed_distance",
            "signed_cylinder_distance",
            "obstacle_signed_distance",
        ):
            name = "cylinder_signed_distance"
        elif name in (
            "cylinder_unsigned",
            "cylinder_unsigned_distance",
            "unsigned_cylinder_distance",
            "obstacle_unsigned_distance",
        ):
            name = "cylinder_unsigned_distance"
        elif name in ("cylinder_normal", "obstacle_normal", "radial_unit_vector"):
            name = "cylinder_normal"
        else:
            raise ValueError(
                "Unsupported relative geometry channel "
                f"'{item}'. Use domain_distances, cylinder_signed_distance, "
                "cylinder_unsigned_distance, or cylinder_normal."
            )
        if name not in out:
            out.append(name)
    if len(out) == 0:
        raise ValueError("features.relative_geometry_input.channels cannot be empty.")
    return out


def _parse_bbox(raw: Any, *, field_name: str) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("", "auto", "infer", "from_data"):
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        vals = [float(v) for v in raw]
        xmin, xmax, ymin, ymax = vals
        if not all(np.isfinite(v) for v in vals):
            raise ValueError(f"{field_name} must contain finite values.")
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"{field_name} must be [xmin, xmax, ymin, ymax] with positive spans.")
        return vals
    raise ValueError(f"{field_name} must be [xmin, xmax, ymin, ymax], null, or 'auto'.")


def _relative_geometry_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    feats = cfg.get("features", {}) or {}
    blk = feats.get("relative_geometry_input", None)
    bcfg = _boundary_mask_cfg(cfg)

    enabled_raw = feats.get("include_relative_geometry", False)
    channels_raw: Any = None
    domain_bbox_raw: Any = cfg.get("data", {}).get("bbox", None)
    domain_clip_raw: Any = None
    cylinder_scale_raw: Any = "radius"
    cylinder_clip_raw: Any = None
    center_raw: Any = bcfg.get("center_xy", [0.0, 0.0])
    radius_raw: Any = bcfg.get("radius", 0.5)
    band_raw: Any = bcfg.get("band", 0.08)
    infer_raw: Any = bcfg.get("infer_from_data", False)
    infer_band_scale_raw: Any = bcfg.get("infer_band_scale", 1.5)
    infer_min_nodes_raw: Any = bcfg.get("infer_min_nodes", 16)

    if isinstance(blk, dict):
        enabled_raw = blk.get("enabled", enabled_raw)
        channels_raw = blk.get("channels", channels_raw)
        domain_bbox_raw = blk.get("domain_bbox", blk.get("bbox", domain_bbox_raw))
        domain_clip_raw = blk.get("domain_distance_clip", blk.get("distance_clip", domain_clip_raw))
        cylinder_scale_raw = blk.get(
            "cylinder_distance_scale",
            blk.get("distance_scale", cylinder_scale_raw),
        )
        cylinder_clip_raw = blk.get(
            "cylinder_distance_clip",
            blk.get("distance_clip", cylinder_clip_raw),
        )
        center_raw = blk.get("center_xy", blk.get("center", center_raw))
        radius_raw = blk.get("radius", radius_raw)
        band_raw = blk.get("band", blk.get("tolerance", band_raw))
        infer_raw = blk.get("infer_from_data", blk.get("infer", infer_raw))
        infer_band_scale_raw = blk.get("infer_band_scale", infer_band_scale_raw)
        infer_min_nodes_raw = blk.get("infer_min_nodes", infer_min_nodes_raw)
    elif isinstance(blk, (bool, int)):
        enabled_raw = bool(blk)

    enabled = bool(enabled_raw)
    if not enabled:
        return {
            "enabled": False,
            "channels": [],
            "domain_bbox": None,
            "domain_distance_clip": None,
            "cylinder_distance_scale": "radius",
            "cylinder_distance_clip": None,
            "center_xy": [0.0, 0.0],
            "radius": 0.5,
            "band": 0.08,
            "infer_from_data": False,
            "infer_band_scale": 1.5,
            "infer_min_nodes": 16,
        }

    channels = _parse_relative_geometry_channels(channels_raw)
    domain_bbox = _parse_bbox(domain_bbox_raw, field_name="features.relative_geometry_input.domain_bbox")
    domain_clip = _as_optional_float(domain_clip_raw)
    cylinder_clip = _as_optional_float(cylinder_clip_raw)

    cylinder_scale = str(cylinder_scale_raw).strip().lower().replace("-", "_")
    if cylinder_scale in ("", "off"):
        cylinder_scale = "none"
    if cylinder_scale in ("tol", "tolerance"):
        cylinder_scale = "band"
    if cylinder_scale in ("domain_short", "domain_min", "short_domain"):
        cylinder_scale = "domain"
    if cylinder_scale not in ("none", "radius", "band", "domain"):
        raise ValueError(
            "features.relative_geometry_input.cylinder_distance_scale must be one of "
            "['none','radius','band','domain']."
        )

    if isinstance(center_raw, (list, tuple)) and len(center_raw) >= 2:
        center_xy = [float(center_raw[0]), float(center_raw[1])]
    else:
        raise ValueError(
            "features.relative_geometry_input.center_xy must be a list/tuple with two values [cx, cy]."
        )
    radius = float(radius_raw)
    band = float(band_raw)
    infer_band_scale = float(infer_band_scale_raw)
    infer_min_nodes = int(infer_min_nodes_raw)
    uses_cylinder = any(ch.startswith("cylinder_") for ch in channels)

    if enabled:
        if domain_clip is not None and domain_clip <= 0.0:
            raise ValueError("features.relative_geometry_input.domain_distance_clip must be > 0 when provided.")
        if cylinder_clip is not None and cylinder_clip <= 0.0:
            raise ValueError("features.relative_geometry_input.cylinder_distance_clip must be > 0 when provided.")
        if uses_cylinder:
            if bool(infer_raw):
                if infer_band_scale <= 0.0:
                    raise ValueError("features.relative_geometry_input.infer_band_scale must be > 0.")
                if infer_min_nodes < 8:
                    raise ValueError("features.relative_geometry_input.infer_min_nodes must be >= 8.")
            else:
                if radius <= 0.0:
                    raise ValueError("features.relative_geometry_input.radius must be > 0.")
                if band <= 0.0:
                    raise ValueError("features.relative_geometry_input.band must be > 0.")

    return {
        "enabled": enabled,
        "channels": channels,
        "domain_bbox": domain_bbox,
        "domain_distance_clip": domain_clip,
        "cylinder_distance_scale": cylinder_scale,
        "cylinder_distance_clip": cylinder_clip,
        "center_xy": center_xy,
        "radius": radius,
        "band": band,
        "infer_from_data": bool(infer_raw),
        "infer_band_scale": infer_band_scale,
        "infer_min_nodes": infer_min_nodes,
    }


def _relative_geometry_in_channels(cfg: Dict[str, Any]) -> int:
    rcfg = _relative_geometry_cfg(cfg)
    if not bool(rcfg.get("enabled", False)):
        return 0
    dim = 0
    for ch in rcfg.get("channels", []):
        if ch == "domain_distances":
            dim += 4
        elif ch in ("cylinder_signed_distance", "cylinder_unsigned_distance"):
            dim += 1
        elif ch == "cylinder_normal":
            dim += 2
        else:
            raise ValueError(f"Unsupported relative geometry channel: {ch}")
    return dim


def _infer_cylinder_params_from_geometry(
    *,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    infer_band_scale: float,
    infer_min_nodes: int,
) -> Dict[str, float]:
    if pos.ndim != 2 or int(pos.size(1)) < 2:
        raise ValueError(f"Expected pos shape [N,>=2], got {tuple(pos.shape)}")
    if edge_index.ndim != 2:
        raise ValueError(f"Expected edge_index rank 2, got {tuple(edge_index.shape)}")

    pxy = pos[:, :2].detach().to(device="cpu", dtype=torch.float32).numpy()
    ei = edge_index.detach().to(device="cpu", dtype=torch.long)
    if ei.size(0) != 2 and ei.size(1) == 2:
        ei = ei.t().contiguous()
    if ei.size(0) != 2:
        raise ValueError(f"Expected edge_index [2,E] or [E,2], got {tuple(edge_index.shape)}")

    src = ei[0].numpy()
    dst = ei[1].numpy()
    n = int(pxy.shape[0])

    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    udeg, cnt = np.unique(deg, return_counts=True)
    interior_deg = int(udeg[int(np.argmax(cnt))]) if len(udeg) > 0 else 8
    cand = np.flatnonzero(deg < interior_deg).astype(np.int64)
    if cand.size < max(32, infer_min_nodes):
        raise RuntimeError(
            f"Too few candidate boundary nodes for inference (cand={cand.size}, interior_deg={interior_deg})."
        )

    cand_mask = np.zeros((n,), dtype=bool)
    cand_mask[cand] = True
    valid = cand_mask[src] & cand_mask[dst]
    es = src[valid]
    ed = dst[valid]

    # Build compact node-id mapping for candidate subgraph.
    id_of = -np.ones((n,), dtype=np.int64)
    id_of[cand] = np.arange(cand.size, dtype=np.int64)
    cs = id_of[es]
    cd = id_of[ed]
    good = (cs >= 0) & (cd >= 0)
    cs = cs[good]
    cd = cd[good]

    m = int(cand.size)
    adj: List[List[int]] = [[] for _ in range(m)]
    for u, v in zip(cs.tolist(), cd.tolist()):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

    # Connected components on candidate boundary graph.
    seen = np.zeros((m,), dtype=bool)
    comps: List[np.ndarray] = []
    for i in range(m):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        out: List[int] = []
        while stack:
            a = stack.pop()
            out.append(a)
            for b in adj[a]:
                if not seen[b]:
                    seen[b] = True
                    stack.append(b)
        comps.append(np.asarray(out, dtype=np.int64))

    if len(comps) == 0:
        raise RuntimeError("No connected components found for boundary candidates.")

    x = pxy[:, 0]
    y = pxy[:, 1]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = max(1e-6, min(x_max - x_min, y_max - y_min))
    edge_eps = 0.01 * span

    best_nodes: Optional[np.ndarray] = None
    best_size = -1

    # Prefer the largest non-outer boundary component (typically cylinder loop).
    for comp in comps:
        if int(comp.size) < int(infer_min_nodes):
            continue
        nodes = cand[comp]
        cx = x[nodes]
        cy = y[nodes]
        touches_outer = (
            (float(np.min(cx)) <= (x_min + edge_eps))
            or (float(np.max(cx)) >= (x_max - edge_eps))
            or (float(np.min(cy)) <= (y_min + edge_eps))
            or (float(np.max(cy)) >= (y_max - edge_eps))
        )
        if touches_outer:
            continue
        if int(nodes.size) > best_size:
            best_nodes = nodes
            best_size = int(nodes.size)

    # Fallback: smallest large component by bbox area (often interior hole boundary).
    if best_nodes is None:
        best_area = None
        for comp in comps:
            if int(comp.size) < int(infer_min_nodes):
                continue
            nodes = cand[comp]
            cx = x[nodes]
            cy = y[nodes]
            area = float((np.max(cx) - np.min(cx)) * (np.max(cy) - np.min(cy)))
            if best_area is None or area < best_area:
                best_area = area
                best_nodes = nodes
        if best_nodes is None:
            raise RuntimeError("Could not identify an interior boundary component for cylinder inference.")

    bx = x[best_nodes]
    by = y[best_nodes]
    cxi = float(np.mean(bx))
    cyi = float(np.mean(by))
    rr = np.sqrt((bx - cxi) ** 2 + (by - cyi) ** 2)
    rad = float(np.mean(rr))
    rstd = float(np.std(rr))
    if not np.isfinite(rad) or rad <= 0.0:
        raise RuntimeError("Inferred cylinder radius is invalid.")

    # Estimate geometric spacing from boundary edges in the selected component.
    comp_mask = np.zeros((n,), dtype=bool)
    comp_mask[best_nodes] = True
    ecomp = comp_mask[src] & comp_mask[dst]
    if np.any(ecomp):
        dxy = pxy[dst[ecomp], :] - pxy[src[ecomp], :]
        elen = np.sqrt(np.sum(dxy * dxy, axis=1))
        edge_med = float(np.median(elen)) if elen.size > 0 else 0.0
    else:
        edge_med = 0.0
    band = float(max(3.0 * rstd, float(infer_band_scale) * edge_med, 1e-6))

    return {
        "center_x": cxi,
        "center_y": cyi,
        "radius": rad,
        "band": band,
    }


def _boundary_infer_cache_key(pos: torch.Tensor, edge_index: torch.Tensor) -> Tuple[Any, ...]:
    pxy = pos[:, :2]
    x_min = float(pxy[:, 0].min().item())
    x_max = float(pxy[:, 0].max().item())
    y_min = float(pxy[:, 1].min().item())
    y_max = float(pxy[:, 1].max().item())
    ei = edge_index
    if ei.ndim == 2 and ei.size(0) == 2:
        e = int(ei.size(1))
    elif ei.ndim == 2 and ei.size(1) == 2:
        e = int(ei.size(0))
    else:
        e = int(ei.numel())
    return (
        int(pos.size(0)),
        int(pos.size(1)),
        int(e),
        round(x_min, 5),
        round(x_max, 5),
        round(y_min, 5),
        round(y_max, 5),
    )


def _resolve_cylinder_geometry(
    *,
    pos: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    device: torch.device,
    center_xy: Sequence[float],
    radius: float,
    band: float,
    infer_from_data: bool,
    infer_band_scale: float,
    infer_min_nodes: int,
) -> Tuple[torch.Tensor, float, float]:
    if bool(infer_from_data):
        if edge_index is None:
            raise RuntimeError("Cylinder geometry inference requires edge_index, but none was provided.")
        key = _boundary_infer_cache_key(pos, edge_index)
        if key in _BOUNDARY_INFER_CACHE:
            inf = _BOUNDARY_INFER_CACHE[key]
        else:
            inf = _infer_cylinder_params_from_geometry(
                pos=pos,
                edge_index=edge_index,
                infer_band_scale=float(infer_band_scale),
                infer_min_nodes=int(infer_min_nodes),
            )
            _BOUNDARY_INFER_CACHE[key] = inf
            if len(_BOUNDARY_INFER_CACHE) > 32:
                _BOUNDARY_INFER_CACHE.pop(next(iter(_BOUNDARY_INFER_CACHE)))
        cxy = torch.tensor([inf["center_x"], inf["center_y"]], device=device, dtype=torch.float32).view(1, 2)
        return cxy, float(inf["radius"]), float(inf["band"])

    cxy = torch.tensor(center_xy, device=device, dtype=torch.float32).view(1, 2)
    return cxy, float(radius), float(band)


def _build_boundary_mask_node_feature(
    *,
    pos: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    bcfg = _boundary_mask_cfg(cfg)
    if not bool(bcfg.get("enabled", False)):
        return None

    if pos.ndim != 2 or int(pos.size(1)) < 2:
        raise ValueError(
            f"Boundary mask input requires positions with at least 2 columns; got shape {tuple(pos.shape)}"
        )

    pxy = pos[:, :2].to(device=device, dtype=torch.float32)
    cxy, radius, band = _resolve_cylinder_geometry(
        pos=pos,
        edge_index=edge_index,
        device=device,
        center_xy=bcfg["center_xy"],
        radius=float(bcfg["radius"]),
        band=float(bcfg["band"]),
        infer_from_data=bool(bcfg.get("infer_from_data", False)),
        infer_band_scale=float(bcfg.get("infer_band_scale", 1.5)),
        infer_min_nodes=int(bcfg.get("infer_min_nodes", 16)),
    )
    rad = torch.linalg.norm(pxy - cxy, dim=1)
    signed_dist = rad - float(radius)
    unsigned_dist = torch.abs(signed_dist)
    mask = (unsigned_dist <= float(band)).to(dtype=dtype).view(-1, 1)

    mode = str(bcfg.get("feature_mode", "mask"))
    out: List[torch.Tensor] = []
    if mode in ("mask", "mask_signed_distance", "mask_unsigned_distance"):
        out.append(mask)

    if mode in ("signed_distance", "mask_signed_distance"):
        dist = signed_dist
    elif mode in ("unsigned_distance", "mask_unsigned_distance"):
        dist = unsigned_dist
    else:
        dist = None

    if dist is not None:
        scale_mode = str(bcfg.get("distance_scale", "band"))
        if scale_mode == "band":
            scale = max(float(abs(band)), 1e-12)
        elif scale_mode == "radius":
            scale = max(float(abs(radius)), 1e-12)
        else:
            scale = 1.0
        d = dist / float(scale)

        dclip = bcfg.get("distance_clip", None)
        if dclip is not None:
            c = float(dclip)
            if mode in ("signed_distance", "mask_signed_distance"):
                d = torch.clamp(d, min=-c, max=c)
            else:
                d = torch.clamp(d, min=0.0, max=c)
        out.append(d.to(dtype=dtype).view(-1, 1))

    if len(out) == 0:
        return None
    if len(out) == 1:
        return out[0]
    return torch.cat(out, dim=1)


def _build_relative_geometry_node_features(
    *,
    pos: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    rcfg = _relative_geometry_cfg(cfg)
    if not bool(rcfg.get("enabled", False)):
        return None
    if pos.ndim != 2 or int(pos.size(1)) < 2:
        raise ValueError(
            f"Relative geometry input requires positions with at least 2 columns; got shape {tuple(pos.shape)}"
        )

    pxy = pos[:, :2].to(device=device, dtype=torch.float32)
    x = pxy[:, 0]
    y = pxy[:, 1]
    bbox = rcfg.get("domain_bbox", None)
    if bbox is None:
        xmin = float(x.min().item())
        xmax = float(x.max().item())
        ymin = float(y.min().item())
        ymax = float(y.max().item())
    else:
        xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    x_span = max(float(xmax - xmin), 1e-12)
    y_span = max(float(ymax - ymin), 1e-12)
    domain_scale = max(min(x_span, y_span), 1e-12)

    out: List[torch.Tensor] = []
    channels = list(rcfg.get("channels", []))
    if "domain_distances" in channels:
        domain = torch.stack(
            [
                (x - float(xmin)) / x_span,
                (float(xmax) - x) / x_span,
                (y - float(ymin)) / y_span,
                (float(ymax) - y) / y_span,
            ],
            dim=1,
        )
        dclip = rcfg.get("domain_distance_clip", None)
        if dclip is not None:
            domain = torch.clamp(domain, min=0.0, max=float(dclip))
        out.append(domain.to(dtype=dtype))

    needs_cylinder = any(ch in channels for ch in (
        "cylinder_signed_distance",
        "cylinder_unsigned_distance",
        "cylinder_normal",
    ))
    if needs_cylinder:
        cxy, radius, band = _resolve_cylinder_geometry(
            pos=pos,
            edge_index=edge_index,
            device=device,
            center_xy=rcfg["center_xy"],
            radius=float(rcfg["radius"]),
            band=float(rcfg["band"]),
            infer_from_data=bool(rcfg.get("infer_from_data", False)),
            infer_band_scale=float(rcfg.get("infer_band_scale", 1.5)),
            infer_min_nodes=int(rcfg.get("infer_min_nodes", 16)),
        )
        rel = pxy - cxy
        rad = torch.linalg.norm(rel, dim=1)
        signed_dist = rad - float(radius)

        scale_mode = str(rcfg.get("cylinder_distance_scale", "radius"))
        if scale_mode == "band":
            scale = max(float(abs(band)), 1e-12)
        elif scale_mode == "radius":
            scale = max(float(abs(radius)), 1e-12)
        elif scale_mode == "domain":
            scale = domain_scale
        else:
            scale = 1.0

        cclip = rcfg.get("cylinder_distance_clip", None)
        if "cylinder_signed_distance" in channels:
            d = signed_dist / float(scale)
            if cclip is not None:
                c = float(cclip)
                d = torch.clamp(d, min=-c, max=c)
            out.append(d.to(dtype=dtype).view(-1, 1))

        if "cylinder_unsigned_distance" in channels:
            d = torch.abs(signed_dist) / float(scale)
            if cclip is not None:
                d = torch.clamp(d, min=0.0, max=float(cclip))
            out.append(d.to(dtype=dtype).view(-1, 1))

        if "cylinder_normal" in channels:
            nrm = rel / rad.clamp_min(1e-12).view(-1, 1)
            out.append(nrm.to(dtype=dtype))

    if len(out) == 0:
        return None
    if len(out) == 1:
        return out[0]
    return torch.cat(out, dim=1)


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
    dual_volume_raw = None
    dual_volume_name = None
    for vk in ("dual_volume", "dual_volumes", "control_volume", "control_volumes", "cell_area", "cell_areas", "node_area", "node_areas"):
        vv = _extract_attr(step, vk, None)
        if vv is not None:
            dual_volume_raw = vv
            dual_volume_name = vk
            break
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

    pos_t = _as_2d_float(pos, "pos")
    out = {
        "x": _as_2d_float(x, "x"),
        "y": None if y is None else _as_2d_float(y, "y"),
        "pos": pos_t,
        "edge_index": _as_edge_index(edge_index),
        "dual_volume": (
            None
            if dual_volume_raw is None
            else _as_optional_node_scalar(dual_volume_raw, str(dual_volume_name), int(pos_t.size(0)))
        ),
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
    dual_volume: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
        None if dual_volume is None else dual_volume.index_select(0, keep_idx),
    )


@dataclass
class PointPair:
    x: torch.Tensor
    y: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    dual_volume: Optional[torch.Tensor] = None
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
            src_reynolds = _infer_reynolds_number(raw_obj, raw_steps, src_path)
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
                dual_volume0 = s0.get("dual_volume", None)

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
                        xs, ys, ps, eis, dvs = _subgraph_by_index(
                            x_sel,
                            y_sel,
                            pos_sel,
                            ei0,
                            keep,
                            dual_volume=dual_volume0,
                        )
                        if xs.size(0) == 0:
                            continue
                        pairs.append(
                            PointPair(
                                x=xs,
                                y=ys,
                                pos=ps,
                                edge_index=eis,
                                dual_volume=dvs,
                                t_src=s0["time"],
                                t_dst=s1["time"],
                                meta={
                                    "pair_t": t,
                                    "z_group": gidx,
                                    "source_index": src_idx,
                                    "source_path": os.path.abspath(src_path),
                                    "reynolds_number": src_reynolds,
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
                            dual_volume=dual_volume0,
                            t_src=s0["time"],
                            t_dst=s1["time"],
                            meta={
                                "pair_t": t,
                                "source_index": src_idx,
                                "source_path": os.path.abspath(src_path),
                                "reynolds_number": src_reynolds,
                            },
                        )
                    )

        if len(pairs) == 0:
            raise RuntimeError("No training pairs were built from the dataset.")

        self.pairs = pairs
        self.x_dim = int(self.pairs[0].x.size(1))
        self.y_dim = int(self.pairs[0].y.size(1))
        self.pos_dim = int(self.pairs[0].pos.size(1))
        self.has_dual_volume = any(p.dual_volume is not None for p in self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pairs[idx]
        return {
            "x": p.x,
            "y": p.y,
            "pos": p.pos,
            "edge_index": p.edge_index,
            "dual_volume": p.dual_volume,
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
            src_reynolds = _infer_reynolds_number(raw_obj, raw_steps, src_path)
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
                dual_volume = s.get("dual_volume", None)

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
                        xs, ys, ps, eis, dvs = _subgraph_by_index(
                            x_sel,
                            y_sel,
                            pos_sel,
                            ei,
                            keep,
                            dual_volume=dual_volume,
                        )
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
                                "dual_volume": dvs,
                                "time": s["time"],
                                "meta": {
                                    "t": t,
                                    "z_group": gidx,
                                    "source_index": src_idx,
                                    "source_path": os.path.abspath(src_path),
                                    "reynolds_number": src_reynolds,
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
                            "dual_volume": dual_volume,
                            "time": s["time"],
                            "meta": {
                                "t": t,
                                "source_index": src_idx,
                                "source_path": os.path.abspath(src_path),
                                "reynolds_number": src_reynolds,
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
        self.has_dual_volume = any(
            s.get("dual_volume", None) is not None
            for seq in self.sequences
            for s in seq
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_id, start = self.windows[idx]
        seq = self.sequences[seq_id]
        chunk = seq[start : start + self.window_size]

        x_list = [s["x"] for s in chunk]
        pos_list = [s["pos"] for s in chunk]
        edge_index_list = [s["edge_index"] for s in chunk]
        dual_volume_list = [s.get("dual_volume", None) for s in chunk]
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
            "dual_volume_list": dual_volume_list,
            "t_list": t_list,
            "meta": {
                "seq_id": seq_id,
                "start_t": start,
                "window_size": self.window_size,
                "step_meta": [s.get("meta", {}) for s in chunk],
                "reynolds_number": chunk[0].get("meta", {}).get("reynolds_number", None),
            },
            # Back-compat convenience for one-step paths.
            "x": x_list[0],
            "y": y_list[0],
            "pos": pos_list[0],
            "edge_index": edge_index_list[0],
            "dual_volume": dual_volume_list[0],
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
    pos_mu: Optional[torch.Tensor] = None
    pos_std: Optional[torch.Tensor] = None
    pos_mode: str = "none"


def _canonical_norm_mode(raw: Any, *, field_name: str) -> str:
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
        f"Unsupported normalization mode for {field_name}. "
        "Use 'zscore' or 'minmax_11'."
    )


def _normalization_mode_from_cfg(cfg: Dict[str, Any]) -> str:
    feats = cfg.get("features", {}) or {}
    norm_blk = feats.get("normalization", None)
    raw = None
    if isinstance(norm_blk, dict):
        raw = norm_blk.get("normalization_mode", None)
    if raw is None:
        raw = feats.get("normalization_mode", feats.get("norm_mode", "zscore"))
    return _canonical_norm_mode(
        raw,
        field_name="features.normalization.normalization_mode",
    )


def _pos_normalization_cfg(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    feats = cfg.get("features", {}) or {}
    pos_blk = feats.get("pos_normalization", None)

    enabled_raw = feats.get("pos_normalize", False)
    mode_raw = feats.get("pos_normalization_mode", None)
    if isinstance(pos_blk, dict):
        enabled_raw = pos_blk.get("enabled", enabled_raw)
        mode_raw = pos_blk.get("mode", mode_raw)
    elif isinstance(pos_blk, (bool, int)):
        enabled_raw = bool(pos_blk)

    enabled = bool(enabled_raw)
    if mode_raw is None:
        mode_raw = _normalization_mode_from_cfg(cfg)

    if (not enabled) and (mode_raw is None):
        return False, "zscore"
    mode = _canonical_norm_mode(
        mode_raw,
        field_name="features.pos_normalization.mode",
    )
    return enabled, mode


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
    compute_xy: bool = True,
    component_mode: str = "independent",
    shared_channels: Optional[Sequence[int]] = None,
    compute_pos: bool = False,
    pos_mode: str = "zscore",
    rollout_steps: Optional[int] = None,
) -> NormStats:
    mode = str(mode).strip().lower()
    if compute_xy and mode not in ("zscore", "minmax_11"):
        raise ValueError(f"Unsupported normalization mode: {mode}")
    component_mode = str(component_mode).strip().lower()
    if component_mode not in ("independent", "shared"):
        raise ValueError(f"Unsupported component scale mode: {component_mode}")
    pos_mode = str(pos_mode).strip().lower()
    if compute_pos and pos_mode not in ("zscore", "minmax_11"):
        raise ValueError(f"Unsupported position normalization mode: {pos_mode}")

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
    p_sum = None
    p_sq_sum = None
    p_min = None
    p_max = None
    n_p = 0

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

    def _accum_pos(pos_raw: torch.Tensor) -> None:
        nonlocal p_sum, p_sq_sum, p_min, p_max, n_p
        p = pos_raw.to(device=device, dtype=torch.float32)
        if pos_mode == "zscore":
            if p_sum is None:
                p_sum = p.sum(dim=0)
                p_sq_sum = (p * p).sum(dim=0)
            else:
                p_sum += p.sum(dim=0)
                p_sq_sum += (p * p).sum(dim=0)
        else:
            p_lo = p.min(dim=0).values
            p_hi = p.max(dim=0).values
            if p_min is None:
                p_min = p_lo
                p_max = p_hi
            else:
                p_min = torch.minimum(p_min, p_lo)
                p_max = torch.maximum(p_max, p_hi)
        n_p += int(p.size(0))

    for idx in indices:
        ex = dataset[int(idx)]
        if ("x" in ex) and ("y" in ex) and torch.is_tensor(ex["x"]) and torch.is_tensor(ex["y"]):
            if compute_xy:
                _accum_xy(ex["x"], ex["y"])
            if compute_pos:
                if ("pos" not in ex) or (not torch.is_tensor(ex["pos"])):
                    raise KeyError("Position normalization requested, but dataset example has no tensor key 'pos'.")
                _accum_pos(ex["pos"])
            continue

        x_list = ex.get("x_list", None)
        y_list = ex.get("y_list", None)
        pos_list = ex.get("pos_list", None)
        if isinstance(x_list, list) and isinstance(y_list, list) and len(y_list) > 0:
            use_steps = len(y_list)
            if rollout_steps is not None:
                use_steps = min(use_steps, max(1, int(rollout_steps)))
            for k in range(use_steps):
                if compute_xy:
                    _accum_xy(x_list[k], y_list[k])
                if compute_pos:
                    if (not isinstance(pos_list, list)) or k >= len(pos_list) or (not torch.is_tensor(pos_list[k])):
                        raise KeyError(
                            "Position normalization requested, but window example has no valid pos_list[k] tensor."
                        )
                    _accum_pos(pos_list[k])
            continue

        raise KeyError("Dataset example must provide x/y tensors or x_list/y_list tensors.")

    x_mu = None
    x_std = None
    y_mu = None
    y_std = None
    if compute_xy and (n_x > 0) and (n_y > 0):
        if mode == "zscore":
            if x_sum is not None and y_sum is not None:
                x_mu = x_sum / float(n_x)
                y_mu = y_sum / float(n_y)
                x_var = (x_sq_sum / float(n_x)) - (x_mu * x_mu)
                y_var = (y_sq_sum / float(n_y)) - (y_mu * y_mu)
                x_std = torch.sqrt(torch.clamp(x_var, min=1e-12))
                y_std = torch.sqrt(torch.clamp(y_var, min=1e-12))
        else:
            if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
                x_mu = 0.5 * (x_min + x_max)
                y_mu = 0.5 * (y_min + y_max)
                x_std = (0.5 * (x_max - x_min)).clamp_min(1e-12)
                y_std = (0.5 * (y_max - y_min)).clamp_min(1e-12)

    if (
        compute_xy
        and x_std is not None
        and y_std is not None
        and component_mode == "shared"
        and shared_channels is not None
    ):
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

    pos_mu = None
    pos_std = None
    out_pos_mode = "none"
    if compute_pos and (n_p > 0):
        out_pos_mode = pos_mode
        if pos_mode == "zscore":
            if p_sum is not None:
                pos_mu = p_sum / float(n_p)
                p_var = (p_sq_sum / float(n_p)) - (pos_mu * pos_mu)
                pos_std = torch.sqrt(torch.clamp(p_var, min=1e-12))
        else:
            if p_min is not None and p_max is not None:
                pos_mu = 0.5 * (p_min + p_max)
                pos_std = (0.5 * (p_max - p_min)).clamp_min(1e-12)

    return NormStats(
        x_mu,
        x_std,
        y_mu,
        y_std,
        mode=mode,
        pos_mu=pos_mu,
        pos_std=pos_std,
        pos_mode=out_pos_mode,
    )


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
    dual_volume: Optional[torch.Tensor] = None,
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

    if dual_volume is not None:
        dv = dual_volume.to(device=pos.device, dtype=torch.float32)
        if dv.ndim == 2:
            dv = dv[:, 0]
        elif dv.ndim != 1:
            raise ValueError(f"dual_volume must be [N] or [N,1], got shape {tuple(dv.shape)}")
        if int(dv.numel()) != int(pos.size(0)):
            raise ValueError(
                f"dual_volume node count mismatch: got {int(dv.numel())}, expected {int(pos.size(0))}"
            )
        valid_dv = torch.isfinite(dv) & (dv > 0)
        if bool(valid_dv.any()):
            area = torch.where(valid_dv, dv, area)

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
    dual_volume: Optional[torch.Tensor],
    cfg: Dict[str, Any],
    compute_adv: bool,
    compute_diff: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    geom = _geometry_from_pos_edge(pos, edge_index, dual_volume=dual_volume)
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
    dual_volume: Optional[torch.Tensor],
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

    geom = _geometry_from_pos_edge(pos, edge_index, dual_volume=dual_volume)
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
    dual_volume: Optional[torch.Tensor] = None,
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
    dv_f = None if dual_volume is None else dual_volume.to(device=device, dtype=torch.float32)

    backend = _physics_backend(cfg)
    if backend == "mls":
        r_adv_abs, r_diff_abs, _area = _physics_terms_mls_abs_point(
            x_abs=x_abs_f,
            pos=pos_f,
            edge_index=ei_f,
            dual_volume=dv_f,
            cfg=cfg,
            compute_adv=need_adv,
            compute_diff=need_diff,
        )
    else:
        r_adv_abs, r_diff_abs, _area = _physics_terms_dec_abs_point(
            x_abs=x_abs_f,
            pos=pos_f,
            edge_index=ei_f,
            dual_volume=dv_f,
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


def _model_type_from_cfg(cfg: Dict[str, Any]) -> str:
    mcfg = cfg.get("model", {}) or {}
    raw = mcfg.get("type", mcfg.get("name", "sageconv"))
    key = str(raw).strip().lower().replace("-", "_")
    sage_names = {"featurenet", "feature_net", "graphsage", "graph_sage", "sage", "sageconv"}
    mesh_names = {"meshgraphnet", "mesh_graph_net", "mgn"}
    flux_names = {"fluxgraphnet", "flux_graph_net", "fluxgnn", "flux"}
    if key in sage_names:
        return "sageconv"
    if key in mesh_names:
        return "meshgraphnet"
    if key in flux_names:
        return "fluxgraphnet"
    raise ValueError(
        "Unsupported model.type/model.name. Use 'sageconv', 'meshgraphnet', or 'fluxgraphnet'. "
        f"Got {raw!r}."
    )


def _model_name_from_cfg(cfg: Dict[str, Any]) -> str:
    """Back-compat wrapper for older scripts that asked for the model name."""
    return _model_type_from_cfg(cfg)


def _build_model(cfg: Dict[str, Any], in_dim: int, out_dim: int, device: torch.device) -> torch.nn.Module:
    mcfg = cfg.get("model", {}) or {}
    model_type = _model_type_from_cfg(cfg)
    if model_type == "sageconv":
        return FeatureNet(
            in_channels=in_dim,
            out_channels=out_dim,
            hidden=int(mcfg.get("hidden", 128)),
            layers=int(mcfg.get("layers", 3)),
            dropout=float(mcfg.get("dropout", 0.0)),
            make_score_head=False,
        ).to(device)

    if model_type == "meshgraphnet":
        edge_pos_dim = int(mcfg.get("edge_pos_dim", 2))
        edge_attr_channels = mcfg.get("edge_attr_channels", mcfg.get("edge_in_channels", None))
        if edge_attr_channels is None:
            edge_attr_channels = edge_pos_dim + 1
        return MeshGraphNet(
            in_channels=in_dim,
            out_channels=out_dim,
            hidden=int(mcfg.get("hidden", 128)),
            layers=int(mcfg.get("layers", mcfg.get("processor_steps", 3))),
            edge_attr_channels=int(edge_attr_channels),
            edge_pos_dim=edge_pos_dim,
            mlp_hidden_layers=int(mcfg.get("mlp_hidden_layers", 1)),
            activation=str(mcfg.get("activation", "relu")),
            activation_negative_slope=float(mcfg.get("activation_negative_slope", 0.01)),
            activation_elu_alpha=float(mcfg.get("activation_elu_alpha", 1.0)),
            use_layernorm=bool(mcfg.get("use_layernorm", mcfg.get("layer_norm", False))),
            layernorm_eps=float(mcfg.get("layernorm_eps", 1e-6)),
            decoder_layer_norm=bool(mcfg.get("decoder_layer_norm", False)),
            dropout=float(mcfg.get("dropout", 0.0)),
            aggregation=str(mcfg.get("aggregation", "sum")),
            use_skip=bool(mcfg.get("use_skip", False)),
            make_score_head=False,
        ).to(device)

    if model_type == "fluxgraphnet":
        edge_pos_dim = int(mcfg.get("edge_pos_dim", 2))
        edge_attr_channels = mcfg.get("edge_attr_channels", mcfg.get("edge_in_channels", None))
        if edge_attr_channels is None:
            edge_attr_channels = (2 * edge_pos_dim) + 1

        rcfg = _relative_geometry_cfg(cfg)
        bcfg = _boundary_mask_cfg(cfg)
        domain_bbox = mcfg.get("domain_bbox", rcfg.get("domain_bbox", None))
        if domain_bbox == "auto":
            domain_bbox = None
        center_xy = mcfg.get(
            "cylinder_center_xy",
            rcfg.get("center_xy", bcfg.get("center_xy", [0.0, 0.0])),
        )
        cylinder_radius = float(
            mcfg.get("cylinder_radius", rcfg.get("radius", bcfg.get("radius", 0.5)))
        )
        cylinder_width = float(
            mcfg.get(
                "cylinder_boundary_width",
                rcfg.get("band", bcfg.get("band", 0.08)),
            )
        )
        return FluxGraphNet(
            in_channels=in_dim,
            out_channels=out_dim,
            hidden=int(mcfg.get("hidden", 128)),
            layers=int(mcfg.get("layers", mcfg.get("processor_steps", 3))),
            state_channel=int(mcfg.get("state_channel", 0)),
            predict_type=mcfg.get("predict_type", "state"),
            edge_attr_channels=int(edge_attr_channels),
            edge_pos_dim=edge_pos_dim,
            mlp_hidden_layers=int(mcfg.get("mlp_hidden_layers", 1)),
            activation=str(mcfg.get("activation", "relu")),
            activation_negative_slope=float(mcfg.get("activation_negative_slope", 0.01)),
            activation_elu_alpha=float(mcfg.get("activation_elu_alpha", 1.0)),
            use_layernorm=bool(mcfg.get("use_layernorm", mcfg.get("layer_norm", False))),
            layernorm_eps=float(mcfg.get("layernorm_eps", 1e-6)),
            dropout=float(mcfg.get("dropout", 0.0)),
            aggregation=str(mcfg.get("aggregation", "sum")),
            flux_scale=float(mcfg.get("flux_scale", 1.0)),
            use_dual_volume=bool(mcfg.get("use_dual_volume", True)),
            volume_floor=float(mcfg.get("volume_floor", 1e-12)),
            use_open_boundary_source=bool(mcfg.get("use_open_boundary_source", True)),
            open_boundary_mode=str(mcfg.get("open_boundary_mode", "learned_source")),
            open_boundary_modes_by_side=mcfg.get("open_boundary_modes_by_side", None),
            open_boundary_source_channels=mcfg.get("open_boundary_source_channels", None),
            open_boundary_flux_sides=mcfg.get("open_boundary_flux_sides", None),
            open_boundary_flux_scale=float(mcfg.get("open_boundary_flux_scale", 0.05)),
            open_boundary_flux_outflow_only=bool(mcfg.get("open_boundary_flux_outflow_only", True)),
            boundary_width=float(mcfg.get("boundary_width", 0.02)),
            domain_bbox=domain_bbox,
            cylinder_center_xy=center_xy,
            cylinder_radius=cylinder_radius,
            cylinder_boundary_width=cylinder_width,
            velocity_channels=mcfg.get("velocity_channels", [0, 1]),
            make_score_head=False,
        ).to(device)

    raise RuntimeError(f"Unexpected normalized model type: {model_type!r}")


def _forward_model(
    model: torch.nn.Module,
    x_model: torch.Tensor,
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    dual_volume: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if bool(getattr(model, "uses_edge_geometry", False)):
        if bool(getattr(model, "uses_dual_volume", False)):
            return model(x_model, edge_index, pos=pos, dual_volume=dual_volume)
        return model(x_model, edge_index, pos=pos)
    return model(x_model, edge_index)


def _run_epoch(
    model: torch.nn.Module,
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
        dual_volume_raw = batch.get("dual_volume", None)
        dual_volume = (
            None
            if dual_volume_raw is None
            else dual_volume_raw.to(device=device, dtype=torch.float32)
        )
        dt_phys = _safe_dt_scalar(batch.get("t_src", None), batch.get("t_dst", None), default_dt=1.0)

        x_in = _maybe_norm(x, norm.x_mu, norm.x_std)
        y_tgt = _maybe_norm(y, norm.y_mu, norm.y_std)
        pos_in = _maybe_norm(pos, norm.pos_mu, norm.pos_std) if include_pos else None

        x_parts = [x_in]
        re_extra = _build_reynolds_node_feature(
            n_nodes=int(x_in.size(0)),
            cfg=cfg,
            meta=batch.get("meta", None),
            source_reynolds=None,
            device=device,
            dtype=x_in.dtype,
        )
        if re_extra is not None and re_extra.numel() > 0:
            x_parts.append(re_extra)
        rel_geo = _build_relative_geometry_node_features(
            pos=pos,
            edge_index=ei,
            cfg=cfg,
            device=device,
            dtype=x_in.dtype,
        )
        if rel_geo is not None and rel_geo.numel() > 0:
            x_parts.append(rel_geo)
        bnd_extra = _build_boundary_mask_node_feature(
            pos=pos,
            edge_index=ei,
            cfg=cfg,
            device=device,
            dtype=x_in.dtype,
        )
        if bnd_extra is not None and bnd_extra.numel() > 0:
            x_parts.append(bnd_extra)
        if include_pos:
            assert pos_in is not None
            x_parts.append(pos_in)

        phy_extra = _build_physics_extra_features(
            x_abs=x,
            pos=pos,
            edge_index=ei,
            dual_volume=dual_volume,
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

        y_pred_norm, _score, _h = _forward_model(model, x_model, ei, pos, dual_volume=dual_volume)
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
    model: torch.nn.Module,
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
        dual_volume_list = batch.get("dual_volume_list", None)
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
            dual_volume = None
            if isinstance(dual_volume_list, list) and k < len(dual_volume_list):
                dual_raw = dual_volume_list[k]
                if dual_raw is not None:
                    dual_volume = dual_raw.to(device=device, dtype=torch.float32)

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
            pos_in = _maybe_norm(pos, norm.pos_mu, norm.pos_std) if include_pos else None
            if isinstance(t_list, list) and (k + 1) < len(t_list):
                dt_phys = _safe_dt_scalar(t_list[k], t_list[k + 1], default_dt=1.0)
            else:
                dt_phys = 1.0

            x_parts = [x_in]
            re_extra = _build_reynolds_node_feature(
                n_nodes=int(x_in.size(0)),
                cfg=cfg,
                meta=batch.get("meta", None),
                source_reynolds=None,
                device=device,
                dtype=x_in.dtype,
            )
            if re_extra is not None and re_extra.numel() > 0:
                x_parts.append(re_extra)
            rel_geo = _build_relative_geometry_node_features(
                pos=pos,
                edge_index=ei,
                cfg=cfg,
                device=device,
                dtype=x_in.dtype,
            )
            if rel_geo is not None and rel_geo.numel() > 0:
                x_parts.append(rel_geo)
            bnd_extra = _build_boundary_mask_node_feature(
                pos=pos,
                edge_index=ei,
                cfg=cfg,
                device=device,
                dtype=x_in.dtype,
            )
            if bnd_extra is not None and bnd_extra.numel() > 0:
                x_parts.append(bnd_extra)
            if include_pos:
                assert pos_in is not None
                x_parts.append(pos_in)
            phy_extra = _build_physics_extra_features(
                x_abs=x_in_abs,
                pos=pos,
                edge_index=ei,
                dual_volume=dual_volume,
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
                y_pred_norm, _score, _h = _forward_model(model, x_model, ei, pos, dual_volume=dual_volume)
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
    include_reynolds, reynolds_mode = _reynolds_input_cfg(cfg)
    relative_geometry_cfg = _relative_geometry_cfg(cfg)
    include_relative_geometry = bool(relative_geometry_cfg.get("enabled", False))
    relative_geometry_dim = _relative_geometry_in_channels(cfg)
    bmask_cfg = _boundary_mask_cfg(cfg)
    include_boundary_mask = bool(bmask_cfg.get("enabled", False))
    boundary_extra_dim = _boundary_extra_in_channels(cfg)
    normalize = bool(feat_cfg.get("normalize", True))
    norm_mode = _normalization_mode_from_cfg(cfg)
    pos_normalize, pos_norm_mode = _pos_normalization_cfg(cfg)
    if (not include_pos) and pos_normalize:
        print("[WARN] features.pos_normalization.enabled=true but features.include_pos=false; disabling pos normalization.")
        pos_normalize = False
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
    if normalize or pos_normalize:
        stats = _compute_norm_stats(
            norm_dataset,
            norm_indices,
            device=device,
            mode=norm_mode,
            compute_xy=bool(normalize),
            component_mode=component_mode,
            shared_channels=shared_channels,
            compute_pos=bool(pos_normalize and include_pos),
            pos_mode=pos_norm_mode,
            rollout_steps=(rollout_steps if use_window_mode else 1),
        )
    else:
        stats = NormStats(None, None, None, None, mode=norm_mode, pos_mode="none")

    physics_extra_dim = _physics_extra_in_channels(cfg, dataset_for_dims.x_dim)
    in_dim = (
        dataset_for_dims.x_dim
        + (1 if include_reynolds else 0)
        + int(relative_geometry_dim)
        + int(boundary_extra_dim)
        + (dataset_for_dims.pos_dim if include_pos else 0)
        + int(physics_extra_dim)
    )
    out_dim = dataset_for_dims.y_dim
    model = _build_model(cfg, in_dim=in_dim, out_dim=out_dim, device=device)
    has_dual_volume = bool(getattr(dataset_for_dims, "has_dual_volume", False))

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
                f"dual_volume={has_dual_volume} "
                f"window_size={window_size} stride={stride} rollout_steps={rollout_steps} "
                f"autoregressive={multi_step_autoreg} reverse_time={reverse_time}"
            )
        else:
            print(
                f"[INFO] dataset pairs={n_total} "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) "
                f"files(train={n_train_files}, val={n_val_files}, test={n_test_files}) "
                f"x_dim={dataset_for_dims.x_dim} y_dim={dataset_for_dims.y_dim} pos_dim={dataset_for_dims.pos_dim} "
                f"dual_volume={has_dual_volume} "
                f"reverse_time={reverse_time}"
            )
    else:
        if use_window_mode:
            print(
                f"[INFO] dataset windows={n_total} "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) "
                f"x_dim={dataset_for_dims.x_dim} y_dim={dataset_for_dims.y_dim} pos_dim={dataset_for_dims.pos_dim} "
                f"dual_volume={has_dual_volume} "
                f"window_size={window_size} stride={stride} rollout_steps={rollout_steps} "
                f"autoregressive={multi_step_autoreg} reverse_time={reverse_time}"
            )
        else:
            print(
                f"[INFO] dataset pairs={n_total} "
                f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) "
                f"x_dim={dataset_for_dims.x_dim} y_dim={dataset_for_dims.y_dim} pos_dim={dataset_for_dims.pos_dim} "
                f"dual_volume={has_dual_volume} "
                f"reverse_time={reverse_time}"
            )
    if normalize and stats.x_mu is not None:
        print(
            f"[INFO] normalization enabled "
            f"(mode={stats.mode}, component_mode={component_mode}; train split stats computed)."
        )
        if component_mode == "shared":
            print(f"[INFO] shared component scaling channels={shared_channels}")
    elif normalize:
        print(
            f"[WARN] feature normalization requested but stats were unavailable "
            f"(mode={norm_mode}, component_mode={component_mode})."
        )
    else:
        print(
            f"[INFO] normalization disabled "
            f"(configured mode={norm_mode}, component_mode={component_mode})."
        )
    if pos_normalize and stats.pos_mu is not None:
        print(
            f"[INFO] position normalization enabled "
            f"(mode={stats.pos_mode}; train split stats computed)."
        )
    elif pos_normalize:
        print(f"[WARN] position normalization requested but stats were unavailable (mode={pos_norm_mode}).")
    else:
        print(f"[INFO] position normalization disabled (configured mode={pos_norm_mode}).")
    if include_reynolds:
        print(f"[INFO] Reynolds conditioning enabled: mode={reynolds_mode} (channel_dim=1).")
    if include_relative_geometry:
        print(
            "[INFO] relative geometry input enabled: "
            f"channels={list(relative_geometry_cfg.get('channels', []))} "
            f"input_dim={relative_geometry_dim} "
            f"domain_bbox={relative_geometry_cfg.get('domain_bbox', None)} "
            f"cylinder_distance_scale={relative_geometry_cfg.get('cylinder_distance_scale', 'radius')} "
            f"cylinder_distance_clip={relative_geometry_cfg.get('cylinder_distance_clip', None)} "
            f"center={relative_geometry_cfg.get('center_xy', [0.0, 0.0])} "
            f"radius={float(relative_geometry_cfg.get('radius', 0.5))}"
        )
    if include_boundary_mask:
        if bool(bmask_cfg.get("infer_from_data", False)):
            print(
                "[INFO] boundary geometry input enabled: "
                f"type={bmask_cfg.get('type','cylinder')} infer_from_data=true "
                f"feature_mode={bmask_cfg.get('feature_mode','mask')} "
                f"distance_scale={bmask_cfg.get('distance_scale','band')} "
                f"distance_clip={bmask_cfg.get('distance_clip',None)} "
                f"channels={boundary_extra_dim} "
                f"infer_band_scale={float(bmask_cfg.get('infer_band_scale',1.5))} "
                f"infer_min_nodes={int(bmask_cfg.get('infer_min_nodes',16))}"
            )
        else:
            print(
                "[INFO] boundary geometry input enabled: "
                f"type={bmask_cfg.get('type','cylinder')} "
                f"feature_mode={bmask_cfg.get('feature_mode','mask')} "
                f"distance_scale={bmask_cfg.get('distance_scale','band')} "
                f"distance_clip={bmask_cfg.get('distance_clip',None)} "
                f"channels={boundary_extra_dim} "
                f"center={bmask_cfg.get('center_xy',[0.0,0.0])} "
                f"radius={float(bmask_cfg.get('radius',0.5))} "
                f"band={float(bmask_cfg.get('band',0.08))}"
            )
    if _physics_inputs_enabled(cfg):
        print(
            f"[INFO] physics inputs enabled: backend={_physics_backend(cfg)} "
            f"extra_in_channels={physics_extra_dim}"
        )
    if _model_type_from_cfg(cfg) == "fluxgraphnet":
        mcfg = cfg.get("model", {}) or {}
        print(
            "[INFO] FluxGraphNet enabled: "
            "edge fluxes are canonicalized over unique undirected mesh edges; "
            f"open_boundary_mode={mcfg.get('open_boundary_mode', 'learned_source')} "
            f"use_dual_volume={bool(mcfg.get('use_dual_volume', True))} "
            f"domain_bbox={mcfg.get('domain_bbox', relative_geometry_cfg.get('domain_bbox', None))} "
            f"cylinder_radius={float(mcfg.get('cylinder_radius', relative_geometry_cfg.get('radius', 0.5)))}"
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
                            "pos_mode": str(stats.pos_mode),
                            "pos_mu": None if stats.pos_mu is None else stats.pos_mu.detach().cpu(),
                            "pos_std": None if stats.pos_std is None else stats.pos_std.detach().cpu(),
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
                "pos_mode": str(stats.pos_mode),
                "pos_mu": None if stats.pos_mu is None else stats.pos_mu.detach().cpu(),
                "pos_std": None if stats.pos_std is None else stats.pos_std.detach().cpu(),
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
                "has_dual_volume": bool(has_dual_volume),
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
                "pos_normalize": bool(pos_normalize),
                "pos_normalization_mode": str(pos_norm_mode),
                "include_relative_geometry": bool(include_relative_geometry),
                "relative_geometry_channels": list(relative_geometry_cfg.get("channels", [])),
                "relative_geometry_input_channels": int(relative_geometry_dim),
                "relative_geometry_domain_bbox": relative_geometry_cfg.get("domain_bbox", None),
                "relative_geometry_domain_distance_clip": relative_geometry_cfg.get("domain_distance_clip", None),
                "relative_geometry_cylinder_distance_scale": str(
                    relative_geometry_cfg.get("cylinder_distance_scale", "radius")
                ),
                "relative_geometry_cylinder_distance_clip": relative_geometry_cfg.get("cylinder_distance_clip", None),
                "relative_geometry_cylinder_infer_from_data": bool(
                    relative_geometry_cfg.get("infer_from_data", False)
                ),
                "relative_geometry_cylinder_center_xy": [
                    float(x) for x in relative_geometry_cfg.get("center_xy", [0.0, 0.0])
                ],
                "relative_geometry_cylinder_radius": float(relative_geometry_cfg.get("radius", 0.5)),
                "relative_geometry_cylinder_band": float(relative_geometry_cfg.get("band", 0.08)),
                "include_reynolds": bool(include_reynolds),
                "reynolds_mode": str(reynolds_mode),
                "include_boundary_mask": bool(include_boundary_mask),
                "boundary_mask_type": str(bmask_cfg.get("type", "cylinder")),
                "boundary_mask_feature_mode": str(bmask_cfg.get("feature_mode", "mask")),
                "boundary_mask_distance_scale": str(bmask_cfg.get("distance_scale", "band")),
                "boundary_mask_distance_clip": bmask_cfg.get("distance_clip", None),
                "boundary_mask_input_channels": int(boundary_extra_dim),
                "boundary_mask_infer_from_data": bool(bmask_cfg.get("infer_from_data", False)),
                "boundary_mask_infer_band_scale": float(bmask_cfg.get("infer_band_scale", 1.5)),
                "boundary_mask_infer_min_nodes": int(bmask_cfg.get("infer_min_nodes", 16)),
                "boundary_mask_center_xy": [float(x) for x in bmask_cfg.get("center_xy", [0.0, 0.0])],
                "boundary_mask_radius": float(bmask_cfg.get("radius", 0.5)),
                "boundary_mask_band": float(bmask_cfg.get("band", 0.08)),
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
                "cached_dual_volume_area": bool(has_dual_volume),
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
                "type": _model_type_from_cfg(cfg),
                "hidden": int((cfg.get("model", {}) or {}).get("hidden", 128)),
                "layers": int((cfg.get("model", {}) or {}).get("layers", 3)),
                "edge_attr_channels": int(
                    (cfg.get("model", {}) or {}).get(
                        "edge_attr_channels",
                        (cfg.get("model", {}) or {}).get(
                            "edge_in_channels",
                            (
                                (2 * int((cfg.get("model", {}) or {}).get("edge_pos_dim", 2)) + 1)
                                if _model_type_from_cfg(cfg) == "fluxgraphnet"
                                else int((cfg.get("model", {}) or {}).get("edge_pos_dim", 2)) + 1
                            ),
                        ),
                    )
                ),
                "edge_pos_dim": int((cfg.get("model", {}) or {}).get("edge_pos_dim", 2)),
                "mlp_hidden_layers": int((cfg.get("model", {}) or {}).get("mlp_hidden_layers", 1)),
                "activation": str((cfg.get("model", {}) or {}).get("activation", "relu")),
                "activation_negative_slope": float(
                    (cfg.get("model", {}) or {}).get("activation_negative_slope", 0.01)
                ),
                "activation_elu_alpha": float(
                    (cfg.get("model", {}) or {}).get("activation_elu_alpha", 1.0)
                ),
                "use_skip": bool((cfg.get("model", {}) or {}).get("use_skip", False)),
                "use_layernorm": bool(
                    (cfg.get("model", {}) or {}).get(
                        "use_layernorm",
                        (cfg.get("model", {}) or {}).get("layer_norm", False),
                    )
                ),
                "layernorm_eps": float((cfg.get("model", {}) or {}).get("layernorm_eps", 1e-6)),
                "dropout": float((cfg.get("model", {}) or {}).get("dropout", 0.0)),
                "predict_type": str((cfg.get("model", {}) or {}).get("predict_type", "state")),
                "use_open_boundary_source": bool(
                    (cfg.get("model", {}) or {}).get("use_open_boundary_source", True)
                ),
                "open_boundary_mode": str(
                    (cfg.get("model", {}) or {}).get("open_boundary_mode", "learned_source")
                ),
                "open_boundary_modes_by_side": (cfg.get("model", {}) or {}).get(
                    "open_boundary_modes_by_side",
                    None,
                ),
                "boundary_width": float((cfg.get("model", {}) or {}).get("boundary_width", 0.02)),
                "use_dual_volume": bool((cfg.get("model", {}) or {}).get("use_dual_volume", True)),
            },
            "derived": {
                "dataset_mode": "window" if use_window_mode else "pair",
                "rollout_steps": int(rollout_steps),
                "split_mode": ("pre_split_dir" if use_pre_split_dirs else "random_split"),
                "reverse_time": bool(reverse_time),
                "physics_inputs_enabled": bool(_physics_inputs_enabled(cfg)),
                "physics_backend": _physics_backend(cfg),
                "relative_geometry_input_channels": int(relative_geometry_dim),
                "boundary_input_channels": int(boundary_extra_dim),
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
            "relative_geometry_input_channels": int(relative_geometry_dim),
            "physics_extra_in_channels": int(physics_extra_dim),
            "boundary_input_channels": int(boundary_extra_dim),
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

#!/usr/bin/env python3
"""
Plot mesh connectivity from a PyTorch/PyG .pt time series.

Expected per-timestep fields:
  - pos:        (N,2+) node/cell-center coordinates
  - edge_index: (2,E) or (E,2) graph connectivity (used to infer cell size)
Optional:
  - level:      (N,) refinement level (for coloring)
  - H, W, bbox, dx, dy (if present, improve cell-size reconstruction)

Examples:
  python utils/plot_mesh_from_pt.py \
      --pt cache/Re_600_normalized.pt \
      --step 0 \
      --out misc_plots/mesh_step0.png

  # Explicit zoom box (xmin,xmax,ymin,ymax)
  python utils/plot_mesh_from_pt.py \
      --pt cache/Re_600_normalized.pt \
      --step 0 \
      --out misc_plots/mesh_zoom.png \
      --zoom-bbox 0.2,0.35,0.45,0.60
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch


def _load_pt(path: str) -> Any:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".pt") or m.endswith(".pth")]
            if not members:
                raise RuntimeError(f"No .pt/.pth file found inside zip: {path}")
            with zf.open(members[0], "r") as f:
                buf = io.BytesIO(f.read())
            return torch.load(buf, map_location="cpu", weights_only=False)

    return torch.load(path, map_location="cpu", weights_only=False)


def _as_np_2col(x: Any, name: str, pos_cols: Tuple[int, int]) -> np.ndarray:
    t = torch.as_tensor(x).detach().cpu()
    if t.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor/array, got {tuple(t.shape)}")

    c0, c1 = int(pos_cols[0]), int(pos_cols[1])
    if c0 == c1:
        raise ValueError(f"pos column indices must be distinct, got {pos_cols}")
    if c0 < 0 or c1 < 0:
        raise ValueError(f"pos column indices must be non-negative, got {pos_cols}")
    if max(c0, c1) >= int(t.shape[1]):
        raise ValueError(
            f"{name} has shape {tuple(t.shape)}; requested columns {pos_cols} are out of bounds"
        )
    return t[:, [c0, c1]].to(torch.float32).numpy()


def _as_edge_index(x: Any) -> np.ndarray:
    ei = torch.as_tensor(x).detach().cpu().to(torch.long)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {tuple(ei.shape)}")
    if ei.shape[0] == 2:
        out = ei
    elif ei.shape[1] == 2:
        out = ei.t().contiguous()
    else:
        raise ValueError(f"edge_index must be (2,E) or (E,2), got {tuple(ei.shape)}")
    return out.numpy()


def _to_scalar(x: Any) -> float | None:
    if x is None:
        return None
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return None
            return float(x.detach().cpu().view(-1)[0].item())
        return float(x)
    except Exception:
        return None


def _to_bbox(x: Any) -> Tuple[float, float, float, float] | None:
    if x is None:
        return None
    try:
        t = torch.as_tensor(x).detach().cpu().view(-1).to(torch.float64)
        if t.numel() != 4:
            return None
        return (float(t[0].item()), float(t[1].item()), float(t[2].item()), float(t[3].item()))
    except Exception:
        return None


def _extract_step(raw_step: Any, pos_cols: Tuple[int, int]) -> Dict[str, Any]:
    if isinstance(raw_step, dict):
        pos = raw_step.get("pos", raw_step.get("xy", None))
        ei = raw_step.get("edge_index", raw_step.get("ei", None))
        lvl = raw_step.get("level", raw_step.get("levels", None))
        H = raw_step.get("H", None)
        W = raw_step.get("W", None)
        bbox = raw_step.get("bbox", None)
        dx = raw_step.get("dx", None)
        dy = raw_step.get("dy", None)
        gp = raw_step.get("global_params", None)
    else:
        pos = getattr(raw_step, "pos", getattr(raw_step, "xy", None))
        ei = getattr(raw_step, "edge_index", getattr(raw_step, "ei", None))
        lvl = getattr(raw_step, "level", getattr(raw_step, "levels", None))
        H = getattr(raw_step, "H", getattr(raw_step, "coarse_H", None))
        W = getattr(raw_step, "W", getattr(raw_step, "coarse_W", None))
        bbox = getattr(raw_step, "bbox", None)
        dx = getattr(raw_step, "dx", None)
        dy = getattr(raw_step, "dy", None)
        gp = getattr(raw_step, "global_params", None)

    if pos is None:
        raise KeyError("Timestep is missing 'pos' (or 'xy').")
    if ei is None:
        raise KeyError("Timestep is missing 'edge_index' (or 'ei').")

    if isinstance(gp, dict):
        H = H if H is not None else gp.get("H", gp.get("coarse_H", None))
        W = W if W is not None else gp.get("W", gp.get("coarse_W", None))
        bbox = bbox if bbox is not None else gp.get("bbox", None)
        dx = dx if dx is not None else gp.get("dx", None)
        dy = dy if dy is not None else gp.get("dy", None)

    out = {
        "pos": _as_np_2col(pos, "pos", pos_cols),
        "edge_index": _as_edge_index(ei),
        "level": None,
        "H": _to_scalar(H),
        "W": _to_scalar(W),
        "bbox": _to_bbox(bbox),
        "dx": _to_scalar(dx),
        "dy": _to_scalar(dy),
    }

    if lvl is not None:
        lv = torch.as_tensor(lvl).detach().cpu().view(-1).to(torch.long).numpy()
        if lv.shape[0] == out["pos"].shape[0]:
            out["level"] = lv

    return out


def _parse_pos_cols(text: str) -> Tuple[int, int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--pos-cols must contain exactly two comma-separated integers, got: {text}")
    try:
        c0 = int(parts[0])
        c1 = int(parts[1])
    except Exception as exc:
        raise ValueError(f"--pos-cols must be integers, got: {text}") from exc
    return c0, c1


def _parse_pair(text: str, *, arg_name: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{arg_name} must contain exactly two comma-separated numbers, got: {text}")
    try:
        a = float(parts[0])
        b = float(parts[1])
    except Exception as exc:
        raise ValueError(f"{arg_name} must be numeric, got: {text}") from exc
    return a, b


def _parse_zoom_bbox(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"--zoom-bbox must be xmin,xmax,ymin,ymax, got: {text}")
    try:
        xmin = float(parts[0]); xmax = float(parts[1]); ymin = float(parts[2]); ymax = float(parts[3])
    except Exception as exc:
        raise ValueError(f"--zoom-bbox must be numeric, got: {text}") from exc
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"--zoom-bbox must satisfy xmax>xmin and ymax>ymin, got: {text}")
    return xmin, xmax, ymin, ymax


def _resolve_zoom_bbox(
    *,
    zoom_bbox_text: str | None,
    zoom_center_text: str | None,
    zoom_width: float | None,
    zoom_height: float | None,
) -> Tuple[float, float, float, float] | None:
    if zoom_bbox_text:
        return _parse_zoom_bbox(zoom_bbox_text)

    if zoom_center_text is None:
        return None

    cx, cy = _parse_pair(zoom_center_text, arg_name="--zoom-center")
    if zoom_width is None:
        raise ValueError("--zoom-center requires --zoom-width (and optional --zoom-height).")
    if zoom_width <= 0:
        raise ValueError("--zoom-width must be > 0.")

    h = float(zoom_height if zoom_height is not None else zoom_width)
    if h <= 0:
        raise ValueError("--zoom-height must be > 0.")

    half_w = 0.5 * float(zoom_width)
    half_h = 0.5 * h
    return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)


def _pick_step(series: Any, step_idx: int) -> Tuple[Any, int, int]:
    if isinstance(series, list):
        if len(series) == 0:
            raise RuntimeError("Series is empty.")
        idx = int(step_idx)
        if idx < 0:
            idx = len(series) + idx
        if idx < 0 or idx >= len(series):
            raise IndexError(f"step={step_idx} is out of range for series length {len(series)}")
        return series[idx], idx, len(series)

    if isinstance(series, dict):
        for k in ("timesteps", "snapshots", "steps", "data_list", "sequence"):
            v = series.get(k, None)
            if isinstance(v, list) and len(v) > 0:
                return _pick_step(v, step_idx)
        # treat dict as a single timestep
        return series, 0, 1

    # treat any other object as a single timestep
    return series, 0, 1


def _estimate_cell_half_sizes(
    pos: np.ndarray,
    edge_index: np.ndarray,
    *,
    level: np.ndarray | None,
    H: float | None,
    W: float | None,
    bbox: Tuple[float, float, float, float] | None,
    dx0: float | None,
    dy0: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-cell half-width/half-height in plotted coordinates.

    Priority:
      1) if level + (dx,dy) available -> dyadic exact sizes
      2) if level + (H,W,bbox) available -> dyadic exact sizes
      3) fallback from incident edge spacings
    """
    n = int(pos.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Exact dyadic size path.
    if level is not None:
        lvl = level.astype(np.float32, copy=False)
        if (dx0 is None or dy0 is None) and (bbox is not None and H is not None and W is not None):
            x0, x1, y0, y1 = bbox
            if float(H) > 0 and float(W) > 0:
                dx0 = float(x1 - x0) / float(W)
                dy0 = float(y1 - y0) / float(H)
        if dx0 is not None and dy0 is not None and dx0 > 0 and dy0 > 0:
            scale = np.power(2.0, lvl)
            hx = (0.5 * float(dx0) / scale).astype(np.float32, copy=False)
            hy = (0.5 * float(dy0) / scale).astype(np.float32, copy=False)
            return hx, hy

    # Fallback: infer from edge spacings.
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n) & (src != dst)
    src = src[valid]
    dst = dst[valid]

    eps = 1e-12
    hx_min = np.full((n,), np.inf, dtype=np.float32)
    hy_min = np.full((n,), np.inf, dtype=np.float32)
    h_iso_min = np.full((n,), np.inf, dtype=np.float32)

    if src.size > 0:
        dxy = np.abs(pos[dst] - pos[src]).astype(np.float32, copy=False)
        dxv = dxy[:, 0]
        dyv = dxy[:, 1]
        dist = np.hypot(dxv, dyv).astype(np.float32, copy=False)

        horiz = (dxv > eps) & (dxv >= dyv)
        vert = (dyv > eps) & (dyv > dxv)
        anyd = dist > eps

        if np.any(horiz):
            v = dxv[horiz]
            s = src[horiz]
            d = dst[horiz]
            np.minimum.at(hx_min, s, v)
            np.minimum.at(hx_min, d, v)

        if np.any(vert):
            v = dyv[vert]
            s = src[vert]
            d = dst[vert]
            np.minimum.at(hy_min, s, v)
            np.minimum.at(hy_min, d, v)

        if np.any(anyd):
            v = dist[anyd]
            s = src[anyd]
            d = dst[anyd]
            np.minimum.at(h_iso_min, s, v)
            np.minimum.at(h_iso_min, d, v)

    # Global fallback scale from extents.
    if np.isfinite(h_iso_min).any():
        global_h = float(np.nanmedian(h_iso_min[np.isfinite(h_iso_min)]))
    else:
        x0, y0 = pos.min(axis=0)
        x1, y1 = pos.max(axis=0)
        area = max((x1 - x0) * (y1 - y0), 1e-12)
        global_h = float(np.sqrt(area / max(n, 1)))

    hx = np.where(np.isfinite(hx_min), hx_min, np.where(np.isfinite(h_iso_min), h_iso_min, global_h))
    hy = np.where(np.isfinite(hy_min), hy_min, np.where(np.isfinite(h_iso_min), h_iso_min, global_h))

    # Convert spacing -> half-size. Slight shrink avoids overdraw at interfaces.
    hx = (0.5 * hx * 0.95).astype(np.float32, copy=False)
    hy = (0.5 * hy * 0.95).astype(np.float32, copy=False)

    hx = np.clip(hx, 1e-12, np.inf)
    hy = np.clip(hy, 1e-12, np.inf)
    return hx, hy


def _induced_subgraph(
    pos: np.ndarray,
    edge_index: np.ndarray,
    level: np.ndarray | None,
    *,
    max_nodes: int,
    max_edges: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n = pos.shape[0]
    e = edge_index.shape[1]
    if max_nodes <= 0 and max_edges <= 0:
        return pos, edge_index, level

    rng = np.random.default_rng(seed)

    pos_out = pos
    lvl_out = level
    ei_out = edge_index

    if max_nodes > 0 and n > max_nodes:
        keep = np.sort(rng.choice(n, size=max_nodes, replace=False))
        remap = np.full((n,), -1, dtype=np.int64)
        remap[keep] = np.arange(keep.shape[0], dtype=np.int64)

        src = ei_out[0]
        dst = ei_out[1]
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        src = src[valid]
        dst = dst[valid]
        in_sub = (remap[src] >= 0) & (remap[dst] >= 0)
        src2 = remap[src[in_sub]]
        dst2 = remap[dst[in_sub]]

        pos_out = pos_out[keep]
        lvl_out = None if lvl_out is None else lvl_out[keep]
        ei_out = np.stack([src2, dst2], axis=0) if src2.size else np.zeros((2, 0), dtype=np.int64)

    if max_edges > 0 and ei_out.shape[1] > max_edges:
        sel = rng.choice(ei_out.shape[1], size=max_edges, replace=False)
        ei_out = ei_out[:, sel]

    # Ensure edges are valid after sampling.
    nn = pos_out.shape[0]
    if ei_out.shape[1] > 0:
        src = ei_out[0]
        dst = ei_out[1]
        valid = (src >= 0) & (src < nn) & (dst >= 0) & (dst < nn) & (src != dst)
        ei_out = ei_out[:, valid]

    return pos_out, ei_out, lvl_out


def _cell_segments(pos: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> np.ndarray:
    n = int(pos.shape[0])
    x = pos[:, 0].astype(np.float32, copy=False)
    y = pos[:, 1].astype(np.float32, copy=False)
    hx = hx.astype(np.float32, copy=False)
    hy = hy.astype(np.float32, copy=False)

    x0 = x - hx
    x1 = x + hx
    y0 = y - hy
    y1 = y + hy

    segs = np.empty((4 * n, 2, 2), dtype=np.float32)
    # bottom
    segs[0 * n : 1 * n, 0, 0] = x0
    segs[0 * n : 1 * n, 0, 1] = y0
    segs[0 * n : 1 * n, 1, 0] = x1
    segs[0 * n : 1 * n, 1, 1] = y0
    # right
    segs[1 * n : 2 * n, 0, 0] = x1
    segs[1 * n : 2 * n, 0, 1] = y0
    segs[1 * n : 2 * n, 1, 0] = x1
    segs[1 * n : 2 * n, 1, 1] = y1
    # top
    segs[2 * n : 3 * n, 0, 0] = x1
    segs[2 * n : 3 * n, 0, 1] = y1
    segs[2 * n : 3 * n, 1, 0] = x0
    segs[2 * n : 3 * n, 1, 1] = y1
    # left
    segs[3 * n : 4 * n, 0, 0] = x0
    segs[3 * n : 4 * n, 0, 1] = y1
    segs[3 * n : 4 * n, 1, 0] = x0
    segs[3 * n : 4 * n, 1, 1] = y0
    return segs


def _filter_cells_to_zoom(
    pos: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    level: np.ndarray | None,
    zoom_bbox: Tuple[float, float, float, float] | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Keep cells whose rectangles intersect the zoom box.
    """
    if zoom_bbox is None:
        return pos, hx, hy, level

    xmin, xmax, ymin, ymax = zoom_bbox
    x = pos[:, 0]
    y = pos[:, 1]

    keep = ((x + hx) >= xmin) & ((x - hx) <= xmax) & ((y + hy) >= ymin) & ((y - hy) <= ymax)
    if not np.any(keep):
        raise RuntimeError(
            "Zoom region contains no cells. "
            f"bbox=({xmin:.6g},{xmax:.6g},{ymin:.6g},{ymax:.6g})"
        )

    pos2 = pos[keep]
    hx2 = hx[keep]
    hy2 = hy[keep]
    lvl2 = None if level is None else level[keep]
    return pos2, hx2, hy2, lvl2


def _plot(
    pos: np.ndarray,
    edge_index: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    level: np.ndarray | None,
    *,
    out_path: str,
    step: int,
    total_steps: int,
    node_size: float,
    edge_width: float,
    edge_alpha: float,
    color_by_level: bool,
    dpi: int,
    zoom_bbox: Tuple[float, float, float, float] | None = None,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    segs = _cell_segments(pos, hx, hy)
    lc = LineCollection(segs, linewidths=edge_width, alpha=edge_alpha, colors="black")
    ax.add_collection(lc)

    if color_by_level and (level is not None):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=level, s=max(node_size, 0.1), cmap="viridis", linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("level")
    elif node_size > 0:
        ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c="#1f77b4", linewidths=0)

    if zoom_bbox is not None:
        xmin, xmax, ymin, ymax = zoom_bbox
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        # Auto limits with small padding.
        xmin, ymin = pos.min(axis=0)
        xmax, ymax = pos.max(axis=0)
        dx = max(1e-9, xmax - xmin)
        dy = max(1e-9, ymax - ymin)
        ax.set_xlim(xmin - 0.03 * dx, xmax + 0.03 * dx)
        ax.set_ylim(ymin - 0.03 * dy, ymax + 0.03 * dy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Mesh Step {step} / {max(total_steps - 1, 0)}"
        f"  (cells={pos.shape[0]})"
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot mesh graph from a .pt/.pth/.zip time series.")
    ap.add_argument("--pt", required=True, help="Path to input .pt/.pth or .zip containing one .pt/.pth")
    ap.add_argument("--step", type=int, default=0, help="Timestep index to plot (supports negative index)")
    ap.add_argument("--out", required=True, help="Output image path (e.g., mesh_step0.png)")
    ap.add_argument("--max-nodes", type=int, default=250000, help="Node cap for plotting; <=0 disables")
    ap.add_argument("--max-edges", type=int, default=600000, help="Edge cap for plotting; <=0 disables")
    ap.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    ap.add_argument("--node-size", type=float, default=0.0, help="Optional center marker size (0 disables)")
    ap.add_argument("--edge-width", type=float, default=0.30, help="Cell outline line width")
    ap.add_argument("--edge-alpha", type=float, default=0.45, help="Cell outline transparency")
    ap.add_argument("--color-by-level", action="store_true", help="Color nodes by level if available")
    ap.add_argument(
        "--pos-cols",
        default="0,1",
        help="Two comma-separated columns from pos to plot (e.g., 0,1 or 0,2)",
    )
    ap.add_argument(
        "--zoom-bbox",
        default=None,
        help="Optional zoom box: xmin,xmax,ymin,ymax",
    )
    ap.add_argument(
        "--zoom-center",
        default=None,
        help="Optional zoom center: x,y (requires --zoom-width)",
    )
    ap.add_argument(
        "--zoom-width",
        type=float,
        default=None,
        help="Zoom window width when using --zoom-center",
    )
    ap.add_argument(
        "--zoom-height",
        type=float,
        default=None,
        help="Zoom window height when using --zoom-center (defaults to --zoom-width)",
    )
    ap.add_argument("--dpi", type=int, default=180, help="Output DPI")
    args = ap.parse_args()

    pos_cols = _parse_pos_cols(args.pos_cols)
    zoom_bbox = _resolve_zoom_bbox(
        zoom_bbox_text=args.zoom_bbox,
        zoom_center_text=args.zoom_center,
        zoom_width=args.zoom_width,
        zoom_height=args.zoom_height,
    )

    series = _load_pt(args.pt)
    raw_step, idx, total = _pick_step(series, args.step)
    step = _extract_step(raw_step, pos_cols=pos_cols)

    pos, ei, lvl = _induced_subgraph(
        step["pos"],
        step["edge_index"],
        step["level"],
        max_nodes=int(args.max_nodes),
        max_edges=int(args.max_edges),
        seed=int(args.seed),
    )

    if pos.shape[0] == 0:
        raise RuntimeError("No nodes left to plot after filtering.")

    hx, hy = _estimate_cell_half_sizes(
        pos=pos,
        edge_index=ei,
        level=lvl,
        H=step.get("H"),
        W=step.get("W"),
        bbox=step.get("bbox"),
        dx0=step.get("dx"),
        dy0=step.get("dy"),
    )
    pos, hx, hy, lvl = _filter_cells_to_zoom(pos, hx, hy, lvl, zoom_bbox)

    _plot(
        pos=pos,
        edge_index=ei,
        hx=hx,
        hy=hy,
        level=lvl,
        out_path=args.out,
        step=idx,
        total_steps=total,
        node_size=float(args.node_size),
        edge_width=float(args.edge_width),
        edge_alpha=float(args.edge_alpha),
        color_by_level=bool(args.color_by_level),
        dpi=int(args.dpi),
        zoom_bbox=zoom_bbox,
    )

    print(f"[INFO] plotted pos columns: {pos_cols[0]},{pos_cols[1]}")
    if zoom_bbox is not None:
        print(
            "[INFO] zoom bbox: "
            f"{zoom_bbox[0]:.6g},{zoom_bbox[1]:.6g},{zoom_bbox[2]:.6g},{zoom_bbox[3]:.6g}"
        )
    print(f"[OK] Wrote mesh plot: {args.out}")


if __name__ == "__main__":
    main()

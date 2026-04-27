#!/usr/bin/env python3
"""
mesh_gif.py

Create a GIF showing the evolution of the predicted mesh geometry stored in a rollout
precompute HDF5 (e.g., precomp_rollout.h5).

Each frame renders the dyadic quad partition implied by:
  - pred_centers: (N,2) cell centers
  - pred_levels : (N,) refinement level (0..Lmax)
and meta attrs:
  - dx, dy: level-0 cell sizes
  - bbox  : [xmin,xmax,ymin,ymax]

Optionally filters by mask_pred_parent_flat_u8 + pred_parents (keeps only cells whose
coarse parent lies inside the domain mask).

Dependencies:
  pip install h5py numpy matplotlib imageio

Example:
  python mesh_gif.py \
      --h5 ./cache/all/precomp_rollout.h5 \
      --out_gif pred_mesh.gif \
      --fps 10 \
      --stride 1 \
      --max_cells 200000 \
      --color_by_level 1
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np

import h5py
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import imageio.v2 as imageio


# ------------------------- H5 helpers -------------------------

def _is_timestep_group(name: str) -> bool:
    return name.startswith("t") and len(name) >= 2 and name[1:].isdigit()

def _sorted_timestep_groups(f: h5py.File) -> List[str]:
    names = [k for k in f.keys() if _is_timestep_group(k)]
    names.sort()
    return names

def _read_meta(f: h5py.File):
    if "meta" not in f:
        raise RuntimeError("H5 missing /meta group. Expected a precomp_rollout-style file.")
    meta = f["meta"]

    def _get_attr(key, default=None):
        v = meta.attrs.get(key, default)
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8")
        return v

    H = int(_get_attr("H", -1))
    W = int(_get_attr("W", -1))
    dx = float(_get_attr("dx", np.nan))
    dy = float(_get_attr("dy", np.nan))
    T  = int(_get_attr("T", -1))

    bbox = _get_attr("bbox", None)
    if bbox is None:
        # Some writers store bbox as dataset; try that too.
        if "bbox" in meta:
            bbox = meta["bbox"][...]
        else:
            bbox = None

    if bbox is not None:
        bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if bbox.size != 4:
            bbox = None

    return H, W, dx, dy, T, bbox


# ------------------------- geometry rendering -------------------------

def _segments_from_centers_levels(
    centers: np.ndarray,  # (N,2)
    levels: np.ndarray,   # (N,)
    dx0: float,
    dy0: float,
) -> np.ndarray:
    """
    Build line segments for axis-aligned dyadic quads implied by (center, level).
    Returns segs: (4*N, 2, 2) float32, where each row is a segment with endpoints.
    """
    c = np.asarray(centers, dtype=np.float32)
    L = np.asarray(levels, dtype=np.int64)

    # cell size at level L: dx0 / 2^L, dy0 / 2^L
    scale = np.power(2.0, L.astype(np.float32))
    hx = (dx0 / scale) * 0.5  # half-width
    hy = (dy0 / scale) * 0.5  # half-height

    x = c[:, 0]
    y = c[:, 1]

    x0 = x - hx
    x1 = x + hx
    y0 = y - hy
    y1 = y + hy

    # corners: (x0,y0), (x1,y0), (x1,y1), (x0,y1)
    # segments: bottom, right, top, left
    # Build vectorized segments array
    N = c.shape[0]
    segs = np.empty((4 * N, 2, 2), dtype=np.float32)

    # bottom: (x0,y0) -> (x1,y0)
    segs[0*N:1*N, 0, 0] = x0
    segs[0*N:1*N, 0, 1] = y0
    segs[0*N:1*N, 1, 0] = x1
    segs[0*N:1*N, 1, 1] = y0

    # right: (x1,y0) -> (x1,y1)
    segs[1*N:2*N, 0, 0] = x1
    segs[1*N:2*N, 0, 1] = y0
    segs[1*N:2*N, 1, 0] = x1
    segs[1*N:2*N, 1, 1] = y1

    # top: (x1,y1) -> (x0,y1)
    segs[2*N:3*N, 0, 0] = x1
    segs[2*N:3*N, 0, 1] = y1
    segs[2*N:3*N, 1, 0] = x0
    segs[2*N:3*N, 1, 1] = y1

    # left: (x0,y1) -> (x0,y0)
    segs[3*N:4*N, 0, 0] = x0
    segs[3*N:4*N, 0, 1] = y1
    segs[3*N:4*N, 1, 0] = x0
    segs[3*N:4*N, 1, 1] = y0

    return segs

def _sample_cells(
    centers: np.ndarray,
    levels: np.ndarray,
    max_cells: int,
    seed: int,
    strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce cell count for faster rendering.
    strategy:
      - "first": first max_cells
      - "random": random subset
      - "per_level": stratified random subset across levels
    """
    N = centers.shape[0]
    if max_cells <= 0 or N <= max_cells:
        return centers, levels

    rng = np.random.default_rng(seed)

    if strategy == "first":
        idx = np.arange(max_cells, dtype=np.int64)

    elif strategy == "random":
        idx = rng.choice(N, size=max_cells, replace=False)

    elif strategy == "per_level":
        L = np.asarray(levels, dtype=np.int64)
        uniq = np.unique(L)
        # allocate roughly proportional to counts, with at least 1 per level
        counts = np.array([(L == u).sum() for u in uniq], dtype=np.int64)
        frac = counts / max(1, counts.sum())
        alloc = np.maximum(1, np.floor(frac * max_cells).astype(np.int64))
        # fix total to exactly max_cells
        while alloc.sum() > max_cells:
            j = int(np.argmax(alloc))
            if alloc[j] > 1:
                alloc[j] -= 1
            else:
                break
        while alloc.sum() < max_cells:
            j = int(np.argmax(counts))
            alloc[j] += 1

        picks = []
        for u, a in zip(uniq, alloc):
            idx_u = np.flatnonzero(L == u)
            if idx_u.size == 0:
                continue
            if idx_u.size <= a:
                picks.append(idx_u)
            else:
                picks.append(rng.choice(idx_u, size=int(a), replace=False))
        idx = np.concatenate(picks, axis=0)
        if idx.size > max_cells:
            idx = rng.choice(idx, size=max_cells, replace=False)

    else:
        raise ValueError(f"Unknown sample strategy: {strategy}")

    return centers[idx], levels[idx]

def _apply_parent_mask_if_available(
    g: h5py.Group,
    centers: np.ndarray,
    levels: np.ndarray,
    *,
    use_mask: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If present, uses:
      - pred_parents: (N,)
      - mask_pred_parent_flat_u8: (H*W,)
    to filter cells (keep those with mask[parent] != 0).
    """
    if not use_mask:
        return centers, levels

    if ("pred_parents" not in g) or ("mask_pred_parent_flat_u8" not in g):
        return centers, levels

    parents = g["pred_parents"][...].astype(np.int64, copy=False)
    mask_flat = g["mask_pred_parent_flat_u8"][...].astype(np.uint8, copy=False)

    if parents.ndim != 1 or mask_flat.ndim != 1:
        return centers, levels
    if parents.shape[0] != centers.shape[0]:
        return centers, levels

    parents = np.clip(parents, 0, mask_flat.shape[0] - 1)
    keep = mask_flat[parents] != 0

    if keep.any():
        return centers[keep], levels[keep]
    return centers, levels


# ------------------------- frame rendering -------------------------

def _render_frame(
    centers: np.ndarray,
    levels: np.ndarray,
    *,
    dx0: float,
    dy0: float,
    bbox: Optional[np.ndarray],
    title: str,
    linewidth: float,
    color_by_level: bool,
    figsize: Tuple[float, float],
    dpi: int,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    segs = _segments_from_centers_levels(centers, levels, dx0=dx0, dy0=dy0)

    if color_by_level:
        # map each cell to a color, then repeat for its 4 segments
        L = np.asarray(levels, dtype=np.int64)
        Lmin = int(L.min()) if L.size else 0
        Lmax = int(L.max()) if L.size else 1
        denom = max(1, (Lmax - Lmin))
        t = (L - Lmin) / denom  # 0..1
        # pick a simple colormap
        cmap = plt.get_cmap("viridis")
        c = cmap(t)  # (N,4)
        c4 = np.repeat(c, repeats=4, axis=0)  # (4N,4)
        lc = LineCollection(segs, colors=c4, linewidths=linewidth, antialiased=True)
    else:
        lc = LineCollection(segs, colors="black", linewidths=linewidth, antialiased=True)

    ax.add_collection(lc)

    if bbox is not None:
        xmin, xmax, ymin, ymax = map(float, bbox.tolist())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        # tight bounds from centers + approximate max half-size
        if centers.shape[0] > 0:
            ax.set_xlim(float(centers[:, 0].min()), float(centers[:, 0].max()))
            ax.set_ylim(float(centers[:, 1].min()), float(centers[:, 1].max()))

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    fig.tight_layout(pad=0.1)
    fig.canvas.draw()

    # Robust across matplotlib versions: grab RGBA buffer and drop alpha
    buf = np.asarray(fig.canvas.buffer_rgba())          # (H,W,4) uint8
    img = np.ascontiguousarray(buf[..., :3])            # (H,W,3) uint8

    plt.close(fig)
    return img


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Make a GIF of predicted mesh evolution from precomp_rollout*.h5")
    ap.add_argument("--h5", type=str, required=True, help="Input H5 (e.g., precomp_rollout.h5)")
    ap.add_argument("--out_gif", type=str, required=True, help="Output GIF path")

    ap.add_argument("--t_start", type=int, default=1, help="First timestep index (default 1 => t00001)")
    ap.add_argument("--t_end", type=int, default=-1, help="Last timestep index inclusive (default: inferred)")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (use >1 to skip timesteps)")

    ap.add_argument("--fps", type=float, default=10.0, help="GIF frames per second")
    ap.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    ap.add_argument("--fig_w", type=float, default=6.5, help="Figure width (inches)")
    ap.add_argument("--fig_h", type=float, default=6.5, help="Figure height (inches)")

    ap.add_argument("--linewidth", type=float, default=0.20, help="Mesh line width")
    ap.add_argument("--max_cells", type=int, default=200000, help="Max cells per frame (<=0 means no limit)")
    ap.add_argument("--sample_strategy", type=str, default="per_level",
                    choices=["first", "random", "per_level"],
                    help="How to downsample cells when N is huge")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")

    ap.add_argument("--use_mask", type=int, default=1, help="Use pred_parents + mask_pred_parent_flat_u8 if present (0/1)")
    ap.add_argument("--color_by_level", type=int, default=0, help="Color edges by refinement level (0/1)")

    args = ap.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(args.h5)

    os.makedirs(os.path.dirname(args.out_gif) or ".", exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        H, W, dx0, dy0, T, bbox = _read_meta(f)
        groups = _sorted_timestep_groups(f)
        if not groups:
            raise RuntimeError("No timestep groups like t00001 found in the H5.")

        # Determine available timestep indices from group names
        # group name is t{t:05d}
        t_available = [int(g[1:]) for g in groups]
        t_min = min(t_available)
        t_max = max(t_available)

        t_start = max(args.t_start, t_min)
        t_end = t_max if args.t_end < 0 else min(args.t_end, t_max)
        if t_end < t_start:
            raise ValueError(f"Invalid range: t_start={t_start}, t_end={t_end}, available [{t_min},{t_max}]")

        if not np.isfinite(dx0) or not np.isfinite(dy0):
            raise RuntimeError("meta attrs dx/dy missing or invalid; cannot infer cell sizes from levels.")

        frames: List[np.ndarray] = []

        for t in range(t_start, t_end + 1, max(1, int(args.stride))):
            gname = f"t{t:05d}"
            if gname not in f:
                continue
            g = f[gname]

            if ("pred_centers" not in g) or ("pred_levels" not in g):
                continue

            centers = g["pred_centers"][...].astype(np.float32, copy=False)
            levels  = g["pred_levels"][...].astype(np.int64, copy=False)

            # optional: parent mask
            centers, levels = _apply_parent_mask_if_available(
                g, centers, levels, use_mask=bool(args.use_mask)
            )

            # optional: sampling for speed
            centers, levels = _sample_cells(
                centers, levels,
                max_cells=int(args.max_cells),
                seed=int(args.seed) + int(t),  # vary slightly by frame for stable-looking sampling
                strategy=str(args.sample_strategy),
            )

            title = f"{os.path.basename(args.h5)} | {gname} | N={centers.shape[0]}"

            img = _render_frame(
                centers, levels,
                dx0=float(dx0), dy0=float(dy0),
                bbox=bbox,
                title=title,
                linewidth=float(args.linewidth),
                color_by_level=bool(args.color_by_level),
                figsize=(float(args.fig_w), float(args.fig_h)),
                dpi=int(args.dpi),
            )
            frames.append(img)

            if len(frames) == 1 or (t - t_start) % (10 * max(1, int(args.stride))) == 0:
                print(f"[frame] {gname} -> {len(frames)} frames")

    if not frames:
        raise RuntimeError("No frames produced (check timestep range and dataset names).")

    duration = 1.0 / float(args.fps) if float(args.fps) > 0 else 0.1
    imageio.mimsave(args.out_gif, frames, duration=duration)
    print(f"[ok] wrote GIF: {args.out_gif}  (frames={len(frames)}, fps={args.fps})")


if __name__ == "__main__":
    main()

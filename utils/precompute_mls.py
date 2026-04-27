#!/usr/bin/env python3
"""
precompute_mls_from_precomp_h5.py

Reads an existing rollout-precompute HDF5 (e.g., precomp_rollout.h5) and produces a NEW HDF5
that additionally stores MLS *geometry-dependent* precomputables per timestep group.

What this script can precompute (geometry-only; independent of x/vel):
  - Gradient MLS geometry:
      * M_inv  : per-node inverse moment matrix (typically 2x2) used by MLS gradient solves
      * dX     : per-edge relative position vectors used in the gradient solve
      * (optional) augmented edge_index (2-hop stencil) if your MLS solver uses it
      * (optional) neighbor damping vector (per-node) if your MLS solver uses it
  - Diffusion/Laplacian MLS geometry:
      * lap_weights : per-edge weights used by apply_laplacian()
      * (optional) augmented edge_index used for building the moment matrix in weight solve (2-hop)
      * (optional) neighbor damping vector (per-node) if used

Notes:
  - This DOES NOT precompute r_adv or r_diff directly, because those depend on x_abs and velocity.
  - It is meant to let you bypass expensive per-step MLS geometry rebuilds inside training/rollout.

Typical usage:
  python precompute_mls_from_precomp_h5.py \
      --in_h5 /path/to/precomp_rollout.h5 \
      --out_h5 /path/to/precomp_rollout_mls.h5 \
      --poly_order 2 \
      --use_2hop 1 \
      --use_neighbor_damping 1 \
      --damping_alpha 0.5

Integration idea (separate patch):
  - Load these MLS datasets in your H5 precomp reader/collate and attach them to the batch.
  - Modify your MLS forward paths (AdvectionMLS/DiffusionMLS wrappers) to accept precomputed
    (M_inv,dX,edge_aug,weights,...) to avoid recomputing/caching-by-geometry during training.

If you want me to wire this into your existing H5 reader + CollateWithPrecompute + training loop,
paste the code that reads H5 into `self.precomp[...]` (the “reader” that fills pred_centers, pred_ei, etc.).
"""

from __future__ import annotations

import argparse
import os
import shutil
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:
    raise ImportError("This script requires h5py. Install with: pip install h5py") from e

import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError(
        "This script requires torch_geometric. Install PyG for your torch build."
    ) from e


# ----------------------------- HDF5 helpers -----------------------------

def _write_ds(g: "h5py.Group", name: str, arr: np.ndarray, *, compress: bool = True):
    if name in g:
        del g[name]
    kwargs = {}
    if compress:
        kwargs = dict(compression="gzip", compression_opts=4, shuffle=True, chunks=True)
    g.create_dataset(name, data=arr, **kwargs)


def _ensure_group(root: "h5py.File|h5py.Group", name: str) -> "h5py.Group":
    if name in root:
        return root[name]
    return root.create_group(name)


def _is_timestep_group(name: str) -> bool:
    return name.startswith("t") and len(name) >= 2 and name[1:].isdigit()


def _sorted_timestep_groups(f: "h5py.File") -> list[str]:
    names = [k for k in f.keys() if _is_timestep_group(k)]
    names.sort()
    return names


# ----------------------------- MLS precompute -----------------------------

def _import_mls_module(mls_path: Optional[str]):
    """
    Import your project's mls module.
    - If mls_path is None: assumes `import mls` works (module on PYTHONPATH).
    - If mls_path is a file: loads it as a module.
    """
    if mls_path is None:
        import mls  # type: ignore
        return mls

    import importlib.util
    import pathlib
    p = pathlib.Path(mls_path)
    if not p.exists():
        raise FileNotFoundError(f"--mls_path not found: {mls_path}")
    spec = importlib.util.spec_from_file_location("mls", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {mls_path}")
    mls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mls)  # type: ignore
    return mls


def _try_construct_solver(cls, kwargs: Dict[str, Any]):
    """
    Construct solver in a way that tolerates signature drift across your mls.py versions.
    """
    try:
        return cls(**kwargs)
    except TypeError:
        # Drop unknown kwargs progressively
        safe = {}
        for k, v in kwargs.items():
            try:
                _ = cls(**{k: v})
                safe[k] = v
            except TypeError:
                pass
        return cls(**safe)


def _get_augmented_edge_index(mls_mod, edge_index: torch.Tensor, num_nodes: int, device: torch.device,
                             use_2hop: bool) -> torch.Tensor:
    """
    Best-effort retrieval/compute of a 2-hop augmented stencil.
    """
    if not use_2hop:
        return edge_index

    # Prefer a module-level helper if it exists
    if hasattr(mls_mod, "compute_2hop_extension"):
        try:
            ei_aug = mls_mod.compute_2hop_extension(edge_index.to(device), num_nodes=num_nodes)
            if torch.is_tensor(ei_aug) and ei_aug.ndim == 2 and ei_aug.size(0) == 2:
                return ei_aug
        except Exception:
            pass

    # Otherwise: no augmentation available
    return edge_index


def _get_neighbor_damping(mls_mod, data: Data, alpha: float, device: torch.device, enabled: bool) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    if hasattr(mls_mod, "compute_neighbor_damping"):
        try:
            d = mls_mod.compute_neighbor_damping(data.to(device), alpha=float(alpha))
            if torch.is_tensor(d):
                return d
        except Exception:
            return None
    return None


def _precompute_grad_geometry(
    mls_mod,
    grad_solver,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    use_2hop: bool,
    use_neighbor_damping: bool,
    damping_alpha: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict with at least:
      - grad_M_inv: (N,2,2) or whatever your solver uses
      - grad_dX   : (E_used,2) (or matching your solver)
      - grad_ei_used: (2,E_used)
      - (optional) grad_node_damp: (N,)
    """
    pos = pos.to(device=device, dtype=torch.float32)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    N = int(pos.shape[0])

    ei_used = _get_augmented_edge_index(mls_mod, edge_index, num_nodes=N, device=device, use_2hop=use_2hop)

    data = Data(pos=pos, edge_index=ei_used)

    node_damp = _get_neighbor_damping(mls_mod, data, alpha=damping_alpha, device=device, enabled=use_neighbor_damping)

    out: Dict[str, torch.Tensor] = {"grad_ei_used": ei_used}

    # Prefer an explicit precompute if your solver exposes it
    for meth_name in ("precompute_geometry", "_precompute_geometry"):
        if hasattr(grad_solver, meth_name):
            try:
                M_inv, dX = getattr(grad_solver, meth_name)(data)
                if torch.is_tensor(M_inv) and torch.is_tensor(dX):
                    out["grad_M_inv"] = M_inv
                    out["grad_dX"] = dX
                    break
            except Exception:
                pass

    # Fallback: attempt to use solver's internal construction (for your uploaded SolveGradientsLST-like API)
    if "grad_M_inv" not in out or "grad_dX" not in out:
        # mimic common MLS grad geometry: dX along edges, per-node 2x2 moment inverse
        row, col = ei_used[0], ei_used[1]
        dX = pos[row] - pos[col]  # (E,2)
        # Moment matrix per node: sum over neighbors of dX dX^T
        # M[n] = Σ_e (dX_e)(dX_e)^T for edges incoming to row==n
        E = dX.shape[0]
        M = torch.zeros((N, 2, 2), device=device, dtype=torch.float32)
        outer = dX[:, :, None] * dX[:, None, :]  # (E,2,2)
        M.index_add_(0, row, outer)
        # Invert with damping epsilon
        eps = 1e-8
        M = M + eps * torch.eye(2, device=device, dtype=torch.float32).view(1, 2, 2)
        M_inv = torch.linalg.inv(M)
        out["grad_M_inv"] = M_inv
        out["grad_dX"] = dX

    if node_damp is not None:
        out["grad_node_damp"] = node_damp

    return out


def _precompute_lap_weights(
    mls_mod,
    lap_solver,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    poly_order: int,
    use_2hop: bool,
    use_neighbor_damping: bool,
    damping_alpha: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict with at least:
      - lap_weights: (E,) weights for apply_laplacian on *original* edge_index
      - (optional) lap_ei_aug: (2,E_aug) if use_2hop used in weight solve moment matrix
      - (optional) lap_node_damp: (N,)
    """
    pos = pos.to(device=device, dtype=torch.float32)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    N = int(pos.shape[0])

    # If your lap solver builds an augmented stencil internally, we can store it too (best-effort)
    ei_aug = _get_augmented_edge_index(mls_mod, edge_index, num_nodes=N, device=device, use_2hop=use_2hop)

    data_aug = Data(pos=pos, edge_index=ei_aug)
    node_damp = _get_neighbor_damping(mls_mod, data_aug, alpha=damping_alpha, device=device, enabled=use_neighbor_damping)

    out: Dict[str, torch.Tensor] = {}
    out["lap_ei_base"] = edge_index
    if ei_aug is not edge_index:
        out["lap_ei_aug"] = ei_aug
    if node_damp is not None:
        out["lap_node_damp"] = node_damp

    # Preferred: use lap solver's API if available
    if hasattr(lap_solver, "compute_weights"):
        try:
            w = lap_solver.compute_weights(data_aug, edge_index=edge_index, use_2hop_extension=use_2hop)
            if torch.is_tensor(w):
                out["lap_weights"] = w
                return out
        except Exception:
            pass

    # Fallback: if lap_solver itself is callable and returns weights (like SolveWeightLST2d.forward)
    try:
        w = lap_solver(data_aug)
        if torch.is_tensor(w):
            out["lap_weights"] = w
            return out
    except Exception:
        pass

    raise RuntimeError("Could not compute lap_weights with the provided MLS module/solver API.")


# ----------------------------- main pipeline -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Add MLS precomputables into a new HDF5 derived from precomp_rollout.h5")
    ap.add_argument("--in_h5", type=str, required=True, help="Input precomputed rollout H5 (e.g., precomp_rollout.h5)")
    ap.add_argument("--out_h5", type=str, default=None, help="Output H5 (default: <in>_mls.h5)")
    ap.add_argument("--overwrite", type=int, default=0, help="Overwrite output if it exists (0/1)")

    ap.add_argument("--config_json", type=str, default=None, help="Optional config JSON path (only used for metadata)")
    ap.add_argument("--mls_path", type=str, default=None, help="Optional path to your project's mls.py file")

    ap.add_argument("--device", type=str, default="cpu", help="Device for MLS precompute (cpu recommended)")
    ap.add_argument("--poly_order", type=int, default=2, help="Polynomial order for laplacian weights (typically 2)")

    ap.add_argument("--use_2hop", type=int, default=1, help="Use 2-hop extension (0/1)")
    ap.add_argument("--use_neighbor_damping", type=int, default=1, help="Use neighbor damping (0/1)")
    ap.add_argument("--damping_alpha", type=float, default=0.5, help="Neighbor damping alpha")

    ap.add_argument("--compute_grad", type=int, default=1, help="Precompute gradient geometry (0/1)")
    ap.add_argument("--compute_lap", type=int, default=1, help="Precompute laplacian weights (0/1)")
    ap.add_argument("--weights_dtype", type=str, default="float32", choices=["float32", "float16"],
                    help="dtype for stored lap_weights")

    ap.add_argument("--compress", type=int, default=1, help="gzip compress new datasets (0/1)")
    ap.add_argument("--progress_every", type=int, default=10, help="Print progress every N groups")

    args = ap.parse_args()

    in_h5 = args.in_h5
    if args.out_h5 is None:
        root, ext = os.path.splitext(in_h5)
        args.out_h5 = root + "_mls" + (ext if ext else ".h5")
    out_h5 = args.out_h5

    if not os.path.exists(in_h5):
        raise FileNotFoundError(f"--in_h5 not found: {in_h5}")

    if os.path.exists(out_h5):
        if not bool(args.overwrite):
            raise FileExistsError(f"Output exists: {out_h5} (use --overwrite 1)")
        os.remove(out_h5)

    # Copy input -> output, then append MLS datasets into output.
    shutil.copy2(in_h5, out_h5)

    # Load optional cfg for metadata
    cfg: Optional[Dict[str, Any]] = None
    if args.config_json is not None and os.path.exists(args.config_json):
        with open(args.config_json, "r") as f:
            cfg = json.load(f)

    # Import MLS module
    mls_mod = _import_mls_module(args.mls_path)

    dev = torch.device(args.device)
    use_2hop = bool(args.use_2hop)
    use_neighbor_damping = bool(args.use_neighbor_damping)
    damping_alpha = float(args.damping_alpha)
    poly_order = int(args.poly_order)

    # Construct solvers (best-effort across versions)
    grad_solver = None
    lap_solver = None

    if bool(args.compute_grad):
        if not hasattr(mls_mod, "SolveGradientsLST"):
            raise RuntimeError("MLS module missing SolveGradientsLST")
        grad_solver = _try_construct_solver(
            mls_mod.SolveGradientsLST,
            dict(
                cache_by_geometry=False,          # tolerated if supported
                use_2hop_extension=use_2hop,      # tolerated if supported
                use_neighbor_damping=use_neighbor_damping,
                damping_alpha=damping_alpha,
            ),
        )

    if bool(args.compute_lap):
        if not hasattr(mls_mod, "SolveWeightLST2d"):
            raise RuntimeError("MLS module missing SolveWeightLST2d")
        lap_solver = _try_construct_solver(
            mls_mod.SolveWeightLST2d,
            dict(
                polynomial_order=poly_order,
                cache_by_geometry=False,          # tolerated if supported
                use_2hop_extension=use_2hop,      # tolerated if supported
                use_neighbor_damping=use_neighbor_damping,
                damping_alpha=damping_alpha,
            ),
        )

    # Append datasets
    with h5py.File(out_h5, "a") as f:
        # Add meta about MLS precompute
        meta = _ensure_group(f, "meta")
        meta.attrs["mls_precompute"] = np.bytes_("1")
        meta.attrs["mls_device"] = np.bytes_(str(args.device))
        meta.attrs["mls_poly_order"] = int(poly_order)
        meta.attrs["mls_use_2hop"] = int(use_2hop)
        meta.attrs["mls_use_neighbor_damping"] = int(use_neighbor_damping)
        meta.attrs["mls_damping_alpha"] = float(damping_alpha)
        if cfg is not None:
            meta.create_dataset("mls_cfg_json", data=np.bytes_(json.dumps(cfg, sort_keys=True, default=str)))

        groups = _sorted_timestep_groups(f)
        if len(groups) == 0:
            raise RuntimeError("No timestep groups found in H5 (expected groups like t00001, t00002, ...).")

        for gi, gname in enumerate(groups):
            g = f[gname]

            if "pred_centers" not in g or "pred_ei" not in g:
                continue

            pos_np = g["pred_centers"][...].astype(np.float32, copy=False)
            ei_np = g["pred_ei"][...]
            if ei_np.size == 0:
                continue

            # pred_ei stored int32 -> torch.long
            ei_np = ei_np.astype(np.int64, copy=False)

            pos = torch.from_numpy(pos_np)
            edge_index = torch.from_numpy(ei_np)

            mls_g = _ensure_group(g, "mls")
            mls_g.attrs["poly_order"] = int(poly_order)
            mls_g.attrs["use_2hop"] = int(use_2hop)
            mls_g.attrs["use_neighbor_damping"] = int(use_neighbor_damping)
            mls_g.attrs["damping_alpha"] = float(damping_alpha)

            # --- gradient geometry ---
            if grad_solver is not None:
                grad = _precompute_grad_geometry(
                    mls_mod,
                    grad_solver,
                    pos,
                    edge_index,
                    use_2hop=use_2hop,
                    use_neighbor_damping=use_neighbor_damping,
                    damping_alpha=damping_alpha,
                    device=dev,
                )
                # Store
                _write_ds(mls_g, "grad_ei_used", grad["grad_ei_used"].detach().cpu().to(torch.int32).numpy(),
                          compress=bool(args.compress))
                _write_ds(mls_g, "grad_M_inv", grad["grad_M_inv"].detach().cpu().to(torch.float32).numpy(),
                          compress=bool(args.compress))
                _write_ds(mls_g, "grad_dX", grad["grad_dX"].detach().cpu().to(torch.float32).numpy(),
                          compress=bool(args.compress))
                if "grad_node_damp" in grad:
                    _write_ds(mls_g, "grad_node_damp", grad["grad_node_damp"].detach().cpu().to(torch.float32).numpy(),
                              compress=bool(args.compress))
                mls_g.attrs["grad_layout"] = np.bytes_("M_inv: (N,*,*), dX: (E_used,2), ei_used: (2,E_used)")

            # --- laplacian weights ---
            if lap_solver is not None:
                lap = _precompute_lap_weights(
                    mls_mod,
                    lap_solver,
                    pos,
                    edge_index,
                    poly_order=poly_order,
                    use_2hop=use_2hop,
                    use_neighbor_damping=use_neighbor_damping,
                    damping_alpha=damping_alpha,
                    device=dev,
                )

                w = lap["lap_weights"].detach().cpu()
                if args.weights_dtype == "float16":
                    w_np = w.to(torch.float16).numpy()
                    mls_g.attrs["lap_weights_dtype"] = np.bytes_("float16")
                else:
                    w_np = w.to(torch.float32).numpy()
                    mls_g.attrs["lap_weights_dtype"] = np.bytes_("float32")

                _write_ds(mls_g, "lap_weights", w_np, compress=bool(args.compress))
                _write_ds(mls_g, "lap_ei_base", lap["lap_ei_base"].detach().cpu().to(torch.int32).numpy(),
                          compress=bool(args.compress))
                if "lap_ei_aug" in lap:
                    _write_ds(mls_g, "lap_ei_aug", lap["lap_ei_aug"].detach().cpu().to(torch.int32).numpy(),
                              compress=bool(args.compress))
                if "lap_node_damp" in lap:
                    _write_ds(mls_g, "lap_node_damp", lap["lap_node_damp"].detach().cpu().to(torch.float32).numpy(),
                              compress=bool(args.compress))

                mls_g.attrs["lap_layout"] = np.bytes_("weights: (E_base,), ei_base: (2,E_base), ei_aug optional")

            if (gi % int(args.progress_every)) == 0:
                N = int(pos_np.shape[0])
                E = int(ei_np.shape[1])
                print(f"[MLS-PRECOMP] {gname}: N={N} E={E}  ({gi+1}/{len(groups)})")

        f.flush()

    print(f"[MLS-PRECOMP] Wrote: {out_h5}")


if __name__ == "__main__":
    main()

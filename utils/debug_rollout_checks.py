# utils/debug_rollout_checks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import torch


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure x is (N, F). Accepts (N,), (N,1), (N,F)."""
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1D or 2D tensor, got shape={tuple(x.shape)}")


def _broadcast_dt(dt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    dt may be scalar (), (1,), (N,), (N,1), or (N,F)-broadcastable.
    Return shape broadcastable to x (N,F).
    """
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=x.device, dtype=x.dtype)
    if dt.ndim == 0:
        return dt
    if dt.ndim == 1:
        # (N,) or (1,)
        if dt.numel() == 1:
            return dt.reshape(())
        if dt.shape[0] == x.shape[0]:
            return dt[:, None]  # (N,1)
        raise ValueError(f"dt 1D shape {tuple(dt.shape)} not compatible with x {tuple(x.shape)}")
    if dt.ndim == 2:
        return dt
    raise ValueError(f"dt has unsupported ndim={dt.ndim}")


def standardize(x_phys: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    x = _to_2d(x_phys)
    mu = _to_2d(mu).to(device=x.device, dtype=x.dtype)
    sigma = _to_2d(sigma).to(device=x.device, dtype=x.dtype)
    return (x - mu) / (sigma + 1e-12)


def inv_standardize(x_norm: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    x = _to_2d(x_norm)
    mu = _to_2d(mu).to(device=x.device, dtype=x.dtype)
    sigma = _to_2d(sigma).to(device=x.device, dtype=x.dtype)
    return x * (sigma + 1e-12) + mu


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach()
    finite = torch.isfinite(x)
    out: Dict[str, float] = {}
    out["finite_frac"] = float(finite.float().mean().item())
    if finite.any():
        xf = x[finite]
        out["min"] = float(xf.min().item())
        out["max"] = float(xf.max().item())
        out["mean"] = float(xf.mean().item())
        out["absmax"] = float(xf.abs().max().item())
    else:
        out["min"] = float("nan")
        out["max"] = float("nan")
        out["mean"] = float("nan")
        out["absmax"] = float("nan")
    return out


def err_stats(err: torch.Tensor) -> Dict[str, Any]:
    """
    err: (N,F). Returns global and per-feature summaries.
    """
    err = _to_2d(err).detach()
    ae = err.abs()
    finite = torch.isfinite(err).all(dim=1)
    out: Dict[str, Any] = {}
    out["finite_rows_frac"] = float(finite.float().mean().item())
    if finite.any():
        e = err[finite]
        ae = ae[finite]
        out["absmax"] = float(ae.max().item())
        out["mae"] = float(ae.mean().item())
        # per-feature
        out["per_feature_mae"] = ae.mean(dim=0).cpu().tolist()
        out["per_feature_absmax"] = ae.max(dim=0).values.cpu().tolist()
        # p95 over rows for each feature
        out["per_feature_p95"] = torch.quantile(ae, 0.95, dim=0).cpu().tolist()
    else:
        out["absmax"] = float("nan")
        out["mae"] = float("nan")
        out["per_feature_mae"] = []
        out["per_feature_absmax"] = []
        out["per_feature_p95"] = []
    return out


@dataclass
class DebugRolloutChecks:
    enabled: bool = True

    # Toggle each check independently (can also be toggled via env vars)
    check_target_consistency: bool = True
    check_perfect_model: bool = True
    check_time_index_audit: bool = True
    check_units_audit: bool = True

    # Behavior
    only_k: Optional[set[int]] = None  # e.g., {0,1} to reduce noise
    atol: float = 1e-5
    rtol: float = 1e-3

    # Perfect-model mode: if True, override y_pred with target (for the update only)
    perfect_model_mode: bool = False

    @staticmethod
    def from_env(default_enabled: bool = True) -> "DebugRolloutChecks":
        def env_bool(name: str, default: bool) -> bool:
            v = os.getenv(name, None)
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "y", "on")

        enabled = env_bool("DBG_ROLLOUT", default_enabled)
        return DebugRolloutChecks(
            enabled=enabled,
            check_target_consistency=env_bool("DBG_TARGET", True),
            check_perfect_model=env_bool("DBG_PERFECT", True),
            check_time_index_audit=env_bool("DBG_TIMEIDX", True),
            check_units_audit=env_bool("DBG_UNITS", True),
            perfect_model_mode=env_bool("DBG_PERFECT_MODE", False),
        )

    def _k_ok(self, k: int) -> bool:
        return (self.only_k is None) or (k in self.only_k)

    @torch.no_grad()
    def run_time_index_audit(
        self,
        *,
        k: int,
        base_t: int,
        t_curr: int,
        t_next: int,
        mesh_curr_id: Optional[Any] = None,
        mesh_next_id: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not (self.enabled and self.check_time_index_audit and self._k_ok(k)):
            return
        msg = {
            "k": k,
            "base_t": base_t,
            "t_curr": t_curr,
            "t_next": t_next,
            "mesh_curr_id": str(mesh_curr_id),
            "mesh_next_id": str(mesh_next_id),
        }
        if extra:
            msg.update({f"extra.{kk}": str(vv) for kk, vv in extra.items()})
        print("[DBG-TIMEIDX]", msg)

        # Hard sanity checks you can disable if needed:
        if t_next != t_curr + 1:
            print(f"[DBG-TIMEIDX][WARN] t_next != t_curr+1 (t_curr={t_curr}, t_next={t_next})")
        if t_curr != base_t + k:
            print(f"[DBG-TIMEIDX][WARN] t_curr != base_t+k (base_t={base_t}, k={k}, t_curr={t_curr})")

    @torch.no_grad()
    def run_target_consistency_check(
        self,
        *,
        k: int,
        x_teacher_t: torch.Tensor,      # (N,F) teacher at t on the comparison mesh
        x_teacher_tp1: torch.Tensor,    # (N,F) teacher at t+1 on the comparison mesh
        target: torch.Tensor,           # (N,F) rate_target OR delta_target
        dt: Optional[torch.Tensor],     # required if mode="rate"
        mode: str,                      # "rate" or "delta"
        label: str = "",
    ) -> None:
        if not (self.enabled and self.check_target_consistency and self._k_ok(k)):
            return

        xt = _to_2d(x_teacher_t)
        x1 = _to_2d(x_teacher_tp1)
        tgt = _to_2d(target).to(device=xt.device, dtype=xt.dtype)

        if mode not in ("rate", "delta"):
            raise ValueError(f"mode must be 'rate' or 'delta', got {mode}")

        if mode == "rate":
            if dt is None:
                raise ValueError("dt is required for mode='rate'")
            dtb = _broadcast_dt(dt, xt).to(device=xt.device, dtype=xt.dtype)
            x1_hat = xt + tgt * dtb
        else:
            x1_hat = xt + tgt

        err = x1_hat - x1
        st = err_stats(err)

        # Also compare magnitudes for quick “is my scale off?” detection
        tgt_st = tensor_stats(tgt)
        xt_st = tensor_stats(xt)
        x1_st = tensor_stats(x1)

        print(f"[DBG-TARGET]{' '+label if label else ''} k={k} mode={mode}")
        print(f"  xt   stats: {xt_st}")
        print(f"  x1   stats: {x1_st}")
        print(f"  tgt  stats: {tgt_st}")
        print(f"  err  stats: {st}")

    @torch.no_grad()
    def maybe_override_y_pred_with_target(
        self,
        *,
        k: int,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        label: str = "",
    ) -> torch.Tensor:
        """
        Perfect-model test: for update only, replace y_pred with target.
        You still log original y_pred stats separately.
        """
        if not (self.enabled and self.check_perfect_model and self.perfect_model_mode and self._k_ok(k)):
            return y_pred

        yp = _to_2d(y_pred)
        tgt = _to_2d(target).to(device=yp.device, dtype=yp.dtype)

        print(f"[DBG-PERFECT]{' '+label if label else ''} k={k} OVERRIDING y_pred <- target for update")
        return tgt

    @torch.no_grad()
    def run_units_audit(
        self,
        *,
        k: int,
        x_norm: Optional[torch.Tensor],        # (N,F)
        x_phys: Optional[torch.Tensor],        # (N,F)
        mu: Optional[torch.Tensor],            # (1,F) or (F,) or (N,F)
        sigma: Optional[torch.Tensor],         # same shapes as mu
        label: str = "",
        max_rel_err_warn: float = 1e-2,
    ) -> None:
        """
        If you have both x_norm and x_phys and (mu,sigma), verify:
          inv_standardize(x_norm) ≈ x_phys
          standardize(x_phys) ≈ x_norm
        """
        if not (self.enabled and self.check_units_audit and self._k_ok(k)):
            return
        if mu is None or sigma is None:
            print(f"[DBG-UNITS]{' '+label if label else ''} k={k} (mu/sigma not provided) SKIP")
            return
        if x_norm is None or x_phys is None:
            print(f"[DBG-UNITS]{' '+label if label else ''} k={k} (need both x_norm and x_phys) SKIP")
            return

        xn = _to_2d(x_norm)
        xp = _to_2d(x_phys)

        xp_from_xn = inv_standardize(xn, mu, sigma)
        xn_from_xp = standardize(xp, mu, sigma)

        # Relative error in physical space (scale-sensitive, but very revealing)
        denom = xp.abs().clamp_min(1e-12)
        rel = (xp_from_xn - xp).abs() / denom
        rel_max = float(rel.max().item())
        rel_mean = float(rel.mean().item())

        print(f"[DBG-UNITS]{' '+label if label else ''} k={k}")
        print(f"  x_norm stats: {tensor_stats(xn)}")
        print(f"  x_phys stats: {tensor_stats(xp)}")
        print(f"  inv(x_norm)->phys rel_err: mean={rel_mean:.3e} max={rel_max:.3e}")

        # Additional: compare normalized recon
        diff_n = (xn_from_xp - xn).abs()
        print(f"  std(x_phys)->norm abs_err: mean={float(diff_n.mean().item()):.3e} max={float(diff_n.max().item()):.3e}")

        if rel_max > max_rel_err_warn:
            print(f"[DBG-UNITS][WARN] Large rel_err suggests scaler mismatch / feature order mismatch / double inverse scaling.")

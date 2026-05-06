#!/usr/bin/env python3
"""
Plot training history curves from train_log.csv produced by train_basic_point.py.

Examples:
  python utils/plot_training_history.py --run-dir runs_karman_basic/test --no-show

  python utils/plot_training_history.py \
      --log-csv runs_karman_basic/test/train_log.csv \
      --metric both \
      --out runs_karman_basic/test/training_curves.png \
      --no-show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot train/val curves from train_log.csv.")
    ap.add_argument("--log-csv", type=Path, default=None, help="Path to train_log.csv.")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing train_log.csv (alternative to --log-csv).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="loss",
        choices=("loss", "mae", "both"),
        help="Which metric(s) to plot.",
    )
    ap.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    ap.add_argument("--logy", action="store_true", help="Use logarithmic y-axis.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: <run_dir>/loss_curves.png or training_curves.png for --metric both.",
    )
    ap.add_argument("--no-show", action="store_true", help="Do not display plot window.")
    return ap.parse_args()


def _resolve_log_csv_path(args: argparse.Namespace) -> Path:
    if args.log_csv is not None and args.run_dir is not None:
        raise ValueError("Use either --log-csv or --run-dir, not both.")
    if args.log_csv is None and args.run_dir is None:
        raise ValueError("Provide one of --log-csv or --run-dir.")
    if args.log_csv is not None:
        return args.log_csv
    return Path(args.run_dir) / "train_log.csv"


def _resolve_out_path(args: argparse.Namespace, log_csv_path: Path) -> Path:
    if args.out is not None:
        return args.out
    if args.metric == "both":
        return log_csv_path.parent / "training_curves.png"
    return log_csv_path.parent / "loss_curves.png"


def _load_rows(log_csv_path: Path) -> List[Dict[str, str]]:
    with open(log_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"No rows found in CSV: {log_csv_path}")
    required = {"epoch", "split", "loss", "mae"}
    cols = set(rows[0].keys())
    missing = required - cols
    if missing:
        raise ValueError(f"CSV missing required columns {sorted(missing)}: {log_csv_path}")
    return rows


def _to_float_or_nan(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _extract_series(rows: List[Dict[str, str]], metric: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    by_split: Dict[str, Dict[int, float]] = {}
    for r in rows:
        split = str(r.get("split", "")).strip().lower()
        if not split:
            continue
        ep = _to_float_or_nan(str(r.get("epoch", "")))
        v = _to_float_or_nan(str(r.get(metric, "")))
        if not np.isfinite(ep):
            continue
        epi = int(round(ep))
        by_split.setdefault(split, {})[epi] = v

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split, mp in by_split.items():
        eps = sorted(mp.keys())
        vals = [mp[e] for e in eps]
        out[split] = (
            np.asarray(eps, dtype=np.float64),
            np.asarray(vals, dtype=np.float64),
        )
    return out


def _plot_one(ax, rows: List[Dict[str, str]], metric: str, title: str, logy: bool) -> None:
    series = _extract_series(rows, metric=metric)
    if len(series) == 0:
        raise ValueError(f"No split data found for metric '{metric}'.")

    color_map = {
        "train": "tab:blue",
        "val": "tab:orange",
        "validation": "tab:orange",
        "test": "tab:green",
    }
    order = ["train", "val", "validation", "test"]
    seen = set()
    for split in order + sorted(series.keys()):
        if split not in series or split in seen:
            continue
        seen.add(split)
        ep, vv = series[split]
        ax.plot(ep, vv, lw=2.0, label=split, color=color_map.get(split, None))

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if logy:
        ax.set_yscale("log")


def main() -> None:
    args = parse_args()
    log_csv_path = _resolve_log_csv_path(args)
    if not log_csv_path.exists():
        raise FileNotFoundError(f"train_log.csv not found: {log_csv_path}")

    out_path = _resolve_out_path(args, log_csv_path=log_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(log_csv_path)

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.metric == "both":
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
        _plot_one(
            axes[0],
            rows,
            metric="loss",
            title=(args.title or "Training/Validation Loss"),
            logy=args.logy,
        )
        _plot_one(
            axes[1],
            rows,
            metric="mae",
            title="Training/Validation MAE",
            logy=args.logy,
        )
    else:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
        default_title = "Training/Validation Loss" if args.metric == "loss" else "Training/Validation MAE"
        _plot_one(
            ax,
            rows,
            metric=args.metric,
            title=(args.title or default_title),
            logy=args.logy,
        )

    fig.savefig(out_path, dpi=170)
    print(f"Saved training curve plot: {out_path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

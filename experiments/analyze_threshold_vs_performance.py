"""
Regression: pruning-threshold disparity vs OOD performance degradation.

Tests whether OOD performance degradation (gts / quality deltas) can be
predicted from how different the firing-rate pruning thresholds are between
the scoped domain and the OOD domain — no distribution-level overlap needed.

Predictors
----------
1. log_threshold_ratio = log10(t_ood / t_scoped)
   Positive ↔ OOD threshold is higher: OOD neurons fire strongly and are
   more likely to survive scoping → expect less degradation.
   Negative ↔ OOD threshold is lower: OOD neurons fire weakly and are
   pruned away by scoping → expect more degradation.

2. log_count_ratio = log10(n_active_ood / n_active_scoped)
   where n_active_X = #{i : dist_X[i] >= threshold_X}.
   More OOD-active neurons relative to scoped-active neurons → more of
   OOD's neurons at risk of being pruned.
   (Requires loading firing-rate distributions.)

3. ood_survival_fraction = #{i : A[i] >= t_scoped  AND  B[i] >= t_ood}
                           ─────────────────────────────────────────────
                                     #{i : B[i] >= t_ood}
   Fraction of OOD's important neurons that survive the scoped domain's
   pruning threshold.
   (Requires loading firing-rate distributions.)

Usage
-----
    python experiments/analyze_threshold_vs_performance.py
    python experiments/analyze_threshold_vs_performance.py --device cuda
    python experiments/analyze_threshold_vs_performance.py --output-dir results/thr_reg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))  # experiments/ → direct imports
sys.path.insert(0, str(REPO_ROOT))              # repo root → sae_scoping package

from analyze_overlap_vs_performance import (  # noqa: E402
    PERF_DATA,
    FIRING_RATE_THRESHOLDS,
    PAIR_STYLES,
    _domain_cache_path,
    _load_normalized,
)


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------

def _ols(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """(slope, intercept, r², p).  Uses only finite entries; requires ≥ 3."""
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 3:
        return (float("nan"),) * 4
    res = scipy_stats.linregress(xs[mask], ys[mask])
    return float(res.slope), float(res.intercept), float(res.rvalue ** 2), float(res.pvalue)


# ---------------------------------------------------------------------------
# Scatter panel helper
# ---------------------------------------------------------------------------

def _scatter_panel(
    ax: plt.Axes,
    valid_pairs: list[tuple],
    xs: np.ndarray,
    ys: np.ndarray,
    xlabel: str,
    title: str,
    label_fn,
) -> tuple[float, float]:
    """Plot scatter + OLS regression line. Returns (r², p)."""
    slope, intercept, r2, p = _ols(xs, ys)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs_v = xs[mask]
    ys_v = ys[mask]
    pairs_v = [row for row, m in zip(valid_pairs, mask) if m]

    if len(xs_v) > 1 and np.isfinite(r2):
        span = float(xs_v.max() - xs_v.min())
        margin = span * 0.03 if span > 0 else max(abs(float(xs_v.mean())) * 0.05, 0.1)
        x_line = np.linspace(float(xs_v.min()) - margin, float(xs_v.max()) + margin, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", lw=1.5,
                label=f"OLS  R²={r2:.3f}  p={p:.3f}", zorder=2)

    for row, x, y in zip(pairs_v, xs_v, ys_v):
        color, marker = PAIR_STYLES.get((row[0], row[1]), ("#888888", "o"))
        ax.scatter(x, y, color=color, marker=marker, s=90, zorder=3,
                   label=label_fn(row))

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(f"{title}  (R²={r2:.3f})", fontsize=10)
    ax.grid(True, alpha=0.3)
    return r2, p


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _make_plots(
    title_tag: str,
    valid_pairs: list[tuple],
    log_thr: np.ndarray,
    log_cnt: np.ndarray,
    surv: np.ndarray,
    output_dir: Path,
    label_fn,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for perf_metric, y_col, ylabel in [
        ("gts",     5, "Ground-truth similarity delta (%)"),
        ("quality", 6, "Quality delta (%)"),
    ]:
        ys = np.array([row[y_col] for row in valid_pairs], dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)

        panels = [
            (axes[0], log_thr, "log₁₀(t_ood / t_scoped)",
             "log(t_ood / t_scoped)"),
            (axes[1], log_cnt, "log₁₀(n_active_ood / n_active_scoped)",
             "log(n_active_ood / n_active_scoped)"),
            (axes[2], surv,    "OOD survival fraction  (above scoped threshold)",
             "OOD survival fraction"),
        ]
        for ax, xs, xlabel, name in panels:
            r2, p = _scatter_panel(ax, valid_pairs, xs, ys, xlabel, name, label_fn)
            print(f"  {perf_metric} / {name}: R²={r2:.3f}  p={p:.3f}")

        axes[0].set_ylabel(ylabel)
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles, labels, fontsize=7.5, ncol=2,
                       loc="best", framealpha=0.9)
        fig.suptitle(
            f"Pruning-threshold disparity vs OOD degradation — {title_tag} ({perf_metric})",
            fontsize=11,
        )
        plt.tight_layout()
        out = output_dir / f"scatter_{perf_metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  → {out}")


# ---------------------------------------------------------------------------
# Per-pair predictor computation
# ---------------------------------------------------------------------------

def _compute_pair(
    scoped: str,
    ood: str,
    sae_dir: str,
    model: str,
    device: torch.device,
) -> tuple[float | None, float | None, float | None]:
    """Returns (log_thr_ratio, log_count_ratio, ood_survival_fraction).

    log_thr_ratio requires both thresholds to be set (> 0).
    log_count_ratio and ood_survival_fraction additionally require
    the firing-rate distributions to be loadable.
    """
    t_a = FIRING_RATE_THRESHOLDS.get((model, scoped), 0.0)
    t_b = FIRING_RATE_THRESHOLDS.get((model, ood),    0.0)

    log_thr = float(np.log10(t_b / t_a)) if t_a > 0 and t_b > 0 else None

    # Distribution-based metrics
    log_cnt = None
    surv    = None
    if t_a > 0 and t_b > 0:
        path_a = _domain_cache_path(sae_dir, scoped)
        path_b = _domain_cache_path(sae_dir, ood)
        da = _load_normalized(path_a, device)
        db = _load_normalized(path_b, device)
        if da is not None and db is not None:
            n_a = int((da >= t_a).sum().item())
            n_b = int((db >= t_b).sum().item())
            if n_a > 0 and n_b > 0:
                log_cnt = float(np.log10(n_b / n_a))
                surviving = int(((da >= t_a) & (db >= t_b)).sum().item())
                surv = surviving / n_b

    return log_thr, log_cnt, surv


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def _run_model(
    model: str,
    data: list[tuple],
    output_dir: Path,
    device: torch.device,
) -> tuple[list[tuple], np.ndarray, np.ndarray, np.ndarray] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n── {model}  ({len(data)} pairs) ──")

    log_thrs: list[float | None] = []
    log_cnts: list[float | None] = []
    survs:    list[float | None] = []

    for scoped, ood, layer, sae_dir, _model, gts, q in data:
        lt, lc, sf = _compute_pair(scoped, ood, sae_dir, model, device)
        log_thrs.append(lt)
        log_cnts.append(lc)
        survs.append(sf)
        if lt is None:
            print(f"  [warn] {scoped}→{ood}: no threshold entry for ({model!r}, {scoped!r}) or ({model!r}, {ood!r})")

    valid_idx = [i for i, v in enumerate(log_thrs) if v is not None]
    if len(valid_idx) < 3:
        print(f"  [skip] only {len(valid_idx)} valid pairs — need ≥ 3 for regression.")
        return None

    valid_pairs = [data[i] for i in valid_idx]

    def _arr(lst: list) -> np.ndarray:
        return np.array([lst[i] if lst[i] is not None else np.nan for i in valid_idx], dtype=float)

    xs_lt = _arr(log_thrs)
    xs_lc = _arr(log_cnts)
    xs_sf = _arr(survs)

    n_dist = int(np.isfinite(xs_lc).sum())
    print(f"  {len(valid_pairs)} / {len(data)} pairs with threshold data  |  {n_dist} with distributions")

    # Summary table
    print(f"\n  {'Pair':<26} | log(t_b/t_a) | log(n_b/n_a) | survival |  gts  | quality")
    print("  " + "─" * 82)
    for row, lt, lc, sf in zip(valid_pairs, xs_lt, xs_lc, xs_sf):
        s, o, layer, _, _m, gts, q = row
        lc_s = f"{lc:+.3f}" if np.isfinite(lc) else "   N/A"
        sf_s = f"{sf:.4f}"  if np.isfinite(sf) else "   N/A"
        print(f"  {s:<10} → {o:<10} | {lt:+12.3f} | {lc_s:>12} | {sf_s:>8} | {gts:5.1f} | {q:5.1f}")

    _make_plots(
        title_tag=model,
        valid_pairs=valid_pairs,
        log_thr=xs_lt,
        log_cnt=xs_lc,
        surv=xs_sf,
        output_dir=output_dir,
        label_fn=lambda row: f"{row[0][:3]}→{row[1][:3]}  (L{row[2]})",
    )

    return valid_pairs, xs_lt, xs_lc, xs_sf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_dir: Path, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(set(row[4] for row in PERF_DATA))
    print(f"Device: {device}  |  total pairs: {len(PERF_DATA)}  |  models: {models}")

    all_pairs: list[tuple] = []
    all_lt:    list[float] = []
    all_lc:    list[float] = []
    all_sf:    list[float] = []

    for model in models:
        model_data = [row for row in PERF_DATA if row[4] == model]
        result = _run_model(model, model_data, output_dir / model, device)
        if result is not None:
            vp, lt, lc, sf = result
            all_pairs.extend(vp)
            all_lt.extend(lt.tolist())
            all_lc.extend(lc.tolist())
            all_sf.extend(sf.tolist())

    if len(all_pairs) >= 3:
        print(f"\n── combined  ({len(all_pairs)} pairs) ──")
        _make_plots(
            title_tag="all models (combined)",
            valid_pairs=all_pairs,
            log_thr=np.array(all_lt),
            log_cnt=np.array(all_lc),
            surv=np.array(all_sf),
            output_dir=output_dir / "combined",
            label_fn=lambda row: f"{row[0][:3]}→{row[1][:3]}  ({row[4]})",
        )

    print(f"\nDone. Outputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "experiments" / "sae_scoping" / "threshold_regression",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    run(args.output_dir, torch.device(args.device))


if __name__ == "__main__":
    main()

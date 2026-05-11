"""
Regression: directed SAE feature overlap vs out-of-domain performance degradation.

Each entry in PERF_DATA specifies:
  - layer   : SAE layer used in the scoping experiment  [required, error if absent]
  - sae_dir : path to the scoped domain's firing-rate cache directory [required]
              The script replaces the 'stemqa_<X>' component to find OOD caches.

Computes cross-domain coverage AUC(scoped→ood) from the exact SAE cache given,
then runs separate OLS regressions against gts and quality deltas.

CODE domain is excluded (no stemqa_code cache).
Physics-scoped OOD is excluded (all null).

Usage:
    python experiments/analyze_overlap_vs_performance.py
    python experiments/analyze_overlap_vs_performance.py --device cuda
    python experiments/analyze_overlap_vs_performance.py --output-dir results/reg

Input: SAE firing-rate distributions for domain A and domain B.

  Each distribution is a 1-D vector of length N (SAE width, e.g. 262,144) where entry i is the fraction of tokens (across all domain A
  examples) that activated feature i. Both distributions sum to 1.

  ---
  Metric 1: Cross-domain coverage AUC

  1. Sort features by their A-domain firing rate (descending) → gives a ranked list of "A's most important features"
  2. For each k in {1, 2, 4, ..., N} (log-spaced):
    - Take the top-k features by A's ranking
    - Sum their firing rates in B → coverage(k) = Σ_{i ∈ topK(A)} B[i]
    - This is "what fraction of B's total activation mass is explained by A's top-k features?"
  3. Compute AUC of the curve coverage(k) vs k via trapezoidal integration, normalized by the area of a flat curve at 1.0

  Interpretation: High value → A's most-active features are also responsible for most of B's activations. Directed: measures how well
  A's features "cover" B.

  ---
  Metric 2: Mean Reciprocal Rank (MRR) AUC

  1. Rank all N features in B by their firing rate (rank 1 = most active)
  2. Sort features by A's firing rate (descending)
  3. For each k in {1, 2, 4, ..., N}:
    - Take A's top-k features
    - Compute MRR(k) = (1/k) Σ_{i ∈ topK(A)} 1 / rank_B(i)
    - Normalize by the best-case MRR: H_k / k (harmonic number / k), achieved when A's top-k exactly equals B's top-k
  4. Compute AUC of normalized MRR(k) vs k

  Interpretation: High value → A's top features tend to rank near the top in B too. More sensitive to exact rank coincidences at small k
   (the 1/rank weighting gives strong signal when the same feature is #1 in both domains).

  ---
  Regression:
  Both metrics are regressed separately against OOD performance degradation (gts delta and quality delta) using OLS, to test whether
  higher feature overlap predicts less OOD degradation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sae_scoping.data_science import default_ks

# ---------------------------------------------------------------------------
# Performance data
# (scoped, ood, layer, sae_dir, model, gts, q)
# ---------------------------------------------------------------------------

_BASE = "/data/aruna_sankaranarayanan/SAEScoping/experiments/.cache"

# ── gemma-3-12b ──────────────────────────────────────────────────────────────
_SAE_BIO_12B  = (_BASE + "/stemqa_biology"
                 "/ignore_padding_True/google--gemma-3-12b-it--width_262k/layer_15--width_262k--l0_small")
_SAE_CHEM_12B = (_BASE + "/stemqa_chemistry"
                 "/ignore_padding_True/google--gemma-3-12b-it--width_262k/layer_15--width_262k--l0_small")
_SAE_MATH_12B = (_BASE + "/stemqa_math"
                 "/ignore_padding_True/google--gemma-3-12b-it/layer_31--width_16k--l0_medium")
_SAE_CODE_12B = (_BASE + "/coding_gemma3_12b"
                 "/ignore_padding_True/layer_31--width_16k--l0_medium")

# ── gemma-2-9b ───────────────────────────────────────────────────────────────
_SAE_BIO_9B   = (_BASE + "/stemqa_biology"
                 "/ignore_padding_True/google--gemma-2-9b-it/layer_31--width_16k--canonical")
_SAE_CHEM_9B  = (_BASE + "/stemqa_chemistry"
                 "/ignore_padding_True/google--gemma-2-9b-it/layer_31--width_16k--canonical")
_SAE_MATH_9B  = (_BASE + "/stemqa_math"
                 "/ignore_padding_True/google--gemma-2-9b-it/layer_31--width_16k--canonical")
_SAE_PHYS_9B  = (_BASE + "/stemqa_physics"
                 "/ignore_padding_True/google--gemma-2-9b-it/layer_31--width_16k--canonical")
_SAE_CODE_9B  = (_BASE + "/coding_gemma2_9b"
                 "/ignore_padding_True/layer_31--width_16k--canonical")

PERF_DATA: list[tuple[str, str, int, str, str, float, float]] = [
    # (scoped, ood, layer, sae_dir, model, gts, q)

    # ── gemma-3-12b ──────────────────────────────────────────────────────────
    ("biology",   "chemistry", 15, _SAE_BIO_12B,  "gemma-3-12b", -16.0,  -6.2),
    ("biology",   "math",      15, _SAE_BIO_12B,  "gemma-3-12b",  -9.5,  -4.0),
    ("biology",   "physics",   15, _SAE_BIO_12B,  "gemma-3-12b", -11.5,  -7.2),
    ("chemistry", "physics",   15, _SAE_CHEM_12B, "gemma-3-12b", -16.0,  -6.0),
    ("chemistry", "math",      15, _SAE_CHEM_12B, "gemma-3-12b", -19.0,  -6.0),
    ("chemistry", "biology",   15, _SAE_CHEM_12B, "gemma-3-12b", -19.0,  -6.5),
    ("math",      "physics",   31, _SAE_MATH_12B, "gemma-3-12b", -21.0, -17.0),
    ("math",      "chemistry", 31, _SAE_MATH_12B, "gemma-3-12b", -29.0, -20.0),
    ("math",      "biology",   31, _SAE_MATH_12B, "gemma-3-12b", -61.5, -46.0),
    ("code",      "physics",   31, _SAE_CODE_12B, "gemma-3-12b", -67.0, -85.0),
    ("code",      "math",      31, _SAE_CODE_12B, "gemma-3-12b", -60.0, -67.0),
    ("code",      "biology",   31, _SAE_CODE_12B, "gemma-3-12b",-100.0,-100.0),
    ("code",      "chemistry", 31, _SAE_CODE_12B, "gemma-3-12b", -67.0, -85.0),
    # physics-scoped OOD: all null → excluded

    # ── gemma-2-9b ───────────────────────────────────────────────────────────
    ("biology",   "chemistry", 31, _SAE_BIO_9B,   "gemma-2-9b",  -16.5,  -7.0),
    ("biology",   "physics",   31, _SAE_BIO_9B,   "gemma-2-9b",  -19.0,  -9.0),
    ("biology",   "math",      31, _SAE_BIO_9B,   "gemma-2-9b",  -18.0, -10.0),
    ("chemistry", "physics",   31, _SAE_CHEM_9B,  "gemma-2-9b",   -8.0,  -5.0),
    ("chemistry", "math",      31, _SAE_CHEM_9B,  "gemma-2-9b",   -5.0,  -4.0),
    ("chemistry", "biology",   31, _SAE_CHEM_9B,  "gemma-2-9b",  -17.0,  -9.3),
    ("math",      "physics",   31, _SAE_MATH_9B,  "gemma-2-9b",  -13.0,  -9.0),
    ("math",      "chemistry", 31, _SAE_MATH_9B,  "gemma-2-9b",  -23.0, -14.0),
    ("math",      "biology",   31, _SAE_MATH_9B,  "gemma-2-9b",  -47.0, -34.0),
    ("physics",   "chemistry", 31, _SAE_PHYS_9B,  "gemma-2-9b",   -7.5,  -4.3),
    ("physics",   "math",      31, _SAE_PHYS_9B,  "gemma-2-9b",   -9.0,  -6.0),
    ("physics",   "biology",   31, _SAE_PHYS_9B,  "gemma-2-9b",  -23.0, -13.0),
    ("code",      "chemistry", 31, _SAE_CODE_9B,  "gemma-2-9b",  -35.0, -34.0),
    ("code",      "math",      31, _SAE_CODE_9B,  "gemma-2-9b",  -40.0, -35.0),
    ("code",      "biology",   31, _SAE_CODE_9B,  "gemma-2-9b",  -57.0, -50.0),
    ("code",      "physics",   31, _SAE_CODE_9B,  "gemma-2-9b",  -35.0, -42.0),
]

PAIR_STYLES: dict[tuple[str, str], tuple[str, str]] = {
    ("biology",   "chemistry"): ("#1f77b4", "o"),
    ("biology",   "math"):      ("#aec7e8", "s"),
    ("biology",   "physics"):   ("#6baed6", "^"),
    ("chemistry", "physics"):   ("#2ca02c", "o"),
    ("chemistry", "math"):      ("#98df8a", "s"),
    ("chemistry", "biology"):   ("#74c476", "^"),
    ("math",      "physics"):   ("#d62728", "o"),
    ("math",      "chemistry"): ("#ff9896", "s"),
    ("math",      "biology"):   ("#e6550d", "^"),
    ("physics",   "chemistry"): ("#9467bd", "o"),
    ("physics",   "math"):      ("#c5b0d5", "s"),
    ("physics",   "biology"):   ("#7b4173", "^"),
    ("code",      "biology"):   ("#3d0073", "o"),
    ("code",      "physics"):   ("#6b21a8", "s"),
    ("code",      "chemistry"): ("#a855f7", "^"),
    ("code",      "math"):      ("#d8b4fe", "D"),
}

_N_SAMPLES = 10_000

# Maps coding-SAE directory names → model slug inserted into stemqa OOD paths.
_CODING_SAE_MODEL_SLUG: dict[str, str] = {
    "coding_gemma3_12b": "google--gemma-3-12b-it",
    "coding_gemma2_9b":  "google--gemma-2-9b-it",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_perf_data(data: list[tuple]) -> None:
    for entry in data:
        if len(entry) != 7:
            raise ValueError(
                f"Each PERF_DATA entry must have 7 fields "
                f"(scoped, ood, layer, sae_dir, model, gts, q); "
                f"got {len(entry)} in {entry!r}"
            )
        scoped, ood, layer, sae_dir, model, gts, q = entry
        if not isinstance(layer, int):
            raise ValueError(
                f"'layer' must be a non-null int; got {layer!r} "
                f"in entry ({scoped!r}, {ood!r}, ...)"
            )
        if not sae_dir:
            raise ValueError(
                f"'sae_dir' must be a non-empty string; "
                f"got {sae_dir!r} in entry ({scoped!r}, {ood!r}, ...)"
            )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _domain_cache_path(sae_dir: str, domain: str, n: int = _N_SAMPLES) -> Path:
    """Replace the 'stemqa_<X>' or coding-SAE component in sae_dir with 'stemqa_<domain>'.

    Coding-SAE paths (coding_gemma3_12b, coding_gemma2_9b) lack the model-slug directory
    that stemqa paths have, so when mapping to a non-code domain we inject the appropriate
    model slug after 'ignore_padding_True'.
    """
    p = Path(sae_dir)
    dom = domain.lower()

    coding_key = next((k for k in _CODING_SAE_MODEL_SLUG if k in str(p)), None)
    is_coding_sae = coding_key is not None
    inject_model_slug = is_coding_sae and dom != "code"
    model_slug = _CODING_SAE_MODEL_SLUG.get(coding_key, "") if coding_key else ""

    new_parts: list[str] = []
    replaced = False
    for part in p.parts:
        if part.startswith("stemqa_") and not replaced:
            new_parts.append(f"stemqa_{dom}")
            replaced = True
        elif coding_key and part == coding_key:
            new_parts.append(coding_key if dom == "code" else f"stemqa_{dom}")
            replaced = True
        elif inject_model_slug and part == "ignore_padding_True":
            new_parts.append("ignore_padding_True")
            new_parts.append(model_slug)
        else:
            new_parts.append(part)
    if not replaced:
        raise ValueError(
            f"sae_dir {sae_dir!r} has no 'stemqa_*' or coding-SAE component; "
            f"cannot derive domain cache path."
        )
    return Path(*new_parts) / f"n{n}" / "firing_rates.safetensors"


def _load_normalized(path: Path, device: torch.device) -> torch.Tensor | None:
    if not path.exists():
        return None
    with safe_open(str(path), framework="pt") as f:
        dist = f.get_tensor("distribution").float().to(device)
    total = dist.sum().item()
    return (dist / total) if total > 0 else None


# ---------------------------------------------------------------------------
# Overlap metrics  (all tensor ops on device)
# ---------------------------------------------------------------------------

def _cross_coverage_auc(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    ks: torch.Tensor,
) -> float:
    """Fraction of B's mass retained when keeping top-k A neurons. Normalized to [0,1]."""
    n = dist_a.numel()
    sorted_idx = torch.argsort(dist_a, descending=True)
    cumsum_b = torch.cumsum(dist_b[sorted_idx], dim=0)
    curve = cumsum_b[ks.clamp(max=n) - 1].float()
    ks_f = ks.float()
    area = torch.trapz(curve, ks_f)
    max_area = torch.trapz(torch.ones_like(curve), ks_f)
    return (area / max_area).item() if max_area.item() > 0 else 0.0


def _mean_reciprocal_rank_auc(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    ks: torch.Tensor,
) -> float:
    """MRR of A's top-K features in B's ranking, normalized by best-case MRR (H_K / K)."""
    n = dist_a.numel()
    rank_b = (torch.argsort(torch.argsort(dist_b, descending=True)) + 1).float()
    sorted_a = torch.argsort(dist_a, descending=True)

    harmonic = torch.zeros(n + 1, dtype=torch.float32, device=dist_a.device)
    harmonic[1:] = torch.cumsum(
        1.0 / torch.arange(1, n + 1, dtype=torch.float32, device=dist_a.device), dim=0
    )

    mrr_scores = torch.empty(len(ks), dtype=torch.float32, device=dist_a.device)
    best_scores = torch.empty(len(ks), dtype=torch.float32, device=dist_a.device)
    for i, k in enumerate(ks.tolist()):
        mrr_scores[i] = (1.0 / rank_b[sorted_a[:k]]).mean()
        best_scores[i] = harmonic[k] / k

    ks_f = ks.float()
    auc = torch.trapz(mrr_scores, ks_f)
    best_auc = torch.trapz(best_scores, ks_f)
    return (auc / best_auc).item() if best_auc.item() > 0 else 0.0


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

def _ols(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """(slope, intercept, r2, p_value)."""
    if len(xs) < 3:
        return (float("nan"),) * 4
    res = scipy_stats.linregress(xs, ys)
    return float(res.slope), float(res.intercept), float(res.rvalue ** 2), float(res.pvalue)


# ---------------------------------------------------------------------------
# Scatter + regression helper
# ---------------------------------------------------------------------------

def _scatter_regression(
    title_tag: str,
    valid_pairs: list[tuple],
    xs_cov: np.ndarray,
    xs_mrr: np.ndarray,
    output_dir: Path,
    label_fn,
) -> None:
    """Save a 2-panel scatter (coverage | MRR) for each perf metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for perf_metric, y_col, ylabel in [
        ("gts",     5, "Ground-truth similarity delta (%)"),
        ("quality", 6, "Quality delta (%)"),
    ]:
        ys = np.array([row[y_col] for row in valid_pairs])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

        for ax, xs, xlabel, metric_name in [
            (axes[0], xs_cov, "Cross-domain coverage AUC", "cross-coverage"),
            (axes[1], xs_mrr, "Mean reciprocal rank AUC  (norm. by best case)", "MRR"),
        ]:
            slope, intercept, r2, p = _ols(xs, ys)
            x_line = np.linspace(xs.min() * 0.99, xs.max() * 1.01, 100)
            y_line = slope * x_line + intercept

            for row, x, y in zip(valid_pairs, xs, ys):
                s, o = row[0], row[1]
                color, marker = PAIR_STYLES.get((s, o), ("#888888", "o"))
                ax.scatter(x, y, color=color, marker=marker, s=90, zorder=3,
                           label=label_fn(row))
            ax.plot(x_line, y_line, "k--", lw=1.5,
                    label=f"OLS  R²={r2:.3f}  p={p:.3f}")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_title(f"{metric_name}  (R²={r2:.3f})", fontsize=10)
            ax.grid(True, alpha=0.3)
            print(f"  {perf_metric} / {metric_name}: R²={r2:.3f}  p={p:.3f}")

        axes[0].set_ylabel(ylabel)
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels, fontsize=7.5, ncol=2,
                       loc="upper left", framealpha=0.9)
        fig.suptitle(
            f"SAE feature overlap vs OOD degradation — {title_tag} ({perf_metric})",
            fontsize=11,
        )
        plt.tight_layout()
        out = output_dir / f"scatter_{perf_metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  → {out}")


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def _run_model(
    model: str,
    data: list[tuple],
    output_dir: Path,
    device: torch.device,
) -> tuple[list[tuple], np.ndarray, np.ndarray] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n── {model}  ({len(data)} pairs) ──")

    # Load distributions (deduplicated by sae_dir × domain)
    cache: dict[tuple[str, str], torch.Tensor | None] = {}
    for scoped, ood, layer, sae_dir, _model, gts, q in data:
        for dom in (scoped, ood):
            key = (sae_dir, dom)
            if key not in cache:
                p = _domain_cache_path(sae_dir, dom)
                d = _load_normalized(p, device)
                if d is None:
                    print(f"  [warn] missing: {p}")
                cache[key] = d

    # Compute both metrics per pair
    cross_covs: list[float | None] = []
    mrr_scores: list[float | None] = []
    for scoped, ood, layer, sae_dir, _model, gts, q in data:
        da = cache[(sae_dir, scoped)]
        db = cache[(sae_dir, ood)]
        if da is None or db is None:
            print(f"  [skip] {scoped}→{ood} layer={layer}")
            cross_covs.append(None)
            mrr_scores.append(None)
            continue
        ks = default_ks(da.numel()).to(device)
        cross_covs.append(_cross_coverage_auc(da, db, ks))
        mrr_scores.append(_mean_reciprocal_rank_auc(da, db, ks))

    valid_idx = [i for i, v in enumerate(cross_covs) if v is not None]
    if len(valid_idx) < 3:
        print(f"  [skip model] only {len(valid_idx)} valid pairs — need ≥ 3 for regression.")
        return None

    valid_pairs = [data[i] for i in valid_idx]
    xs_cov = np.array([cross_covs[i] for i in valid_idx])
    xs_mrr = np.array([mrr_scores[i]  for i in valid_idx])
    print(f"  {len(valid_pairs)} / {len(data)} pairs computed")

    _scatter_regression(
        title_tag=model,
        valid_pairs=valid_pairs,
        xs_cov=xs_cov,
        xs_mrr=xs_mrr,
        output_dir=output_dir,
        label_fn=lambda row: f"{row[0][:3]}→{row[1][:3]}  (L{row[2]})",
    )

    # Summary table
    print(f"\n  {'Pair':<26} | layer | coverage |   MRR   |  gts  | quality")
    print("  " + "─" * 68)
    for row, cov, mrr in zip(valid_pairs, xs_cov, xs_mrr):
        s, o, layer, _, _m, gts, q = row
        print(f"  {s:<10} → {o:<10} |  {layer:2d}   |  {cov:.4f}  | {mrr:.4f} | {gts:5.1f} | {q:5.1f}")

    return valid_pairs, xs_cov, xs_mrr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_dir: Path, device: torch.device) -> None:
    _validate_perf_data(PERF_DATA)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(set(row[4] for row in PERF_DATA))
    print(f"Device: {device}  |  total pairs: {len(PERF_DATA)}  |  models: {models}")

    all_pairs: list[tuple] = []
    all_cov:   list[float] = []
    all_mrr:   list[float] = []

    for model in models:
        model_data = [row for row in PERF_DATA if row[4] == model]
        result = _run_model(model, model_data, output_dir / model, device)
        if result is not None:
            vp, xc, xm = result
            all_pairs.extend(vp)
            all_cov.extend(xc.tolist())
            all_mrr.extend(xm.tolist())

    # Combined regression across all models
    if len(all_pairs) >= 3:
        print(f"\n── combined  ({len(all_pairs)} pairs) ──")
        _scatter_regression(
            title_tag="all models (combined)",
            valid_pairs=all_pairs,
            xs_cov=np.array(all_cov),
            xs_mrr=np.array(all_mrr),
            output_dir=output_dir / "combined",
            label_fn=lambda row: f"{row[0][:3]}→{row[1][:3]}  ({row[4]})",
        )

    print(f"\nDone. Outputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "experiments" / "sae_scoping" / "overlap_regression",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    run(args.output_dir, torch.device(args.device))


if __name__ == "__main__":
    main()

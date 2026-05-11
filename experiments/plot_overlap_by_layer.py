"""
Plot cross-domain coverage AUC and MRR AUC vs layer for gemma-3-12b-it (width_262k).

Sweeps all 12 directed domain pairs (bio, chem, math, physics) across layers 0-47.
Produces a figure with 8 subplots: 4 rows (source domain) × 2 cols (coverage AUC | MRR AUC).
Each subplot shows 3 lines, one per OOD target domain.

Usage:
    python experiments/plot_overlap_by_layer.py
    python experiments/plot_overlap_by_layer.py --device cuda
    python experiments/plot_overlap_by_layer.py --output results/overlap_by_layer.png
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sae_scoping.data_science import default_ks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CACHE_ROOT = (
    Path("/data/aruna_sankaranarayanan/SAEScoping/experiments/.cache")
)
_MODEL_SLUG = "google--gemma-3-12b-it--width_16k"
_LAYER_TAG  = "layer_{i}--width_16k--l0_small"
_N_SAMPLES  = 10_000

DOMAINS = ["biology", "chemistry", "math", "physics"]

PAIR_STYLES: dict[tuple[str, str], tuple[str, str]] = {
    ("biology",   "chemistry"): ("#1f77b4", "-"),
    ("biology",   "math"):      ("#aec7e8", "--"),
    ("biology",   "physics"):   ("#6baed6", ":"),
    ("chemistry", "biology"):   ("#74c476", "-"),
    ("chemistry", "math"):      ("#98df8a", "--"),
    ("chemistry", "physics"):   ("#2ca02c", ":"),
    ("math",      "biology"):   ("#e6550d", "-"),
    ("math",      "chemistry"): ("#ff9896", "--"),
    ("math",      "physics"):   ("#d62728", ":"),
    ("physics",   "biology"):   ("#7b4173", "-"),
    ("physics",   "chemistry"): ("#9467bd", "--"),
    ("physics",   "math"):      ("#c5b0d5", ":"),
}

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _firing_rate_path(domain: str, layer: int) -> Path:
    layer_tag = _LAYER_TAG.format(i=layer)
    return (
        _CACHE_ROOT
        / f"stemqa_{domain}"
        / "ignore_padding_True"
        / _MODEL_SLUG
        / layer_tag
        / f"n{_N_SAMPLES}"
        / "firing_rates.safetensors"
    )


def _load_normalized(path: Path, device: torch.device) -> torch.Tensor | None:
    if not path.exists():
        return None
    with safe_open(str(path), framework="pt") as f:
        dist = f.get_tensor("distribution").float().to(device)
    total = dist.sum().item()
    return (dist / total) if total > 0 else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cross_coverage_auc(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    ks: torch.Tensor,
) -> float:
    n = dist_a.numel()
    sorted_idx = torch.argsort(dist_a, descending=True)
    cumsum_b = torch.cumsum(dist_b[sorted_idx], dim=0)
    curve = cumsum_b[ks.clamp(max=n) - 1].float()
    ks_f = ks.float()
    area = torch.trapz(curve, ks_f)
    max_area = torch.trapz(torch.ones_like(curve), ks_f)
    return (area / max_area).item() if max_area.item() > 0 else float("nan")


def _mrr_auc(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    ks: torch.Tensor,
) -> float:
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
    return (auc / best_auc).item() if best_auc.item() > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_path: Path, device: torch.device) -> None:
    # Discover available layers from biology cache (all domains share the same set)
    layer_dir = _CACHE_ROOT / "stemqa_biology" / "ignore_padding_True" / _MODEL_SLUG
    layers = sorted(
        int(d.name.split("--")[0].replace("layer_", ""))
        for d in layer_dir.iterdir()
        if d.is_dir() and d.name.startswith("layer_")
    )
    print(f"Device: {device}  |  layers: {len(layers)} ({layers[0]}–{layers[-1]})")

    directed_pairs = [
        (a, b) for a in DOMAINS for b in DOMAINS if a != b
    ]

    # Results: pair → list of (layer, cov, mrr)
    results: dict[tuple[str, str], list[tuple[int, float, float]]] = {
        pair: [] for pair in directed_pairs
    }

    for layer in layers:
        # Load all domain distributions for this layer
        dists: dict[str, torch.Tensor | None] = {}
        for dom in DOMAINS:
            p = _firing_rate_path(dom, layer)
            dists[dom] = _load_normalized(p, device)

        missing = [d for d, v in dists.items() if v is None]
        if missing:
            print(f"  [layer {layer:2d}] missing: {missing} — skipping")
            continue

        # Compute both metrics for all directed pairs
        ks = default_ks(next(v for v in dists.values() if v is not None).numel()).to(device)
        for a, b in directed_pairs:
            da, db = dists[a], dists[b]
            cov = _cross_coverage_auc(da, db, ks)
            mrr = _mrr_auc(da, db, ks)
            results[(a, b)].append((layer, cov, mrr))

        print(f"  layer {layer:2d} done")

    # Plot: 4 rows (source domain) × 2 cols (coverage | MRR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(DOMAINS), 2,
        figsize=(12, 4 * len(DOMAINS)),
        sharey=False,
        sharex=True,
    )

    # OOD-target colors (consistent across rows)
    target_colors = {
        "biology":   "#1f77b4",
        "chemistry": "#2ca02c",
        "math":      "#d62728",
        "physics":   "#9467bd",
    }

    for row, src in enumerate(DOMAINS):
        ax_cov = axes[row, 0]
        ax_mrr = axes[row, 1]

        for tgt in DOMAINS:
            if tgt == src:
                continue
            pts = results.get((src, tgt), [])
            if not pts:
                continue
            ls_arr  = np.array([p[0] for p in pts])
            cov_arr = np.array([p[1] for p in pts])
            mrr_arr = np.array([p[2] for p in pts])
            color = target_colors[tgt]
            label = f"→{tgt[:4]}"
            ax_cov.plot(ls_arr, cov_arr, color=color, lw=1.5, label=label)
            ax_mrr.plot(ls_arr, mrr_arr, color=color, lw=1.5, label=label)

        for ax, metric_title in [
            (ax_cov, "Coverage AUC"),
            (ax_mrr, "MRR AUC"),
        ]:
            ax.set_ylabel("AUC", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
            ax.set_title(f"{src.capitalize()} → * | {metric_title}", fontsize=10)

        ax_cov.set_xlabel("Layer", fontsize=9)
        ax_mrr.set_xlabel("Layer", fontsize=9)
        for ax in (ax_cov, ax_mrr):
            ax.tick_params(labelbottom=True)
            ax.set_xticks(range(layers[0], layers[-1] + 1, 2))

    fig.suptitle(
        f"SAE feature overlap by layer — gemma-3-12b-it ({_MODEL_SLUG.split('--', 1)[1]})",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "experiments" / "sae_scoping" / "overlap_by_layer" / "overlap_by_layer_gemma3_12b.png",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    run(args.output, torch.device(args.device))


if __name__ == "__main__":
    main()

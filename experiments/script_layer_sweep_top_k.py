"""Layer sweep of top-k intersection AUC across all SAE layers.

Phase 1 – Compute (--compute, needs GPU):
    For every uncached (model, layer, domain) tuple, run forward passes to
    collect SAE feature firing rates and write firing_rates.safetensors.

Phase 2 – Analyze (--analyze, CPU-OK):
    Load all cached distributions, normalize, compute pairwise top-k AUC at
    each layer, and write a CSV plus two matplotlib figures (line plot + heatmap
    grid).

Cache layout (matches existing convention):
    experiments/.cache/stemqa_<domain>/ignore_padding_True/<model_slug>/
        <layer_tag>/n<n>/firing_rates.safetensors
    key "distribution": float32 tensor of shape [d_sae]; raw or normalized —
        always normalized on load so both old (sum≈L0) and new (sum=1) files work.

Available SAE layers:
    gemma-2-9b-it   : 9, 20, 31  (gemma-scope-9b-it-res-canonical, width_16k)
    gemma-3-12b-it  : 0–47       (gemma-scope-2-12b-it-res-all, width_16k l0_small)

Usage:
    # compute all missing then analyze:
    python experiments/script_layer_sweep_top_k.py --gemma3 --compute --analyze

    # analyze only (reads existing cache, no GPU needed):
    python experiments/script_layer_sweep_top_k.py --gemma2 --gemma3 --analyze

    # compute gemma-3 missing chemistry/math/physics at biology-cached layers:
    python experiments/script_layer_sweep_top_k.py --gemma3 --compute --domains chemistry,math,physics
"""

from __future__ import annotations

import csv
import gc
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sae_scoping.data_science import (
    default_ks,
    overlap_curve_auc,
    pairwise_overlap_auc_matrix,
    top_k_overlap_curve,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAINS = ["biology", "chemistry", "math", "physics"]
N_SAMPLES = 10_000
CACHE_DIR = REPO_ROOT / "experiments" / ".cache"
OUTPUT_DIR = REPO_ROOT / "experiments" / "sae_scoping" / "top_k_layer_sweep_2026_04_27"

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

GEMMA2_CFG: dict[str, Any] = dict(
    model_name="google/gemma-2-9b-it",
    sae_release="gemma-scope-9b-it-res-canonical",  # only layers 9, 20, 31 available
    model_slug="google--gemma-2-9b-it",
    layers=[9, 20, 31],
    sae_id=lambda L: f"layer_{L}/width_16k/canonical",
    hookpoint=lambda L: f"model.layers.{L}",
    layer_tag=lambda L: f"layer_{L}--width_16k--canonical",
)

def _gemma3_cfg(width: str = "262k") -> dict[str, Any]:
    return dict(
        model_name="google/gemma-3-12b-it",
        sae_release="gemma-scope-2-12b-it-res-all",
        model_slug=f"google--gemma-3-12b-it--width_{width}",
        layers=list(range(48)),
        sae_id=lambda L: f"layer_{L}_width_{width}_l0_small",
        hookpoint=lambda L: f"model.language_model.layers.{L}",
        layer_tag=lambda L: f"layer_{L}--width_{width}--l0_small",
    )

GEMMA3_CFG = _gemma3_cfg("262k")

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def cache_path(domain: str, model_slug: str, layer_tag: str, n: int = N_SAMPLES) -> Path:
    return (
        CACHE_DIR
        / f"stemqa_{domain}"
        / "ignore_padding_True"
        / model_slug
        / layer_tag
        / f"n{n}"
        / "firing_rates.safetensors"
    )


def load_normalized(path: Path) -> torch.Tensor:
    """Load the 'distribution' tensor and normalize to sum=1 (handles both
    raw-rate files (sum≈L0) and already-normalized files (sum=1))."""
    with safe_open(str(path), framework="pt") as f:
        dist = f.get_tensor("distribution").float()
    total = dist.sum().item()
    if total <= 0:
        raise ValueError(f"All-zero or negative distribution in {path}")
    return dist / total


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_domain_dataset(domain: str, tokenizer: Any, n_samples: int):
    from datasets import load_dataset

    stream = load_dataset("4gate/StemQAMixture", domain, split="train", streaming=False)
    stream = stream.shuffle(seed=42)
    rows = []
    for example in stream:
        if len(rows) >= n_samples:
            break
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(example["question"])},
                {"role": "assistant", "content": str(example["answer"])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text})
    from datasets import Dataset
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Phase 1: compute
# ---------------------------------------------------------------------------


def run_compute(
    cfg: dict,
    domains: list[str],
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> None:
    from sae_lens import SAE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sae_scoping.trainers.sae_enhanced.rank import rank_neurons

    model_name = cfg["model_name"]
    model_slug = cfg["model_slug"]
    print(f"\n=== Compute: {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets ...")
    datasets = {d: load_domain_dataset(d, tokenizer, n_samples) for d in domains}

    print(f"Loading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()

    for layer in cfg["layers"]:
        sae_id = cfg["sae_id"](layer)
        hookpoint = cfg["hookpoint"](layer)
        layer_tag = cfg["layer_tag"](layer)

        missing = [
            d
            for d in domains
            if not cache_path(d, model_slug, layer_tag, n_samples).exists()
        ]
        if not missing:
            print(f"  Layer {layer:3d}: all cached")
            continue

        print(f"  Layer {layer:3d}: computing {missing} ...")
        try:
            sae = SAE.from_pretrained(
                release=cfg["sae_release"], sae_id=sae_id, device=str(device)
            )
            sae = sae.to(device)
        except Exception as e:
            print(f"    SAE load failed for layer {layer}: {e} — skipping")
            continue

        for domain in missing:
            out_path = cache_path(domain, model_slug, layer_tag, n_samples)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with torch.no_grad():
                    ranking, distribution = rank_neurons(
                        dataset=datasets[domain],
                        sae=sae,
                        model=model,
                        tokenizer=tokenizer,
                        T=0.0,
                        hookpoint=hookpoint,
                        batch_size=batch_size,
                        token_selection="attention_mask",
                        return_distribution=True,
                    )
                save_file(
                    {
                        "distribution": distribution.cpu().float(),
                        "ranking": ranking.cpu(),
                    },
                    str(out_path),
                )
                print(f"    Saved: {out_path}")
            except Exception as e:
                print(f"    Failed ({domain}, layer {layer}): {e}")

        sae = sae.to("cpu")
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Cross-domain coverage helpers
# ---------------------------------------------------------------------------


def cross_domain_coverage_curve(
    dist_a: torch.Tensor, dist_b: torch.Tensor, ks: torch.Tensor
) -> torch.Tensor:
    """For each k, select the top-k neurons by dist_a and return the fraction
    of dist_b's total mass that falls on those neurons.

    Concretely: if dist_a = bio and dist_b = math, the curve answers
    "if I keep the top-k bio neurons, what fraction of math activity do I retain?"
    Compare this to the self-coverage curve (dist_a=dist_b=bio) to see whether
    the top bio neurons are domain-general (high cross-coverage) or bio-specific
    (low cross-coverage).
    """
    sorted_idx = torch.argsort(dist_a, descending=True)
    dist_b_sorted = dist_b[sorted_idx]
    cumsum_b = torch.cumsum(dist_b_sorted, dim=0)
    ks_clamped = ks.clamp(max=dist_a.numel())
    return cumsum_b[ks_clamped - 1]


def cross_coverage_auc(curve: torch.Tensor, ks: torch.Tensor, n: int) -> float:
    """Area under the cross-coverage curve, normalized to [0, 1]."""
    ks_np = ks.numpy().astype(float)
    curve_np = curve.numpy().astype(float)
    area = float(np.trapz(curve_np, ks_np))
    max_area = float(np.trapz(np.ones(len(ks_np)), ks_np))
    return area / max_area if max_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 2: analyze
# ---------------------------------------------------------------------------


def run_analyze(
    cfg: dict,
    domains: list[str],
    n_samples: int,
    output_dir: Path,
) -> None:
    model_slug = cfg["model_slug"]
    model_name = cfg["model_name"]
    print(f"\n=== Analyze: {model_name} ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather distributions per layer
    layer_dists: dict[int, dict[str, torch.Tensor]] = {}
    for layer in cfg["layers"]:
        layer_tag = cfg["layer_tag"](layer)
        dists: dict[str, torch.Tensor] = {}
        for domain in domains:
            p = cache_path(domain, model_slug, layer_tag, n_samples)
            if p.exists():
                try:
                    dists[domain] = load_normalized(p)
                except Exception as e:
                    print(f"  Warning: could not load {p}: {e}")
        if len(dists) >= 2:
            layer_dists[layer] = dists

    if not layer_dists:
        print(f"  No layers with ≥2 domains cached for {model_name}; nothing to analyze.")
        return

    available_layers = sorted(layer_dists)
    print(f"  Layers with ≥2 domains: {available_layers}")

    # Compute pairwise AUC at each layer
    rows: list[dict] = []
    for layer in available_layers:
        dists = layer_dists[layer]
        n = next(iter(dists.values())).numel()
        ks = default_ks(n)
        for a, b in combinations(sorted(dists), 2):
            curve = top_k_overlap_curve(dists[a], dists[b], ks=ks)
            auc = overlap_curve_auc(curve, ks, n)
            rows.append(
                dict(
                    model=model_slug,
                    layer=layer,
                    domain_a=a,
                    domain_b=b,
                    auc=round(auc, 6),
                )
            )

    # CSV
    csv_path = output_dir / f"top_k_auc_{model_slug}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "layer", "domain_a", "domain_b", "auc"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  → {csv_path}")

    # Compute cross-domain coverage at each layer (ordered pairs, including self)
    # cross_coverage(A→B, k): fraction of B's activity retained when keeping top-k A neurons.
    # Self-coverage (A→A) is the upper bound; compare cross (A→B) against it to see
    # whether top-A neurons are domain-general or domain-specific.
    coverage_rows: list[dict] = []
    for layer in available_layers:
        dists = layer_dists[layer]
        n = next(iter(dists.values())).numel()
        ks = default_ks(n)
        ks_np = ks.numpy()
        for a in sorted(dists):
            for b in sorted(dists):
                curve = cross_domain_coverage_curve(dists[a], dists[b], ks)
                auc = cross_coverage_auc(curve, ks, n)
                # Also record coverage at k=15% of SAE width as a point metric
                k_15pct = max(1, int(0.15 * n))
                k_15pct_idx = int(np.searchsorted(ks_np, k_15pct))
                k_15pct_idx = min(k_15pct_idx, len(curve) - 1)
                coverage_rows.append(
                    dict(
                        model=model_slug,
                        layer=layer,
                        selector=a,
                        target=b,
                        auc=round(auc, 6),
                        coverage_at_15pct=round(float(curve[k_15pct_idx].item()), 6),
                    )
                )

    coverage_csv = output_dir / f"cross_domain_coverage_{model_slug}.csv"
    with open(coverage_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "layer", "selector", "target", "auc", "coverage_at_15pct"]
        )
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"  CSV  → {coverage_csv}")

    # Line plot: AUC vs layer
    _plot_auc_by_layer(rows, model_name, model_slug, output_dir)

    # Cross-coverage curves per layer (self vs cross, for each selector domain)
    _plot_cross_coverage_by_layer(available_layers, layer_dists, model_name, model_slug, output_dir)

    # Per-layer overlap curves
    _plot_overlap_curves_by_layer(available_layers, layer_dists, model_name, model_slug, output_dir)

    # Heatmaps for layers where all requested domains are present
    full_layers = [L for L in available_layers if set(layer_dists[L]) >= set(domains)]
    if full_layers:
        _plot_heatmaps(full_layers, layer_dists, domains, model_name, model_slug, output_dir)
    else:
        print(f"  No layer has all {domains}; skipping heatmap grid.")


def _plot_cross_coverage_by_layer(
    available_layers: list[int],
    layer_dists: dict[int, dict[str, torch.Tensor]],
    model_name: str,
    model_slug: str,
    output_dir: Path,
) -> None:
    """For each selector domain, plot self-coverage and cross-coverage curves
    at every layer. Reveals whether top-k selector neurons are domain-specific
    (self >> cross) or domain-general (self ≈ cross).
    """
    selector_domains = sorted({d for dists in layer_dists.values() for d in dists})

    for selector in selector_domains:
        layers_with_selector = [L for L in available_layers if selector in layer_dists[L]]
        if not layers_with_selector:
            continue

        n_layers = len(layers_with_selector)
        n_cols = min(6, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False
        )
        axes_flat = axes.reshape(-1)

        legend_handles: list = []
        legend_labels: list[str] = []

        for ax, layer in zip(axes_flat, layers_with_selector):
            dists = layer_dists[layer]
            n = dists[selector].numel()
            ks = default_ks(n)
            ks_np = ks.numpy()
            k_15 = int(0.15 * n)

            for target in sorted(dists):
                curve = cross_domain_coverage_curve(dists[selector], dists[target], ks)
                linestyle = "-" if target == selector else "--"
                label = f"→ {target}" + (" (self)" if target == selector else "")
                (line,) = ax.plot(
                    ks_np, curve.numpy(), linestyle=linestyle, linewidth=1.2, label=label
                )
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.axvline(k_15, color="gray", linewidth=0.8, linestyle=":")
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_title(f"Layer {layer}", fontsize=9)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("k (neurons kept)", fontsize=7)
            ax.set_ylabel("Fraction of target activity", fontsize=7)

        for ax in axes_flat[n_layers:]:
            ax.set_visible(False)

        bottom_pad = 0.0
        if legend_handles:
            ncol = min(len(legend_labels), 4)
            fig.legend(
                legend_handles, legend_labels,
                loc="lower center", ncol=ncol, fontsize=8,
                bbox_to_anchor=(0.5, 0),
            )
            bottom_pad = 0.06

        fig.suptitle(
            f"Cross-domain coverage: top-k {selector} neurons — {model_name}\n"
            "(dashed = cross-domain; solid = self; dotted line = k=15%)",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, bottom_pad, 1, 1])
        plot_path = output_dir / f"cross_coverage_{selector}_{model_slug}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Plot → {plot_path}")


def _plot_auc_by_layer(
    rows: list[dict],
    model_name: str,
    model_slug: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    pairs = sorted({(r["domain_a"], r["domain_b"]) for r in rows})
    for a, b in pairs:
        pts = sorted(
            (r["layer"], r["auc"])
            for r in rows
            if r["domain_a"] == a and r["domain_b"] == b
        )
        if pts:
            ls, aucs = zip(*pts)
            ax.plot(ls, aucs, marker="o", markersize=4, label=f"{a} vs {b}")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-k overlap AUC")
    ax.set_ylim(0, 1)
    ax.set_title(f"Top-k overlap AUC by layer — {model_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / f"top_k_auc_by_layer_{model_slug}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {plot_path}")


def _plot_overlap_curves_by_layer(
    available_layers: list[int],
    layer_dists: dict[int, dict[str, torch.Tensor]],
    model_name: str,
    model_slug: str,
    output_dir: Path,
) -> None:
    n_layers = len(available_layers)
    n_cols = min(6, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    legend_handles: list = []
    legend_labels: list[str] = []

    for ax, layer in zip(axes_flat, available_layers):
        dists = layer_dists[layer]
        n = next(iter(dists.values())).numel()
        ks = default_ks(n)
        ks_np = ks.numpy()

        for a, b in combinations(sorted(dists), 2):
            curve = top_k_overlap_curve(dists[a], dists[b], ks=ks)
            (line,) = ax.plot(ks_np, curve.numpy(), linewidth=1, label=f"{a} vs {b}")
            label = f"{a} vs {b}"
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        # ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("k (neurons kept)", fontsize=7)
        ax.set_ylabel("Overlap@k", fontsize=7)

    for ax in axes_flat[n_layers:]:
        ax.set_visible(False)

    bottom_pad = 0.0
    if legend_handles:
        ncol = min(len(legend_labels), 3)
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center", ncol=ncol, fontsize=8,
            bbox_to_anchor=(0.5, 0),
        )
        bottom_pad = 0.06

    fig.suptitle(f"Top-k overlap curves by layer — {model_name}", fontsize=11)
    fig.tight_layout(rect=[0, bottom_pad, 1, 1])
    plot_path = output_dir / f"overlap_curves_by_layer_{model_slug}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Plot → {plot_path}")


def _plot_heatmaps(
    layers: list[int],
    layer_dists: dict[int, dict[str, torch.Tensor]],
    domains: list[str],
    model_name: str,
    model_slug: str,
    output_dir: Path,
) -> None:
    sorted_domains = sorted(domains)
    n_cols = min(4, len(layers))
    n_rows = (len(layers) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax, layer in zip(axes_flat, layers):
        dists_sorted = {
            d: layer_dists[layer][d]
            for d in sorted_domains
            if d in layer_dists[layer]
        }
        names, matrix = pairwise_overlap_auc_matrix(dists_sorted)
        mat = matrix.numpy()
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"Layer {layer}", fontsize=9)
        for i in range(len(names)):
            for j in range(len(names)):
                val = mat[i, j]
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val < 0.5 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_flat[len(layers):]:
        ax.set_visible(False)

    fig.suptitle(f"AUC heatmaps — {model_name}", fontsize=11)
    fig.tight_layout()
    heatmap_path = output_dir / f"top_k_heatmaps_{model_slug}.png"
    fig.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    print(f"  Heatmaps → {heatmap_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--gemma2", is_flag=True, default=False, help="Include gemma-2-9b-it")
@click.option("--gemma3", is_flag=True, default=False, help="Include gemma-3-12b-it")
@click.option("--16k", "width_16k", is_flag=True, default=False, help="Use width_16k SAEs (default: width_262k)")
@click.option("--compute", is_flag=True, default=False, help="Compute missing firing rates (GPU)")
@click.option("--analyze", is_flag=True, default=False, help="Analyze cache, write CSV + plots")
@click.option("--n-samples", default=N_SAMPLES, show_default=True, help="Ranking samples per domain")
@click.option("--batch-size", default=4, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option(
    "--output-dir",
    default=str(OUTPUT_DIR),
    show_default=True,
    help="Where to write CSV and plots",
)
@click.option(
    "--domains",
    default=",".join(DOMAINS),
    show_default=True,
    help="Comma-separated list of domains to process",
)
@click.option(
    "--layers",
    default=None,
    help="Layer range to compute, e.g. '0-23' or '24-47'. "
         "Omit to process all layers for the chosen model.",
)
def main(
    gemma2: bool,
    gemma3: bool,
    width_16k: bool,
    compute: bool,
    analyze: bool,
    n_samples: int,
    batch_size: int,
    device: str,
    output_dir: str,
    domains: str,
    layers: str | None,
) -> None:
    if not compute and not analyze:
        raise click.UsageError("Pass at least one of --compute or --analyze.")

    domain_list = [d.strip() for d in domains.split(",") if d.strip()]

    # Parse optional layer range
    layer_subset: set[int] | None = None
    if layers is not None:
        if "-" in layers:
            lo, hi = layers.split("-", 1)
            layer_subset = set(range(int(lo), int(hi) + 1))
        else:
            layer_subset = {int(x) for x in layers.split(",")}

    # Default to gemma-3 only when no model flag given
    if not gemma2 and not gemma3:
        gemma3 = True
    cfgs = []
    if gemma2:
        cfg = dict(GEMMA2_CFG)
        if layer_subset is not None:
            cfg["layers"] = [L for L in cfg["layers"] if L in layer_subset]
        cfgs.append(cfg)
    if gemma3:
        cfg = _gemma3_cfg("16k" if width_16k else "262k")
        if layer_subset is not None:
            cfg["layers"] = [L for L in cfg["layers"] if L in layer_subset]
        cfgs.append(cfg)

    dev = torch.device(device)
    out = Path(output_dir)

    if compute:
        for cfg in cfgs:
            run_compute(cfg, domain_list, n_samples, batch_size, dev)

    if analyze:
        for cfg in cfgs:
            run_analyze(cfg, domain_list, n_samples, out)


if __name__ == "__main__":
    main()

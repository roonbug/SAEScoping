"""
Analyze firing rate magnitudes of overlapping neurons between two STEM domains
across all Gemma3 layers/SAE configs.

Three analyses:

1. THRESHOLD ANALYSIS: for neurons active above a fixed firing rate threshold in
   both domains, compare domain-A vs domain-B firing rate magnitudes.

2. K-SWEEP ANALYSIS: for each K in 1–30%, select the top-K% neurons by domain-A
   firing rate. Evaluate domain-A-specificity by comparing their A vs B rates.

   Key metrics per (layer, K):
   - pct_a_dominant: fraction of top-K A neurons where A_rate > B_rate
   - ol_pct: fraction of top-K A neurons that are also in the top-K B neurons
   - mean_excess: mean(A_rate − B_rate) for top-K A neurons
   - marginal_excess: (A_rate − B_rate) of the K-th neuron added

   Visualisations: per-layer line plots, heatmaps, crossover chart.

3. DECODER SUBSPACE ANALYSIS (requires --sae-release): tests whether the
   firing-rate explanation is sufficient by checking if top-K domain-A neurons
   write to the same subspace of the residual stream as top-K domain-B neurons.

   If domain-A scoping improves domain-B performance despite A-selective firing
   rates, the hypothesis is that A-selected decoder columns span a subspace that
   is causally important for B. This analysis measures that overlap directly.

   Key metric: subspace_overlap — mean squared cosine of principal angles between
   the subspace spanned by top-K A decoder columns and top-K B decoder columns.
   0 = orthogonal (A and B write to completely different directions), 1 = identical.

   The SAE ID is derived from the layer tag by replacing '--' with '_', e.g.
   layer_20--width_262k--l0_small → layer_20_width_262k_l0_small. Pass
   --sae-release to enable (e.g. "gemma-scope-2-12b-it-res-all").

Usage:
  python analyze_overlap_firing_rates.py
  python analyze_overlap_firing_rates.py --domain-a chemistry --domain-b math
  python analyze_overlap_firing_rates.py --domain-a biology --domain-b math --threshold 1e-4
  python analyze_overlap_firing_rates.py --domain-a biology --domain-b math --output-dir results/bio_math
  python analyze_overlap_firing_rates.py --domain-a biology --domain-b math \\
      --sae-release gemma-scope-2-12b-it-res-all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file


CACHE_ROOT = Path(__file__).parent / ".cache"
DEFAULT_MODEL_SLUG = "google--gemma-3-12b-it--width_16k"
K_PCTS = list(range(1, 31))  # 1 % … 30 %


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_pairs(domain_a: str, domain_b: str, model_slug: str) -> list[dict]:
    """Return all (layer, SAE config) pairs that have both domain_a and domain_b caches."""
    root_a = CACHE_ROOT / f"stemqa_{domain_a}" / "ignore_padding_True" / model_slug
    root_b = CACHE_ROOT / f"stemqa_{domain_b}" / "ignore_padding_True" / model_slug
    for root, name in [(root_a, domain_a), (root_b, domain_b)]:
        if not root.exists():
            raise FileNotFoundError(f"{name} cache not found: {root}")

    pairs = []
    for path_a in sorted(root_a.rglob("firing_rates.safetensors")):
        rel = path_a.parent.relative_to(root_a)
        path_b = root_b / rel / "firing_rates.safetensors"
        if not path_b.exists():
            continue
        sae_tag = rel.parts[0]
        m = re.match(r"layer_(\d+)", sae_tag)
        layer = int(m.group(1)) if m else -1
        pairs.append({
            "layer": layer,
            "sae_tag": sae_tag,
            "rel": str(rel),
            "path_a": path_a,
            "path_b": path_b,
            "label": f"L{layer} {sae_tag.split('_')[1]}",
        })
    pairs.sort(key=lambda x: (x["layer"], x["sae_tag"]))
    return pairs


def load_dists(path_a: Path, path_b: Path) -> tuple[np.ndarray, np.ndarray] | None:
    dist_a = load_file(str(path_a))["distribution"].float().numpy()
    dist_b = load_file(str(path_b))["distribution"].float().numpy()
    if len(dist_a) != len(dist_b):
        return None
    return dist_a, dist_b


# ── Analysis 1: threshold-based ───────────────────────────────────────────────

def analyze_threshold(
    label: str,
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    domain_a: str,
    domain_b: str,
    threshold: float,
) -> dict | None:
    active_a = dist_a >= threshold
    active_b = dist_b >= threshold
    overlap = active_a & active_b

    n_a = int(active_a.sum())
    n_b = int(active_b.sum())
    n_overlap = int(overlap.sum())
    if n_overlap == 0:
        print(f"  [SKIP] {label}: no overlap at threshold {threshold}")
        return None

    rates_a = dist_a[overlap]
    rates_b = dist_b[overlap]
    ratios = rates_b / (rates_a + 1e-12)

    return {
        "label": label,
        "domain_a": domain_a,
        "domain_b": domain_b,
        "sae_width": int(len(dist_a)),
        f"n_{domain_a}_active": n_a,
        f"n_{domain_b}_active": n_b,
        "n_overlap": n_overlap,
        f"overlap_pct_of_{domain_a}": float(n_overlap / n_a * 100),
        f"overlap_pct_of_{domain_b}": float(n_overlap / n_b * 100),
        f"mean_{domain_b}_{domain_a}_ratio": float(ratios.mean()),
        f"median_{domain_b}_{domain_a}_ratio": float(np.median(ratios)),
        f"pct_{domain_b}_dominant": float((ratios > 1).mean() * 100),
        f"mean_{domain_a}_rate": float(rates_a.mean()),
        f"mean_{domain_b}_rate": float(rates_b.mean()),
        "_ratios": ratios,
        "_rates_a": rates_a,
        "_rates_b": rates_b,
    }


# ── Analysis 2: K-sweep ───────────────────────────────────────────────────────

def compute_k_sweep(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    k_pcts: list[float],
) -> list[dict]:
    """
    For each K%, select the top-K% neurons by domain-A firing rate.
    Evaluate domain-A-specificity by comparing their A vs B rates.
    B is not independently thresholded — it is compared on the same kept set.
    """
    n = len(dist_a)
    order_a = np.argsort(dist_a)[::-1]   # descending by A rate
    order_b = np.argsort(dist_b)[::-1]

    # rank_b[i] = rank of neuron i by B rate (0 = highest B neuron)
    rank_b = np.empty(n, dtype=np.int32)
    rank_b[order_b] = np.arange(n, dtype=np.int32)

    rows = []
    for k_pct in k_pcts:
        k = max(1, int(k_pct / 100 * n))
        top_k_idx = order_a[:k]

        topk_a = dist_a[top_k_idx]
        topk_b = dist_b[top_k_idx]

        # Fraction of top-K A neurons where A > B individually
        pct_a_dom = float((topk_a > topk_b).mean())

        # Fraction of top-K A neurons that are also in the top-K B neurons
        ol_pct = float((rank_b[top_k_idx] < k).mean())

        # Mean A − B for selected neurons (negative = B-biased)
        mean_excess = float((topk_a - topk_b).mean())

        # Marginal neuron: the K-th neuron in the A ranking
        kth_idx = order_a[k - 1]
        marginal_excess = float(dist_a[kth_idx] - dist_b[kth_idx])

        rows.append({
            "k_pct": k_pct,
            "k": k,
            "pct_a_dominant": pct_a_dom,
            "ol_pct": ol_pct,
            "mean_excess": mean_excess,
            "marginal_excess": marginal_excess,
            "mean_a": float(topk_a.mean()),
            "mean_b": float(topk_b.mean()),
        })
    return rows


# ── Tables ────────────────────────────────────────────────────────────────────

def print_table(results: list[dict], domain_a: str, domain_b: str) -> None:
    hdr = (
        f"{'Config':<35} {'n_overlap':>9} {f'ol/{domain_a}%':>9} {f'ol/{domain_b}%':>9}"
        f" {'mean ratio':>11} {f'%{domain_b}-dom':>10} {f'mean {domain_a}':>11} {f'mean {domain_b}':>11}"
    )
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'─'*len(hdr)}")
    for r in results:
        print(
            f"{r['label']:<35}"
            f"{r['n_overlap']:>9,}"
            f"{r[f'overlap_pct_of_{domain_a}']:>8.1f}%"
            f"{r[f'overlap_pct_of_{domain_b}']:>8.1f}%"
            f"{r[f'mean_{domain_b}_{domain_a}_ratio']:>11.3f}"
            f"{r[f'pct_{domain_b}_dominant']:>9.1f}%"
            f"{r[f'mean_{domain_a}_rate']:>11.6f}"
            f"{r[f'mean_{domain_b}_rate']:>11.6f}"
        )
    print(f"{'='*len(hdr)}")


# ── Threshold-based plots ─────────────────────────────────────────────────────

def plot_ratio_histograms(results: list[dict], domain_a: str, domain_b: str, output_dir: Path) -> None:
    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // cols][i % cols]
        log_ratios = np.log2(r["_ratios"] + 1e-12)
        ax.hist(log_ratios, bins=50, color="steelblue", alpha=0.8, edgecolor="none")
        ax.axvline(0, color="red", lw=1.5, ls="--")
        ax.set_title(r["label"], fontsize=9)
        ax.set_xlabel(f"log₂({domain_b} / {domain_a})", fontsize=8)
        ax.set_ylabel("# neurons", fontsize=8)
        ax.text(0.97, 0.95, f"{r[f'pct_{domain_b}_dominant']:.1f}% {domain_b}-dom",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="darkred")
    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(f"log₂({domain_b}/{domain_a}) ratio for overlap neurons", fontsize=11)
    plt.tight_layout()
    path = output_dir / "ratio_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_layer_summary(results: list[dict], domain_a: str, domain_b: str, output_dir: Path) -> None:
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(labels) * 0.9), 7), sharex=True)
    ax1.bar(x, [r[f"pct_{domain_b}_dominant"] for r in results], color="steelblue", alpha=0.85)
    ax1.axhline(50, color="red", ls="--", lw=1.2, label="50%")
    ax1.set_ylabel(f"% overlap neurons where {domain_b} > {domain_a}", fontsize=9)
    ax1.set_title(f"{domain_b}-dominance of overlap neurons", fontsize=11)
    ax1.legend(fontsize=8); ax1.set_ylim(0, 100)
    ax2.bar(x, [r[f"mean_{domain_b}_{domain_a}_ratio"] for r in results], color="salmon", alpha=0.85)
    ax2.axhline(1.0, color="red", ls="--", lw=1.2, label="ratio=1")
    ax2.set_ylabel(f"Mean {domain_b}/{domain_a} ratio (overlap)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    path = output_dir / "layer_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_scatters(results: list[dict], domain_a: str, domain_b: str, output_dir: Path) -> None:
    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // cols][i % cols]
        ra, rb = r["_rates_a"], r["_rates_b"]
        if len(ra) > 4000:
            idx = np.random.default_rng(42).choice(len(ra), 4000, replace=False)
            ra, rb = ra[idx], rb[idx]
        lim = max(ra.max(), rb.max()) * 1.05
        ax.scatter(ra, rb, s=3, alpha=0.3, color="steelblue", linewidths=0)
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"{domain_a} rate", fontsize=8)
        ax.set_ylabel(f"{domain_b} rate", fontsize=8)
        ax.set_title(r["label"], fontsize=9)
    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(f"Overlap neuron firing rates (above diagonal = {domain_b}-dominant)", fontsize=11)
    plt.tight_layout()
    path = output_dir / "scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── K-sweep plots ─────────────────────────────────────────────────────────────

def plot_k_sweep_per_layer(
    sweep_by_label: dict[str, list[dict]],
    domain_a: str,
    domain_b: str,
    output_dir: Path,
) -> None:
    labels = list(sweep_by_label.keys())
    n = len(labels)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for i, label in enumerate(labels):
        ax = axes[i // cols][i % cols]
        sweep = sweep_by_label[label]
        k_pcts = [r["k_pct"] for r in sweep]
        ks     = [r["k"]     for r in sweep]
        pct_a = [r["pct_a_dominant"] for r in sweep]
        ol = [r["ol_pct"] for r in sweep]

        excesses = np.array([r["mean_excess"] for r in sweep])
        e_min, e_max = excesses.min(), excesses.max()
        excess_norm = (excesses - e_min) / (e_max - e_min + 1e-12)

        ax.plot(k_pcts, pct_a,       color="steelblue", lw=1.5, label=f"% {domain_a}-dom")
        ax.plot(k_pcts, ol,          color="salmon",    lw=1.5, label=f"ol/{domain_b}%")
        ax.plot(k_pcts, excess_norm, color="seagreen",  lw=1.5, ls="--", label="mean excess (norm)")
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_title(label, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(k_pcts[0], k_pcts[-1])
        tick_step = max(1, len(k_pcts) // 6)
        tick_idx = list(range(0, len(k_pcts), tick_step))
        ax.set_xticks([k_pcts[j] for j in tick_idx])
        ax.set_xticklabels([f"{k_pcts[j]:.0f}%\n[{ks[j]}]" for j in tick_idx], fontsize=5)
        ax.set_xlabel("K%  [neurons kept]", fontsize=7)
        if i % cols == 0:
            ax.set_ylabel("[0, 1]", fontsize=7)
        ax.tick_params(labelsize=6)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=9, ncol=3)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Top-K% {domain_a} neurons: {domain_a}-specificity by layer\n"
        f"blue=% individually {domain_a}-dominant  |  red={domain_b} overlap fraction  |  green=mean({domain_a}−{domain_b}) norm",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "k_sweep_per_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_k_sweep_heatmaps(
    sweep_by_label: dict[str, list[dict]],
    domain_a: str,
    domain_b: str,
    output_dir: Path,
) -> None:
    labels = list(sweep_by_label.keys())
    k_pcts = [r["k_pct"] for r in next(iter(sweep_by_label.values()))]

    pct_a_grid       = np.array([[r["pct_a_dominant"] for r in sweep_by_label[l]] for l in labels])
    mean_excess_grid = np.array([[r["mean_excess"]    for r in sweep_by_label[l]] for l in labels])
    ol_pct_grid      = np.array([[r["ol_pct"]         for r in sweep_by_label[l]] for l in labels])

    # Use first label's k counts as representative tick labels for the shared x-axis
    ref_sweep = sweep_by_label[labels[0]]
    tick_step = max(1, len(k_pcts) // 8)
    tick_idx = list(range(0, len(k_pcts), tick_step))
    tick_xvals  = [k_pcts[j] for j in tick_idx]
    tick_xlabels = [f"{k_pcts[j]:.0f}%\n[{ref_sweep[j]['k']}]" for j in tick_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(labels) * 0.22)))

    ext = [k_pcts[0] - 0.5, k_pcts[-1] + 0.5, len(labels) - 0.5, -0.5]

    def _hm(ax, data, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=ext)
        ax.set_xticks(tick_xvals)
        ax.set_xticklabels(tick_xlabels, fontsize=7)
        ax.set_xlabel("K%  [neurons kept — first layer]", fontsize=9)
        ax.set_ylabel("Layer / SAE config", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _hm(axes[0], pct_a_grid,
        f"% {domain_a}-dominant\n(↑ more {domain_a}-specific)", "RdYlGn", 0, 1)
    _hm(axes[1], mean_excess_grid,
        f"mean({domain_a} − {domain_b}) firing rate\n(↑ more {domain_a}-specific)", "RdYlGn")
    _hm(axes[2], ol_pct_grid,
        f"ol/{domain_b}% ({domain_b} overlap fraction)\n(↓ less {domain_b} contamination)", "RdYlGn_r", 0, 1)

    fig.suptitle(
        f"Specificity of top-K% {domain_a} neurons\n"
        f"(green = more {domain_a}-specific  |  red = more {domain_b}-contaminated)",
        fontsize=11,
    )
    plt.tight_layout()
    path = output_dir / "k_sweep_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_marginal_crossover(
    sweep_by_label: dict[str, list[dict]],
    domain_a: str,
    output_dir: Path,
) -> None:
    labels = list(sweep_by_label.keys())
    crossovers = []
    for label in labels:
        cross = next((r["k_pct"] for r in sweep_by_label[label] if r["marginal_excess"] > 0), None)
        crossovers.append(cross)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.42), 5))
    xs = np.arange(len(labels))
    vals = [c if c is not None else 31 for c in crossovers]
    colors = ["steelblue" if c is not None else "lightcoral" for c in crossovers]
    ax.bar(xs, vals, color=colors, alpha=0.85)
    ax.axhline(30, color="gray", ls="--", lw=1, label="max K (30%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Crossover K%")
    ax.set_ylim(0, 33)
    ax.set_title(
        f"First K% where the marginal neuron added is {domain_a}-dominant ({domain_a} > other domain)\n"
        "red = never reaches domain-dominant within 0–30%",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / "marginal_crossover_by_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Decoder subspace analysis ─────────────────────────────────────────────────

def load_sae_decoder(sae_release: str, sae_id: str, device: str) -> torch.Tensor:
    """Load W_dec from a pretrained SAE, with disk cache. Returns [d_sae, d_model] float32 on device."""
    from safetensors.torch import save_file as st_save
    cache_path = CACHE_ROOT / "sae_decoders" / sae_release / sae_id.replace("/", "--") / "W_dec.safetensors"
    if cache_path.exists():
        return load_file(str(cache_path))["W_dec"].float().to(device)
    from sae_lens import SAE
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    W_dec = sae.W_dec.detach().float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    st_save({"W_dec": W_dec.cpu()}, str(cache_path))
    return W_dec


def compute_decoder_subspace_overlap(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    W_dec: torch.Tensor,
    k_pcts: list[float],
) -> list[dict]:
    """
    For each K%, compute how much the subspace spanned by the top-K domain-A
    decoder columns overlaps with the subspace spanned by the top-K domain-B
    decoder columns.

    Uses principal angles between subspaces via SVD of Q_a.T @ Q_b, where Q_a
    and Q_b are orthonormal bases obtained by QR-decomposing the stacked decoder
    columns. subspace_overlap = mean(sv²): 0 = orthogonal, 1 = identical.
    """
    device = W_dec.device
    n = len(dist_a)
    order_a = torch.from_numpy(np.argsort(dist_a)[::-1].copy()).to(device)
    order_b = torch.from_numpy(np.argsort(dist_b)[::-1].copy()).to(device)

    rows = []
    for k_pct in k_pcts:
        k = max(1, int(k_pct / 100 * n))
        dec_a = W_dec[order_a[:k]]  # [k, d_model]
        dec_b = W_dec[order_b[:k]]  # [k, d_model]

        Q_a, _ = torch.linalg.qr(dec_a.T)  # [d_model, min(k, d_model)]
        Q_b, _ = torch.linalg.qr(dec_b.T)

        sv = torch.linalg.svdvals(Q_a.T @ Q_b).clamp(0.0, 1.0)

        rows.append({
            "k_pct": k_pct,
            "k": k,
            "subspace_overlap": float((sv ** 2).mean()),
            "mean_cos_principal_angle": float(sv.mean()),
            "min_cos_principal_angle": float(sv.min()),
        })
    return rows


def plot_decoder_subspace_per_layer(
    subspace_by_label: dict[str, list[dict]],
    domain_a: str,
    domain_b: str,
    output_dir: Path,
) -> None:
    labels = list(subspace_by_label.keys())
    n = len(labels)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for i, label in enumerate(labels):
        ax = axes[i // cols][i % cols]
        sweep = subspace_by_label[label]
        k_pcts = [r["k_pct"] for r in sweep]
        ks     = [r["k"]     for r in sweep]
        overlap   = [r["subspace_overlap"] for r in sweep]
        mean_cos  = [r["mean_cos_principal_angle"] for r in sweep]
        min_cos   = [r["min_cos_principal_angle"] for r in sweep]

        ax.plot(k_pcts, overlap,  color="steelblue", lw=1.5, label="subspace overlap (mean sin²)")
        ax.plot(k_pcts, mean_cos, color="salmon",    lw=1.5, label="mean cos(angle)")
        ax.plot(k_pcts, min_cos,  color="seagreen",  lw=1.5, ls="--", label="min cos(angle)")
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_title(label, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(k_pcts[0], k_pcts[-1])
        tick_step = max(1, len(k_pcts) // 6)
        tick_idx = list(range(0, len(k_pcts), tick_step))
        ax.set_xticks([k_pcts[j] for j in tick_idx])
        ax.set_xticklabels([f"{k_pcts[j]:.0f}%\n[{ks[j]}]" for j in tick_idx], fontsize=5)
        ax.set_xlabel("K%  [neurons kept]", fontsize=7)
        if i % cols == 0:
            ax.set_ylabel("[0, 1]", fontsize=7)
        ax.tick_params(labelsize=6)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=9, ncol=3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(
        f"Decoder subspace overlap: top-K {domain_a} vs top-K {domain_b} neurons\n"
        "1 = identical subspace, 0 = orthogonal — high overlap means A-selected features write to B-relevant directions",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = output_dir / "decoder_subspace_per_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_decoder_subspace_heatmap(
    subspace_by_label: dict[str, list[dict]],
    domain_a: str,
    domain_b: str,
    output_dir: Path,
) -> None:
    labels = list(subspace_by_label.keys())
    ref_sub = next(iter(subspace_by_label.values()))
    k_pcts = [r["k_pct"] for r in ref_sub]

    overlap_grid = np.array(
        [[r["subspace_overlap"] for r in subspace_by_label[l]] for l in labels]
    )

    tick_step = max(1, len(k_pcts) // 8)
    tick_idx = list(range(0, len(k_pcts), tick_step))
    tick_xvals   = [k_pcts[j] for j in tick_idx]
    tick_xlabels = [f"{k_pcts[j]:.0f}%\n[{ref_sub[j]['k']}]" for j in tick_idx]

    fig, ax = plt.subplots(figsize=(max(8, len(k_pcts) * 0.4), max(4, len(labels) * 0.22)))
    ext = [k_pcts[0] - 0.5, k_pcts[-1] + 0.5, len(labels) - 0.5, -0.5]
    im = ax.imshow(overlap_grid, aspect="auto", origin="upper", cmap="viridis",
                   vmin=0, vmax=1, extent=ext)
    ax.set_xticks(tick_xvals)
    ax.set_xticklabels(tick_xlabels, fontsize=7)
    ax.set_xlabel("K%  [neurons kept — first layer]", fontsize=9)
    ax.set_ylabel("Layer / SAE config", fontsize=9)
    ax.set_title(
        f"Decoder subspace overlap: top-K {domain_a} vs top-K {domain_b}\n"
        "(1 = identical, 0 = orthogonal)",
        fontsize=10,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = output_dir / "decoder_subspace_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-a", type=str, default="chemistry",
                        help="Selector domain (neurons ranked by this domain's firing rate)")
    parser.add_argument("--domain-b", type=str, default="math",
                        help="Comparison domain (evaluated on the top-K domain-A neurons)")
    parser.add_argument("--model-slug", type=str, default=DEFAULT_MODEL_SLUG,
                        help="Model slug used in the cache directory structure")
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/overlap_analysis_{domain_a}_vs_{domain_b})")
    parser.add_argument(
        "--k-pcts", type=str, default=None,
        help="Comma-separated K%% values for sweep, e.g. '1,5,10,20,30'. Default: 1..30",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for SAE loading and decoder subspace computations (default: cuda if available)")
    parser.add_argument(
        "--sae-release", type=str, default=None,
        help=(
            "SAE release name (e.g. 'gemma-scope-2-12b-it-res-all'). "
            "When provided, enables decoder subspace analysis. "
            "The SAE ID for each layer is derived from the layer tag by replacing '--' with '_' "
            "(e.g. layer_20--width_262k--l0_small → layer_20_width_262k_l0_small)."
        ),
    )
    args = parser.parse_args()

    domain_a = args.domain_a
    domain_b = args.domain_b

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"results/overlap_analysis_{domain_a}_vs_{domain_b}_{args.model_slug}_{args.sae_release or 'no_sae'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    k_pcts = [float(x) for x in args.k_pcts.split(",")] if args.k_pcts else K_PCTS

    print(f"Domain A (selector): {domain_a}  |  Domain B (comparison): {domain_b}")
    print(f"Model: {args.model_slug}  |  Threshold: {args.threshold}  |  K sweep: {k_pcts[0]}–{k_pcts[-1]}%")
    print(f"Cache: {CACHE_ROOT}")
    if args.sae_release:
        print(f"SAE release: {args.sae_release} (decoder subspace analysis enabled, device={args.device})")

    pairs = discover_pairs(domain_a, domain_b, args.model_slug)
    if not pairs:
        print("No matching pairs found.")
        return
    print(f"Found {len(pairs)} pair(s):\n  " + "\n  ".join(p["rel"] for p in pairs))

    threshold_results = []
    sweep_by_label: dict[str, list[dict]] = {}
    subspace_by_label: dict[str, list[dict]] = {}

    for p in pairs:
        print(f"\nLoading {p['label']} ...")
        dists = load_dists(p["path_a"], p["path_b"])
        if dists is None:
            print(f"  [SKIP] SAE width mismatch")
            continue
        dist_a, dist_b = dists

        r = analyze_threshold(p["label"], dist_a, dist_b, domain_a, domain_b, args.threshold)
        if r is not None:
            threshold_results.append(r)

        sweep_by_label[p["label"]] = compute_k_sweep(dist_a, dist_b, k_pcts)

        if args.sae_release:
            sae_id = p["sae_tag"].replace("--", "_")
            print(f"  Loading SAE weights: {sae_id} ...")
            try:
                W_dec = load_sae_decoder(args.sae_release, sae_id, args.device)
                subspace_by_label[p["label"]] = compute_decoder_subspace_overlap(
                    dist_a, dist_b, W_dec, k_pcts
                )
            except Exception as e:
                print(f"  [SKIP decoder subspace] {e}")

    if threshold_results:
        print_table(threshold_results, domain_a, domain_b)
        summary = [{k: v for k, v in r.items() if not k.startswith("_")} for r in threshold_results]
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSaved: {output_dir / 'summary.json'}")
        plot_ratio_histograms(threshold_results, domain_a, domain_b, output_dir)
        plot_layer_summary(threshold_results, domain_a, domain_b, output_dir)
        plot_scatters(threshold_results, domain_a, domain_b, output_dir)

    if sweep_by_label:
        sweep_json = {
            label: [{k: v for k, v in r.items()} for r in rows]
            for label, rows in sweep_by_label.items()
        }
        (output_dir / "k_sweep.json").write_text(json.dumps(sweep_json, indent=2))
        print(f"Saved: {output_dir / 'k_sweep.json'}")
        plot_k_sweep_per_layer(sweep_by_label, domain_a, domain_b, output_dir)
        plot_k_sweep_heatmaps(sweep_by_label, domain_a, domain_b, output_dir)
        plot_marginal_crossover(sweep_by_label, domain_a, output_dir)

    if subspace_by_label:
        subspace_json = {
            label: [{k: v for k, v in r.items()} for r in rows]
            for label, rows in subspace_by_label.items()
        }
        (output_dir / "decoder_subspace.json").write_text(json.dumps(subspace_json, indent=2))
        print(f"Saved: {output_dir / 'decoder_subspace.json'}")
        plot_decoder_subspace_per_layer(subspace_by_label, domain_a, domain_b, output_dir)
        plot_decoder_subspace_heatmap(subspace_by_label, domain_a, domain_b, output_dir)


if __name__ == "__main__":
    main()

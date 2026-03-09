#!/usr/bin/env python3
"""
Generate charts for the Beyond Behavioral Scanning blog post.

Single-family mode (--prefix or legacy):
  Reads:   results/{prefix}_augmented.jsonl  (or results/axis_augmented.jsonl)
  Outputs: charts/{prefix}_displacement_histogram.png
           charts/{prefix}_signature_distribution.png
           charts/{prefix}_threshold_visualization.png

Comparison mode (--compare):
  Reads:   results/{prefix}_augmented.jsonl for each prefix
  Outputs: charts/comparison_displacement.png
           charts/comparison_signatures.png

Usage:
  python3 generate_charts.py                          # legacy, no prefix
  python3 generate_charts.py --prefix dan_full        # single family
  python3 generate_charts.py --compare dan_full promptinject knownbadsignatures continuation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
CHARTS_DIR  = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

# ── Smoke test baselines (update if you re-run extraction) ─────────
ASSISTANT_BASELINE = 7.16  # from latest smoke test
SMOKE_TESTS = {
    "Philosophy drift": 5.97,
    "Assistant-like":   7.16,
    "Therapy drift":    7.87,
    "Persona shift":    9.10,
}

# ── Palette: neutral academic ──────────────────────────────────────
BG           = '#FAFAFA'
TEXT         = '#2D2D2D'
GRID         = '#E0E0E0'
ACCENT_DARK  = '#2B5C8A'   # steel blue     → PERSONA_JAILBREAK
ACCENT_MID   = '#CC7A2F'   # warm ochre     → NEAR_MISS
ACCENT_LIGHT = '#2A7F62'   # teal green     → CLEAN
ACCENT_FAINT = '#7B5EA7'   # muted purple   → NO_AXIS_DATA
THRESH_COLOR = '#C44E52'   # muted red
BASE_COLOR   = '#4C72B0'   # muted blue

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.facecolor':   BG,
    'figure.facecolor': BG,
    'axes.edgecolor':   '#7A7A7A',
    'axes.labelcolor':  TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'text.color':       TEXT,
    'grid.color':       GRID,
    'grid.linewidth':   0.5,
})

OUTPUT_DPI = 200

# Signature → color mapping
SIG_COLORS = {
    'PERSONA_JAILBREAK': ACCENT_DARK,
    'NEAR_MISS':         ACCENT_MID,
    'ALIGNMENT_FAILURE': THRESH_COLOR,
    'CLEAN':             ACCENT_LIGHT,
    'NO_AXIS_DATA':      ACCENT_FAINT,
}

# ── Helpers ────────────────────────────────────────────────────────

def load_jsonl(path, filter_fn=None):
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if filter_fn is None or filter_fn(rec):
                records.append(rec)
    return records


def load_family_data(prefix):
    """Load and parse data for one probe family (prefix).

    Returns a dict with keys:
      augmented_path, augmented_records, prompt_displacements,
      attempt_displacements, attempt_signatures, sig_counts,
      THRESHOLD, n_prompts, total
    """
    if prefix:
        aug_path = RESULTS_DIR / f"{prefix}_augmented.jsonl"
    else:
        aug_path = RESULTS_DIR / "axis_augmented.jsonl"

    if not aug_path.exists():
        print(f"ERROR: {aug_path} not found. Run from the project root.",
              file=sys.stderr)
        sys.exit(1)

    augmented_records = load_jsonl(
        aug_path,
        filter_fn=lambda r: r.get("entry_type") == "attempt"
    )
    print(f"Loaded {len(augmented_records)} attempt records from {aug_path}")

    # Per-prompt displacements (deduplicated; first occurrence wins)
    seen_prompts = {}
    for r in augmented_records:
        p = r.get("prompt", "")
        d = r.get("displacement")
        if p not in seen_prompts and d is not None:
            seen_prompts[p] = d
    prompt_displacements = np.array(list(seen_prompts.values()))

    # Per-attempt data
    attempt_displacements = []
    attempt_signatures = []
    for r in augmented_records:
        sig  = r.get("signature")
        disp = r.get("displacement")
        if sig and disp is not None:
            attempt_signatures.append(sig)
            attempt_displacements.append(disp)
    attempt_displacements = np.array(attempt_displacements)

    # Infer threshold
    thresholds = [r["threshold_used"] for r in augmented_records if "threshold_used" in r]
    threshold = thresholds[0] if thresholds else 7.0

    sig_counts = Counter(attempt_signatures)
    total = sum(sig_counts.values())

    return {
        "augmented_path":       aug_path,
        "augmented_records":    augmented_records,
        "prompt_displacements": prompt_displacements,
        "attempt_displacements": attempt_displacements,
        "attempt_signatures":   attempt_signatures,
        "sig_counts":           sig_counts,
        "THRESHOLD":            threshold,
        "n_prompts":            len(prompt_displacements),
        "total":                total,
    }


def chart_prefix(prefix):
    """Return the file-name prefix string (with trailing underscore if set)."""
    return f"{prefix}_" if prefix else ""


# ============================================================
# CHART 1: Displacement Distribution Histogram
# ============================================================

def plot_displacement_histogram(ax, prompt_displacements, threshold, title=None):
    """Draw histogram onto ax; returns (counts, bins, patches)."""
    result = ax.hist(
        prompt_displacements, bins='fd', color=ACCENT_LIGHT,
        edgecolor=ACCENT_DARK, linewidth=0.6, alpha=0.85
    )
    counts, bins, patches = result

    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= threshold:
            patch.set_facecolor(ACCENT_DARK)
            patch.set_alpha(0.6)

    ax.axvline(threshold, color=THRESH_COLOR, linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold})')
    ax.axvline(ASSISTANT_BASELINE, color=BASE_COLOR, linestyle=':',
               linewidth=1.5, label=f'Assistant baseline ({ASSISTANT_BASELINE})')

    ax.set_xlabel('Displacement along Assistant Axis')
    ax.set_ylabel('Count (unique prompts)')
    if title:
        ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    return result


def save_displacement_histogram(data, prefix):
    n_prompts = data["n_prompts"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    label = prefix if prefix else "DAN"
    plot_displacement_histogram(
        ax,
        data["prompt_displacements"],
        data["THRESHOLD"],
        title=f'{label} Probe Displacement Distribution (n={n_prompts})',
    )
    plt.tight_layout()
    out = CHARTS_DIR / f"{chart_prefix(prefix)}displacement_histogram.png"
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"✓ {out}")


# ============================================================
# CHART 2: Signature Distribution (Horizontal Bar)
# ============================================================

def plot_signature_bars(ax, sig_counts, total, title=None):
    """Draw a horizontal bar chart onto ax."""
    ordered_sigs  = sorted(sig_counts.keys(), key=lambda s: (-sig_counts[s], s))
    counts_list   = [sig_counts[s] for s in ordered_sigs]
    pcts          = [100.0 * c / total for c in counts_list]
    colors_list   = [SIG_COLORS.get(s, ACCENT_FAINT) for s in ordered_sigs]

    y_pos = np.arange(len(ordered_sigs))
    bars  = ax.barh(y_pos, counts_list, color=colors_list,
                    edgecolor='white', linewidth=0.8, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_sigs, fontfamily='monospace', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(f'Attempts (n={total})')
    if title:
        ax.set_title(title, fontweight='bold', fontsize=13)

    for bar, count, pct in zip(bars, counts_list, pcts):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{count}  ({pct:.1f}%)', va='center', fontsize=9, color=TEXT)

    ax.set_xlim(0, max(counts_list) * 1.25)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ordered_sigs, counts_list, pcts


def save_signature_distribution(data, prefix):
    fig, ax = plt.subplots(
        figsize=(8, max(3.0, 0.7 * len(data["sig_counts"])))
    )
    plot_signature_bars(ax, data["sig_counts"], data["total"],
                        title='Attack Signature Distribution')
    plt.tight_layout()
    out = CHARTS_DIR / f"{chart_prefix(prefix)}signature_distribution.png"
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"✓ {out}")


# ============================================================
# CHART 3: Threshold Visualization (Strip Plot)
# ============================================================

def save_threshold_visualization(data, prefix):
    attempt_displacements = data["attempt_displacements"]
    attempt_signatures    = data["attempt_signatures"]
    sig_counts            = data["sig_counts"]
    threshold             = data["THRESHOLD"]

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10, 4.2))

    for disp, sig in zip(attempt_displacements, attempt_signatures):
        jitter = rng.uniform(-0.25, 0.25)
        color  = SIG_COLORS.get(sig, ACCENT_FAINT)
        ax.scatter(disp, jitter, color=color, s=28, alpha=0.7,
                   edgecolors='white', linewidths=0.3, zorder=3)

    ax.axvline(threshold, color=THRESH_COLOR, linestyle='--', linewidth=1.5, zorder=2)
    ax.axvline(ASSISTANT_BASELINE, color=BASE_COLOR, linestyle=':', linewidth=1.5, zorder=2)

    ax.text(threshold - 0.15, 0.48, '← Displaced from Assistant',
            ha='right', fontsize=8.5, color=THRESH_COLOR, fontstyle='italic')
    ax.text(threshold + 0.15, 0.48, 'Assistant-like →',
            ha='left', fontsize=8.5, color=BASE_COLOR, fontstyle='italic')

    smoke_y = {
        'Philosophy drift': -0.35,
        'Assistant-like':   -0.48,
        'Therapy drift':    -0.35,
        'Persona shift':    -0.48,
    }
    for label, val in SMOKE_TESTS.items():
        y_off = smoke_y[label]
        ax.annotate(label, xy=(val, y_off), fontsize=7, ha='center',
                    color=TEXT, fontstyle='italic')
        ax.plot(val, y_off + 0.06, marker='v', markersize=6,
                color='#000000', zorder=4)

    ax.set_xlabel('Displacement along Assistant Axis')
    ax.set_title('Per-Prompt Displacement with Threshold and Baselines',
                 fontweight='bold', fontsize=13)
    ax.set_ylim(-0.62, 0.62)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=7, label=s)
        for s, c in SIG_COLORS.items() if s in sig_counts
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True,
              facecolor=BG, edgecolor=GRID, fontsize=8, ncol=2)

    plt.tight_layout()
    out = CHARTS_DIR / f"{chart_prefix(prefix)}threshold_visualization.png"
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"✓ {out}")


# ============================================================
# Summary statistics printout
# ============================================================

def print_summary(data, prefix):
    sig_counts        = data["sig_counts"]
    prompt_displacements = data["prompt_displacements"]
    ordered_sigs = sorted(sig_counts.keys(), key=lambda s: (-sig_counts[s], s))

    label = prefix if prefix else "(legacy)"
    print(f"\nAll charts saved to {CHARTS_DIR}/ [{label}]")
    print(f"  Unique prompts:     {data['n_prompts']}")
    print(f"  Total attempts:     {data['total']}")
    print(f"  Threshold:          {data['THRESHOLD']}")
    print(f"  Displacement range: [{prompt_displacements.min():.2f}, {prompt_displacements.max():.2f}]")
    print(f"  Displacement mean:  {prompt_displacements.mean():.2f}")
    print(f"  Displacement median:{np.median(prompt_displacements):.2f}")
    print(f"\nSignature counts:")
    for s in ordered_sigs:
        print(f"  {s:24s} {sig_counts[s]:4d}")
    print(f"\nRun charts through cwebp for final WebP output.")


# ============================================================
# COMPARISON MODE
# ============================================================

def save_comparison_displacement(all_data, prefixes):
    """Stacked small-multiples displacement histogram, one per family."""
    n = len(prefixes)
    # Global x-axis range across all families
    global_min = min(d["prompt_displacements"].min() for d in all_data)
    global_max = max(d["prompt_displacements"].max() for d in all_data)
    x_pad = (global_max - global_min) * 0.05
    x_lim = (global_min - x_pad, global_max + x_pad)

    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, data, prefix in zip(axes, all_data, prefixes):
        pd = data["prompt_displacements"]
        threshold = data["THRESHOLD"]

        counts, bins, patches = ax.hist(
            pd, bins='fd', color=ACCENT_LIGHT,
            edgecolor=ACCENT_DARK, linewidth=0.6, alpha=0.85
        )
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= threshold:
                patch.set_facecolor(ACCENT_DARK)
                patch.set_alpha(0.6)

        ax.axvline(threshold, color=THRESH_COLOR, linestyle='--', linewidth=1.5,
                   label=f'Threshold ({threshold})')
        ax.axvline(ASSISTANT_BASELINE, color=BASE_COLOR, linestyle=':',
                   linewidth=1.5, label=f'Baseline ({ASSISTANT_BASELINE})')

        ax.set_xlim(x_lim)
        ax.set_ylabel('Count')
        ax.set_title(prefix, fontweight='bold', fontsize=11, loc='left')
        ax.legend(frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8)
        ax.grid(axis='y', alpha=0.4)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[-1].set_xlabel('Displacement along Assistant Axis')
    fig.suptitle('Displacement Distribution by Probe Family',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()

    out = CHARTS_DIR / "comparison_displacement.png"
    fig.savefig(out, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {out}")


def save_comparison_signatures(all_data, prefixes):
    """Grouped horizontal bar chart: families as groups, signatures as bars."""
    # Collect all signatures seen across families
    all_sigs = set()
    for data in all_data:
        all_sigs.update(data["sig_counts"].keys())
    all_sigs = sorted(all_sigs)
    n_sigs    = len(all_sigs)
    n_families = len(prefixes)

    group_height = 0.7
    bar_height   = group_height / n_families

    fig, ax = plt.subplots(
        figsize=(9, max(4.0, 0.9 * n_sigs * n_families))
    )

    # Family color cycle (reuse palette; cycle if more than 4 families)
    family_colors = [ACCENT_DARK, ACCENT_MID, ACCENT_LIGHT, ACCENT_FAINT,
                     BASE_COLOR, THRESH_COLOR]

    y_base = np.arange(n_sigs, dtype=float)

    for fi, (data, prefix) in enumerate(zip(all_data, prefixes)):
        sig_counts = data["sig_counts"]
        total = data["total"] or 1
        offset = (fi - (n_families - 1) / 2.0) * bar_height
        counts_list = [sig_counts.get(s, 0) for s in all_sigs]
        color = family_colors[fi % len(family_colors)]

        bars = ax.barh(y_base + offset, counts_list, height=bar_height * 0.9,
                       color=color, edgecolor='white', linewidth=0.6,
                       label=prefix, alpha=0.85)

    ax.set_yticks(y_base)
    ax.set_yticklabels(all_sigs, fontfamily='monospace', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Attempts')
    ax.set_title('Attack Signature Distribution by Probe Family',
                 fontweight='bold', fontsize=13)
    ax.legend(frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = CHARTS_DIR / "comparison_signatures.png"
    fig.savefig(out, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"✓ {out}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate displacement / signature charts for garak-axis probes."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prefix", metavar="PREFIX",
        help="Probe family prefix (reads results/{prefix}_augmented.jsonl)."
    )
    mode.add_argument(
        "--compare", nargs="+", metavar="PREFIX",
        help="Comparison mode: list of prefixes to overlay."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        # ── Comparison mode ──────────────────────────────────────────
        prefixes = args.compare
        print(f"Comparison mode: {prefixes}")
        all_data = [load_family_data(p) for p in prefixes]

        save_comparison_displacement(all_data, prefixes)
        save_comparison_signatures(all_data, prefixes)

        print(f"\nComparison charts saved to {CHARTS_DIR}/")
        for data, prefix in zip(all_data, prefixes):
            print(f"\n── {prefix} ──")
            print_summary(data, prefix)

    else:
        # ── Single-family (or legacy) mode ───────────────────────────
        prefix = args.prefix  # may be None (legacy)
        data   = load_family_data(prefix)
        print(f"Threshold (from data): {data['THRESHOLD']}")

        save_displacement_histogram(data, prefix)
        save_signature_distribution(data, prefix)
        save_threshold_visualization(data, prefix)
        print_summary(data, prefix)


if __name__ == "__main__":
    main()

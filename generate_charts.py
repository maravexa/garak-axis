#!/usr/bin/env python3
"""
Generate charts for the Beyond Behavioral Scanning blog post.

Reads from:
  results/axis_capture.jsonl    — per-prompt displacement (unique prompts)
  results/axis_augmented.jsonl  — per-attempt signatures with displacement

Outputs to:
  charts/displacement_histogram.png
  charts/signature_distribution.png
  charts/threshold_visualization.png
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

AXIS_AUGMENTED = RESULTS_DIR / "axis_augmented.jsonl"

if not AXIS_AUGMENTED.exists():
    print(f"ERROR: {AXIS_AUGMENTED} not found. Run from the project root.", file=sys.stderr)
    sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────
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

# axis_augmented: one record per attempt, filter to attempts only
augmented_records = load_jsonl(
    AXIS_AUGMENTED,
    filter_fn=lambda r: r.get("entry_type") == "attempt"
)
print(f"Loaded {len(augmented_records)} attempt records from {AXIS_AUGMENTED}")

# Per-prompt displacements (deduplicated from augmented for histogram)
_seen_prompts = {}
for r in augmented_records:
    p = r.get("prompt", "")
    d = r.get("displacement")
    if p not in _seen_prompts and d is not None:
        _seen_prompts[p] = d
prompt_displacements = np.array(list(_seen_prompts.values()))

# Per-attempt data (for signature chart and threshold viz)
attempt_displacements = []
attempt_signatures = []
for r in augmented_records:
    sig = r.get("signature")
    disp = r.get("displacement")
    if sig and disp is not None:
        attempt_signatures.append(sig)
        attempt_displacements.append(disp)

attempt_displacements = np.array(attempt_displacements)

# Infer threshold from data (all records should share the same value)
thresholds = [r["threshold_used"] for r in augmented_records if "threshold_used" in r]
THRESHOLD = thresholds[0] if thresholds else 7.0
print(f"Threshold (from data): {THRESHOLD}")

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
ACCENT_DARK  = '#CC7A2F'   # warm ochre    → PERSONA_JAILBREAK
ACCENT_MID   = '#2B5C8A'   # steel blue    → NEAR_MISS
ACCENT_LIGHT = '#2A7F62'   # teal green    → CLEAN
ACCENT_FAINT = '#7B5EA7'   # muted purple  → NO_AXIS_DATA
THRESH_COLOR = '#C44E52'   # muted red
BASE_COLOR   = '#4C72B0'   # muted blue

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.facecolor': BG,
    'figure.facecolor': BG,
    'axes.edgecolor': '#7A7A7A',
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
    'text.color': TEXT,
    'grid.color': GRID,
    'grid.linewidth': 0.5,
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

# ============================================================
# CHART 1: Displacement Distribution Histogram
# ============================================================
n_prompts = len(prompt_displacements)
fig, ax = plt.subplots(figsize=(8, 4.5))
counts, bins, patches = ax.hist(
    prompt_displacements, bins='fd', color=ACCENT_LIGHT,
    edgecolor=ACCENT_DARK, linewidth=0.6, alpha=0.85
)

# Darken bars above threshold
for patch, left_edge in zip(patches, bins[:-1]):
    if left_edge >= THRESHOLD:
        patch.set_facecolor(ACCENT_DARK)
        patch.set_alpha(0.6)

ax.axvline(THRESHOLD, color=THRESH_COLOR, linestyle='--', linewidth=1.5,
           label=f'Threshold ({THRESHOLD})')
ax.axvline(ASSISTANT_BASELINE, color=BASE_COLOR, linestyle=':',
           linewidth=1.5, label=f'Assistant baseline ({ASSISTANT_BASELINE})')

ax.set_xlabel('Displacement along Assistant Axis')
ax.set_ylabel('Count (unique prompts)')
ax.set_title(f'DAN Probe Displacement Distribution (n={n_prompts})',
             fontweight='bold', fontsize=13)
ax.legend(frameon=True, facecolor=BG, edgecolor=GRID, fontsize=9)
ax.grid(axis='y', alpha=0.4)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
out = CHARTS_DIR / "displacement_histogram.png"
fig.savefig(out, dpi=OUTPUT_DPI)
plt.close(fig)
print(f"✓ {out}")

# ============================================================
# CHART 2: Signature Distribution (Horizontal Bar)
# ============================================================
sig_counts = Counter(attempt_signatures)

# Stable ordering: largest first, then alphabetical
ordered_sigs = sorted(sig_counts.keys(), key=lambda s: (-sig_counts[s], s))
counts_list = [sig_counts[s] for s in ordered_sigs]
total = sum(counts_list)
pcts = [100.0 * c / total for c in counts_list]
colors_list = [SIG_COLORS.get(s, ACCENT_FAINT) for s in ordered_sigs]

fig, ax = plt.subplots(figsize=(8, max(3.0, 0.7 * len(ordered_sigs))))
y_pos = np.arange(len(ordered_sigs))
bars = ax.barh(y_pos, counts_list, color=colors_list,
               edgecolor='white', linewidth=0.8, height=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(ordered_sigs, fontfamily='monospace', fontsize=10)
ax.invert_yaxis()
ax.set_xlabel(f'Attempts (n={total})')
ax.set_title('Attack Signature Distribution', fontweight='bold', fontsize=13)

for bar, count, pct in zip(bars, counts_list, pcts):
    ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f'{count}  ({pct:.1f}%)', va='center', fontsize=9, color=TEXT)

ax.set_xlim(0, max(counts_list) * 1.25)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = CHARTS_DIR / "signature_distribution.png"
fig.savefig(out, dpi=OUTPUT_DPI)
plt.close(fig)
print(f"✓ {out}")

# ============================================================
# CHART 3: Threshold Visualization (Strip Plot)
# ============================================================
rng = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(10, 4.2))

for disp, sig in zip(attempt_displacements, attempt_signatures):
    jitter = rng.uniform(-0.25, 0.25)
    color = SIG_COLORS.get(sig, ACCENT_FAINT)
    ax.scatter(disp, jitter, color=color, s=28, alpha=0.7,
               edgecolors='white', linewidths=0.3, zorder=3)

ax.axvline(THRESHOLD, color=THRESH_COLOR, linestyle='--', linewidth=1.5, zorder=2)
ax.axvline(ASSISTANT_BASELINE, color=BASE_COLOR, linestyle=':', linewidth=1.5, zorder=2)

# Zone labels
ax.text(THRESHOLD - 0.15, 0.48, '← Displaced from Assistant',
        ha='right', fontsize=8.5, color=THRESH_COLOR, fontstyle='italic')
ax.text(THRESHOLD + 0.15, 0.48, 'Assistant-like →',
        ha='left', fontsize=8.5, color=BASE_COLOR, fontstyle='italic')

# Smoke test reference points (staggered y to avoid overlap)
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
            color=TEXT, zorder=4)

ax.set_xlabel('Displacement along Assistant Axis')
ax.set_title('Per-Prompt Displacement with Threshold and Baselines',
             fontweight='bold', fontsize=13)
ax.set_ylim(-0.62, 0.62)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=7, label=s)
    for s, c in SIG_COLORS.items() if s in sig_counts
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True,
          facecolor=BG, edgecolor=GRID, fontsize=8, ncol=2)

plt.tight_layout()
out = CHARTS_DIR / "threshold_visualization.png"
fig.savefig(out, dpi=OUTPUT_DPI)
plt.close(fig)
print(f"✓ {out}")

# ── Summary ────────────────────────────────────────────────────────
print(f"\nAll charts saved to {CHARTS_DIR}/")
print(f"  Unique prompts:  {n_prompts}")
print(f"  Total attempts:  {total}")
print(f"  Threshold:       {THRESHOLD}")
print(f"  Displacement range: [{prompt_displacements.min():.2f}, {prompt_displacements.max():.2f}]")
print(f"  Displacement mean:  {prompt_displacements.mean():.2f}")
print(f"  Displacement median: {np.median(prompt_displacements):.2f}")
print(f"\nSignature counts:")
for s in ordered_sigs:
    print(f"  {s:24s} {sig_counts[s]:4d}")
print(f"\nRun charts through cwebp for final WebP output.")

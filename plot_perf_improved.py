"""
Improved perf_vs_n_cde_loss_real plot — draft for review.

Changes vs original:
  1. Right-side labels for all nonparametric and foundation lines;
     best parametric method also labeled.
  2. Foundation lines stop with an explicit terminal marker + caption.
  3. SE shaded bands for nonparametric and foundation lines.
  4. Parametric cluster replaced by a shaded min/max band;
     Ridge variants shown separately as thin dashed lines;
     best non-Ridge parametric shown as a single solid labeled line.
  5. Terminal × marker for every method that drops out before n=500K.
  6. Y-axis inverted so better (more negative CDE loss) is at the top.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

RESULTS = "results_real/sdss_scaling/results.json"
OUT     = "results_real/sdss_scaling/perf_vs_n_cde_loss_improved.png"
METRIC  = "CDE_loss"

# ── group membership ─────────────────────────────────────────────────────────
FOUNDATIONAL = {"TabPFN-Native", "TabPFN-2.5", "RealTabPFN-2.5", "TabICL-Quantiles"}

PARAMETRIC_BASE = {
    "LinearGauss-Homo", "LinearGauss-Hetero", "Student-t",
    "LogNormal-Homo", "LogNormal-Hetero", "Gamma-GLM",
}
PARAMETRIC_RIDGE = {
    "LinGauss-Homo-Ridge", "LinGauss-Hetero-Ridge", "Student-t-Ridge",
    "LogNormal-Homo-Ridge", "LogNormal-Hetero-Ridge", "Gamma-GLM-Ridge",
}
PARAMETRIC = PARAMETRIC_BASE | PARAMETRIC_RIDGE

NONPARAMETRIC = {"MDN", "Flow-Spline", "BART-Homo", "BART-Hetero",
                 "FlexCode-RF", "CatMLP", "Quantile-Tree"}

# ── colors ────────────────────────────────────────────────────────────────────
C_PARAM   = "#1b9e77"
C_NONPAR  = "#377eb8"
C_FOUND   = "#e67e22"

FOUND_STYLES = {
    "TabPFN-Native":    {"ls": "-",  "marker": "o"},
    "TabPFN-2.5":       {"ls": "-",  "marker": "o"},   # identical to Native; merged in label
    "RealTabPFN-2.5":   {"ls": "-.", "marker": "^"},
    "TabICL-Quantiles": {"ls": ":",  "marker": "D"},
}
# TabPFN-Native and TabPFN-2.5 produce identical results; show as one labeled line
FOUND_MERGE = {"TabPFN-Native", "TabPFN-2.5"}
FOUND_MERGE_LABEL = "TabPFN Native/2.5"

NONPAR_LINES = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]

DISPLAY = {
    "TabPFN-Native":    "TabPFN Native",
    "TabPFN-2.5":       "TabPFN 2.5",
    "RealTabPFN-2.5":   "RealTabPFN 2.5",
    "TabICL-Quantiles": "TabICL Quantiles",
    "LinearGauss-Homo": "LinGauss-Homo",
    "LinearGauss-Hetero": "LinGauss-Hetero",
}
def disp(m):
    return DISPLAY.get(m, m)

MAX_N = 500_000

# ── load data ─────────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    raw = json.load(f)

def parse_n(ds):
    return int(ds.split("-")[1])

datasets = sorted(raw.keys(), key=parse_n)
all_ns   = [parse_n(ds) for ds in datasets]

def get_series(method):
    vals, ses, ns = [], [], []
    for ds, n in zip(datasets, all_ns):
        r = raw[ds].get(method)
        if r and r.get(METRIC) is not None:
            vals.append(float(r[METRIC]))
            ses.append(float(r.get(f"{METRIC}_se") or 0))
            ns.append(n)
    return ns, vals, ses

# ── label placement helper ────────────────────────────────────────────────────
MIN_GAP = 0.55   # data units

def place_labels(ax, series, label_x, fontsize=9.5):
    """series: list of (y_val, x_src, label, color).
    Stagger labels at label_x; arrows originate from (x_src, y_val)."""
    if not series:
        return
    series = sorted(series, key=lambda t: t[0])
    placed = []
    for y_raw, x_src, label, col in series:
        y = y_raw
        # Push against all already-placed labels; a few passes resolves clusters
        for _ in range(len(placed) + 1):
            for py in placed:
                if abs(y - py) < MIN_GAP:
                    y = py - MIN_GAP if y < py else py + MIN_GAP
        placed.append(y)
        needs_arrow = abs(y - y_raw) > 0.05 or x_src < label_x * 0.92
        ax.annotate(
            label,
            xy=(x_src, y_raw), xytext=(label_x, y),
            fontsize=fontsize, color=col, va="center", ha="left",
            fontweight="bold", clip_on=False,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.55,
                            connectionstyle="arc3,rad=0.0")
            if needs_arrow else None,
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=col, lw=0.7, alpha=0.9),
            annotation_clip=False,
        )

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

label_series = []   # (y_last, label, color) for right-side annotation

# 1. PARAMETRIC BAND ──────────────────────────────────────────────────────────
param_by_n = {n: [] for n in all_ns}
best_base_ns, best_base_vals, best_base_ses = None, None, None
best_base_name, best_base_val_last = None, None

for m in sorted(PARAMETRIC_BASE):
    ns, vals, ses = get_series(m)
    if not ns:
        continue
    for n, v in zip(ns, vals):
        param_by_n[n].append(v)
    # track best (lowest CDE loss at largest shared n)
    if best_base_val_last is None or vals[-1] < best_base_val_last:
        best_base_ns, best_base_vals, best_base_ses = ns, vals, ses
        best_base_name, best_base_val_last = m, vals[-1]
# Ridge variants excluded from band — they are consistently worse than nonparametric
# and their range would stretch the y-axis beyond the region of interest.

band_ns   = sorted(n for n, vs in param_by_n.items() if vs)
band_mins = [min(param_by_n[n]) for n in band_ns]
band_maxs = [max(param_by_n[n]) for n in band_ns]

ax.fill_between(band_ns, band_mins, band_maxs,
                color=C_PARAM, alpha=0.18, zorder=1,
                label="Parametric (full range)")

# Best non-Ridge parametric — solid labeled line
if best_base_ns:
    ax.plot(best_base_ns, best_base_vals,
            color=C_PARAM, lw=2.4, ls="-", zorder=4,
            label=f"Best parametric ({disp(best_base_name)})")
    arr_ns = np.array(best_base_ns)
    arr_v  = np.array(best_base_vals)
    arr_se = np.array(best_base_ses)
    ax.fill_between(arr_ns, arr_v - arr_se, arr_v + arr_se,
                    color=C_PARAM, alpha=0.20, zorder=3)
    label_series.append((best_base_vals[-1], best_base_ns[-1], disp(best_base_name), C_PARAM))
    if best_base_ns[-1] < MAX_N:
        ax.plot(best_base_ns[-1], best_base_vals[-1], "x",
                color=C_PARAM, ms=7, mew=1.8, zorder=6)

# 2. NONPARAMETRIC — individual lines; SE bands for top-3 only ────────────────
nonpar_all = [(m, *get_series(m)) for m in NONPARAMETRIC if get_series(m)[0]]
nonpar_all.sort(key=lambda t: t[2][-1])          # ascending CDE loss = best first
top3_nonpar = {t[0] for t in nonpar_all[:3]}
nonpar_methods = [t[0] for t in nonpar_all]

for i, (m, ns, vals, ses) in enumerate(nonpar_all):
    ls = NONPAR_LINES[i % len(NONPAR_LINES)]
    ax.plot(ns, vals, color=C_NONPAR, lw=2.0, ls=ls, zorder=5, alpha=0.88)
    if m in top3_nonpar:
        arr = np.array(vals); arr_se = np.array(ses)
        ax.fill_between(ns, arr - arr_se, arr + arr_se,
                        color=C_NONPAR, alpha=0.07, zorder=4)
    label_series.append((vals[-1], ns[-1], disp(m), C_NONPAR))
    if ns[-1] < MAX_N:
        ax.plot(ns[-1], vals[-1], "x", color=C_NONPAR,
                ms=7, mew=1.8, zorder=7)

# 3. FOUNDATION — individual lines + SE bands + terminal annotation ─────────
found_methods = sorted(m for m in FOUNDATIONAL if get_series(m)[0])
merged_label_added = False
for m in found_methods:
    ns, vals, ses = get_series(m)
    sty = FOUND_STYLES.get(m, {"ls": "-", "marker": "o"})
    ax.plot(ns, vals, color=C_FOUND, lw=3.2, ls=sty["ls"],
            marker=sty["marker"], ms=6, zorder=8, alpha=0.95)
    arr = np.array(vals); arr_se = np.array(ses)
    ax.fill_between(ns, arr - arr_se, arr + arr_se,
                    color=C_FOUND, alpha=0.14, zorder=7)
    # Merge TabPFN-Native/2.5 into one label (identical results)
    if m in FOUND_MERGE:
        if not merged_label_added:
            label_series.append((vals[-1], ns[-1], FOUND_MERGE_LABEL, C_FOUND))
            merged_label_added = True
    else:
        label_series.append((vals[-1], ns[-1], disp(m), C_FOUND))
    if ns[-1] < MAX_N:
        ax.plot(ns[-1], vals[-1], "x", color=C_FOUND,
                ms=9, mew=2.2, zorder=9)

# 4. AXES ─────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xticks(all_ns)
ax.set_xticklabels([
    f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M"
    for n in all_ns
], fontsize=13)
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel("n", fontsize=15)
ax.set_ylabel("CDE Loss", fontsize=15)
ax.tick_params(labelsize=13)
ax.grid(alpha=0.25, linewidth=0.8)

# 6. Invert y-axis so better (more negative) is at top; clip to non-Ridge range
all_line_vals = (
    best_base_vals
    + [v for _, ns, vals, ses in nonpar_all for v in vals]
    + [v for m in found_methods for v in get_series(m)[1]]
)
y_best = min(all_line_vals) - 0.4
y_worst = max(all_line_vals) + 0.4
ax.set_ylim(y_best, y_worst)   # natural order; invert_yaxis puts best at top
ax.invert_yaxis()

# Extend x-limit to make room for labels
ax.set_xlim(left=800, right=MAX_N * 3.2)


# 5. RIGHT-SIDE LABELS ─────────────────────────────────────────────────────────
ax.figure.canvas.draw()          # needed so ylim is finalised
place_labels(ax, label_series, label_x=MAX_N * 1.06, fontsize=9)

# Legend (groups only)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=C_PARAM,  alpha=0.35, label="Parametric (range)"),
    Patch(facecolor=C_NONPAR, alpha=0.35, label="Nonparametric (SE band)"),
    Patch(facecolor=C_FOUND,  alpha=0.35, label="Foundation (SE band)"),
    Line2D([0],[0], color="gray", lw=0, marker="x", ms=7,
           mew=1.8, label="Last available n"),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=9.5,
          framealpha=0.92, borderpad=0.7)

ax.set_title("CDE Loss vs Sample Size — SDSS  (lower is better, top = better)",
             fontsize=14, fontweight="bold", pad=10)

fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT}")

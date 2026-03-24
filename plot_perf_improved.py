"""
Improved perf-vs-n plots for SDSS scaling study.

Generates:
  - Three individual figures (CDE Loss, CRPS, Log-Likelihood)
  - One combined 3-panel figure for the paper
    (CDE Loss on top, CRPS and Log-Likelihood on the bottom row)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

RESULTS = "results_real/sdss_scaling/results.json"
OUTDIR  = "results_real/sdss_scaling"

# ── metric configs ───────────────────────────────────────────────────────────
METRICS = [
    {"key": "CDE_loss",  "label": "CDE Loss",       "lower_is_better": True},
    {"key": "CRPS",      "label": "CRPS",            "lower_is_better": True},
    {"key": "log_lik",   "label": "Log-Likelihood",  "lower_is_better": False},
]

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

NONPARAMETRIC = {"MDN", "Flow-Spline", "BART-Homo", "BART-Hetero",
                 "FlexCode-RF", "FlexZBoost", "CatMLP", "Quantile-Tree"}

# ── colors ───────────────────────────────────────────────────────────────────
C_PARAM   = "#1b9e77"
C_NONPAR  = "#377eb8"
C_FOUND   = "#e67e22"

FOUND_STYLES = {
    "TabPFN-Native":    {"ls": "-",  "marker": "o"},
    "TabPFN-2.5":       {"ls": "-",  "marker": "o"},
    "RealTabPFN-2.5":   {"ls": "-.", "marker": "^"},
    "TabICL-Quantiles": {"ls": ":",  "marker": "D"},
}
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

# ── load data ────────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    raw = json.load(f)

def parse_n(ds):
    return int(ds.split("-")[1])

datasets = sorted(raw.keys(), key=parse_n)
all_ns   = [parse_n(ds) for ds in datasets]

def get_series(method, metric_key):
    vals, ses, ns = [], [], []
    for ds, n in zip(datasets, all_ns):
        r = raw[ds].get(method)
        if r and r.get(metric_key) is not None:
            vals.append(float(r[metric_key]))
            ses.append(float(r.get(f"{metric_key}_se") or 0))
            ns.append(n)
    return ns, vals, ses


def place_labels(ax, series, label_x, fontsize, min_gap):
    """Stagger right-side labels to avoid overlap."""
    if not series:
        return
    series = sorted(series, key=lambda t: t[0])
    placed = []
    for y_raw, x_src, label, col in series:
        y = y_raw
        for _ in range(len(placed) + 1):
            for py in placed:
                if abs(y - py) < min_gap:
                    y = py - min_gap if y < py else py + min_gap
        placed.append(y)
        needs_arrow = abs(y - y_raw) > min_gap * 0.08 or x_src < label_x * 0.92
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


# ── core panel drawing ───────────────────────────────────────────────────────
def draw_panel(ax, metric_key, metric_label, lower_is_better, *,
               fs_tick=18, fs_axis=22, fs_label=14, fs_annotation=14,
               fs_legend=16, show_xlabel=True, show_legend=True,
               panel_tag=None, label_gap_factor=1.0, xtick_rotation=0,
               foundation_inline=True):
    """
    Draw one metric panel into `ax`.
    fs_* parameters control font sizes so the function works for both
    standalone and multi-panel figures.
    """

    def is_better(a, b):
        return a < b if lower_is_better else a > b

    label_series = []

    # 1. PARAMETRIC BAND ──────────────────────────────────────────────────────
    param_by_n = {n: [] for n in all_ns}
    best_base_ns, best_base_vals, best_base_ses = None, None, None
    best_base_name, best_base_val_last = None, None

    for m in sorted(PARAMETRIC_BASE):
        ns, vals, ses = get_series(m, metric_key)
        if not ns:
            continue
        for n, v in zip(ns, vals):
            param_by_n[n].append(v)
        if best_base_val_last is None or is_better(vals[-1], best_base_val_last):
            best_base_ns, best_base_vals, best_base_ses = ns, vals, ses
            best_base_name, best_base_val_last = m, vals[-1]

    band_ns   = sorted(n for n, vs in param_by_n.items() if vs)
    band_mins = [min(param_by_n[n]) for n in band_ns]
    band_maxs = [max(param_by_n[n]) for n in band_ns]

    ax.fill_between(band_ns, band_mins, band_maxs,
                    color=C_PARAM, alpha=0.18, zorder=1)

    band_mid_idx = len(band_ns) // 2
    band_mid_x = band_ns[band_mid_idx]
    band_mid_y = (band_mins[band_mid_idx] + band_maxs[band_mid_idx]) / 2
    ax.annotate("Other parametric\nmethods",
                xy=(band_mid_x, band_mid_y),
                fontsize=fs_annotation, color=C_PARAM,
                va="center", ha="center",
                fontweight="bold", fontstyle="italic",
                alpha=0.70, zorder=2)

    if best_base_ns:
        ax.plot(best_base_ns, best_base_vals,
                color=C_PARAM, lw=2.4, ls="-", zorder=4)
        arr_ns = np.array(best_base_ns)
        arr_v  = np.array(best_base_vals)
        arr_se = np.array(best_base_ses)
        ax.fill_between(arr_ns, arr_v - arr_se, arr_v + arr_se,
                        color=C_PARAM, alpha=0.20, zorder=3)
        label_series.append((best_base_vals[-1], best_base_ns[-1],
                             disp(best_base_name), C_PARAM))
        if best_base_ns[-1] < MAX_N:
            ax.plot(best_base_ns[-1], best_base_vals[-1], "x",
                    color=C_PARAM, ms=7, mew=1.8, zorder=6)

    # 2. NONPARAMETRIC ────────────────────────────────────────────────────────
    nonpar_all = [(m, *get_series(m, metric_key))
                  for m in NONPARAMETRIC if get_series(m, metric_key)[0]]
    nonpar_all.sort(key=lambda t: t[2][-1],
                    reverse=not lower_is_better)
    top3_nonpar = {t[0] for t in nonpar_all[:3]}

    for i, (m, ns, vals, ses) in enumerate(nonpar_all):
        ls = NONPAR_LINES[i % len(NONPAR_LINES)]
        is_top = m in top3_nonpar
        ax.plot(ns, vals, color=C_NONPAR,
                lw=2.0 if is_top else 1.6,
                ls=ls, zorder=5, alpha=0.88 if is_top else 0.45)
        if is_top:
            arr = np.array(vals); arr_se = np.array(ses)
            ax.fill_between(ns, arr - arr_se, arr + arr_se,
                            color=C_NONPAR, alpha=0.07, zorder=4)
            label_series.append((vals[-1], ns[-1], disp(m), C_NONPAR))
        if ns[-1] < MAX_N:
            ax.plot(ns[-1], vals[-1], "x", color=C_NONPAR,
                    ms=7 if is_top else 5, mew=1.8 if is_top else 1.2,
                    zorder=7, alpha=0.88 if is_top else 0.45)

    # 3. FOUNDATION ───────────────────────────────────────────────────────────
    found_methods = sorted(m for m in FOUNDATIONAL
                           if get_series(m, metric_key)[0])
    merged_label_added = False
    found_term_labels = []
    for m in found_methods:
        ns, vals, ses = get_series(m, metric_key)
        sty = FOUND_STYLES.get(m, {"ls": "-", "marker": "o"})
        ax.plot(ns, vals, color=C_FOUND, lw=3.2, ls=sty["ls"],
                marker=sty["marker"], ms=6, zorder=8, alpha=0.95)
        arr = np.array(vals); arr_se = np.array(ses)
        ax.fill_between(ns, arr - arr_se, arr + arr_se,
                        color=C_FOUND, alpha=0.14, zorder=7)
        terminates = ns[-1] < MAX_N
        if terminates:
            ax.plot(ns[-1], vals[-1], "x", color=C_FOUND,
                    ms=9, mew=2.2, zorder=9)

        if m in FOUND_MERGE:
            if merged_label_added:
                continue
            lbl = FOUND_MERGE_LABEL
            merged_label_added = True
        else:
            lbl = disp(m)

        if terminates:
            found_term_labels.append((ns[-1], vals[-1], lbl))
        else:
            label_series.append((vals[-1], ns[-1], lbl, C_FOUND))

    # 4. AXES ─────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xticks(all_ns)
    ax.set_xticklabels([
        f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M"
        for n in all_ns
    ], fontsize=fs_tick, rotation=xtick_rotation,
       ha="right" if xtick_rotation else "center")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    if show_xlabel:
        ax.set_xlabel("Sample size (n)", fontsize=fs_axis)
    ax.set_ylabel(metric_label, fontsize=fs_axis)
    ax.tick_params(labelsize=fs_tick)
    ax.grid(alpha=0.25, linewidth=0.8)

    labeled_vals = list(best_base_vals) if best_base_vals else []
    labeled_vals += [v for m, ns, vals, ses in nonpar_all
                     if m in top3_nonpar for v in vals]
    labeled_vals += [v for m in found_methods
                     for v in get_series(m, metric_key)[1]]
    y_lo = min(labeled_vals) - 0.4 * (max(labeled_vals) - min(labeled_vals) + 1e-9)
    y_hi = max(labeled_vals) + 0.4 * (max(labeled_vals) - min(labeled_vals) + 1e-9)

    if lower_is_better:
        ax.set_ylim(y_lo, y_hi)
        ax.invert_yaxis()
    else:
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlim(left=800, right=MAX_N * 2.6)

    # 5a. INLINE FOUNDATION LABELS ────────────────────────────────────────────
    y_range = abs(y_hi - y_lo)
    found_gap = y_range * 0.07 * label_gap_factor
    if foundation_inline:
        found_term_labels.sort(key=lambda t: t[1],
                               reverse=not lower_is_better)
        placed_ys = []
        for x_end, y_end, lbl in found_term_labels:
            x_txt = x_end * 1.4
            y_txt = y_end
            for _ in range(len(placed_ys) + 1):
                for py in placed_ys:
                    if abs(y_txt - py) < found_gap:
                        y_txt = py + found_gap if lower_is_better else py - found_gap
            placed_ys.append(y_txt)
            ax.annotate(
                lbl, xy=(x_end, y_end), xytext=(x_txt, y_txt),
                fontsize=fs_label, color=C_FOUND, va="center", ha="left",
                fontweight="bold", zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=C_FOUND, lw=0.8, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=C_FOUND, lw=0.6, alpha=0.5),
                annotation_clip=False,
            )
    else:
        # Add foundation terminal labels to the right-side label pool
        for x_end, y_end, lbl in found_term_labels:
            label_series.append((y_end, x_end, lbl, C_FOUND))

    # 5b. RIGHT-SIDE LABELS ───────────────────────────────────────────────────
    ax.figure.canvas.draw()
    min_gap = y_range * 0.06 * label_gap_factor
    place_labels(ax, label_series, label_x=MAX_N * 1.06,
                 fontsize=fs_label, min_gap=min_gap)

    # 6. LEGEND ───────────────────────────────────────────────────────────────
    if show_legend:
        legend_handles = [
            Patch(facecolor=C_PARAM,  alpha=0.35, label="Parametric (range)"),
            Patch(facecolor=C_NONPAR, alpha=0.35, label="Nonparametric (SE band)"),
            Patch(facecolor=C_FOUND,  alpha=0.35, label="Foundation (SE band)"),
            Line2D([0],[0], color="gray", lw=0, marker="x", ms=8,
                   mew=2.0, label="Last available n"),
        ]
        legend_loc = "lower left" if lower_is_better else "upper left"
        ax.legend(handles=legend_handles, loc=legend_loc, fontsize=fs_legend,
                  framealpha=0.92, borderpad=0.7)

    # 7. PANEL TAG (e.g. "(a)") ───────────────────────────────────────────────
    if panel_tag is not None:
        ax.text(0.02, 0.97, panel_tag, transform=ax.transAxes,
                fontsize=fs_label + 2, fontweight="bold", va="top")

    return metric_label   # for title use


# ── standalone figures ───────────────────────────────────────────────────────
def make_plot(metric_key, metric_label, lower_is_better):
    fig, ax = plt.subplots(figsize=(13, 9))
    draw_panel(ax, metric_key, metric_label, lower_is_better,
               fs_tick=18, fs_axis=22, fs_label=14, fs_annotation=14,
               fs_legend=16, show_xlabel=True, show_legend=True)
    ax.set_title(f"{metric_label} vs Sample Size — SDSS  (top = better)",
                 fontsize=21, fontweight="bold", pad=12)
    fig.tight_layout()
    stem = f"perf_vs_n_{metric_key.lower()}_improved"
    out_png = os.path.join(OUTDIR, f"{stem}.png")
    out_pdf = os.path.join(OUTDIR, f"{stem}.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


# ── combined 3-panel figure ──────────────────────────────────────────────────
def make_combined_figure():
    # Layout: CDE Loss on top (~80% width, centred), CRPS + Log-Lik on bottom row.
    # Use a 2×2 GridSpec so each bottom panel gets half the figure with a
    # proper wspace gap (~25% of panel width ≈ 2.7 in) — wide enough for the
    # right-side label boxes not to bleed into the adjacent panel.
    fig = plt.figure(figsize=(26, 22))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.25)
    ax_top = fig.add_subplot(gs[0, :])   # full-width; narrowed below
    ax_bl  = fig.add_subplot(gs[1, 0])
    ax_br  = fig.add_subplot(gs[1, 1])

    panels = [
        (ax_top, "CDE_loss",  "CDE Loss",      True,  True,  True,  "(a)", 1.0,  0, True),
        (ax_bl,  "CRPS",      "CRPS",          True,  True,  False, "(b)", 1.5, 30, False),
        (ax_br,  "log_lik",   "Log-Likelihood",False, True,  False, "(c)", 1.5, 30, False),
    ]

    for ax, mk, ml, lib, show_xl, show_leg, tag, lgf, xrot, fi in panels:
        draw_panel(ax, mk, ml, lib,
                   fs_tick=20, fs_axis=24, fs_label=16, fs_annotation=16,
                   fs_legend=18, show_xlabel=show_xl, show_legend=show_leg,
                   panel_tag=tag, label_gap_factor=lgf, xtick_rotation=xrot,
                   foundation_inline=fi)
        ax.set_title(f"{ml}  (top = better)", fontsize=23,
                     fontweight="bold", pad=8)

    # Narrow panel (a) to ~80% of row width, centred
    pos = ax_top.get_position()
    inset = pos.width * 0.10   # 10% inset each side → 80% width
    ax_top.set_position([pos.x0 + inset, pos.y0, pos.width - 2 * inset, pos.height])

    out_png = os.path.join(OUTDIR, "perf_vs_n_combined_improved.png")
    out_pdf = os.path.join(OUTDIR, "perf_vs_n_combined_improved.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


# ── generate all outputs ─────────────────────────────────────────────────────
for cfg in METRICS:
    make_plot(cfg["key"], cfg["label"], cfg["lower_is_better"])

make_combined_figure()

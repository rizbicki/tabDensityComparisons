"""
Plots and tables for the post-hoc PIT recalibration analysis.

Reads ``recal_per_cell.csv`` (from ``run_recalibration_analysis.py``) and
produces scale-free summaries of recalibration's effect, by method group
(Foundation / MDN / other baselines):

  * calibration *levels* before vs after (PIT-KS, |coverage-0.90|): these are
    unit-free and directly comparable across datasets;
  * *win-rates* (fraction of dataset cells where a metric improves) for all six
    metrics -- the right scale-free summary for the proper scoring rules, whose
    raw units differ by orders of magnitude across datasets;
  * average *ranks* before vs after (across methods within each dataset), which
    is how the paper already reports calibration and which answers the
    referee's question of whether recalibration changes the *relative* standing
    of foundation models.

USAGE:
  python plot_recalibration.py [--in-dir results_real/recalibration]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

FOUNDATION = {"TabPFN-2.5", "RealTabPFN-2.5", "TabICL-Quantiles"}
GROUP_STYLE = {
    "Foundation": "#e67e22",
    "MDN": "#8da0cb",
    "Other baselines": "#777777",
}

# Parametric / nonparametric split used in the perf-vs-n panels, reused for the
# rank dumbbell so its colors match the rest of the paper.
NONPARAMETRIC = {"MDN", "Flow-Spline", "BART-Homo", "BART-Hetero",
                 "FlexCode-RF", "FlexZBoost", "CatMLP", "Quantile-Tree"}
C_FOUND, C_PARAM, C_NONPAR = "#e67e22", "#1b9e77", "#377eb8"


def _group3(method):
    if method in FOUNDATION:
        return "foundational"
    return "nonparametric" if method in NONPARAMETRIC else "parametric"


def _rank_color(method):
    return {"foundational": C_FOUND,
            "nonparametric": C_NONPAR,
            "parametric": C_PARAM}[_group3(method)]

# metric -> (display label, better direction)
METRIC_INFO = {
    "PIT_KS": ("PIT KS statistic", "lower"),
    "coverage_abs_err": ("|coverage - 0.90|", "lower"),
    "CRPS": ("CRPS", "lower"),
    "CDE_loss": ("CDE loss", "lower"),
    "log_lik": ("Log-likelihood", "higher"),
    "interval_width": ("90% interval width", "none"),
}
CALIB_METRICS = ["PIT_KS", "coverage_abs_err"]


def _group(method):
    if method in FOUNDATION:
        return "Foundation"
    return "MDN" if method == "MDN" else "Other baselines"


def _add_derived(df):
    df = df.copy()
    df["group3"] = df["method"].map(_group)
    for suff in ("before", "after"):
        df[f"coverage_abs_err__{suff}"] = (df[f"coverage_90__{suff}"] - 0.90).abs()
    return df.dropna(subset=["PIT_KS__after"])  # drop skipped (n=50) cells


def _improved(metric, before, after):
    direction = METRIC_INFO[metric][1]
    return (after > before) if direction == "higher" else (after < before)


# --------------------------------------------------------------------------- #
# aggregations
# --------------------------------------------------------------------------- #
def group_levels(df, metric):
    """Mean before/after of a (scale-free) metric per (group, n)."""
    rows = []
    for (grp, n), g in df.groupby(["group3", "n"], observed=True):
        rows.append(dict(group=grp, n=n,
                         before=g[f"{metric}__before"].mean(),
                         after=g[f"{metric}__after"].mean()))
    return pd.DataFrame(rows)


def group_winrate(df, metric):
    """Fraction of cells improved per (group, n)."""
    rows = []
    for (grp, n), g in df.groupby(["group3", "n"], observed=True):
        imp = _improved(metric, g[f"{metric}__before"], g[f"{metric}__after"])
        rows.append(dict(group=grp, n=n, win_rate=float(imp.mean()), n_cells=len(g)))
    return pd.DataFrame(rows)


def group_ranks(df, metric):
    """Average rank (1 = best) within each dataset, before vs after, per group."""
    direction = METRIC_INFO[metric][1]
    sign = -1.0 if direction == "higher" else 1.0  # rank so 1 = best
    parts = []
    for (ds, n), cell in df.groupby(["dataset", "n"], observed=True):
        cell = cell.copy()
        cell["rank_before"] = rankdata(sign * cell[f"{metric}__before"].values)
        cell["rank_after"] = rankdata(sign * cell[f"{metric}__after"].values)
        parts.append(cell[["group3", "n", "rank_before", "rank_after"]])
    allc = pd.concat(parts, ignore_index=True)
    return (allc.groupby(["group3", "n"], observed=True)[["rank_before", "rank_after"]]
            .mean().reset_index().rename(columns={"group3": "group"}))


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_levels(ax, df, metric):
    label = METRIC_INFO[metric][0]
    gl = group_levels(df, metric)
    for grp, color in GROUP_STYLE.items():
        s = gl[gl["group"] == grp].sort_values("n")
        if s.empty:
            continue
        ax.plot(s["n"], s["before"], color=color, ls="--", lw=1.6, alpha=0.55,
                marker="o", mfc="white", ms=6, label=f"{grp} (before)")
        ax.plot(s["n"], s["after"], color=color, ls="-", lw=2.6,
                marker="o", ms=6, label=f"{grp} (after)")
    ax.set_xscale("log")
    ax.set_xlabel("n (training size)")
    ax.set_ylabel(label + "  (lower better)")
    ax.set_title(label)
    ax.grid(True, alpha=0.25)


def plot_winrate(ax, df, metric):
    label = METRIC_INFO[metric][0]
    wr = group_winrate(df, metric)
    for grp, color in GROUP_STYLE.items():
        s = wr[wr["group"] == grp].sort_values("n")
        if s.empty:
            continue
        ax.plot(s["n"], s["win_rate"], color=color, lw=2.4, marker="o", ms=6, label=grp)
    ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("n (training size)")
    ax.set_ylabel("fraction of datasets improved")
    ax.set_title(label)
    ax.grid(True, alpha=0.25)


def plot_ranks(ax, df, metric):
    label = METRIC_INFO[metric][0]
    gr = group_ranks(df, metric)
    for grp, color in GROUP_STYLE.items():
        s = gr[gr["group"] == grp].sort_values("n")
        if s.empty:
            continue
        ax.plot(s["n"], s["rank_before"], color=color, ls="--", lw=1.6, alpha=0.55,
                marker="o", mfc="white", ms=6, label=f"{grp} (before)")
        ax.plot(s["n"], s["rank_after"], color=color, ls="-", lw=2.6,
                marker="o", ms=6, label=f"{grp} (after)")
    ax.set_xscale("log")
    ax.invert_yaxis()  # rank 1 (best) at top
    ax.set_xlabel("n (training size)")
    ax.set_ylabel("average rank (1 = best)")
    ax.set_title(f"{label}: average rank")
    ax.grid(True, alpha=0.25)


def make_calibration_figure(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, met in zip(axes, CALIB_METRICS):
        plot_levels(ax, df, met)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Calibration before vs after post-hoc PIT recalibration "
                 "(dashed = before, solid = after)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "recal_calibration_levels.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def make_winrate_panel(df, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, met in zip(axes.ravel(), METRIC_INFO):
        plot_winrate(ax, df, met)
    axes.ravel()[0].legend(fontsize=9, loc="best")
    fig.suptitle("Recalibration win-rate: fraction of datasets improved "
                 "(0.5 = no net effect)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "recal_winrate_all_metrics.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def make_rank_figure(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, met in zip(axes, CALIB_METRICS):
        plot_ranks(ax, df, met)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Relative calibration standing before vs after recalibration",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "recal_calibration_ranks.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def calibration_table(df):
    rows = []
    for met in CALIB_METRICS:
        lv = group_levels(df, met)
        wr = group_winrate(df, met)
        m = lv.merge(wr, on=["group", "n"])
        m["metric"] = met
        m["delta"] = m["after"] - m["before"]
        rows.append(m)
    return (pd.concat(rows, ignore_index=True)
            .sort_values(["metric", "n", "group"])
            [["metric", "n", "group", "n_cells", "before", "after", "delta", "win_rate"]])


def winrate_table(df):
    rows = []
    for met in METRIC_INFO:
        wr = group_winrate(df, met)
        wr["metric"] = met
        rows.append(wr)
    out = pd.concat(rows, ignore_index=True)
    return out.pivot_table(index=["metric", "group"], columns="n",
                           values="win_rate").reset_index()


def write_markdown(tab, path):
    cols = list(tab.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in tab.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines))


def write_latex(tab, path):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Metric & $n$ & Group & Before & After & $\Delta$ & Win rate \\",
             r"\midrule"]
    pretty = {"PIT_KS": "PIT KS", "coverage_abs_err": r"$|\mathrm{cov}-0.90|$"}
    for met in CALIB_METRICS:
        for _, r in tab[tab["metric"] == met].iterrows():
            lines.append(f"{pretty[met]} & {int(r['n'])} & {r['group']} & "
                         f"{r['before']:.3f} & {r['after']:.3f} & "
                         f"{r['delta']:+.3f} & {r['win_rate']*100:.0f}\\% \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}"]
    path.write_text("\n".join(lines))


def method_levels(df, metric):
    """Per-method mean before/after of a metric, at each n."""
    g = (df.groupby(["method", "group3", "n"], observed=True)
         [[f"{metric}__before", f"{metric}__after"]].mean().reset_index())
    return g.rename(columns={f"{metric}__before": "before", f"{metric}__after": "after"})


def _method_order(df, metric, ref_n):
    """Foundation, then MDN, then other baselines sorted by before-level."""
    lv = method_levels(df, metric)
    ref = lv[lv["n"] == ref_n].set_index("method")
    found = [m for m in FOUNDATION if m in ref.index]
    base = [m for m in ref.index if m not in FOUNDATION and m != "MDN"]
    base = sorted(base, key=lambda m: ref.loc[m, "before"])
    order = base + (["MDN"] if "MDN" in ref.index else []) + found
    return order  # bottom -> top on the y-axis


def make_dumbbell(df, out_dir, metric="PIT_KS", ns=(1000, 5000, 20000)):
    """Per-method before->after movement (dumbbell), one panel per n.

    Shows every method individually -- direction and magnitude of the change --
    so the *differential* improvement across methods is read off directly.
    """
    lv = method_levels(df, metric)
    ns = [n for n in ns if n in set(lv["n"])]
    order = _method_order(df, metric, ns[len(ns) // 2])
    ypos = {m: i for i, m in enumerate(order)}
    fig, axes = plt.subplots(1, len(ns), figsize=(5.2 * len(ns), 9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, ns):
        sub = lv[lv["n"] == n]
        for _, r in sub.iterrows():
            if r["method"] not in ypos:
                continue
            y = ypos[r["method"]]
            c = GROUP_STYLE[_group(r["method"])]
            ax.plot([r["before"], r["after"]], [y, y], color=c, lw=2.0, alpha=0.6,
                    zorder=2)
            ax.scatter(r["before"], y, facecolors="white", edgecolors=c, s=55,
                       lw=1.8, zorder=3)
            ax.scatter(r["after"], y, color=c, s=55, zorder=4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_title(f"n = {n}")
        ax.set_xlabel(METRIC_INFO[metric][0] + "  (lower = better)")
        ax.grid(True, axis="x", alpha=0.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", mfc="white", color=c, lw=0, mec=c,
                      label=f"{g} (before)") for g, c in GROUP_STYLE.items()]
    handles += [Line2D([0], [0], marker="o", color=c, lw=0, label=f"{g} (after)")
                for g, c in GROUP_STYLE.items()]
    axes[0].legend(handles=handles, fontsize=7, loc="lower right", ncol=1)
    fig.suptitle(f"Per-method effect of recalibration on {METRIC_INFO[metric][0]} "
                 f"(open = before, filled = after)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"recal_dumbbell_{metric}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def make_calibration_group_dumbbell(df, out_dir):
    """Group-level before->after dumbbell, mirroring Table~\\ref{tab:recal_calibration}.

    One panel per calibration metric; within a panel each (group, n) cell is a
    before->after segment (open = before, filled = after). Reproduces exactly the
    means tabulated by ``group_levels`` -- the same numbers as the first table.
    """
    groups = list(GROUP_STYLE)  # Foundation, MDN, Other baselines
    fig, axes = plt.subplots(1, len(CALIB_METRICS), figsize=(6.5 * len(CALIB_METRICS), 5.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    ns = sorted(df["n"].unique())
    # y layout: blocks of n (top = largest), 3 group rows within each block.
    yticks, ylabels = [], []
    ypos = {}  # (n, group) -> y
    row = 0
    for n in ns:  # build bottom -> top so largest n ends on top after layout
        for gi, grp in enumerate(reversed(groups)):
            ypos[(n, grp)] = row
            row += 1
        yticks.append(row - len(groups) + (len(groups) - 1) / 2.0)
        ylabels.append(f"$n={n:,}$")
        row += 1  # gap between n blocks

    for ax, met in zip(axes, CALIB_METRICS):
        gl = group_levels(df, met)
        for _, r in gl.iterrows():
            key = (r["n"], r["group"])
            if key not in ypos:
                continue
            y = ypos[key]
            c = GROUP_STYLE[r["group"]]
            ax.plot([r["before"], r["after"]], [y, y], color=c, lw=2.4, alpha=0.6,
                    zorder=2)
            ax.scatter(r["before"], y, facecolors="white", edgecolors=c, s=70,
                       lw=1.8, zorder=3)
            ax.scatter(r["after"], y, color=c, s=70, zorder=4)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=11)
        ax.set_xlabel(METRIC_INFO[met][0] + "  (lower = better)", fontsize=12)
        ax.set_title(METRIC_INFO[met][0], fontsize=14)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlim(left=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", mfc="white", color=c, lw=0, mec=c,
                      label=f"{g} (before)") for g, c in GROUP_STYLE.items()]
    handles += [Line2D([0], [0], marker="o", color=c, lw=0, label=f"{g} (after)")
                for g, c in GROUP_STYLE.items()]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Calibration before → after recalibration, by method group "
                 "(open → filled)", fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    path = out_dir / "recal_calibration_group_dumbbell.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def make_calibration_method_dumbbell(df, out_dir, ns=(500, 5000, 20000)):
    """Per-method calibration levels before->after: rows = metric, cols = n.

    All-methods, shared-order layout (like the rank dumbbell) but on the raw
    (scale-free) calibration scales rather than ranks. Each method's actual
    before->after movement on PIT KS and |coverage-0.90| is read off directly,
    at three sample sizes, generalizing the group-level table to every method.
    """
    ns = [n for n in ns if n in set(df["n"])]
    ref_n = ns[len(ns) // 2]
    # Group parametric -> nonparametric -> foundational (bottom -> top), sorted
    # by PIT-KS before-level within each group, matching the rank dumbbell.
    lv_ref = method_levels(df, "PIT_KS")
    ref = lv_ref[lv_ref["n"] == ref_n].set_index("method")
    group_rank = {"parametric": 0, "nonparametric": 1, "foundational": 2}
    order = sorted(ref.index,
                   key=lambda m: (group_rank[_group3(m)], ref.loc[m, "before"], m))
    ypos = {m: i for i, m in enumerate(order)}

    nrow, ncol = len(CALIB_METRICS), len(ns)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 5.5 * nrow),
                             sharey=True)
    axes = np.atleast_2d(axes)
    for i, met in enumerate(CALIB_METRICS):
        lv = method_levels(df, met)
        for j, n in enumerate(ns):
            ax = axes[i, j]
            for _, r in lv[lv["n"] == n].iterrows():
                if r["method"] not in ypos:
                    continue
                y, c = ypos[r["method"]], _rank_color(r["method"])
                ax.plot([r["before"], r["after"]], [y, y], color=c, lw=2.3, alpha=0.6, zorder=2)
                ax.scatter(r["before"], y, facecolors="white", edgecolors=c, s=70, lw=1.9, zorder=3)
                ax.scatter(r["after"], y, color=c, s=70, zorder=4)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=12)
            ax.set_xlim(left=0)
            ax.grid(True, axis="x", alpha=0.25)
            if i == 0:
                ax.set_title(f"$n = {n:,}$", fontsize=16)
            if i == nrow - 1:
                ax.set_xlabel(METRIC_INFO[met][0] + "  (lower = better)", fontsize=13)
        axes[i, 0].set_ylabel(METRIC_INFO[met][0], fontsize=15)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", mfc="white", color=c, lw=0, mec=c,
                      label=f"{lab} (before)")
               for lab, c in [("Foundation", C_FOUND), ("Parametric", C_PARAM),
                              ("Nonparametric", C_NONPAR)]]
    handles += [Line2D([0], [0], marker="o", color=c, lw=0, label=f"{lab} (after)")
                for lab, c in [("Foundation", C_FOUND), ("Parametric", C_PARAM),
                               ("Nonparametric", C_NONPAR)]]
    fig.legend(handles=handles, fontsize=12, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Per-method calibration before → after recalibration "
                 "(open → filled)", fontsize=19)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    path = out_dir / "recal_calibration_method_dumbbell.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _avg_rank(df, metric, ns):
    """Average rank (1 = best) per method, before and after, pooled over ns."""
    sign = -1.0 if METRIC_INFO[metric][1] == "higher" else 1.0
    sub = df[df["n"].isin(ns)]
    parts = []
    for _, cell in sub.groupby(["dataset", "n"], observed=True):
        cell = cell.copy()
        cell["rb"] = rankdata(sign * cell[f"{metric}__before"].values)
        cell["ra"] = rankdata(sign * cell[f"{metric}__after"].values)
        parts.append(cell[["method", "group3", "rb", "ra"]])
    allc = pd.concat(parts, ignore_index=True)
    return allc.groupby(["method", "group3"], observed=True)[["rb", "ra"]].mean().reset_index()


def make_rank_dumbbell(df, out_dir, ns=(5000, 10000, 20000)):
    """Average-rank before->after per method, faceted by metric (shared method order).

    Scale-free, so it works for every metric. Answers 'do foundation methods stay
    better?': their rows should remain on the left (best ranks) on CDE loss /
    log-likelihood / CRPS after recalibration.
    """
    ns = [n for n in ns if n in set(df["n"])]
    ref = _avg_rank(df, "CDE_loss", ns).set_index("method")
    # Match the rank plots' ordering: grouped parametric -> nonparametric ->
    # foundational (bottom -> top), sorted by average rank within each group.
    group_rank = {"parametric": 0, "nonparametric": 1, "foundational": 2}
    order = sorted(ref.index,
                   key=lambda m: (group_rank[_group3(m)], ref.loc[m, "rb"], m))
    ypos = {m: i for i, m in enumerate(order)}

    metrics = list(METRIC_INFO)
    fig, axes = plt.subplots(3, 2, figsize=(15, 22), sharey=True)
    for ax, met in zip(axes.ravel(), metrics):
        for _, r in _avg_rank(df, met, ns).iterrows():
            if r["method"] not in ypos:
                continue
            y, c = ypos[r["method"]], _rank_color(r["method"])
            ax.plot([r["rb"], r["ra"]], [y, y], color=c, lw=2.5, alpha=0.6, zorder=2)
            ax.scatter(r["rb"], y, facecolors="white", edgecolors=c, s=80, lw=2.0, zorder=3)
            ax.scatter(r["ra"], y, color=c, s=80, zorder=4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=15)
        ax.set_title(METRIC_INFO[met][0], fontsize=19)
        ax.set_xlabel("average rank (1 = best, left = better)", fontsize=16)
        ax.tick_params(axis="x", labelsize=15)
        ax.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Average rank before → after recalibration (open → filled)",
                 fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "recal_rank_dumbbell.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def foundation_best_fraction(df, metric):
    """Fraction of datasets where the best foundation model beats the best
    non-foundation method, before vs after recalibration, per n."""
    higher = METRIC_INFO[metric][1] == "higher"
    rows = []
    for (_, n), cell in df.groupby(["dataset", "n"], observed=True):
        f = cell[cell["group3"] == "Foundation"]
        o = cell[cell["group3"] != "Foundation"]
        if f.empty or o.empty:
            continue
        for suff in ("before", "after"):
            fv, ov = f[f"{metric}__{suff}"], o[f"{metric}__{suff}"]
            win = (fv.max() > ov.max()) if higher else (fv.min() < ov.min())
            rows.append(dict(n=n, phase=suff, win=float(win)))
    return (pd.DataFrame(rows).groupby(["n", "phase"])["win"].mean().reset_index())


def foundation_best_wide(df, metrics=("CDE_loss", "log_lik", "CRPS",
                                      "PIT_KS", "coverage_abs_err")):
    """Wide table: metric x n, value = (before, after) fraction foundation best."""
    ns = sorted(df["n"].unique())
    rows = []
    for met in metrics:
        piv = foundation_best_fraction(df, met).pivot(index="n", columns="phase",
                                                       values="win")
        row = {"metric": met}
        for n in ns:
            row[f"{n}_before"] = piv.loc[n, "before"]
            row[f"{n}_after"] = piv.loc[n, "after"]
        rows.append(row)
    return pd.DataFrame(rows), ns


def write_foundation_best_latex(df, path):
    pretty = {"CDE_loss": "CDE loss", "log_lik": "Log-likelihood", "CRPS": "CRPS",
              "PIT_KS": "PIT KS", "coverage_abs_err": r"$|\mathrm{cov}-0.90|$"}
    wide, ns = foundation_best_wide(df, tuple(pretty))
    head = "Metric & " + " & ".join(f"$n={n:,}$" for n in ns) + r" \\"
    lines = [r"\begin{tabular}{l" + "c" * len(ns) + "}", r"\toprule", head, r"\midrule"]
    for _, r in wide.iterrows():
        cells = [f"{r[f'{n}_before']*100:.0f}$\\to${r[f'{n}_after']*100:.0f}\\%"
                 for n in ns]
        lines.append(f"{pretty[r['metric']]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


def make_foundation_best_panel(df, out_dir):
    metrics = list(METRIC_INFO)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, met in zip(axes.ravel(), metrics):
        g = foundation_best_fraction(df, met)
        for phase, ls in (("before", "--"), ("after", "-")):
            s = g[g["phase"] == phase].sort_values("n")
            ax.plot(s["n"], s["win"], ls=ls, lw=2.6, marker="o", color="#e67e22",
                    label=phase)
        ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.5)
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_title(METRIC_INFO[met][0])
        ax.set_xlabel("n (training size)")
        ax.set_ylabel("frac. datasets foundation is best")
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend(fontsize=10, loc="best")
    fig.suptitle("Does a foundation model stay best after recalibration?  "
                 "(best foundation vs best non-foundation, per dataset)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "recal_foundation_stays_best.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def make_scatter_before_after(df, out_dir, metric="CRPS", min_n=1000, log=True):
    """Before-vs-after scatter (one point per dataset x method x n), y=x line.

    Scale-free for positive metrics via log-log axes: points below the diagonal
    improved, distance from it is the magnitude. Group clustering relative to the
    diagonal shows who benefits.
    """
    sub = df[df["n"] >= min_n].copy()
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    for grp, c in GROUP_STYLE.items():
        s = sub[sub["group3"] == grp]
        ax.scatter(s[f"{metric}__before"], s[f"{metric}__after"], s=16, alpha=0.45,
                   color=c, label=f"{grp} (n_cells={len(s)})", edgecolors="none")
    lo = float(np.nanmin(sub[[f"{metric}__before", f"{metric}__after"]].values))
    hi = float(np.nanmax(sub[[f"{metric}__before", f"{metric}__after"]].values))
    if log:
        lo = max(lo, 1e-6)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, label="no change (y = x)")
    ax.set_xlabel(f"{METRIC_INFO[metric][0]} before")
    ax.set_ylabel(f"{METRIC_INFO[metric][0]} after")
    ax.set_title(f"{METRIC_INFO[metric][0]}: below the diagonal = improved "
                 f"(n $\\geq$ {min_n})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = out_dir / f"recal_scatter_{metric}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results_real/recalibration")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    df = _add_derived(pd.read_csv(in_dir / "recal_per_cell.csv"))

    figs = [make_calibration_group_dumbbell(df, in_dir),
            make_calibration_method_dumbbell(df, in_dir),
            make_dumbbell(df, in_dir, "PIT_KS"),
            make_dumbbell(df, in_dir, "coverage_abs_err"),
            make_rank_dumbbell(df, in_dir),
            make_winrate_panel(df, in_dir),
            make_scatter_before_after(df, in_dir, "PIT_KS", min_n=1000, log=False),
            make_scatter_before_after(df, in_dir, "CRPS", min_n=1000, log=True)]

    ctab = calibration_table(df)
    wtab = winrate_table(df)
    fwide, _ = foundation_best_wide(df)
    ctab.to_csv(in_dir / "recal_calibration_summary.csv", index=False)
    wtab.to_csv(in_dir / "recal_winrate_summary.csv", index=False)
    fwide.to_csv(in_dir / "recal_foundation_best.csv", index=False)
    write_latex(ctab, in_dir / "recal_calibration_table.tex")
    write_foundation_best_latex(df, in_dir / "recal_foundation_best_table.tex")
    write_markdown(ctab, in_dir / "recal_calibration_summary.md")

    print("Wrote figures:\n  " + "\n  ".join(str(p) for p in figs))
    print("Wrote tables:\n  " + "\n  ".join(str(in_dir / f) for f in (
        "recal_calibration_summary.csv", "recal_winrate_summary.csv",
        "recal_foundation_best.csv", "recal_calibration_table.tex",
        "recal_foundation_best_table.tex")))

    print("\n=== Calibration levels & win-rate (mean over datasets) ===")
    print(ctab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Fraction of datasets where a foundation model is best "
          "(before -> after) ===")
    print(fwide.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

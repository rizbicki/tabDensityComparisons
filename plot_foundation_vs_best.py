#!/usr/bin/env python3
"""
Figure showing cases where foundation models underperform:
estimated conditional densities for the best foundation model
vs the best non-foundation model, on selected test instances.

Each panel is zoomed to the region where the densities have mass.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────
CACHE_DIR = Path("results_real/cache")
OUT_DIR   = Path("results_real")
N_INST    = 5

FOUNDATION_COLOR = "#e67e22"
NONFOUND_COLOR   = "#377eb8"
OBS_COLOR        = "#333333"

FONT_SCALE = 1.30
TITLE_FONTSIZE = 12 * FONT_SCALE
TICK_FONTSIZE = 12 * FONT_SCALE
LEGEND_FONTSIZE = 14 * FONT_SCALE

ROWS = [
    ("Digits-50",            "TabPFN-2.5",       "CatMLP"),
    ("VideoTranscoding-50",  "RealTabPFN-2.5",   "LogNormal-Homo-Ridge"),
    ("BlackFriday-20000",    "TabPFN-2.5",       "CatMLP"),
]

# ── load CDE-loss scores ─────────────────────────────────────────────
with open("results_real/results.json") as f:
    all_results = json.load(f)


def _fmt_loss(ds_key, method):
    r = all_results[ds_key][method]
    v, se = r["CDE_loss"], r["CDE_loss_se"]
    if ds_key.startswith("BlackFriday"):
        return f"{v:.5f} \u00b1 {se:.5f}"
    if abs(v) >= 100:
        return f"{v:.1f} \u00b1 {se:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f} \u00b1 {se:.2f}"
    return f"{v:.4f} \u00b1 {se:.4f}"


def _pick_instances(z_te, n=N_INST):
    order = np.argsort(z_te)
    positions = np.linspace(0, len(order) - 1, n, dtype=int)
    return order[positions]


def _load(cache_name):
    return np.load(CACHE_DIR / f"{cache_name}.npz", allow_pickle=True)


def _zoom_xlim(zg_f, cde_f_i, zg_nf, cde_nf_i, z_obs, pad_frac=0.15):
    """Compute per-panel x-limits that cover the region with density mass."""
    # find range where either density is above 1% of its max
    thresh_f  = 0.01 * cde_f_i.max()  if cde_f_i.max()  > 0 else 0
    thresh_nf = 0.01 * cde_nf_i.max() if cde_nf_i.max() > 0 else 0

    mask_f  = cde_f_i  > thresh_f
    mask_nf = cde_nf_i > thresh_nf

    lo = min(
        zg_f[mask_f][0]   if mask_f.any()  else z_obs,
        zg_nf[mask_nf][0] if mask_nf.any() else z_obs,
        z_obs,
    )
    hi = max(
        zg_f[mask_f][-1]   if mask_f.any()  else z_obs,
        zg_nf[mask_nf][-1] if mask_nf.any() else z_obs,
        z_obs,
    )
    span = hi - lo if hi > lo else abs(z_obs) * 0.5 or 1.0
    return lo - pad_frac * span, hi + pad_frac * span


# ── build figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    len(ROWS), N_INST,
    figsize=(20, 3.8 * len(ROWS)),
    gridspec_kw={"hspace": 0.40, "wspace": 0.18},
)

for ri, (cache_key, found_m, nonfound_m) in enumerate(ROWS):
    data  = _load(cache_key)
    z_te  = data["z_te"]
    idxs  = _pick_instances(z_te, N_INST)
    cde_f  = data[f"cde_{found_m}"]
    zg_f   = data[f"zgrid_{found_m}"]
    cde_nf = data[f"cde_{nonfound_m}"]
    zg_nf  = data[f"zgrid_{nonfound_m}"]

    ds_display = cache_key.rsplit("-", 1)[0]
    n_display  = cache_key.rsplit("-", 1)[1]
    d_dim      = data["X_te"].shape[1]
    loss_f     = _fmt_loss(cache_key, found_m)
    loss_nf    = _fmt_loss(cache_key, nonfound_m)

    title_str = (
        f"{ds_display}  ($n$={n_display},  $d$={d_dim})        "
        f"{found_m}: {loss_f}     vs.     {nonfound_m}: {loss_nf}"
    )

    for col in range(N_INST):
        ax = axes[ri, col]
        idx = idxs[col]

        ax.fill_between(zg_f,  cde_f[idx],  color=FOUNDATION_COLOR, alpha=0.25)
        ax.fill_between(zg_nf, cde_nf[idx], color=NONFOUND_COLOR,   alpha=0.25)
        ax.plot(zg_f,  cde_f[idx],  color=FOUNDATION_COLOR, lw=2.4)
        ax.plot(zg_nf, cde_nf[idx], color=NONFOUND_COLOR,   lw=2.4)

        ax.axvline(z_te[idx], color=OBS_COLOR, ls="--", lw=1.5, alpha=0.7)

        # zoom x-axis to where the action is
        xlo, xhi = _zoom_xlim(zg_f, cde_f[idx], zg_nf, cde_nf[idx], z_te[idx])
        ax.set_xlim(xlo, xhi)

        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))

        if col == 0:
            ax.set_title(title_str, fontsize=TITLE_FONTSIZE, fontweight="bold",
                         loc="left", pad=12)

# ── shared legend ────────────────────────────────────────────────────
legend_elements = [
    Patch(facecolor=FOUNDATION_COLOR, alpha=0.45, edgecolor=FOUNDATION_COLOR,
          linewidth=2, label="Foundation model (best)"),
    Patch(facecolor=NONFOUND_COLOR, alpha=0.45, edgecolor=NONFOUND_COLOR,
          linewidth=2, label="Non-foundation model (best)"),
    Line2D([0], [0], color=OBS_COLOR, ls="--", lw=1.5, alpha=0.7,
           label="Observed $y$"),
]
fig.legend(handles=legend_elements, loc="upper center",
           ncol=3, fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.52, 1.01))

out_path = OUT_DIR / "foundation_failure_modes.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved \u2192 {out_path}")

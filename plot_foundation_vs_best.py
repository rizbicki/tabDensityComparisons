#!/usr/bin/env python3
"""
Generate a 4×4 figure comparing the best foundation model vs the best
non-foundation model on 4 test instances, for 4 dataset/n scenarios
chosen to highlight when foundation models win or lose by CDE loss.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────
FOUNDATION = {"TabPFN-Native", "TabPFN-2.5", "RealTabPFN-2.5", "TabICL-Quantiles"}

CASES = [
    # (cache_file, best_foundation, best_nonfoundation, row_label)
    ("Ailerons-50",        "TabPFN-2.5",       "Student-t-Ridge",
     "Ailerons ($n$=50)\nFoundation better"),
    ("Digits-50",          "TabPFN-2.5",       "CatMLP",
     "Digits ($n$=50)\nFoundation worse"),
    ("SDSS-20000",         "TabICL-Quantiles", "Flow-Spline",
     "SDSS ($n$=20 000)\nFoundation better"),
    ("HealthInsurance-20000", "TabPFN-2.5",    "Flow-Spline",
     "HealthInsurance ($n$=20 000)\nFoundation worse"),
]

CACHE_DIR = Path("results_real/cache")
OUT_DIR   = Path("results_real")
N_INST    = 4          # number of test instances per case

FOUNDATION_COLOR   = "#e67e22"
NONFOUND_COLOR     = "#377eb8"
OBS_COLOR          = "#222222"

# ── helpers ──────────────────────────────────────────────────────────

def _pick_instances(z_te, n=N_INST):
    """Pick n instances spread across the z range."""
    order = np.argsort(z_te)
    positions = np.linspace(0, len(order) - 1, n, dtype=int)
    return order[positions]


def _load(cache_name):
    """Load cache npz; cache_name without .npz extension."""
    # handle LaTeX escaping in CASES
    raw = cache_name.replace("\\", "")
    data = np.load(CACHE_DIR / f"{raw}.npz", allow_pickle=True)
    return data


# ── main ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    len(CASES), N_INST,
    figsize=(3.4 * N_INST, 3.0 * len(CASES)),
    constrained_layout=True,
)

for row, (cache_name, found_m, nonfound_m, label) in enumerate(CASES):
    data = _load(cache_name)
    z_te = data["z_te"]
    idxs = _pick_instances(z_te, N_INST)

    cde_f  = data[f"cde_{found_m}"]
    zg_f   = data[f"zgrid_{found_m}"]
    cde_nf = data[f"cde_{nonfound_m}"]
    zg_nf  = data[f"zgrid_{nonfound_m}"]

    for col, idx in enumerate(idxs):
        ax = axes[row, col]

        # estimated densities
        ax.plot(zg_f,  cde_f[idx],  color=FOUNDATION_COLOR, lw=2.0,
                label=found_m    if col == 0 else None)
        ax.plot(zg_nf, cde_nf[idx], color=NONFOUND_COLOR,   lw=2.0,
                label=nonfound_m if col == 0 else None)

        # observed y
        ax.axvline(z_te[idx], color=OBS_COLOR, ls="--", lw=1.2, alpha=0.8,
                   label="observed $y$" if col == 0 else None)

        ax.set_yticks([])
        if row == len(CASES) - 1:
            ax.set_xlabel("$y$", fontsize=11)
        if col == 0:
            ax.set_ylabel(label, fontsize=10)

        # per-instance title
        ax.set_title(f"instance {col + 1}", fontsize=9, color="grey")

    # legend on first column only
    axes[row, 0].legend(fontsize=7.5, loc="best", framealpha=0.85)

fig.suptitle(
    r"Best foundation model vs.\ best non-foundation model",
    fontsize=13, fontweight="bold", y=1.01,
)

out_path = OUT_DIR / "foundation_vs_best_density_examples.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")

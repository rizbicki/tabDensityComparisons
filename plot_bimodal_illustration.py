#!/usr/bin/env python
"""
Bimodal-DGP illustration with x-dependent means, variances, AND mixture
weights.  Layout: 3 rows (instances) × 3 cols (n = 50, 100, 1000).
Reading across a row shows convergence as n grows for a fixed test point.
Each row is surrounded by a coloured box to visually group panels of the
same instance.

Methods coloured by group (same scheme as perf_vs_n_foundational plots):
  Parametric   : LinearGauss-Homo
  Nonparametric: MDN, Flow-Spline
  Foundational : TabPFN 2.5, TabICL Quantiles

Output: results_simulated/bimodal_illustration.png
"""

import sys, warnings
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    FlexCodeEstimator, RFFlexRegressor,
    normalizing_flow_density_tuned,
)
from models.native import tabpfn_native_density

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print("- TabPFN not available")



# ── DGP: bimodal with x-dependent mean, variance, AND mixture weight ─────────
_MAX_N = 25_000

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_bimodal_full(n=1000, d=5, seed=42):
    """Bimodal DGP where mean, variance, and mixture weight all depend on X.

    mu1(x) =  2*x0 + x1          mu2(x) = -x0 + 0.5*x1
    sigma1(x) = 0.3 + 0.2*sig(x0)  sigma2(x) = 0.3 + 0.3*sig(x1)
    w(x) = sig(x0 - 0.5*x1)      (probability of component 1)
    """
    rng = np.random.RandomState(seed + 1000)
    X_all = rng.randn(_MAX_N, d)
    eps1_all = rng.randn(_MAX_N)
    eps2_all = rng.randn(_MAX_N)
    u_all = rng.rand(_MAX_N)

    X = X_all[:n]
    mu1 = 2.0 * X[:, 0] + X[:, 1]
    mu2 = -X[:, 0] + 0.5 * X[:, 1]
    s1 = 0.3 + 0.2 * _sigmoid(X[:, 0])
    s2 = 0.3 + 0.3 * _sigmoid(X[:, 1])
    w  = _sigmoid(X[:, 0] - 0.5 * X[:, 1])

    comp = (u_all[:n] < w).astype(int)
    z = np.where(comp, mu1 + s1 * eps1_all[:n], mu2 + s2 * eps2_all[:n])

    def true_density(X_test, z_grid):
        m1 = 2.0 * X_test[:, 0] + X_test[:, 1]
        m2 = -X_test[:, 0] + 0.5 * X_test[:, 1]
        sig1 = 0.3 + 0.2 * _sigmoid(X_test[:, 0])
        sig2 = 0.3 + 0.3 * _sigmoid(X_test[:, 1])
        wt   = _sigmoid(X_test[:, 0] - 0.5 * X_test[:, 1])
        p1 = stats.norm.pdf(z_grid[None, :], m1[:, None], sig1[:, None])
        p2 = stats.norm.pdf(z_grid[None, :], m2[:, None], sig2[:, None])
        return wt[:, None] * p1 + (1.0 - wt[:, None]) * p2

    tag = f"BimodalFull-d{d}-{n}"
    return X, z, tag, true_density


# ── colour / style ────────────────────────────────────────────────────────────
GROUP_COLOR = {
    'flexcode':      '#e41a1c',
    'flow':          '#377eb8',
    'foundational':  '#e67e22',
}

# (key, display_name, group, linestyle, linewidth)
METHOD_SPEC = [
    ('FlexCode-RF',        'FlexCode-RF',      'flexcode',      '-',   2.2),
    ('Flow-Spline',        'Flow-Spline',      'flow',          '-',   2.2),
    ('TabPFN-2.5',         'TabPFN 2.5',       'foundational',  '-',   2.6),
]
METHOD_MAP = {k: (d, g, ls, lw) for k, d, g, ls, lw in METHOD_SPEC}

# ── three fixed test points (original X space, d=5) ──────────────────────────
# Chosen to show different bimodal shapes (varying dominance, separation, width)
# mu1 = 2*x0+x1,  mu2 = -x0+0.5*x1,  w = sig(x0 - 0.5*x1)
# s1 = 0.3+0.2*sig(x0),  s2 = 0.3+0.3*sig(x1)
TEST_POINTS = [
    np.array([[ 0.8,  0.5]]),   # w≈0.63, mu1=2.1, mu2=-0.55 — mild asymmetry
    np.array([[-0.5,  1.5]]),   # w≈0.22, mu1=0.5, mu2=1.25  — mode 2 dominant, close modes
    np.array([[ 1.0,  2.0]]),   # w≈0.50, mu1=4.0,  mu2=0.0  — equal weights, well-separated modes
]

D       = 2
SEED    = 42
N_GRID  = 300
N_SIZES = [50, 200, 2000]
OUT_DIR = Path('results_simulated')
OUT_DIR.mkdir(exist_ok=True)


# ── fit and evaluate one method at one test point ─────────────────────────────
def _eval(key, X_tr, z_tr, X_demo, z_lo, z_hi):
    kw = dict(n_grid=N_GRID, z_min=z_lo, z_max=z_hi)
    try:
        if key == 'FlexCode-RF':
            max_basis = min(30, max(15, int(np.sqrt(len(X_tr)))))
            model = FlexCodeEstimator(RFFlexRegressor, max_basis=max_basis, name='FlexCode-RF')
            model.fit_cv(X_tr, z_tr, n_folds=5)
            cdes, zg = model.predict(X_demo, n_grid=kw.get('n_grid', N_GRID))
            return zg, cdes[0]
        elif key == 'Flow-Spline':
            cdes, zg = normalizing_flow_density_tuned(
                X_tr, z_tr, X_demo, device='cpu', random_state=SEED, **kw)
        elif key == 'TabPFN-2.5' and HAS_TABPFN:
            m = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2_5, device='cuda',
                    ignore_pretraining_limits=True)
            m.fit(X_tr, z_tr)
            cdes, zg = tabpfn_native_density(m, X_demo, **kw)
        else:
            return None, None
        return zg, cdes[0]
    except Exception as e:
        print(f"  [{key}] FAILED: {e}")
        return None, None


# ── precompute z-ranges for each test point ───────────────────────────────────
panels_meta = []
for idx, X_demo_raw in enumerate(TEST_POINTS):
    x0, x1 = X_demo_raw[0, 0], X_demo_raw[0, 1]
    mu1 = 2.0 * x0 + x1
    mu2 = -x0 + 0.5 * x1
    s1  = 0.3 + 0.2 * _sigmoid(x0)
    s2  = 0.3 + 0.3 * _sigmoid(x1)
    z_lo = min(mu1 - 4*s1, mu2 - 4*s2) - 0.5
    z_hi = max(mu1 + 4*s1, mu2 + 4*s2) + 0.5
    z_grid = np.linspace(z_lo, z_hi, N_GRID)
    panels_meta.append(dict(inst_idx=idx, z_lo=z_lo, z_hi=z_hi,
                            z_grid=z_grid, X_demo_raw=X_demo_raw))


# ── collect all results: data[inst_idx][n_idx] ───────────────────────────────
# We index as data[instance][n_column] = dict with curves + true_dens
n_instances = len(TEST_POINTS)
n_ncols     = len(N_SIZES)
data = [[None]*n_ncols for _ in range(n_instances)]

for n_idx, N in enumerate(N_SIZES):
    print(f"\n{'='*60}")
    print(f"  n = {N}")
    print(f"{'='*60}")

    X_all, z_all, _, true_density_fn = make_bimodal_full(n=N, d=D, seed=SEED)
    X_train, _, z_train, _ = train_test_split(
        X_all, z_all, test_size=0.25, random_state=SEED)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)

    for inst_idx, meta in enumerate(panels_meta):
        X_demo = scaler.transform(meta['X_demo_raw'])
        true_dens = true_density_fn(meta['X_demo_raw'], meta['z_grid'])[0]

        curves = {}
        for key, *_ in METHOD_SPEC:
            print(f"  inst {inst_idx+1}  {key}...", end=" ", flush=True)
            zg, dens = _eval(key, X_tr, z_train, X_demo,
                             meta['z_lo'], meta['z_hi'])
            if zg is not None:
                curves[key] = (zg, dens)
                print("ok")
            else:
                print("skip")

        data[inst_idx][n_idx] = dict(
            true_dens=true_dens, curves=curves, **meta)


# ── legend handles ────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color='black', lw=2.2, ls='--', label='True'),
]
for key, disp, group, ls, lw in METHOD_SPEC:
    legend_handles.append(
        Line2D([0], [0], color=GROUP_COLOR[group], lw=lw, ls=ls,
               alpha=0.9, label=disp))


# ── draw one panel ────────────────────────────────────────────────────────────
def _draw_panel(ax, panel):
    ax.plot(panel['z_grid'], panel['true_dens'],
            color='black', lw=2.2, ls='--', zorder=6)

    for key, (disp, group, ls, lw) in METHOD_MAP.items():
        if key not in panel['curves']:
            continue
        zg, dens = panel['curves'][key]
        ax.plot(zg, dens, color=GROUP_COLOR[group], ls=ls, lw=lw,
                alpha=0.88,
                zorder={'flexcode': 3, 'flow': 4, 'foundational': 5}[group])

    ax.set_xlim(panel['z_lo'], panel['z_hi'])
    ax.set_ylim(0, max(panel['true_dens']) * 1.9)
    ax.tick_params(labelsize=9)


# ── Figure: 3 rows (instances) × 3 cols (n values) ───────────────────────────
ROW_BOX_COLORS = ['#e8f0e4', '#e4ecf3', '#fdf3e5']   # soft green, blue, orange

fig, axes = plt.subplots(
    n_instances, n_ncols,
    figsize=(9, 6.8),
    squeeze=False,
)
fig.subplots_adjust(hspace=0.45, wspace=0.12, bottom=0.13, top=0.90,
                    left=0.08, right=0.92)

for inst_idx in range(n_instances):
    for n_idx in range(n_ncols):
        ax = axes[inst_idx][n_idx]
        _draw_panel(ax, data[inst_idx][n_idx])

        # column header (top row only)
        if inst_idx == 0:
            ax.set_title(f'$n = {N_SIZES[n_idx]}$', fontsize=12, pad=5)

        # hide y-tick labels on interior columns
        if n_idx > 0:
            ax.set_yticklabels([])

        # y-axis label (left column only)
        if n_idx == 0:
            ax.set_ylabel('Density', fontsize=11)

    # row label on the right
    axes[inst_idx][n_ncols - 1].annotate(
        f'Instance {inst_idx + 1}',
        xy=(1.05, 0.5), xycoords='axes fraction',
        va='center', ha='left', fontsize=11, fontweight='bold',
        rotation=0,
    )

# ── boxes around each row of panels ──────────────────────────────────────────
renderer = fig.canvas.get_renderer()
for inst_idx in range(n_instances):
    # get bounding box that spans all panels in this row
    bb_left  = axes[inst_idx][0].get_position()
    bb_right = axes[inst_idx][n_ncols - 1].get_position()
    pad_x, pad_y = 0.006, 0.005
    x0 = bb_left.x0  - pad_x
    y0 = bb_left.y0  - pad_y
    w  = bb_right.x1 - bb_left.x0 + 2 * pad_x
    h  = bb_left.y1  - bb_left.y0 + 2 * pad_y
    rect = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.005',
        transform=fig.transFigure,
        facecolor=ROW_BOX_COLORS[inst_idx % len(ROW_BOX_COLORS)],
        edgecolor='#888888',
        linewidth=1.4,
        alpha=0.55,
        zorder=-1,
    )
    fig.patches.append(rect)

# ── single centred x-axis label ───────────────────────────────────────────────
fig.text(0.5, 0.08, r'$y$', ha='center', va='top', fontsize=12)

# ── horizontal legend below all panels ────────────────────────────────────────
fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=len(legend_handles),
    fontsize=12,
    framealpha=0.9,
    handlelength=2.2,
    columnspacing=1.0,
    handletextpad=0.5,
    bbox_to_anchor=(0.5, 0.0),
)

out_path = OUT_DIR / 'bimodal_illustration.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")

# ── also save individual-row versions ─────────────────────────────────────────
for inst_idx in range(n_instances):
    fig2, axes2 = plt.subplots(1, n_ncols, figsize=(9, 2.3), squeeze=False)
    fig2.subplots_adjust(wspace=0.26, bottom=0.30)
    for n_idx in range(n_ncols):
        ax = axes2[0][n_idx]
        _draw_panel(ax, data[inst_idx][n_idx])
        ax.set_title(f'$n = {N_SIZES[n_idx]}$', fontsize=10, pad=4)
        ax.set_xlabel(r'$y$', fontsize=9)
        if n_idx == 0:
            ax.set_ylabel('Density', fontsize=11)
    fig2.legend(handles=legend_handles, loc='lower center',
                ncol=len(legend_handles), fontsize=8,
                handlelength=2.0, columnspacing=0.9, handletextpad=0.4,
                bbox_to_anchor=(0.5, 0.0))
    fig2.suptitle(f'Instance {inst_idx+1}', fontsize=10, y=1.01)
    out_n = OUT_DIR / f'bimodal_illustration_instance{inst_idx+1}.png'
    fig2.savefig(out_n, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_n}")

print("\nDone.")

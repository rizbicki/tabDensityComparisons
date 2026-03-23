#!/usr/bin/env python
"""
Bimodal-DGP illustration — same as bimodal_illustration but the parametric
model is now CORRECTLY SPECIFIED (same functional form as the true DGP)
and fitted by maximum likelihood.

DGP:
  mu1 = a1*x0 + a2*x1,  mu2 = b1*x0 + b2*x1
  sigma1 = c0 + c1*sigmoid(x0),  sigma2 = d0 + d1*sigmoid(x1)
  w = sigmoid(e1*x0 + e2*x1)

Layout: 3 rows (instances) × 3 cols (n = 50, 200, 1000).

Output: results_simulated/bimodal_illustration_correctly_specified.png
"""

import sys, warnings
import numpy as np
from scipy import stats
from scipy.optimize import minimize
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

from models import normalizing_flow_density_tuned
from models.native import tabpfn_native_density

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print("- TabPFN not available")


# ── DGP ───────────────────────────────────────────────────────────────────────
_MAX_N = 50_000

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -30, 30)))


def make_bimodal_full(n=1000, d=2, seed=42):
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

    return X, z, true_density


# ── Correctly-specified MLE model ─────────────────────────────────────────────
# Parameters (10 total):
#   a1, a2         — mu1 = a1*x0 + a2*x1
#   b1, b2         — mu2 = b1*x0 + b2*x1
#   c0_raw, c1_raw — sigma1 = softplus(c0_raw) + softplus(c1_raw)*sigmoid(x0)
#   d0_raw, d1_raw — sigma2 = softplus(d0_raw) + softplus(d1_raw)*sigmoid(x1)
#   e1, e2         — w = sigmoid(e1*x0 + e2*x1)

def _unpack(params):
    return (params[0], params[1],   # a1, a2
            params[2], params[3],   # b1, b2
            params[4], params[5],   # c0_raw, c1_raw
            params[6], params[7],   # d0_raw, d1_raw
            params[8], params[9])   # e1, e2

def _mixture_components(params, X):
    a1, a2, b1, b2, c0r, c1r, d0r, d1r, e1, e2 = _unpack(params)
    mu1 = a1 * X[:, 0] + a2 * X[:, 1]
    mu2 = b1 * X[:, 0] + b2 * X[:, 1]
    sig1 = _softplus(c0r) + _softplus(c1r) * _sigmoid(X[:, 0])
    sig2 = _softplus(d0r) + _softplus(d1r) * _sigmoid(X[:, 1])
    w    = _sigmoid(e1 * X[:, 0] + e2 * X[:, 1])
    return mu1, mu2, sig1, sig2, w

def _neg_log_lik(params, X, z):
    mu1, mu2, sig1, sig2, w = _mixture_components(params, X)
    p1 = w * stats.norm.pdf(z, mu1, sig1)
    p2 = (1.0 - w) * stats.norm.pdf(z, mu2, sig2)
    ll = np.log(p1 + p2 + 1e-300)
    return -np.sum(ll)

def _fit_correct_mle(X_tr, z_tr, n_restarts=8):
    """Fit the correctly-specified mixture model by MLE with multiple restarts."""
    best_nll = np.inf
    best_params = None
    rng = np.random.RandomState(42)

    # initial guesses: one "informed" + random perturbations
    init_list = [
        np.array([2.0, 1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5]),
    ]
    for _ in range(n_restarts - 1):
        init_list.append(rng.randn(10) * 0.5)

    for i, p0 in enumerate(init_list):
        try:
            res = minimize(_neg_log_lik, p0, args=(X_tr, z_tr),
                           method='L-BFGS-B', options={'maxiter': 2000})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params

def _correct_mle_density(X_tr, z_tr, X_test, n_grid=300, z_min=-5, z_max=5):
    """Fit correctly-specified model and evaluate density on grid."""
    params = _fit_correct_mle(X_tr, z_tr)
    z_grid = np.linspace(z_min, z_max, n_grid)
    mu1, mu2, sig1, sig2, w = _mixture_components(params, X_test)
    p1 = stats.norm.pdf(z_grid[None, :], mu1[:, None], sig1[:, None])
    p2 = stats.norm.pdf(z_grid[None, :], mu2[:, None], sig2[:, None])
    cdes = w[:, None] * p1 + (1.0 - w[:, None]) * p2
    return cdes, z_grid


# ── colour / style ────────────────────────────────────────────────────────────
GROUP_COLOR = {
    'parametric':    '#1b9e77',
    'nonparametric': '#377eb8',
    'foundational':  '#e67e22',
}

METHOD_SPEC = [
    ('CorrectMLE',   'Correct Param.', 'parametric',    '-',   2.2),
    ('Flow-Spline',  'Flow-Spline',  'nonparametric', '-',   2.2),
    ('TabPFN-2.5',   'TabPFN 2.5',   'foundational',  '-',   2.6),
]
# Maximum training-set size for ISE evaluation per method (None = no limit)
METHOD_MAX_N_ISE = {
    'CorrectMLE':  None,
    'Flow-Spline': 2000,   # OOM beyond this on CPU
    'TabPFN-2.5':  None,
}
METHOD_MAP = {k: (d, g, ls, lw) for k, d, g, ls, lw in METHOD_SPEC}

TEST_POINTS = [
    np.array([[ 0.8,  0.5]]),
    np.array([[-0.5,  1.5]]),
    np.array([[-1.2, -0.5]]),
]

D       = 2
SEED    = 42
N_GRID  = 300
N_SIZES_PANEL = [20, 100, 500]          # shown in density panels (left)
N_SIZES_ISE   = [20, 50, 100, 500, 1000, 2000, 5000]  # shown in ISE plot (right)
N_REPS  = 10                             # repetitions for ISE evaluation
OUT_DIR = Path('results_simulated')
OUT_DIR.mkdir(exist_ok=True)


def _eval_batch(key, X_tr, z_tr, X_test, z_lo, z_hi):
    """Return (cdes, z_grid) for all rows in X_test, or (None, None)."""
    kw = dict(n_grid=N_GRID, z_min=z_lo, z_max=z_hi)
    try:
        if key == 'CorrectMLE':
            return _correct_mle_density(X_tr, z_tr, X_test, **kw)
        elif key == 'Flow-Spline':
            return normalizing_flow_density_tuned(
                X_tr, z_tr, X_test, device='cpu', random_state=SEED, **kw)
        elif key == 'TabPFN-2.5' and HAS_TABPFN:
            m = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V2_5, device='cpu',
                    ignore_pretraining_limits=True)
            m.fit(X_tr, z_tr)
            return tabpfn_native_density(m, X_test, **kw)
        else:
            return None, None
    except Exception as e:
        print(f"  [{key}] FAILED: {e}")
        return None, None


# ── precompute z-ranges ───────────────────────────────────────────────────────
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


# ── collect results ───────────────────────────────────────────────────────────
N_TEST_L2 = 1000          # number of test points for L2 evaluation
L2_Z_LO, L2_Z_HI = -8, 8  # wide grid covering all test-point densities

n_instances = len(TEST_POINTS)
n_panel_cols = len(N_SIZES_PANEL)
data = [[None]*n_panel_cols for _ in range(n_instances)]

# ── 1) density panels: single seed, N_SIZES_PANEL ────────────────────────────
for n_idx, N in enumerate(N_SIZES_PANEL):
    print(f"\n{'='*60}")
    print(f"  Panel  n = {N}")
    print(f"{'='*60}")

    X_all, z_all, true_density_fn = make_bimodal_full(n=N, d=D, seed=SEED)
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
            cdes, zg = _eval_batch(key, X_tr, z_train, X_demo,
                                   meta['z_lo'], meta['z_hi'])
            if cdes is not None:
                curves[key] = (zg, cdes[0])
                print("ok")
            else:
                print("skip")

        data[inst_idx][n_idx] = dict(
            true_dens=true_dens, curves=curves, **meta)

# ── 2) ISE evaluation: N_REPS repetitions × N_SIZES_ISE ─────────────────────
# ise_all[(n_idx, key)] = list of N_REPS mean-ISE values
ise_all = {(ni, key): [] for ni in range(len(N_SIZES_ISE))
           for key, *_ in METHOD_SPEC}

l2_zgrid = np.linspace(L2_Z_LO, L2_Z_HI, N_GRID)
dz = l2_zgrid[1] - l2_zgrid[0]

for rep in range(N_REPS):
    rep_seed = SEED + rep * 7
    # fixed test set per repetition (independent of training data)
    l2_rng = np.random.RandomState(999 + rep)
    X_test_l2_raw = l2_rng.randn(N_TEST_L2, D)

    for n_idx, N in enumerate(N_SIZES_ISE):
        print(f"\n  ISE rep {rep+1}/{N_REPS}  n = {N}")

        X_all, z_all, true_density_fn = make_bimodal_full(
            n=N, d=D, seed=rep_seed)
        X_train, _, z_train, _ = train_test_split(
            X_all, z_all, test_size=0.25, random_state=rep_seed)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)

        true_cde_l2 = true_density_fn(X_test_l2_raw, l2_zgrid)
        X_test_l2_sc = scaler.transform(X_test_l2_raw)

        for key, *_ in METHOD_SPEC:
            max_n = METHOD_MAX_N_ISE.get(key)
            if max_n is not None and N > max_n:
                print(f"    {key}... skip (n={N} > max {max_n})")
                continue
            print(f"    {key}...", end=" ", flush=True)
            cdes, zg = _eval_batch(key, X_tr, z_train, X_test_l2_sc,
                                   L2_Z_LO, L2_Z_HI)
            if cdes is not None:
                if not np.allclose(zg, l2_zgrid):
                    cdes_interp = np.zeros_like(true_cde_l2)
                    for i in range(len(cdes)):
                        cdes_interp[i] = np.interp(l2_zgrid, zg, cdes[i],
                                                   left=0, right=0)
                    cdes = cdes_interp
                ise = np.sum((cdes - true_cde_l2)**2, axis=1) * dz
                val = float(np.mean(ise))
                ise_all[(n_idx, key)].append(val)
                print(f"ISE = {val:.4f}")
            else:
                print("skip")

# Compute mean and SE
ise_mean = {}  # (n_idx, key) -> mean
ise_se   = {}  # (n_idx, key) -> standard error
for (n_idx, key), vals in ise_all.items():
    if vals:
        arr = np.array(vals)
        ise_mean[(n_idx, key)] = arr.mean()
        ise_se[(n_idx, key)]   = arr.std(ddof=1) / np.sqrt(len(arr))


# ── legend handles (no ISE in legend — shown in side panel) ──────────────────
legend_handles = [
    Line2D([0], [0], color='black', lw=2.2, ls='--', label='True'),
]
for key, disp, group, ls, lw in METHOD_SPEC:
    legend_handles.append(
        Line2D([0], [0], color=GROUP_COLOR[group], lw=lw, ls=ls,
               alpha=0.9, label=disp))


def _draw_panel(ax, panel):
    ax.plot(panel['z_grid'], panel['true_dens'],
            color='black', lw=2.2, ls='--', zorder=6)
    for key, (disp, group, ls, lw) in METHOD_MAP.items():
        if key not in panel['curves']:
            continue
        zg, dens = panel['curves'][key]
        ax.plot(zg, dens, color=GROUP_COLOR[group], ls=ls, lw=lw,
                alpha=0.88,
                zorder={'parametric': 3, 'nonparametric': 4,
                        'foundational': 5}[group])
    ax.set_xlim(panel['z_lo'], panel['z_hi'])
    ax.set_ylim(0, max(panel['true_dens']) * 1.9)
    ax.tick_params(labelsize=9)


# ── Figure: density panels (left) + ISE-vs-n panel (right) ───────────────────
from matplotlib.gridspec import GridSpec
ROW_BOX_COLORS = ['#e8f0e4', '#e4ecf3', '#fdf3e5']

fig = plt.figure(figsize=(14, 6.8))
gs = GridSpec(n_instances, n_panel_cols + 1, figure=fig,
             width_ratios=[1]*n_panel_cols + [1.2],
             hspace=0.45, wspace=0.45,
             bottom=0.13, top=0.90, left=0.06, right=0.97)

# -- density panels --
axes = [[fig.add_subplot(gs[r, c]) for c in range(n_panel_cols)]
        for r in range(n_instances)]

for inst_idx in range(n_instances):
    for n_idx in range(n_panel_cols):
        ax = axes[inst_idx][n_idx]
        _draw_panel(ax, data[inst_idx][n_idx])
        if inst_idx == 0:
            ax.set_title(f'$n = {N_SIZES_PANEL[n_idx]}$', fontsize=12, pad=5)
        if n_idx > 0:
            ax.set_yticklabels([])
        if n_idx == 0:
            ax.set_ylabel('Density', fontsize=11)

# -- row boxes around density panels --
for inst_idx in range(n_instances):
    bb_left  = axes[inst_idx][0].get_position()
    bb_right = axes[inst_idx][n_panel_cols - 1].get_position()
    pad_x, pad_y = 0.006, 0.005
    x0 = bb_left.x0 - pad_x
    y0 = bb_left.y0 - pad_y
    w  = bb_right.x1 - bb_left.x0 + 2 * pad_x
    h  = bb_left.y1 - bb_left.y0 + 2 * pad_y
    fig.patches.append(FancyBboxPatch(
        (x0, y0), w, h, boxstyle='round,pad=0.005',
        transform=fig.transFigure,
        facecolor=ROW_BOX_COLORS[inst_idx % len(ROW_BOX_COLORS)],
        edgecolor='#888888', linewidth=1.4, alpha=0.55, zorder=-1))

# -- ISE-vs-n panel (spans all rows) with mean ± SE --
ax_ise = fig.add_subplot(gs[:, n_panel_cols])
n_ise = len(N_SIZES_ISE)
for key, disp, group, ls, lw in METHOD_SPEC:
    means = np.array([ise_mean.get((ni, key), np.nan) for ni in range(n_ise)])
    ses   = np.array([ise_se.get((ni, key), 0) for ni in range(n_ise)])
    valid = ~np.isnan(means)
    ns = np.array(N_SIZES_ISE)[valid]
    m  = means[valid]
    se = ses[valid]
    ax_ise.plot(ns, m, color=GROUP_COLOR[group],
                ls=ls, lw=lw, marker='o', markersize=5,
                alpha=0.9, label=disp)
    ax_ise.fill_between(ns, m - se, m + se,
                        color=GROUP_COLOR[group], alpha=0.18)
ax_ise.set_xlabel('$n$', fontsize=12)
ax_ise.set_ylabel('Mean ISE', fontsize=11)
ax_ise.set_title('ISE vs $n$', fontsize=12, pad=5)
ax_ise.set_xscale('log')
ax_ise.set_yscale('log')
ax_ise.set_xticks(N_SIZES_ISE)
ax_ise.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_ise.tick_params(labelsize=9)
ax_ise.legend(fontsize=9, loc='upper right')
ax_ise.grid(True, alpha=0.3)

# -- x-axis label for density panels --
mid_x = (axes[-1][0].get_position().x0 +
         axes[-1][n_panel_cols-1].get_position().x1) / 2
fig.text(mid_x, 0.08, r'$y$', ha='center', va='top', fontsize=12)

# -- shared legend for density panels --
fig.legend(handles=legend_handles, loc='lower center',
           ncol=len(legend_handles), fontsize=11, framealpha=0.9,
           handlelength=2.2, columnspacing=1.0, handletextpad=0.5,
           bbox_to_anchor=(mid_x, 0.0))

out_path = OUT_DIR / 'bimodal_illustration_correctly_specified.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
print("\nDone.")

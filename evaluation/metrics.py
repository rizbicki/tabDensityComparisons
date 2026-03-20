"""
Evaluation metrics for conditional density estimation.
"""

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid


def _normalize_density_rows(cdes, z_grid):
    """Normalize density rows onto the evaluation grid."""
    cdes = np.array(cdes, dtype=float, copy=True)
    integrals = np.trapezoid(cdes, z_grid, axis=1)
    integrals[integrals <= 0] = 1.0
    return cdes / integrals[:, None]


def _cdf_from_density_row(cde_row, z_grid):
    """Numerically integrate a density row into a CDF on the same grid."""
    total = np.trapezoid(cde_row, z_grid)
    if total > 0:
        cde_row = cde_row / total
    cdf = cumulative_trapezoid(cde_row, z_grid, initial=0.0)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return np.clip(cdf, 0.0, 1.0)


def _density_at_targets(cdes, z_grid, z_test):
    """Evaluate densities at targets, using zero mass outside the grid support."""
    return np.array([
        np.interp(z_test[i], z_grid, cdes[i], left=0.0, right=0.0)
        for i in range(len(z_test))
    ])


def eval_cde_loss(cdes, z_grid, z_test):
    cdes = _normalize_density_rows(cdes, z_grid)
    integral_sq = np.trapezoid(cdes ** 2, z_grid, axis=1)
    f_at_z = _density_at_targets(cdes, z_grid, z_test)
    return np.mean(integral_sq - 2 * f_at_z)


def eval_log_lik(cdes, z_grid, z_test):
    cdes = _normalize_density_rows(cdes, z_grid)
    f_at_z = _density_at_targets(cdes, z_grid, z_test)
    return np.mean(np.log(np.maximum(f_at_z, 1e-10)))


def eval_crps(cdes, z_grid, z_test):
    cdes = _normalize_density_rows(cdes, z_grid)
    vals = np.zeros(len(z_test))
    for i in range(len(z_test)):
        cdf = _cdf_from_density_row(cdes[i], z_grid)
        indicator = (z_grid >= z_test[i]).astype(float)
        vals[i] = np.trapezoid((cdf - indicator) ** 2, z_grid)
    return np.mean(vals)


def eval_pit(cdes, z_grid, z_test):
    cdes = _normalize_density_rows(cdes, z_grid)
    pit = np.zeros(len(z_test))
    for i in range(len(z_test)):
        cdf = _cdf_from_density_row(cdes[i], z_grid)
        pit[i] = np.interp(z_test[i], z_grid, cdf, left=0.0, right=1.0)
    return np.clip(pit, 0, 1)


def eval_pit_ks(pit_vals):
    return stats.kstest(pit_vals, 'uniform').statistic


def eval_coverage_width(cdes, z_grid, z_test, level=0.90):
    cdes = _normalize_density_rows(cdes, z_grid)
    covers = np.zeros(len(z_test))
    widths = np.zeros(len(z_test))
    alpha = (1 - level) / 2
    for i in range(len(z_test)):
        cdf = _cdf_from_density_row(cdes[i], z_grid)
        lo = np.interp(alpha, cdf, z_grid, left=z_grid[0], right=z_grid[-1])
        hi = np.interp(1 - alpha, cdf, z_grid, left=z_grid[0], right=z_grid[-1])
        covers[i] = (z_test[i] >= lo) and (z_test[i] <= hi)
        widths[i] = hi - lo
    return np.mean(covers), np.mean(widths)


def compute_all_metrics(cdes, z_grid, z_test):
    """Compute all metrics with per-sample values for standard error estimation."""
    n = len(z_test)
    cdes = _normalize_density_rows(cdes, z_grid)
    f_at_z = _density_at_targets(cdes, z_grid, z_test)

    # CDE loss
    integral_sq = np.trapezoid(cdes ** 2, z_grid, axis=1)
    cde_per = integral_sq - 2 * f_at_z

    # Log-likelihood
    ll_per = np.log(np.maximum(f_at_z, 1e-10))

    # CRPS
    crps_per = np.zeros(n)
    for i in range(n):
        cdf = _cdf_from_density_row(cdes[i], z_grid)
        indicator = (z_grid >= z_test[i]).astype(float)
        crps_per[i] = np.trapezoid((cdf - indicator) ** 2, z_grid)

    # 90% credible interval
    covers = np.zeros(n)
    widths = np.zeros(n)
    alpha = 0.05
    for i in range(n):
        cdf = _cdf_from_density_row(cdes[i], z_grid)
        lo = np.interp(alpha, cdf, z_grid, left=z_grid[0], right=z_grid[-1])
        hi = np.interp(1 - alpha, cdf, z_grid, left=z_grid[0], right=z_grid[-1])
        covers[i] = float(z_test[i] >= lo and z_test[i] <= hi)
        widths[i] = hi - lo

    # PIT KS
    pit_v = eval_pit(cdes, z_grid, z_test)
    pit_ks = eval_pit_ks(pit_v)

    def se(arr):
        return np.std(arr, ddof=1) / np.sqrt(n)

    return {
        'CDE_loss':         np.mean(cde_per),  'CDE_loss_se':        se(cde_per),
        'log_lik':          np.mean(ll_per),   'log_lik_se':         se(ll_per),
        'CRPS':             np.mean(crps_per), 'CRPS_se':            se(crps_per),
        'PIT_KS':           pit_ks,            'PIT_KS_se':          None,
        'coverage_90':      np.mean(covers),   'coverage_90_se':     se(covers),
        'interval_width':   np.mean(widths),   'interval_width_se':  se(widths),
    }

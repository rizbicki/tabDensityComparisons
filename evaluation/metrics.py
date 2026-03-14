"""
Evaluation metrics for conditional density estimation.
"""

import numpy as np
from scipy import stats


def eval_cde_loss(cdes, z_grid, z_test):
    integral_sq = np.trapezoid(cdes ** 2, z_grid, axis=1)
    f_at_z = np.array([np.interp(z_test[i], z_grid, cdes[i])
                       for i in range(len(z_test))])
    return np.mean(integral_sq - 2 * f_at_z)


def eval_log_lik(cdes, z_grid, z_test):
    f_at_z = np.array([np.interp(z_test[i], z_grid, cdes[i])
                       for i in range(len(z_test))])
    return np.mean(np.log(np.maximum(f_at_z, 1e-10)))


def eval_crps(cdes, z_grid, z_test):
    dz = z_grid[1] - z_grid[0]
    vals = np.zeros(len(z_test))
    for i in range(len(z_test)):
        cdf = np.cumsum(cdes[i]) * dz
        cdf = np.minimum(cdf, 1.0)
        indicator = (z_grid >= z_test[i]).astype(float)
        vals[i] = np.trapezoid((cdf - indicator) ** 2, z_grid)
    return np.mean(vals)


def eval_pit(cdes, z_grid, z_test):
    dz = z_grid[1] - z_grid[0]
    pit = np.zeros(len(z_test))
    for i in range(len(z_test)):
        cdf = np.cumsum(cdes[i]) * dz
        pit[i] = np.interp(z_test[i], z_grid, cdf)
    return np.clip(pit, 0, 1)


def eval_pit_ks(pit_vals):
    return stats.kstest(pit_vals, 'uniform').statistic


def eval_coverage_width(cdes, z_grid, z_test, level=0.90):
    dz = z_grid[1] - z_grid[0]
    covers = np.zeros(len(z_test))
    widths = np.zeros(len(z_test))
    alpha = (1 - level) / 2
    for i in range(len(z_test)):
        cdf = np.cumsum(cdes[i]) * dz
        lo = np.interp(alpha, cdf, z_grid)
        hi = np.interp(1 - alpha, cdf, z_grid)
        covers[i] = (z_test[i] >= lo) and (z_test[i] <= hi)
        widths[i] = hi - lo
    return np.mean(covers), np.mean(widths)


def compute_all_metrics(cdes, z_grid, z_test):
    """Compute all metrics with per-sample values for standard error estimation."""
    n = len(z_test)
    dz = z_grid[1] - z_grid[0]

    f_at_z = np.array([np.interp(z_test[i], z_grid, cdes[i]) for i in range(n)])

    # CDE loss
    integral_sq = np.trapezoid(cdes ** 2, z_grid, axis=1)
    cde_per = integral_sq - 2 * f_at_z

    # Log-likelihood
    ll_per = np.log(np.maximum(f_at_z, 1e-10))

    # CRPS
    crps_per = np.zeros(n)
    for i in range(n):
        cdf = np.cumsum(cdes[i]) * dz
        cdf = np.minimum(cdf, 1.0)
        indicator = (z_grid >= z_test[i]).astype(float)
        crps_per[i] = np.trapezoid((cdf - indicator) ** 2, z_grid)

    # 90% credible interval
    covers = np.zeros(n)
    widths = np.zeros(n)
    alpha = 0.05
    for i in range(n):
        cdf = np.cumsum(cdes[i]) * dz
        lo = np.interp(alpha, cdf, z_grid)
        hi = np.interp(1 - alpha, cdf, z_grid)
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

"""
Post-hoc PIT recalibration for conditional density estimators.

Implements distribution-free probabilistic recalibration in the spirit of
Kuleshov et al. (2018): fit a monotone map ``R`` on held-out PIT values so that
``R(F_hat)`` has approximately uniform PIT, then recalibrate the *full density*
via the chain rule

    F_tilde(y | x) = R(F_hat(y | x)),
    f_tilde(y | x) = R'(F_hat(y | x)) * f_hat(y | x).

``R`` is a smooth (C1) monotone PCHIP interpolant of the empirical CDF of the
calibration PIT values, with endpoints pinned at ``(0, 0)`` and ``(1, 1)``.
Smoothness is what lets us recalibrate the *density* (not just the CDF): a raw
isotonic/empirical-CDF map is piecewise constant, so ``R'`` would be a sum of
spikes and the density-based metrics (CDE loss, log-likelihood) would be
ill-defined. PCHIP keeps ``R`` monotone with a bounded, well-defined derivative.

Recalibration is evaluated by K-fold cross-fitting on the test set: each point's
density is recalibrated with a map fit on the *other* folds, so the reported
metrics are out-of-sample for both the base model (the cached predictions are
already on a held-out test split) and the recalibration map.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from .metrics import _cdf_from_density_row, _normalize_density_rows, eval_pit

_EPS = 1e-6


def fit_recalibration_map(u_calib, n_knots=None):
    """Fit a smooth monotone recalibration map ``R: [0, 1] -> [0, 1]``.

    ``R`` is the empirical CDF of the calibration PIT values, evaluated on a
    *coarse, regular* probability grid of ``n_knots`` cells and interpolated with
    a monotone PCHIP, with endpoints pinned at ``(0, 0)`` and ``(1, 1)``. The
    coarse grid is essential: using every calibration point as a knot leaves
    ``R`` close to the truth but makes ``R'`` extremely noisy (the spacing of
    PIT order statistics is irregular), which would corrupt the density-based
    metrics even when the forecast is already calibrated. A modest number of
    regularly spaced knots keeps ``R'`` smooth.

    Returns ``(R, R_prime)`` callables. With fewer than two usable calibration
    points it falls back to the identity map.
    """
    u = np.asarray(u_calib, dtype=float)
    u = u[np.isfinite(u)]
    u = np.clip(u, _EPS, 1.0 - _EPS)
    m = u.size
    if m < 2:
        ident = PchipInterpolator([0.0, 1.0], [0.0, 1.0])
        return ident, ident.derivative()

    if n_knots is None:
        n_knots = int(np.clip(m // 30, 6, 20))

    u_sorted = np.sort(u)
    p = np.linspace(0.0, 1.0, n_knots + 1)              # regular x-knots
    R_at_p = np.searchsorted(u_sorted, p, side="right") / m   # ECDF at knots
    R_at_p[0], R_at_p[-1] = 0.0, 1.0                    # pin endpoints
    R_at_p = np.clip(np.maximum.accumulate(R_at_p), 0.0, 1.0)

    R = PchipInterpolator(p, R_at_p, extrapolate=True)
    return R, R.derivative()


def recalibrate_density_rows(cde_rows, z_grid, R, R_prime):
    """Apply a fitted recalibration map to a block of density rows.

    Returns recalibrated densities on the same ``z_grid`` (renormalized to
    integrate to one).
    """
    cde_rows = _normalize_density_rows(cde_rows, z_grid)
    out = np.empty_like(cde_rows)
    for i in range(cde_rows.shape[0]):
        cdf = _cdf_from_density_row(cde_rows[i], z_grid)        # F_hat on grid
        weight = np.clip(R_prime(cdf), 0.0, None)               # R'(F_hat)
        f_new = weight * cde_rows[i]
        total = np.trapezoid(f_new, z_grid)
        out[i] = f_new / total if total > 0 else cde_rows[i]
    return out


def _kfold_indices(n, n_splits, seed):
    """Shuffled K-fold split indices as a list of (eval_idx, calib_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_splits)
    splits = []
    for k in range(n_splits):
        eval_idx = folds[k]
        calib_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        splits.append((eval_idx, calib_idx))
    return splits


def crossfit_recalibrate(cde, z_grid, z_test, n_splits=5, min_calib=40, seed=0):
    """Cross-fit PIT recalibration over a single test set.

    Each point's density is recalibrated by a map fit on the other folds, so the
    recalibrated densities are out-of-sample for the recalibration map. Returns
    ``(cde_recal, info)`` where ``cde_recal`` has the same shape as ``cde``, or
    ``(None, info)`` when the test set is too small to recalibrate reliably.
    """
    cde = np.asarray(cde, dtype=float)
    z_test = np.asarray(z_test, dtype=float)
    n = len(z_test)
    info = {"n_test": int(n), "n_splits": int(n_splits)}

    if not np.all(np.isfinite(cde)):
        info["status"] = "skipped:nonfinite_density"
        return None, info
    if n < min_calib:
        info["status"] = "skipped:too_small"
        return None, info

    # Cap the number of folds so each calibration set stays usable.
    k = int(min(n_splits, max(2, n // 10)))
    info["n_splits"] = k

    cde_recal = np.empty_like(cde)
    for eval_idx, calib_idx in _kfold_indices(n, k, seed):
        u_calib = eval_pit(cde[calib_idx], z_grid, z_test[calib_idx])
        R, R_prime = fit_recalibration_map(u_calib)
        cde_recal[eval_idx] = recalibrate_density_rows(
            cde[eval_idx], z_grid, R, R_prime
        )
    info["status"] = "ok"
    return cde_recal, info

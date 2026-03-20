"""
Regression checks for finite-grid density normalization.
"""

import numpy as np

from evaluation.metrics import compute_all_metrics
from models.baselines import (
    linear_gaussian_homo_density,
    student_t_density,
)


def _toy_split(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 4)
    beta = np.array([1.0, -0.5, 0.3, 0.2])
    z = X @ beta + 0.8 * rng.randn(300)
    return X[:240], z[:240], X[240:], z[240:]


def test_analytic_grid_densities_are_normalized():
    X_train, z_train, X_test, _ = _toy_split()

    for density_fn in (linear_gaussian_homo_density, student_t_density):
        cdes, z_grid = density_fn(X_train, z_train, X_test, n_grid=200)
        integrals = np.trapezoid(cdes, z_grid, axis=1)
        assert np.allclose(integrals, 1.0, atol=1e-6)


def test_metrics_ignore_uniform_density_scaling():
    X_train, z_train, X_test, z_test = _toy_split()
    cdes, z_grid = linear_gaussian_homo_density(
        X_train, z_train, X_test, n_grid=200
    )

    metrics_ref = compute_all_metrics(cdes, z_grid, z_test)
    metrics_scaled = compute_all_metrics(0.25 * cdes, z_grid, z_test)

    for key in ("CDE_loss", "log_lik", "CRPS", "coverage_90", "interval_width"):
        assert np.isclose(metrics_ref[key], metrics_scaled[key], atol=1e-10)
    assert np.isclose(metrics_ref["PIT_KS"], metrics_scaled["PIT_KS"], atol=1e-10)


if __name__ == "__main__":
    test_analytic_grid_densities_are_normalized()
    test_metrics_ignore_uniform_density_scaling()
    print("density normalization checks passed")

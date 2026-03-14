"""
Synthetic datasets with known true conditional densities.
"""

import numpy as np
from scipy import stats


def make_heteroscedastic(n=1000, d=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta = rng.randn(d) * 0.5
    sigma = 0.5 + np.abs(X[:, 0])
    z = X @ beta + sigma * rng.randn(n)

    def true_density(X_test, z_grid):
        mu = X_test @ beta
        sig = 0.5 + np.abs(X_test[:, 0])
        return stats.norm.pdf(z_grid[None, :], mu[:, None], sig[:, None])

    tag = f"Heteroscedastic-{n}" if n != 1000 else "Heteroscedastic"
    return X, z, tag, true_density


def make_bimodal(n=1000, d=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    mu1, mu2 = X[:, 0] + X[:, 1], -X[:, 0] + X[:, 1]
    comp = rng.binomial(1, 0.5, n)
    z = np.where(comp, mu1 + 0.5 * rng.randn(n), mu2 + 0.5 * rng.randn(n))

    def true_density(X_test, z_grid):
        m1 = X_test[:, 0] + X_test[:, 1]
        m2 = -X_test[:, 0] + X_test[:, 1]
        p1 = stats.norm.pdf(z_grid[None, :], m1[:, None], 0.5)
        p2 = stats.norm.pdf(z_grid[None, :], m2[:, None], 0.5)
        return 0.5 * p1 + 0.5 * p2

    tag = f"Bimodal-{n}" if n != 1000 else "Bimodal"
    return X, z, tag, true_density


def make_skewed(n=1000, d=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    shape = 2 + np.abs(X[:, 0])
    z = rng.gamma(shape, 0.5, n) + X[:, 1]

    def true_density(X_test, z_grid):
        shapes = 2 + np.abs(X_test[:, 0])
        shifts = X_test[:, 1]
        result = np.zeros((len(X_test), len(z_grid)))
        for i in range(len(X_test)):
            z_shifted = z_grid - shifts[i]
            mask = z_shifted > 0
            result[i, mask] = stats.gamma.pdf(z_shifted[mask], a=shapes[i], scale=0.5)
        return result

    tag = f"Skewed-{n}" if n != 1000 else "Skewed"
    return X, z, tag, true_density


def make_linear_gaussian_homo(n=1000, d=5, seed=42):
    """True DGP: Z = X'beta + sigma*eps, eps~N(0,1), sigma constant."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta = rng.randn(d)
    sigma = 1.0
    z = X @ beta + sigma * rng.randn(n)

    def true_density(X_test, z_grid):
        mu = X_test @ beta
        return stats.norm.pdf(z_grid[None, :], mu[:, None], sigma)

    tag = f"LinGauss-Homo-{n}" if n != 1000 else "LinGauss-Homo"
    return X, z, tag, true_density


def make_nonlinear(n=1000, d=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    mu = np.sin(X[:, 0] * 2) + X[:, 1] ** 2 + 0.5 * X[:, 2]
    sigma = 0.3 + 0.3 * np.abs(np.cos(X[:, 0]))
    z = mu + sigma * rng.randn(n)

    def true_density(X_test, z_grid):
        m = np.sin(X_test[:, 0] * 2) + X_test[:, 1] ** 2 + 0.5 * X_test[:, 2]
        sig = 0.3 + 0.3 * np.abs(np.cos(X_test[:, 0]))
        return stats.norm.pdf(z_grid[None, :], m[:, None], sig[:, None])

    tag = f"Nonlinear-{n}" if n != 1000 else "Nonlinear"
    return X, z, tag, true_density

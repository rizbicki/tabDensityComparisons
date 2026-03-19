"""
Synthetic datasets with known true conditional densities.

All generators produce nested samples: the first k observations for
size n are identical to the k observations produced for any n' >= k.
This is achieved by always generating _MAX_N observations from a
fixed RNG seed and slicing to the requested n.

Tags follow the convention "{Base}-d{d}-{n}" so that the per-n
grouping logic (which strips the trailing number) naturally groups
datasets with the same base name and dimensionality together.
"""

import numpy as np
from scipy import stats

_MAX_N = 25_000  # upper bound; we slice to the requested n


def make_heteroscedastic(n=1000, d=5, seed=42):
    param_rng = np.random.RandomState(seed)
    beta = param_rng.randn(d) * 0.5

    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.randn(_MAX_N, d)
    eps_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    sigma = 0.5 + np.abs(X[:, 0])
    z = X @ beta + sigma * eps_all[:n]

    def true_density(X_test, z_grid):
        mu = X_test @ beta
        sig = 0.5 + np.abs(X_test[:, 0])
        return stats.norm.pdf(z_grid[None, :], mu[:, None], sig[:, None])

    tag = f"Heteroscedastic-d{d}-{n}"
    return X, z, tag, true_density


def make_bimodal(n=1000, d=5, seed=42):
    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.randn(_MAX_N, d)
    comp_all = data_rng.binomial(1, 0.5, _MAX_N)
    eps1_all = data_rng.randn(_MAX_N)
    eps2_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    mu1, mu2 = X[:, 0] + X[:, 1], -X[:, 0] + X[:, 1]
    comp = comp_all[:n]
    z = np.where(comp, mu1 + 0.5 * eps1_all[:n], mu2 + 0.5 * eps2_all[:n])

    def true_density(X_test, z_grid):
        m1 = X_test[:, 0] + X_test[:, 1]
        m2 = -X_test[:, 0] + X_test[:, 1]
        p1 = stats.norm.pdf(z_grid[None, :], m1[:, None], 0.5)
        p2 = stats.norm.pdf(z_grid[None, :], m2[:, None], 0.5)
        return 0.5 * p1 + 0.5 * p2

    tag = f"Bimodal-d{d}-{n}"
    return X, z, tag, true_density


def make_bimodal_input_weighted_example(n=1000, d=5, seed=42):
    """Example-only bimodal DGP with input-dependent mixture weights.

    This generator is intentionally not part of the default benchmark
    schedule in ``load_all_datasets``. It exists as a concrete reference
    for how to implement a conditional mixture where the component
    probability depends on X.
    """
    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.randn(_MAX_N, d)
    logits_all = 1.5 * X_all[:, 0] - 0.75 * X_all[:, 1]
    weight_all = stats.norm.cdf(logits_all)
    comp_all = data_rng.binomial(1, weight_all)
    eps1_all = data_rng.randn(_MAX_N)
    eps2_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    mu1 = X[:, 0] + X[:, 1]
    mu2 = -X[:, 0] + X[:, 1]
    comp = comp_all[:n]
    z = np.where(comp, mu1 + 0.5 * eps1_all[:n], mu2 + 0.5 * eps2_all[:n])

    def true_density(X_test, z_grid):
        m1 = X_test[:, 0] + X_test[:, 1]
        m2 = -X_test[:, 0] + X_test[:, 1]
        logits = 1.5 * X_test[:, 0] - 0.75 * X_test[:, 1]
        weight = stats.norm.cdf(logits)
        p1 = stats.norm.pdf(z_grid[None, :], m1[:, None], 0.5)
        p2 = stats.norm.pdf(z_grid[None, :], m2[:, None], 0.5)
        return weight[:, None] * p1 + (1.0 - weight[:, None]) * p2

    tag = f"BimodalInputWeightedExample-d{d}-{n}"
    return X, z, tag, true_density


def make_skewed(n=1000, d=5, seed=42):
    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.randn(_MAX_N, d)
    # gamma draws depend on shape; shape depends on X[:,0] which is fixed
    shape_all = 2 + np.abs(X_all[:, 0])
    gamma_all = np.array([data_rng.gamma(s, 0.5) for s in shape_all])

    X = X_all[:n]
    z = gamma_all[:n] + X[:, 1]

    def true_density(X_test, z_grid):
        shapes = 2 + np.abs(X_test[:, 0])
        shifts = X_test[:, 1]
        result = np.zeros((len(X_test), len(z_grid)))
        for i in range(len(X_test)):
            z_shifted = z_grid - shifts[i]
            mask = z_shifted > 0
            result[i, mask] = stats.gamma.pdf(z_shifted[mask], a=shapes[i], scale=0.5)
        return result

    tag = f"Skewed-d{d}-{n}"
    return X, z, tag, true_density


def make_linear_gaussian_homo(n=1000, d=5, seed=42):
    """True DGP: Z = X'beta + sigma*eps, eps~N(0,1), sigma constant."""
    param_rng = np.random.RandomState(seed)
    beta = param_rng.randn(d)
    sigma = 1.0

    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.randn(_MAX_N, d)
    eps_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    z = X @ beta + sigma * eps_all[:n]

    def true_density(X_test, z_grid):
        mu = X_test @ beta
        return stats.norm.pdf(z_grid[None, :], mu[:, None], sigma)

    tag = f"LinGauss-Homo-d{d}-{n}"
    return X, z, tag, true_density


def make_nonlinear(n=1000, d=5, seed=42):
    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.randn(_MAX_N, d)
    eps_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    mu = np.sin(X[:, 0] * 2) + X[:, 1] ** 2 + 0.5 * X[:, 2]
    sigma = 0.3 + 0.3 * np.abs(np.cos(X[:, 0]))
    z = mu + sigma * eps_all[:n]

    def true_density(X_test, z_grid):
        m = np.sin(X_test[:, 0] * 2) + X_test[:, 1] ** 2 + 0.5 * X_test[:, 2]
        sig = 0.3 + 0.3 * np.abs(np.cos(X_test[:, 0]))
        return stats.norm.pdf(z_grid[None, :], m[:, None], sig[:, None])

    tag = f"Nonlinear-d{d}-{n}"
    return X, z, tag, true_density


def make_interaction(n=1000, d=5, seed=42):
    """DGP with interaction terms: mean and variance depend on X0*X1, X1*X2."""
    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.randn(_MAX_N, d)
    eps_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    mu = 1.0 * X[:, 0] * X[:, 1] + 0.5 * X[:, 1] * X[:, 2] + 0.3 * X[:, 0]
    sigma = 0.3 + 0.5 * np.abs(X[:, 0] * X[:, 1])
    z = mu + sigma * eps_all[:n]

    def true_density(X_test, z_grid):
        m = 1.0 * X_test[:, 0] * X_test[:, 1] + 0.5 * X_test[:, 1] * X_test[:, 2] + 0.3 * X_test[:, 0]
        sig = 0.3 + 0.5 * np.abs(X_test[:, 0] * X_test[:, 1])
        return stats.norm.pdf(z_grid[None, :], m[:, None], sig[:, None])

    tag = f"Interaction-d{d}-{n}"
    return X, z, tag, true_density


def make_friedman1(n=1000, d=10, seed=42):
    """Friedman #1: z = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + N(0,1).

    d >= 5; features beyond the first 5 are irrelevant uniform noise.
    True density is Gaussian with known mean and unit variance.
    """
    d = max(d, 5)  # sklearn requires at least 5 features
    data_rng = np.random.RandomState(seed + 1000)
    X_all = data_rng.uniform(0, 1, (_MAX_N, d))
    eps_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    mu_all = (10 * np.sin(np.pi * X_all[:, 0] * X_all[:, 1])
              + 20 * (X_all[:, 2] - 0.5) ** 2
              + 10 * X_all[:, 3] + 5 * X_all[:, 4])
    z = mu_all[:n] + eps_all[:n]

    def true_density(X_test, z_grid):
        m = (10 * np.sin(np.pi * X_test[:, 0] * X_test[:, 1])
             + 20 * (X_test[:, 2] - 0.5) ** 2
             + 10 * X_test[:, 3] + 5 * X_test[:, 4])
        return stats.norm.pdf(z_grid[None, :], m[:, None], 1.0)

    tag = f"Friedman1-d{d}-{n}"
    return X, z, tag, true_density


def make_friedman2(n=1000, seed=42, **_):
    """Friedman #2: z = sqrt(x1^2 + (x2*x3 - 1/(x2*x4))^2) + N(0, 125).

    Always d=4 (fixed by the DGP).
    True density is Gaussian with known mean and sigma=125.
    """
    data_rng = np.random.RandomState(seed + 1000)
    # Feature distributions as in the original Friedman #2 DGP
    x1 = data_rng.uniform(0, 100, _MAX_N)
    x2 = data_rng.uniform(40 * np.pi, 560 * np.pi, _MAX_N)
    x3 = data_rng.uniform(0, 1, _MAX_N)
    x4 = data_rng.uniform(1, 11, _MAX_N)
    X_all = np.column_stack([x1, x2, x3, x4])
    eps_all = data_rng.randn(_MAX_N)

    X = X_all[:n]
    mu_all = np.sqrt(X_all[:, 0] ** 2
                     + (X_all[:, 1] * X_all[:, 2] - 1.0 / (X_all[:, 1] * X_all[:, 3])) ** 2)
    z = mu_all[:n] + 125.0 * eps_all[:n]

    def true_density(X_test, z_grid):
        m = np.sqrt(X_test[:, 0] ** 2
                    + (X_test[:, 1] * X_test[:, 2]
                       - 1.0 / (X_test[:, 1] * X_test[:, 3])) ** 2)
        return stats.norm.pdf(z_grid[None, :], m[:, None], 125.0)

    tag = f"Friedman2-d4-{n}"
    return X, z, tag, true_density

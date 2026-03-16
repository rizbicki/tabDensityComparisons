"""
Real-world and semi-synthetic dataset loaders.
"""

import numpy as np
from sklearn.datasets import make_friedman1, make_friedman2, fetch_openml

from .synthetic import (
    make_heteroscedastic,
    make_bimodal,
    make_skewed,
    make_nonlinear,
    make_linear_gaussian_homo,
)


def _subsample(X, z, n_max, seed=42):
    """Subsample large datasets.

    Uses a fixed permutation so that smaller subsamples are always
    strict prefixes of larger ones (e.g. the n=1000 rows are the
    first 1000 of the n=2000 rows).
    """
    if len(z) <= n_max:
        return X, z
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(z))
    idx = perm[:n_max]
    return X[idx], z[idx]


# (openml_id, name, true_n)
_REAL_DATASETS = [
    (507, "SpaceGA",    3107),
    (216, "Elevators",  16599),
    (189, "Kin8nm",     8192),
    (225, "Puma8NH",    8192),
    (218, "Bank8FM",    22784),
    (197, "CPUact",     8192),
]


def _load_real_at_n(datasets, target_n):
    """Load each real dataset subsampled to target_n (skip if too small)."""
    for data_id, name, true_n in _REAL_DATASETS:
        if true_n < target_n:
            continue
        try:
            d = fetch_openml(data_id=data_id, as_frame=False, parser='auto')
            X_d, z_d = d.data.astype(float), d.target.astype(float)
            X_d, z_d = _subsample(X_d, z_d, target_n)
            tag = f"{name}-{target_n}" if target_n != true_n else name
            datasets.append((X_d, z_d, tag, None))
        except Exception as e:
            print(f"  [{name}-{target_n} skipped: {e}]")


def load_all_datasets(quick=False):
    # Synthetic: d ∈ {5, 10, 50}, n ∈ {1000, 2000, 4000, 6000}
    synthetic_generators = [
        make_heteroscedastic,
        make_bimodal,
        make_skewed,
        make_nonlinear,
        make_linear_gaussian_homo,
    ]
    datasets = []
    for gen in synthetic_generators:
        for d in [5, 10, 50]:
            for n in [1000, 2000, 4000, 6000, 20000]:
                datasets.append(gen(n=n, d=d))

    if not quick:
        # Semi-synthetic
        X, z = make_friedman1(n_samples=1500, n_features=10, noise=1.0,
                              random_state=42)
        datasets.append((X, z, "Friedman1", None))
        X, z = make_friedman2(n_samples=1500, noise=50.0, random_state=42)
        datasets.append((X, z, "Friedman2", None))

        # Real datasets at n=1000, 2000, 4000, 6000, 20000
        for target_n in [1000, 2000, 4000, 6000, 20000]:
            _load_real_at_n(datasets, target_n)

    return datasets

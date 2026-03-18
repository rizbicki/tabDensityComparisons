"""
Real-world dataset loaders.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

from .synthetic import (
    make_heteroscedastic,
    make_bimodal,
    make_skewed,
    make_nonlinear,
    make_linear_gaussian_homo,
    make_interaction,
    make_friedman1,
    make_friedman2,
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
    (507,   "SpaceGA",    3107),
    (216,   "Elevators",  16599),
    (189,   "Kin8nm",     8192),
    (225,   "Puma8NH",    8192),
    (218,   "Bank8FM",    22784),
    (197,   "CPUact",     8192),
    (537,   "CalHousing", 20640),
    (42225, "Diamonds",   53940),
    (183,   "Abalone",    4177),
    (296,   "Ailerons",   13750),
    (42712, "BikeSharing", 17379),
    (43926, "AmesHousing", 2930),
    (574,   "House16H",   22784),
    (23515, "Sulfur",     10081),
    (42688, "BrazilianHouses", 10692),
    (201,   "Pol",        15000),
    (42570, "MercedesBenz", 4209),
    (46588, "Protein",    45730),
    (44027, "Year",       515345),
    (43144, "SGEMM_GPU",  241600),
    (41540, "BlackFriday",166821),
]


def _load_openml_regression(data_id):
    """Load an OpenML regression dataset, handling mixed feature types."""
    d = fetch_openml(data_id=data_id, as_frame=True, parser='auto')

    X_df = d.data.copy()
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    # One-hot encode categoricals so mixed-type OpenML datasets load cleanly.
    X_df = pd.get_dummies(X_df, dummy_na=False)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')

    z = pd.to_numeric(d.target, errors='coerce')
    keep = (~z.isna()) & (~X_df.isna().any(axis=1))

    X = X_df.loc[keep].to_numpy(dtype=float, copy=False)
    z = z.loc[keep].to_numpy(dtype=float, copy=False)
    return X, z


def _load_real_at_n(datasets, target_n):
    """Load each real dataset subsampled to target_n (skip if too small)."""
    for data_id, name, true_n in _REAL_DATASETS:
        if true_n < target_n:
            continue
        try:
            X_d, z_d = _load_openml_regression(data_id)
            X_d, z_d = _subsample(X_d, z_d, target_n)
            tag = f"{name}-{target_n}" if target_n != true_n else name
            datasets.append((X_d, z_d, tag, None))
        except Exception as e:
            print(f"  [{name}-{target_n} skipped: {e}]")


def _load_sdss(datasets, target_n):
    """Load SDSS galaxy photo-z dataset (ugriz magnitudes → spectroscopic z)."""
    csv_path = Path(__file__).parent / "sdss_galaxies.csv"
    if not csv_path.exists():
        print(f"  [SDSS skipped: {csv_path} not found]")
        return
    try:
        # CSV has a '#Table1' header line, then column names, then data
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=2, dtype=float)
        # cols: objid, u, g, r, i, z, err_u, err_g, err_r, err_i, err_z, redshift
        X_all = data[:, 1:6]   # ugriz magnitudes
        z_all = data[:, 11]    # spectroscopic redshift
        if len(z_all) < target_n:
            return
        X_sub, z_sub = _subsample(X_all, z_all, target_n)
        tag = f"SDSS-{target_n}"
        datasets.append((X_sub, z_sub, tag, None))
    except Exception as e:
        print(f"  [SDSS-{target_n} skipped: {e}]")


def load_real_only_datasets():
    """Load only real (OpenML + SDSS) datasets."""
    datasets = []
    for target_n in [1000, 2000, 4000, 6000, 20000]:
        _load_real_at_n(datasets, target_n)
        _load_sdss(datasets, target_n)
    return datasets


def load_all_datasets(quick=False):
    # Synthetic: d ∈ {5, 10, 50}, n ∈ {1000, 2000, 4000, 6000, 20000}
    synthetic_generators = [
        make_heteroscedastic,
        make_bimodal,
        make_skewed,
        make_nonlinear,
        make_linear_gaussian_homo,
        make_interaction,
        make_friedman1,   # d varies (>=5); extra features are irrelevant noise
    ]
    datasets = []
    for gen in synthetic_generators:
        for d in [5, 10, 50]:
            for n in [1000, 2000, 4000, 6000, 20000]:
                datasets.append(gen(n=n, d=d))

    # Friedman2 has fixed d=4
    for n in [1000, 2000, 4000, 6000, 20000]:
        datasets.append(make_friedman2(n=n))

    if not quick:
        # Real datasets at n=1000, 2000, 4000, 6000, 20000
        for target_n in [1000, 2000, 4000, 6000, 20000]:
            _load_real_at_n(datasets, target_n)

        # SDSS photo-z dataset
        for target_n in [1000, 2000, 4000, 6000, 20000]:
            _load_sdss(datasets, target_n)

    return datasets

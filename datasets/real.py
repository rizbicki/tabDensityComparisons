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
    (227,   "CpuSmall",   8192),
    (197,   "CPUact",     8192),
    (537,   "CalHousing", 20640),
    (42225, "Diamonds",   53940),
    (183,   "Abalone",    4177),
    (296,   "Ailerons",   13750),
    (42712, "BikeSharing", 17379),
    (43926, "AmesHousing", 2930),
    # The paper reports "Digits" as a regression benchmark by using numeric
    # class labels from OpenML's optdigits dataset.
    (28,    "Digits",     5620),
    (574,   "House16H",   22784),
    (42079, "HouseSales", 21613),
    (42208, "NYCTaxi",    581835),
    (23515, "Sulfur",     10081),
    (42688, "BrazilianHouses", 10692),
    # Added benchmark tasks with moderate post-encoding covariate dimension
    # so the full method suite remains practical to run.
    (44964, "Superconductivity", 21263),
    (44975, "WaveEnergy", 72000),
    (44974, "VideoTranscoding", 68784),
    (43873, "SARCOS", 44484),
    (44969, "NavalPropulsion", 11934),
    (44973, "GridStability", 10000),
    (44983, "MiamiHousing", 13932),
    (44993, "HealthInsurance", 22272),
    (44984, "CPS88Wages", 28155),
    (44971, "WhiteWine", 4898),
    (44026, "FIFA", 18063),
    (44981, "Puma32NH", 8192),
    (201,   "Pol",        15000),
    (42570, "MercedesBenz", 4209),
    (46588, "Protein",    45730),
    (688,   "VisualizingSoil", 8641),
    (44027, "Year",       515345),
    (43144, "SGEMM_GPU",  241600),
    (41540, "BlackFriday",166821),
    # High-dimensional datasets (d ≈ 250–525 raw numeric features)
    (46300, "CTSlices",   53500),   # d=384, CT scan slice localization
    (422,   "Topo",       8885),    # d=266, topology prediction
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
    try:
        X_all, z_all = load_sdss_dataset()
        if len(z_all) < target_n:
            return
        X_sub, z_sub = _subsample(X_all, z_all, target_n)
        tag = f"SDSS-{target_n}"
        datasets.append((X_sub, z_sub, tag, None))
    except Exception as e:
        print(f"  [SDSS-{target_n} skipped: {e}]")


def load_sdss_dataset(target_n=None, seed=42):
    """Load SDSS ugriz photometry and spectroscopic redshift target.

    If ``target_n`` is provided, return a deterministic subsample using the
    same permutation logic as the rest of the real-data benchmark.
    """
    csv_path = Path(__file__).parent / "sdss_galaxies.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"SDSS CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=2, dtype=float)
    if data.ndim != 2 or data.shape[1] < 12:
        raise ValueError(
            f"Unexpected SDSS CSV shape {data.shape}; expected 2D array with >=12 columns"
        )

    # Columns: objid, u, g, r, i, z, err_u, err_g, err_r, err_i, err_z, redshift
    X = data[:, 1:6]
    z = data[:, 11]

    keep = np.isfinite(X).all(axis=1) & np.isfinite(z)
    X = X[keep]
    z = z[keep]

    if target_n is None:
        return X, z
    if target_n < 1:
        raise ValueError("target_n must be positive")
    if target_n > len(z):
        raise ValueError(
            f"Requested target_n={target_n:,}, but SDSS only has {len(z):,} usable rows"
        )
    return _subsample(X, z, target_n, seed=seed)


_REAL_TARGET_NS = [50, 500, 1000, 5000, 10000, 20000]


def load_real_only_datasets():
    """Load only real (OpenML + SDSS) datasets."""
    datasets = []
    for target_n in _REAL_TARGET_NS:
        _load_real_at_n(datasets, target_n)
        _load_sdss(datasets, target_n)
    return datasets


def load_all_datasets(include_real=True):
    # Synthetic: d ∈ {5, 10, 50}, n matches the real-data schedule.
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
            for n in _REAL_TARGET_NS:
                datasets.append(gen(n=n, d=d))

    # Friedman2 has fixed d=4
    for n in _REAL_TARGET_NS:
        datasets.append(make_friedman2(n=n))

    if include_real:
        # Real datasets at n=50, 500, 1000, 5000, 10000, 20000
        for target_n in _REAL_TARGET_NS:
            _load_real_at_n(datasets, target_n)

        # SDSS photo-z dataset
        for target_n in _REAL_TARGET_NS:
            _load_sdss(datasets, target_n)

    return datasets

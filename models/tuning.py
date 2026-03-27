"""
Lightweight hyperparameter tuning via random search with K-fold CV.

Uses CDE loss as the selection criterion (proper scoring rule).
"""

import numpy as np
from sklearn.model_selection import KFold
from evaluation.metrics import compute_all_metrics


# ── Search spaces ────────────────────────────────────────────────────────────

MDN_SPACE = {
    'n_components': [2, 3, 5],
    'n_hidden':     [16, 32, 64],
    'lr':           [0.005, 0.01, 0.02],
    'n_epochs':     [300, 500, 800],
}

FLOW_SPACE = {
    'n_bins':       [4, 8, 12],
    'n_layers':     [2, 3, 4],
    'hidden_units': [32, 64, 128],
    'lr':           [1e-3, 2e-3, 5e-3],
    'weight_decay': [1e-6, 1e-5, 1e-4],
}

QUANTILE_GBM_SPACE = {
    'n_estimators':  [50, 100, 200],
    'max_depth':     [3, 4, 6],
    'learning_rate': [0.05, 0.1, 0.2],
}

BART_SPACE = {
    'num_trees':  [20, 30, 50],
    'num_sweeps': [40, 60, 80],
}

QUANTILE_LINEAR_SPACE = {
    'regularization': [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
}

CATEGORICAL_MLP_SPACE = {
    'n_bins':   [30, 50, 100],
    'n_hidden': [32, 64, 128],
    'lr':       [0.005, 0.01, 0.02],
    'n_epochs': [300, 500, 800],
}


def _sample_configs(space, n_configs, rng):
    """Draw n_configs random configurations from a search space."""
    configs = []
    for _ in range(n_configs):
        cfg = {k: rng.choice(v) for k, v in space.items()}
        # Convert numpy types to Python types for clean kwargs passing
        cfg = {k: v.item() if hasattr(v, 'item') else v for k, v in cfg.items()}
        configs.append(cfg)
    return configs


# ── Generic CV tuning ────────────────────────────────────────────────────────

def tune_density_method(density_fn, X_train, z_train, search_space,
                        n_configs=8, n_folds=3, n_grid=200,
                        extra_kwargs=None, random_state=42):
    """Tune a density estimation method via random-search + K-fold CV.

    Parameters
    ----------
    density_fn : callable
        Function with signature (X_train, z_train, X_test, n_grid, z_min,
        z_max, **kwargs) -> (cdes, z_grid).
    X_train, z_train : arrays
        Training data (already scaled).
    search_space : dict
        Keys are parameter names, values are lists of candidates.
    n_configs : int
        Number of random configurations to try.
    n_folds : int
        Number of CV folds.
    n_grid : int
        Grid points for density evaluation.
    extra_kwargs : dict or None
        Fixed kwargs always passed to density_fn (e.g. device='cuda').
    random_state : int
        RNG seed.

    Returns
    -------
    best_params : dict
        Best hyperparameter configuration.
    best_score : float
        Mean CDE loss of best configuration.
    """
    rng = np.random.RandomState(random_state)
    configs = _sample_configs(search_space, n_configs, rng)
    extra_kwargs = extra_kwargs or {}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    best_score = np.inf
    best_params = configs[0]

    for i, cfg in enumerate(configs):
        fold_scores = []
        kwargs = {**cfg, **extra_kwargs}

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            z_tr, z_val = z_train[tr_idx], z_train[val_idx]

            z_lo = z_tr.min() - 0.05 * np.ptp(z_tr)
            z_hi = z_tr.max() + 0.05 * np.ptp(z_tr)

            try:
                cdes, z_grid = density_fn(
                    X_tr, z_tr, X_val, n_grid=n_grid,
                    z_min=z_lo, z_max=z_hi, **kwargs
                )
                metrics = compute_all_metrics(cdes, z_grid, z_val)
                fold_scores.append(metrics['CDE_loss'])
            except Exception as e:
                print(f"    [tune] config {i} fold {fold_idx} failed: {e}")
                fold_scores.append(np.inf)

        mean_score = np.mean(fold_scores)
        print(f"    [tune] config {i}: {cfg} -> CDE={mean_score:.4f}")

        if mean_score < best_score:
            best_score = mean_score
            best_params = cfg

    print(f"    [tune] best: {best_params} -> CDE={best_score:.4f}")
    return best_params, best_score


# ── Tuned wrapper functions ──────────────────────────────────────────────────

def mdn_density_tuned(X_train, z_train, X_test, n_grid=200,
                      z_min=None, z_max=None, n_configs=8, n_folds=3,
                      random_state=42):
    """MDN with random-search tuning over components, hidden size, and lr."""
    from models.baselines import mdn_density

    best_params, _ = tune_density_method(
        mdn_density, X_train, z_train, MDN_SPACE,
        n_configs=n_configs, n_folds=n_folds, n_grid=n_grid,
        random_state=random_state,
    )
    return mdn_density(X_train, z_train, X_test, n_grid=n_grid,
                       z_min=z_min, z_max=z_max, **best_params)


def normalizing_flow_density_tuned(X_train, z_train, X_test, n_grid=200,
                                    z_min=None, z_max=None, device='auto',
                                    n_configs=8, n_folds=3, random_state=42):
    """Spline flow with random-search tuning."""
    from models.baselines import normalizing_flow_density

    best_params, best_score = tune_density_method(
        normalizing_flow_density, X_train, z_train, FLOW_SPACE,
        n_configs=n_configs, n_folds=n_folds, n_grid=n_grid,
        extra_kwargs={'device': device},
        random_state=random_state,
    )
    if not np.isfinite(best_score):
        raise RuntimeError(
            "CUDA out of memory: all Flow-Spline tuning configurations failed")
    return normalizing_flow_density(X_train, z_train, X_test, n_grid=n_grid,
                                    z_min=z_min, z_max=z_max, device=device,
                                    **best_params)


def quantile_gbm_density_tuned(X_train, z_train, X_test, n_grid=200,
                                z_min=None, z_max=None, n_configs=8,
                                n_folds=3, random_state=42):
    """Quantile GBM/XGBoost with random-search tuning."""
    from models.baselines import quantile_gbm_density

    best_params, _ = tune_density_method(
        quantile_gbm_density, X_train, z_train, QUANTILE_GBM_SPACE,
        n_configs=n_configs, n_folds=n_folds, n_grid=n_grid,
        random_state=random_state,
    )
    return quantile_gbm_density(X_train, z_train, X_test, n_grid=n_grid,
                                z_min=z_min, z_max=z_max, **best_params)


def bart_homo_density_tuned(X_train, z_train, X_test, n_grid=200,
                             z_min=None, z_max=None, n_configs=8,
                             n_folds=3, random_state=42):
    """BART-Homo with random-search tuning over num_trees and num_sweeps."""
    from models.baselines import bart_homo_density

    best_params, _ = tune_density_method(
        bart_homo_density, X_train, z_train, BART_SPACE,
        n_configs=n_configs, n_folds=n_folds, n_grid=n_grid,
        random_state=random_state,
    )
    return bart_homo_density(X_train, z_train, X_test, n_grid=n_grid,
                             z_min=z_min, z_max=z_max, **best_params)


def quantile_linear_density_tuned(X_train, z_train, X_test, n_grid=200,
                                   z_min=None, z_max=None, n_configs=6,
                                   n_folds=3, random_state=42):
    """Quantile-Linear with random-search tuning over regularization strength."""
    from models.baselines import quantile_linear_density

    best_params, _ = tune_density_method(
        quantile_linear_density, X_train, z_train, QUANTILE_LINEAR_SPACE,
        n_configs=n_configs, n_folds=n_folds, n_grid=n_grid,
        random_state=random_state,
    )
    return quantile_linear_density(X_train, z_train, X_test, n_grid=n_grid,
                                   z_min=z_min, z_max=z_max, **best_params)


def bart_hetero_density_tuned(X_train, z_train, X_test, n_grid=200,
                               z_min=None, z_max=None, n_configs=8,
                               n_folds=3, random_state=42):
    """BART-Hetero with random-search tuning over num_trees and num_sweeps."""
    from models.baselines import bart_hetero_density

    best_params, _ = tune_density_method(
        bart_hetero_density, X_train, z_train, BART_SPACE,
        n_configs=n_configs, n_folds=n_folds, n_grid=n_grid,
        random_state=random_state,
    )
    return bart_hetero_density(X_train, z_train, X_test, n_grid=n_grid,
                               z_min=z_min, z_max=z_max, **best_params)


def categorical_mlp_density_tuned(X_train, z_train, X_test, n_grid=200,
                                   z_min=None, z_max=None, n_configs=8,
                                   n_folds=3, random_state=42):
    """Categorical MLP with random-search tuning over bins, hidden size, and lr."""
    from models.baselines import categorical_mlp_density

    best_params, _ = tune_density_method(
        categorical_mlp_density, X_train, z_train, CATEGORICAL_MLP_SPACE,
        n_configs=n_configs, n_folds=n_folds, n_grid=n_grid,
        random_state=random_state,
    )
    return categorical_mlp_density(X_train, z_train, X_test, n_grid=n_grid,
                                    z_min=z_min, z_max=z_max, **best_params)

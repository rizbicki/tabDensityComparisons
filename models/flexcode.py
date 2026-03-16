"""
FlexCode: Conditional Density Estimation via orthonormal basis expansion.

Reference: Izbicki & Lee (2017), Electronic Journal of Statistics.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class FlexCodeEstimator:
    """
    FlexCode: f(z|x) = sum_{i=0}^{I-1} beta_i(x) * phi_i(z)

    where beta_i(x) = E[phi_i(Z)|X=x], estimated by regressing phi_i(Z) on X.
    """

    def __init__(self, regressor_factory, max_basis=31, basis_system='cosine',
                 regressor_params=None, name='FlexCode', n_jobs=-1):
        self.regressor_factory = regressor_factory
        self.max_basis = max_basis
        self.basis_system = basis_system
        self.regressor_params = regressor_params or {}
        self.name = name
        self.n_jobs = n_jobs
        self.regressors_ = []
        self.z_min_ = None
        self.z_max_ = None
        self.best_basis_ = None

    def _normalize_z(self, z):
        return (z - self.z_min_) / (self.z_max_ - self.z_min_)

    def _cosine_basis(self, z_norm, i):
        if i == 0:
            return np.ones_like(z_norm)
        return np.sqrt(2) * np.cos(i * np.pi * z_norm)

    def _fit_one_basis(self, X, z_norm, i):
        phi_values = self._cosine_basis(z_norm, i)
        reg = self.regressor_factory(**self.regressor_params)
        reg.fit(X, phi_values)
        return reg

    def fit(self, X, z):
        margin = 0.05 * (z.max() - z.min())
        self.z_min_ = z.min() - margin
        self.z_max_ = z.max() + margin
        z_norm = self._normalize_z(z)

        self.regressors_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one_basis)(X, z_norm, i)
            for i in range(self.max_basis)
        )
        return self

    def tune(self, X_val, z_val):
        z_val_norm = self._normalize_z(z_val)
        n_val = len(z_val)

        beta_hat = np.zeros((n_val, self.max_basis))
        for i, reg in enumerate(self.regressors_):
            beta_hat[:, i] = reg.predict(X_val)

        best_loss = np.inf
        best_I = 1
        scale = 1.0 / (self.z_max_ - self.z_min_)

        cumul_sq = np.zeros(n_val)
        cumul_eval = np.zeros(n_val)

        for I in range(1, self.max_basis + 1):
            i = I - 1
            phi_val = self._cosine_basis(z_val_norm, i)
            cumul_sq += beta_hat[:, i] ** 2
            cumul_eval += beta_hat[:, i] * phi_val
            loss = np.mean(cumul_sq * scale - 2 * cumul_eval * scale)
            if loss < best_loss:
                best_loss = loss
                best_I = I

        self.best_basis_ = best_I
        self.best_loss_ = best_loss
        return best_loss

    def fit_cv(self, X, z, n_folds=5):
        """Fit with K-fold CV for tuning I, then refit on all data."""
        margin = 0.05 * (z.max() - z.min())
        self.z_min_ = z.min() - margin
        self.z_max_ = z.max() + margin
        z_norm = self._normalize_z(z)

        scale = 1.0 / (self.z_max_ - self.z_min_)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        oof_loss = np.zeros(self.max_basis)
        oof_count = 0

        for train_idx, val_idx in kf.split(X):
            X_tr_fold = X[train_idx]
            X_va_fold = X[val_idx]
            z_norm_tr = z_norm[train_idx]
            z_norm_va = z_norm[val_idx]
            n_va = len(val_idx)

            def _fit_and_predict(i):
                phi_tr = self._cosine_basis(z_norm_tr, i)
                reg = self.regressor_factory(**self.regressor_params)
                reg.fit(X_tr_fold, phi_tr)
                return i, reg.predict(X_va_fold)

            fold_beta_hat = np.zeros((n_va, self.max_basis))
            results_list = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_and_predict)(i)
                for i in range(self.max_basis)
            )
            for i, preds in results_list:
                fold_beta_hat[:, i] = preds

            cumul_sq = np.zeros(n_va)
            cumul_eval = np.zeros(n_va)
            for I in range(1, self.max_basis + 1):
                idx = I - 1
                phi_va = self._cosine_basis(z_norm_va, idx)
                cumul_sq += fold_beta_hat[:, idx] ** 2
                cumul_eval += fold_beta_hat[:, idx] * phi_va
                oof_loss[idx] += np.sum(cumul_sq * scale - 2 * cumul_eval * scale)
            oof_count += n_va

        oof_loss /= oof_count
        self.best_basis_ = int(np.argmin(oof_loss)) + 1
        self.best_loss_ = oof_loss[self.best_basis_ - 1]

        # Refit all regressors on the full data
        self.regressors_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one_basis)(X, z_norm, i)
            for i in range(self.max_basis)
        )
        return self

    def predict(self, X, n_grid=200):
        z_grid = np.linspace(self.z_min_, self.z_max_, n_grid)
        z_grid_norm = self._normalize_z(z_grid)
        n_test = X.shape[0]
        I = self.best_basis_ or self.max_basis

        beta_hat = np.zeros((n_test, I))
        for i in range(I):
            beta_hat[:, i] = self.regressors_[i].predict(X)

        phi_grid = np.zeros((I, n_grid))
        for i in range(I):
            phi_grid[i, :] = self._cosine_basis(z_grid_norm, i)

        scale = 1.0 / (self.z_max_ - self.z_min_)
        cdes = (beta_hat @ phi_grid) * scale

        # Project to valid density: non-negative + normalize
        cdes = np.maximum(cdes, 0)
        dz = z_grid[1] - z_grid[0]
        row_sums = cdes.sum(axis=1) * dz
        row_sums[row_sums == 0] = 1.0
        cdes = cdes / row_sums[:, None]

        return cdes, z_grid


# ── Regressor wrappers ──────────────────────────────────────────────────────

class GBMFlexRegressor:
    """sklearn GradientBoosting wrapper."""
    def __init__(self):
        pass
    def fit(self, X, y):
        self.model_ = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.model_.fit(X, y)
        return self
    def predict(self, X):
        return self.model_.predict(X)


class RFFlexRegressor:
    """Random Forest wrapper."""
    def __init__(self):
        pass
    def fit(self, X, y):
        self.model_ = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42
        )
        self.model_.fit(X, y)
        return self
    def predict(self, X):
        return self.model_.predict(X)

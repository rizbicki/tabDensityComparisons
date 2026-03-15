"""
Classical density estimation baselines: Linear-Gaussian, MDN, Quantile methods, GLMs.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV


def _make_reg(regularized=False):
    """Return a RidgeCV (LOO) regressor if regularized, else OLS."""
    if regularized:
        return RidgeCV(alphas=np.logspace(-4, 4, 20), cv=None)  # cv=None → LOO
    return LinearRegression()


def linear_gaussian_homo_density(X_train, z_train, X_test, n_grid=200,
                                  z_min=None, z_max=None, regularized=False):
    """Linear Gaussian with constant variance: f(z|x) = N(x'beta, sigma^2)."""
    reg = _make_reg(regularized).fit(X_train, z_train)
    mu_test = reg.predict(X_test)
    residuals = z_train - reg.predict(X_train)
    sigma = np.std(residuals, ddof=X_train.shape[1] + 1)
    sigma = max(sigma, 1e-8)

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = stats.norm.pdf(z_grid[None, :], mu_test[:, None], sigma)
    return cdes, z_grid


def linear_gaussian_hetero_density(X_train, z_train, X_test, n_grid=200,
                                    z_min=None, z_max=None, regularized=False):
    """Linear Gaussian with input-dependent variance.

    Stage 1: fit E[Z|X] = X'beta.
    Stage 2: fit log(residual^2) = X'gamma to model variance.
    """
    reg_mean = _make_reg(regularized).fit(X_train, z_train)
    mu_train = reg_mean.predict(X_train)
    mu_test = reg_mean.predict(X_test)

    residuals = z_train - mu_train
    log_sq_res = np.log(np.maximum(residuals ** 2, 1e-12))
    reg_var = _make_reg(regularized).fit(X_train, log_sq_res)
    log_var_test = reg_var.predict(X_test)
    sigma_test = np.sqrt(np.exp(log_var_test))
    sigma_test = np.maximum(sigma_test, 1e-8)

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = stats.norm.pdf(z_grid[None, :], mu_test[:, None], sigma_test[:, None])
    return cdes, z_grid


def mdn_density(X_train, z_train, X_test, n_grid=200,
                z_min=None, z_max=None, n_components=2, n_hidden=16,
                n_epochs=500, lr=0.01):
    """Mixture Density Network with few parameters (2-Gaussian mixture).

    Architecture: X -> Linear(d, n_hidden) -> ReLU -> Linear(n_hidden, 3*K)
    where K = n_components.  Outputs: mixing weights (softmax), means, log-stds.
    """
    import torch
    import torch.nn as nn

    d = X_train.shape[1]
    K = n_components

    z_mean, z_std = z_train.mean(), z_train.std()
    z_std = max(z_std, 1e-8)
    z_tr_s = (z_train - z_mean) / z_std

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    z_tr_t = torch.tensor(z_tr_s, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)

    class MDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(d, n_hidden)
            self.out = nn.Linear(n_hidden, 3 * K)

        def forward(self, x):
            h = torch.relu(self.hidden(x))
            params = self.out(h)
            pi_logits = params[:, :K]
            mus = params[:, K:2*K]
            log_sigmas = params[:, 2*K:3*K]
            pi = torch.softmax(pi_logits, dim=1)
            sigmas = torch.exp(log_sigmas.clamp(-5, 3))
            return pi, mus, sigmas

    model = MDN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        pi, mus, sigmas = model(X_tr_t)
        z_exp = z_tr_t.expand_as(mus)
        normal_lp = -0.5 * ((z_exp - mus) / sigmas) ** 2 \
                    - torch.log(sigmas) - 0.5 * np.log(2 * np.pi)
        log_mix = torch.log(pi + 1e-10) + normal_lp
        loss = -torch.logsumexp(log_mix, dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pi, mus, sigmas = model(X_te_t)
    pi = pi.numpy()
    mus_np = mus.numpy() * z_std + z_mean
    sigmas_np = sigmas.numpy() * z_std

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)

    n_test = X_test.shape[0]
    cdes = np.zeros((n_test, n_grid))
    for k in range(K):
        cdes += pi[:, k:k+1] * stats.norm.pdf(
            z_grid[None, :], mus_np[:, k:k+1], sigmas_np[:, k:k+1]
        )
    return cdes, z_grid


def quantile_gbm_density(X_train, z_train, X_test, n_grid=200,
                          z_min=None, z_max=None):
    """Quantile GBM/XGBoost baseline."""
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False

    from sklearn.ensemble import GradientBoostingRegressor

    n_test = X_test.shape[0]
    n_q = 49
    alphas_q = np.linspace(0.02, 0.98, n_q)

    Q_test = np.zeros((n_test, n_q))
    for j, alpha in enumerate(alphas_q):
        if has_xgb:
            reg = xgb.XGBRegressor(
                objective='reg:quantileerror', quantile_alpha=alpha,
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, verbosity=0
            )
        else:
            reg = GradientBoostingRegressor(
                loss='quantile', alpha=alpha,
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42
            )
        reg.fit(X_train, z_train)
        Q_test[:, j] = reg.predict(X_test)

    for i in range(n_test):
        Q_test[i, :] = np.sort(Q_test[i, :])

    if z_min is None:
        z_min = Q_test.min() - 0.1 * np.ptp(Q_test)
    if z_max is None:
        z_max = Q_test.max() + 0.1 * np.ptp(Q_test)

    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = np.zeros((n_test, n_grid))
    dz = z_grid[1] - z_grid[0]

    for i in range(n_test):
        cdf_interp = np.interp(z_grid, Q_test[i, :], alphas_q, left=0.0, right=1.0)
        pdf = np.gradient(cdf_interp, dz)
        pdf = np.maximum(pdf, 0)
        total = pdf.sum() * dz
        if total > 0:
            pdf /= total
        cdes[i, :] = pdf

    return cdes, z_grid


def quantile_linear_density(X_train, z_train, X_test, n_grid=200,
                             z_min=None, z_max=None):
    """Linear quantile regression baseline.

    Fits multiple quantile levels with linear models, then interpolates
    the resulting quantile function to produce a density on a grid.
    """
    from sklearn.linear_model import QuantileRegressor

    n_test = X_test.shape[0]
    n_q = 49
    alphas_q = np.linspace(0.02, 0.98, n_q)

    Q_test = np.zeros((n_test, n_q))
    for j, alpha in enumerate(alphas_q):
        reg = QuantileRegressor(quantile=alpha, alpha=0.0, solver='highs')
        reg.fit(X_train, z_train)
        Q_test[:, j] = reg.predict(X_test)

    # Ensure monotonicity
    for i in range(n_test):
        Q_test[i, :] = np.sort(Q_test[i, :])

    if z_min is None:
        z_min = Q_test.min() - 0.1 * np.ptp(Q_test)
    if z_max is None:
        z_max = Q_test.max() + 0.1 * np.ptp(Q_test)

    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = np.zeros((n_test, n_grid))
    dz = z_grid[1] - z_grid[0]

    for i in range(n_test):
        cdf_interp = np.interp(z_grid, Q_test[i, :], alphas_q, left=0.0, right=1.0)
        pdf = np.gradient(cdf_interp, dz)
        pdf = np.maximum(pdf, 0)
        total = pdf.sum() * dz
        if total > 0:
            pdf /= total
        cdes[i, :] = pdf

    return cdes, z_grid


def gamma_glm_density(X_train, z_train, X_test, n_grid=200,
                       z_min=None, z_max=None, regularized=False):
    """Gamma GLM with log link for the mean.

    Fits a Gamma distribution with:
      - log(mu) = X'beta  (mean varies with covariates via log link)
      - constant shape parameter (estimated from training residuals)

    Responses are shifted so that all values are strictly positive before
    fitting.  The density is shifted back to the original scale for output.
    """
    # Shift z so all values are strictly positive
    shift = 0.0
    z_min_val = z_train.min()
    if z_min_val <= 0:
        shift = -z_min_val + 1e-2 * (np.ptp(z_train) + 1.0)
    z_train_pos = z_train + shift
    z_train_pos = np.maximum(z_train_pos, 1e-10)

    # Fit log(E[Z|X]) = X'beta on log(z)
    log_z = np.log(z_train_pos)
    reg = _make_reg(regularized).fit(X_train, log_z)
    log_mu_train = reg.predict(X_train)
    log_mu_test = reg.predict(X_test)
    mu_train = np.exp(log_mu_train)
    mu_test = np.exp(log_mu_test)

    # Estimate shape parameter
    log_resid = log_z - log_mu_train
    shape = 1.0 / np.maximum(np.var(log_resid, ddof=X_train.shape[1] + 1), 1e-8)

    # Gamma parameterization: shape=a, scale=mu/a
    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)

    z_grid_pos = z_grid + shift
    n_test = X_test.shape[0]
    cdes = np.zeros((n_test, n_grid))

    for i in range(n_test):
        scale_i = mu_test[i] / shape
        mask = z_grid_pos > 0
        cdes[i, mask] = stats.gamma.pdf(z_grid_pos[mask], a=shape, scale=scale_i)

    # Normalize
    dz = z_grid[1] - z_grid[0]
    row_sums = cdes.sum(axis=1) * dz
    row_sums[row_sums == 0] = 1.0
    cdes = cdes / row_sums[:, None]

    return cdes, z_grid


def student_t_density(X_train, z_train, X_test, n_grid=200,
                       z_min=None, z_max=None, regularized=False):
    """Student-t regression: linear mean, constant scale and df.

    Stage 1: fit E[Z|X] = X'beta.
    Stage 2: estimate scale and degrees of freedom by maximising the
             Student-t log-likelihood of the residuals (profile MLE).
    """
    from scipy.optimize import minimize_scalar

    reg = _make_reg(regularized).fit(X_train, z_train)
    mu_train = reg.predict(X_train)
    mu_test = reg.predict(X_test)
    residuals = z_train - mu_train

    def neg_ll(log_df):
        df = np.exp(log_df)
        scale = np.sqrt(np.mean(residuals ** 2) * (df - 2) / df) if df > 2 \
                else np.std(residuals)
        scale = max(scale, 1e-8)
        return -np.sum(stats.t.logpdf(residuals, df=df, scale=scale))

    result = minimize_scalar(neg_ll, bounds=(np.log(2.01), np.log(200)),
                             method='bounded')
    df = np.exp(result.x)
    scale = np.sqrt(np.mean(residuals ** 2) * (df - 2) / df) if df > 2 \
            else np.std(residuals)
    scale = max(scale, 1e-8)

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)

    cdes = stats.t.pdf(z_grid[None, :], df=df, loc=mu_test[:, None], scale=scale)
    return cdes, z_grid


def lognormal_homo_density(X_train, z_train, X_test, n_grid=200,
                            z_min=None, z_max=None, regularized=False):
    """Log-Normal regression with constant variance.

    Model: log(Z - shift) ~ N(X'beta, sigma^2)
    where shift makes all values strictly positive.
    """
    shift = 0.0
    z_min_val = z_train.min()
    if z_min_val <= 0:
        shift = -z_min_val + 1e-2 * (np.ptp(z_train) + 1.0)
    z_pos = z_train + shift
    z_pos = np.maximum(z_pos, 1e-10)

    log_z = np.log(z_pos)
    reg = _make_reg(regularized).fit(X_train, log_z)
    mu_train = reg.predict(X_train)
    mu_test = reg.predict(X_test)

    residuals = log_z - mu_train
    sigma = np.std(residuals, ddof=X_train.shape[1] + 1)
    sigma = max(sigma, 1e-8)

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)

    z_grid_pos = z_grid + shift
    n_test = X_test.shape[0]
    cdes = np.zeros((n_test, n_grid))
    mask = z_grid_pos > 0
    for i in range(n_test):
        cdes[i, mask] = stats.lognorm.pdf(z_grid_pos[mask], s=sigma,
                                           scale=np.exp(mu_test[i]))

    dz = z_grid[1] - z_grid[0]
    row_sums = cdes.sum(axis=1) * dz
    row_sums[row_sums == 0] = 1.0
    cdes = cdes / row_sums[:, None]
    return cdes, z_grid


def lognormal_hetero_density(X_train, z_train, X_test, n_grid=200,
                              z_min=None, z_max=None, regularized=False):
    """Log-Normal regression with input-dependent variance.

    Stage 1: log(Z - shift) = X'beta + eps  (mean in log-space)
    Stage 2: log(eps^2) = X'gamma           (log-variance)
    """
    shift = 0.0
    z_min_val = z_train.min()
    if z_min_val <= 0:
        shift = -z_min_val + 1e-2 * (np.ptp(z_train) + 1.0)
    z_pos = z_train + shift
    z_pos = np.maximum(z_pos, 1e-10)

    log_z = np.log(z_pos)

    reg_mean = _make_reg(regularized).fit(X_train, log_z)
    mu_train = reg_mean.predict(X_train)
    mu_test = reg_mean.predict(X_test)

    residuals = log_z - mu_train
    log_sq_res = np.log(np.maximum(residuals ** 2, 1e-12))
    reg_var = _make_reg(regularized).fit(X_train, log_sq_res)
    log_var_test = reg_var.predict(X_test)
    sigma_test = np.sqrt(np.exp(log_var_test))
    sigma_test = np.maximum(sigma_test, 1e-8)

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)

    z_grid_pos = z_grid + shift
    n_test = X_test.shape[0]
    cdes = np.zeros((n_test, n_grid))
    mask = z_grid_pos > 0
    for i in range(n_test):
        cdes[i, mask] = stats.lognorm.pdf(z_grid_pos[mask], s=sigma_test[i],
                                           scale=np.exp(mu_test[i]))

    dz = z_grid[1] - z_grid[0]
    row_sums = cdes.sum(axis=1) * dz
    row_sums[row_sums == 0] = 1.0
    cdes = cdes / row_sums[:, None]
    return cdes, z_grid


# ── BART-based density estimators ────────────────────────────────────────────

def _fit_xbart(X_train, y_train, num_trees=30, num_sweeps=60, burnin=20):
    """Fit an XBART model and return (model, posterior_mean_sigma)."""
    from xbart import XBART
    model = XBART(num_trees=num_trees, num_sweeps=num_sweeps, burnin=burnin)
    model.fit(X_train, y_train)
    # sigma_draws shape: (num_sweeps, num_trees) – average over trees,
    # then take the mean of post-burnin sweeps
    sigma_arr = np.array(model.sigma_draws)
    sigma_post = sigma_arr[burnin:].mean()
    return model, sigma_post


def bart_homo_density(X_train, z_train, X_test, n_grid=200,
                      z_min=None, z_max=None):
    """BART with constant residual variance: f(z|x) = N(BART(x), sigma^2).

    Uses XBART's built-in sigma draws (posterior mean) for the residual sd.
    """
    model, sigma = _fit_xbart(X_train, z_train)
    mu_test = model.predict(X_test)
    sigma = max(sigma, 1e-8)

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = stats.norm.pdf(z_grid[None, :], mu_test[:, None], sigma)
    return cdes, z_grid


def bart_hetero_density(X_train, z_train, X_test, n_grid=200,
                        z_min=None, z_max=None):
    """Heteroscedastic BART: two-stage density estimation.

    Stage 1: fit BART for E[Z|X].
    Stage 2: fit a second BART for log(residual^2) to model Var(Z|X).
    Result: f(z|x) = N(BART_mean(x), exp(BART_var(x))).
    """
    # Stage 1: mean model
    model_mean, _ = _fit_xbart(X_train, z_train)
    mu_train = model_mean.predict(X_train)
    mu_test = model_mean.predict(X_test)

    # Stage 2: variance model on log-squared residuals
    residuals = z_train - mu_train
    log_sq_res = np.log(np.maximum(residuals ** 2, 1e-12))
    model_var, _ = _fit_xbart(X_train, log_sq_res)
    log_var_test = model_var.predict(X_test)
    sigma_test = np.sqrt(np.exp(log_var_test))
    sigma_test = np.maximum(sigma_test, 1e-8)

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = stats.norm.pdf(z_grid[None, :], mu_test[:, None], sigma_test[:, None])
    return cdes, z_grid

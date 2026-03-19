"""
Classical density estimation baselines: Linear-Gaussian, flows, MDN,
Quantile methods, GLMs, and BART.
"""

import math
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV


def _make_reg(regularized=False):
    """Return a RidgeCV (LOO) regressor if regularized, else OLS."""
    if regularized:
        return RidgeCV(alphas=np.logspace(-4, 4, 20), cv=None)  # cv=None → LOO
    return LinearRegression()


def _normalize_density_rows(cdes, z_grid):
    """Normalize each row so it integrates to one on z_grid."""
    dz = z_grid[1] - z_grid[0]
    row_sums = cdes.sum(axis=1) * dz
    row_sums[row_sums <= 0] = 1.0
    return cdes / row_sums[:, None]


def _flow_torch_device(requested):
    """Map the CLI-style device string onto a PyTorch device."""
    import torch

    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested for Flow-Spline, but no CUDA device is available")
        return torch.device('cuda')
    if requested == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _rational_quadratic_spline(inputs, widths, heights, derivatives,
                               tail_bound, inverse=False,
                               min_bin_width=1e-3, min_bin_height=1e-3,
                               min_derivative=1e-3):
    """Apply a monotone rational-quadratic spline with linear tails."""
    import torch

    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    if not torch.any(inside):
        return outputs, logabsdet

    x_in = inputs[inside]
    w_in = widths[inside]
    h_in = heights[inside]
    d_in = derivatives[inside].clone()
    n_bins = w_in.shape[1]

    width_budget = 2 * tail_bound - min_bin_width * n_bins
    height_budget = 2 * tail_bound - min_bin_height * n_bins
    if width_budget <= 0 or height_budget <= 0:
        raise ValueError("tail_bound is too small for the requested spline configuration")

    w_in = min_bin_width + width_budget * torch.softmax(w_in, dim=1)
    h_in = min_bin_height + height_budget * torch.softmax(h_in, dim=1)
    d_in = min_derivative + torch.nn.functional.softplus(d_in)
    d_in[:, 0] = 1.0
    d_in[:, -1] = 1.0

    cumwidths = torch.cumsum(w_in, dim=1)
    cumwidths = torch.cat(
        [torch.full((w_in.shape[0], 1), -tail_bound, device=w_in.device, dtype=w_in.dtype),
         -tail_bound + cumwidths],
        dim=1,
    )
    cumheights = torch.cumsum(h_in, dim=1)
    cumheights = torch.cat(
        [torch.full((h_in.shape[0], 1), -tail_bound, device=h_in.device, dtype=h_in.dtype),
         -tail_bound + cumheights],
        dim=1,
    )
    delta = h_in / w_in

    bin_locations = cumheights[:, 1:-1] if inverse else cumwidths[:, 1:-1]
    bin_idx = torch.sum(x_in[:, None] >= bin_locations, dim=1).unsqueeze(1)

    input_cumwidths = cumwidths.gather(1, bin_idx).squeeze(1)
    input_bin_widths = w_in.gather(1, bin_idx).squeeze(1)
    input_cumheights = cumheights.gather(1, bin_idx).squeeze(1)
    input_bin_heights = h_in.gather(1, bin_idx).squeeze(1)
    input_delta = delta.gather(1, bin_idx).squeeze(1)
    input_derivatives = d_in.gather(1, bin_idx).squeeze(1)
    input_derivatives_plus_one = d_in[:, 1:].gather(1, bin_idx).squeeze(1)

    if inverse:
        y = x_in
        a = ((y - input_cumheights)
             * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
             + input_bin_heights * (input_delta - input_derivatives))
        b = (input_bin_heights * input_derivatives
             - (y - input_cumheights)
             * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = -input_delta * (y - input_cumheights)
        discriminant = b.pow(2) - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant) + 1e-12)
        theta = torch.clamp(root, 0.0, 1.0)
        x = theta * input_bin_widths + input_cumwidths
        theta_one_minus_theta = theta * (1 - theta)
        denominator = (input_delta
                       + (input_derivatives + input_derivatives_plus_one
                          - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        lad = torch.log(derivative_numerator + 1e-12) - 2 * torch.log(denominator + 1e-12)
        outputs[inside] = x
        logabsdet[inside] = -lad
        return outputs, logabsdet

    theta = (x_in - input_cumwidths) / input_bin_widths
    theta = torch.clamp(theta, 0.0, 1.0)
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_bin_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = (input_delta
                   + (input_derivatives + input_derivatives_plus_one
                      - 2 * input_delta) * theta_one_minus_theta)
    y = input_cumheights + numerator / (denominator + 1e-12)
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    lad = torch.log(derivative_numerator + 1e-12) - 2 * torch.log(denominator + 1e-12)
    outputs[inside] = y
    logabsdet[inside] = lad
    return outputs, logabsdet


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


def normalizing_flow_density(X_train, z_train, X_test, n_grid=200,
                             z_min=None, z_max=None, device='auto',
                             n_bins=8, n_layers=2, hidden_units=64,
                             n_epochs=120, batch_size=512, lr=2e-3,
                             weight_decay=1e-5, patience=12,
                             val_fraction=0.1, random_state=42):
    """Conditional neural spline flow baseline for scalar responses.

    The model learns a monotone spline transform of a standard Gaussian base,
    with the spline parameters produced by an MLP conditioned on x.
    """
    import torch
    import torch.nn as nn

    class _Conditioner(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    class _ConditionalSplineFlow(nn.Module):
        def __init__(self, in_dim, tail_bound):
            super().__init__()
            self.tail_bound = float(tail_bound)
            self.n_bins = int(n_bins)
            self.n_params = 3 * self.n_bins + 1
            self.layers = nn.ModuleList(
                [_Conditioner(in_dim, self.n_params) for _ in range(n_layers)]
            )

        def _split_params(self, raw_params):
            widths = raw_params[:, :self.n_bins]
            heights = raw_params[:, self.n_bins:2 * self.n_bins]
            derivatives = raw_params[:, 2 * self.n_bins:]
            return widths, heights, derivatives

        def log_prob(self, x, z):
            u = z
            logabsdet = torch.zeros_like(z)
            for layer in reversed(self.layers):
                params = layer(x)
                widths, heights, derivatives = self._split_params(params)
                u, lad = _rational_quadratic_spline(
                    u, widths, heights, derivatives,
                    tail_bound=self.tail_bound, inverse=True,
                )
                logabsdet += lad
            base_log_prob = -0.5 * (u.pow(2) + math.log(2 * math.pi))
            return base_log_prob + logabsdet

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    flow_device = _flow_torch_device(device)

    z_mean = float(np.mean(z_train))
    z_scale = float(np.std(z_train))
    z_scale = max(z_scale, 1e-6)
    z_train_std = (z_train - z_mean) / z_scale

    if z_min is None:
        z_min = z_train.min() - 0.1 * np.ptp(z_train)
    if z_max is None:
        z_max = z_train.max() + 0.1 * np.ptp(z_train)
    z_grid = np.linspace(z_min, z_max, n_grid)
    z_grid_std = (z_grid - z_mean) / z_scale

    tail_bound = max(3.0, float(np.quantile(np.abs(z_train_std), 0.995)) + 0.5)

    n_train = X_train.shape[0]
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_train)
    val_size = max(1, int(round(val_fraction * n_train)))
    val_size = min(val_size, max(1, n_train // 5))
    if n_train - val_size < 1:
        val_size = 0

    val_idx = perm[:val_size]
    fit_idx = perm[val_size:] if val_size > 0 else perm

    X_fit_t = torch.tensor(X_train[fit_idx], dtype=torch.float32, device=flow_device)
    z_fit_t = torch.tensor(z_train_std[fit_idx], dtype=torch.float32, device=flow_device)
    X_val_t = torch.tensor(X_train[val_idx], dtype=torch.float32, device=flow_device) if val_size > 0 else None
    z_val_t = torch.tensor(z_train_std[val_idx], dtype=torch.float32, device=flow_device) if val_size > 0 else None
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=flow_device)
    z_grid_t = torch.tensor(z_grid_std, dtype=torch.float32, device=flow_device)

    model = _ConditionalSplineFlow(X_train.shape[1], tail_bound=tail_bound).to(flow_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    batch_size = min(batch_size, len(X_fit_t))
    if len(X_fit_t) >= 20000:
        batch_size = min(1024, len(X_fit_t))
    steps_per_epoch = max(1, int(np.ceil(len(X_fit_t) / batch_size)))
    effective_epochs = min(n_epochs, max(15, int(np.ceil(2500 / steps_per_epoch))))

    best_state = None
    best_val = float('inf')
    stale_epochs = 0

    for _ in range(effective_epochs):
        model.train()
        epoch_perm = torch.randperm(len(X_fit_t), device=flow_device)
        for start in range(0, len(X_fit_t), batch_size):
            idx = epoch_perm[start:start + batch_size]
            loss = -model.log_prob(X_fit_t[idx], z_fit_t[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if X_val_t is None:
                val_loss = float(-model.log_prob(X_fit_t, z_fit_t).mean().item())
            else:
                val_loss = float(-model.log_prob(X_val_t, z_val_t).mean().item())

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_batch = max(16, min(256, 32768 // max(n_grid, 1)))
    grid_pair_batch = 32768
    log_prob_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_test_t), test_batch):
            xb = X_test_t[start:start + test_batch]
            z_block = z_grid_t.unsqueeze(0).expand(len(xb), -1)
            x_block = xb[:, None, :].expand(-1, n_grid, -1).reshape(-1, X_test.shape[1])
            z_flat = z_block.reshape(-1)
            lp_parts = []
            for inner in range(0, len(z_flat), grid_pair_batch):
                lp_parts.append(
                    model.log_prob(
                        x_block[inner:inner + grid_pair_batch],
                        z_flat[inner:inner + grid_pair_batch],
                    ).cpu()
                )
            log_prob_chunks.append(torch.cat(lp_parts).reshape(len(xb), n_grid).numpy())

    log_cdes = np.vstack(log_prob_chunks) - math.log(z_scale)
    log_cdes -= log_cdes.max(axis=1, keepdims=True)
    cdes = np.exp(log_cdes)
    cdes = _normalize_density_rows(cdes, z_grid)
    return cdes, z_grid


def quantile_gbm_density(X_train, z_train, X_test, n_grid=200,
                          z_min=None, z_max=None, n_estimators=100,
                          max_depth=4, learning_rate=0.1):
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
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42, verbosity=0
            )
        else:
            reg = GradientBoostingRegressor(
                loss='quantile', alpha=alpha,
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
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
                             z_min=None, z_max=None, regularization=0.0):
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
        reg = QuantileRegressor(quantile=alpha, alpha=regularization, solver='highs')
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
                      z_min=None, z_max=None, num_trees=30,
                      num_sweeps=60, burnin=20):
    """BART with constant residual variance: f(z|x) = N(BART(x), sigma^2).

    Uses XBART's built-in sigma draws (posterior mean) for the residual sd.
    """
    model, sigma = _fit_xbart(X_train, z_train, num_trees=num_trees,
                              num_sweeps=num_sweeps, burnin=burnin)
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
                        z_min=None, z_max=None, num_trees=30,
                        num_sweeps=60, burnin=20):
    """Heteroscedastic BART: two-stage density estimation.

    Stage 1: fit BART for E[Z|X].
    Stage 2: fit a second BART for log(residual^2) to model Var(Z|X).
    Result: f(z|x) = N(BART_mean(x), exp(BART_var(x))).
    """
    # Stage 1: mean model
    model_mean, _ = _fit_xbart(X_train, z_train, num_trees=num_trees,
                               num_sweeps=num_sweeps, burnin=burnin)
    mu_train = model_mean.predict(X_train)
    mu_test = model_mean.predict(X_test)

    # Stage 2: variance model on log-squared residuals
    residuals = z_train - mu_train
    log_sq_res = np.log(np.maximum(residuals ** 2, 1e-12))
    model_var, _ = _fit_xbart(X_train, log_sq_res, num_trees=num_trees,
                              num_sweeps=num_sweeps, burnin=burnin)
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

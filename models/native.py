"""
Native distribution extraction from tabular foundation models.
"""

import numpy as np


def tabpfn_native_density(model, X_test, n_grid=200, z_min=None, z_max=None):
    """
    Extract TabPFN's native bar-distribution-based predictive density.

    TabPFN regressor predicts a bar distribution over discretized target values.
    We access it via output_type="full", which returns logits and the criterion
    (bar distribution object), then convert to a density on a grid.
    """
    from tabpfn.errors import TabPFNCUDAOutOfMemoryError
    import torch

    n_test = X_test.shape[0]
    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = np.zeros((n_test, n_grid))

    def _full_batched(X, batch_size):
        all_logits = []
        criterion = None
        for start in range(0, len(X), batch_size):
            try:
                out = model.predict(X[start:start + batch_size], output_type="full")
            except TabPFNCUDAOutOfMemoryError:
                if batch_size <= 1:
                    raise
                return _full_batched(X, batch_size // 2)
            if criterion is None:
                criterion = out['criterion']
            lg = out['logits']
            if isinstance(lg, torch.Tensor):
                lg = lg.detach().cpu().numpy()
            all_logits.append(np.array(lg))
        return criterion, np.vstack(all_logits)

    def _quantiles_batched(X, alphas, batch_size):
        batches = []
        for start in range(0, len(X), batch_size):
            try:
                q = model.predict(X[start:start + batch_size],
                                  output_type="quantiles",
                                  quantiles=alphas.tolist())
            except TabPFNCUDAOutOfMemoryError:
                if batch_size <= 1:
                    raise
                return _quantiles_batched(X, alphas, batch_size // 2)
            batches.append(np.column_stack(q))
        return np.vstack(batches)

    try:
        try:
            full_out = model.predict(X_test, output_type="full")
            criterion = full_out['criterion']
            logits = full_out['logits']
            if isinstance(logits, torch.Tensor):
                logits_np = logits.detach().cpu().numpy()
            else:
                logits_np = np.array(logits)
        except TabPFNCUDAOutOfMemoryError:
            criterion, logits_np = _full_batched(X_test, n_test // 2)

        # Convert logits to probabilities via softmax
        logits_np = logits_np - logits_np.max(axis=-1, keepdims=True)
        probs = np.exp(logits_np) / np.exp(logits_np).sum(axis=-1, keepdims=True)

        # Get the bin borders from the criterion
        if hasattr(criterion, 'borders'):
            borders = criterion.borders
            if isinstance(borders, torch.Tensor):
                borders = borders.detach().cpu().numpy()
            borders = np.array(borders).flatten()
        elif hasattr(criterion, 'borders_'):
            borders = np.array(criterion.borders_).flatten()
        else:
            n_bins = probs.shape[1]
            borders = np.linspace(z_min, z_max, n_bins + 1)

        # Convert bar distribution to density on z_grid
        n_bins = probs.shape[1]
        if len(borders) == n_bins + 1:
            centers = (borders[:-1] + borders[1:]) / 2
            widths = borders[1:] - borders[:-1]
        else:
            centers = borders[:n_bins]
            widths = np.ones(n_bins) * (centers[-1] - centers[0]) / n_bins

        for i in range(n_test):
            bin_density = probs[i] / np.maximum(widths, 1e-10)
            cdes[i, :] = np.interp(z_grid, centers, bin_density, left=0, right=0)

        print("[bar dist] ", end="")

    except Exception as e:
        print(f"[quantile fallback: {type(e).__name__}] ", end="")
        alphas = np.linspace(0.01, 0.99, 99)
        try:
            try:
                q_preds = model.predict(X_test, output_type="quantiles",
                                        quantiles=alphas.tolist())
                Q = np.column_stack(q_preds)
            except TabPFNCUDAOutOfMemoryError:
                Q = _quantiles_batched(X_test, alphas, n_test // 2)
        except Exception:
            Q = np.zeros((n_test, len(alphas)))
            for j, a in enumerate(alphas):
                try:
                    Q[:, j] = model.predict(X_test, output_type="quantiles",
                                            quantiles=[a])[0]
                except TabPFNCUDAOutOfMemoryError:
                    Q[:, j] = _quantiles_batched(X_test, np.array([a]),
                                                 n_test // 2)[:, 0]

        for i in range(n_test):
            Q[i, :] = np.sort(Q[i, :])

        dz = z_grid[1] - z_grid[0]
        for i in range(n_test):
            cdf_interp = np.interp(z_grid, Q[i, :], alphas, left=0.0, right=1.0)
            pdf = np.gradient(cdf_interp, dz)
            pdf = np.maximum(pdf, 0)
            total = pdf.sum() * dz
            if total > 0:
                pdf /= total
            cdes[i, :] = pdf

    # Normalize
    dz = z_grid[1] - z_grid[0]
    row_sums = cdes.sum(axis=1) * dz
    row_sums[row_sums == 0] = 1.0
    cdes = cdes / row_sums[:, None]

    return cdes, z_grid


def tabicl_quantile_density(model, X_train, y_train, X_test, n_grid=200,
                             z_min=None, z_max=None):
    """
    Extract TabICLv2's native quantile-based predictive distribution.
    TabICLv2 is pretrained to predict 999 quantiles via pinball loss.
    """
    n_test = X_test.shape[0]
    alphas = np.linspace(0.005, 0.995, 199)

    try:
        quantile_preds = model.predict(X_test, output_type='quantiles',
                                       alphas=alphas.tolist())
        if isinstance(quantile_preds, list):
            quantile_preds = np.column_stack(quantile_preds)
        quantile_preds = np.array(quantile_preds)
        if quantile_preds.ndim == 1:
            raise ValueError("Need multiple quantiles")
        print(f"[{quantile_preds.shape[1]} quantiles] ", end="")
    except Exception as e:
        print(f"[quantile fallback: {type(e).__name__}] ", end="")
        alphas = np.linspace(0.02, 0.98, 49)
        quantile_preds = np.zeros((n_test, len(alphas)))
        for j, a in enumerate(alphas):
            try:
                q = model.predict(X_test, output_type='quantiles', alphas=[a])
                quantile_preds[:, j] = np.array(q).flatten()
            except Exception:
                quantile_preds[:, j] = model.predict(X_test)

    # Ensure monotonicity
    for i in range(n_test):
        quantile_preds[i, :] = np.sort(quantile_preds[i, :])

    # Convert quantiles to density on grid
    if z_min is None:
        z_min = quantile_preds.min() - 0.1 * np.ptp(quantile_preds)
    if z_max is None:
        z_max = quantile_preds.max() + 0.1 * np.ptp(quantile_preds)

    z_grid = np.linspace(z_min, z_max, n_grid)
    cdes = np.zeros((n_test, n_grid))
    dz = z_grid[1] - z_grid[0]

    for i in range(n_test):
        cdf_interp = np.interp(z_grid, quantile_preds[i, :], alphas,
                                left=0.0, right=1.0)
        pdf = np.gradient(cdf_interp, dz)
        pdf = np.maximum(pdf, 0)
        total = pdf.sum() * dz
        if total > 0:
            pdf /= total
        cdes[i, :] = pdf

    return cdes, z_grid

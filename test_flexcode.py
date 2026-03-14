"""
Unit test for the FlexCodeEstimator implementation.

Tests against a known Gaussian: Z|X ~ N(X'beta, 1) with d=3.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

from models import FlexCodeEstimator, RFFlexRegressor

# ============================================================================
# 1. Verify cosine basis
# ============================================================================
print("=" * 60)
print("TEST 1: Cosine basis orthonormality on [0, 1]")
print("=" * 60)

fc_dummy = FlexCodeEstimator(RFFlexRegressor, max_basis=10)
n_quad = 10000
u = np.linspace(0, 1, n_quad)

max_basis_check = 6
ortho_ok = True
for i in range(max_basis_check):
    for j in range(i, max_basis_check):
        phi_i = fc_dummy._cosine_basis(u, i)
        phi_j = fc_dummy._cosine_basis(u, j)
        inner = np.trapezoid(phi_i * phi_j, u)
        expected = 1.0 if i == j else 0.0
        err = abs(inner - expected)
        if err > 0.01:
            print(f"  FAIL: <phi_{i}, phi_{j}> = {inner:.6f}, expected {expected:.1f}")
            ortho_ok = False

if ortho_ok:
    print("  PASS: Cosine basis is orthonormal on [0,1].")
else:
    print("  FAIL: Cosine basis is NOT orthonormal.")

# ============================================================================
# 2. Generate data from known Gaussian
# ============================================================================
print("\n" + "=" * 60)
print("TEST 2: FlexCode with RandomForest on Z|X ~ N(X'beta, 1)")
print("=" * 60)

np.random.seed(42)
d = 3
n = 2000
beta_true = np.array([1.0, -0.5, 0.3])
sigma = 1.0

X = np.random.randn(n, d)
mu_true = X @ beta_true
Z = mu_true + sigma * np.random.randn(n)

X_train, X_rest, Z_train, Z_rest = train_test_split(X, Z, test_size=0.4, random_state=0)
X_val, X_test, Z_val, Z_test = train_test_split(X_rest, Z_rest, test_size=0.5, random_state=0)

print(f"  n_train={len(X_train)}, n_val={len(X_val)}, n_test={len(X_test)}")
print(f"  beta_true={beta_true}, sigma={sigma}")

# ============================================================================
# 3. Fit FlexCode
# ============================================================================
fc = FlexCodeEstimator(
    regressor_factory=RFFlexRegressor,
    max_basis=31,
    basis_system='cosine',
    name='FlexCode-RF'
)
fc.fit(X_train, Z_train)
val_loss = fc.tune(X_val, Z_val)
print(f"  Tuned best_basis = {fc.best_basis_}")
print(f"  Validation CDE loss = {val_loss:.4f}")

# ============================================================================
# 4. Predict densities on test set
# ============================================================================
cdes, z_grid = fc.predict(X_test, n_grid=500)
dz = z_grid[1] - z_grid[0]

integrals = cdes.sum(axis=1) * dz
print(f"\n  Density integrals: mean={integrals.mean():.4f}, "
      f"min={integrals.min():.4f}, max={integrals.max():.4f}")

# ============================================================================
# 5. Compare against true density at test points
# ============================================================================
print("\n" + "=" * 60)
print("TEST 3: Pointwise density comparison (first 5 test obs)")
print("=" * 60)

mu_test = X_test @ beta_true
n_show = 5
for idx in range(n_show):
    true_dens = stats.norm.pdf(z_grid, loc=mu_test[idx], scale=sigma)
    est_dens = cdes[idx, :]

    peak_idx = np.argmin(np.abs(z_grid - mu_test[idx]))
    true_at_peak = true_dens[peak_idx]
    est_at_peak = est_dens[peak_idx]

    l1 = np.trapezoid(np.abs(est_dens - true_dens), z_grid)

    print(f"  obs {idx}: mu_true={mu_test[idx]:.3f}, "
          f"f_true(mode)={true_at_peak:.4f}, f_est(mode)={est_at_peak:.4f}, "
          f"L1={l1:.4f}")

# ============================================================================
# 6. CDE Loss on test set
# ============================================================================
print("\n" + "=" * 60)
print("TEST 4: CDE Loss (Izbicki & Lee) on test set")
print("=" * 60)

integral_sq = (cdes ** 2).sum(axis=1) * dz
f_at_obs = np.array([
    np.interp(Z_test[i], z_grid, cdes[i, :])
    for i in range(len(Z_test))
])

cde_loss_test = np.mean(integral_sq - 2 * f_at_obs)
print(f"  CDE loss on test set = {cde_loss_test:.4f}")

true_dens_at_obs = stats.norm.pdf(Z_test, loc=mu_test, scale=sigma)
true_integral_sq = 1.0 / (2.0 * sigma * np.sqrt(np.pi))
true_cde_loss = true_integral_sq - 2 * np.mean(true_dens_at_obs)
print(f"  CDE loss of true density = {true_cde_loss:.4f}")
print(f"  CDE loss difference (est - true) = {cde_loss_test - true_cde_loss:.4f}")

if cde_loss_test < true_cde_loss + 0.10:
    print("  PASS: FlexCode CDE loss is close to oracle.")
else:
    print(f"  WARNING: CDE loss is {cde_loss_test - true_cde_loss:.4f} above oracle.")

# ============================================================================
# 7. Check phi_0 regression
# ============================================================================
print("\n" + "=" * 60)
print("TEST 5: phi_0 regression check")
print("=" * 60)

phi0_preds = fc.regressors_[0].predict(X_test)
print(f"  phi_0 regressor predictions: mean={phi0_preds.mean():.4f}, "
      f"std={phi0_preds.std():.4f}")

if abs(phi0_preds.mean() - 1.0) < 0.05 and phi0_preds.std() < 0.05:
    print("  PASS: phi_0 regressor correctly learns constant ~1.")
else:
    print("  WARNING: phi_0 regressor may introduce noise.")

# ============================================================================
# 8. Non-negativity and normalization
# ============================================================================
print("\n" + "=" * 60)
print("TEST 6: Non-negativity and normalization")
print("=" * 60)
print(f"  Min density value: {cdes.min():.6f} (should be >= 0)")
print(f"  All non-negative: {np.all(cdes >= 0)}")
print(f"  Integral range: [{integrals.min():.4f}, {integrals.max():.4f}] (should be ~1)")

if np.all(cdes >= 0) and np.all(np.abs(integrals - 1.0) < 0.05):
    print("  PASS: Densities are valid.")
else:
    print("  FAIL: Density validity issues detected.")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Cosine basis orthonormality:  {'PASS' if ortho_ok else 'FAIL'}")
print(f"  Best basis selected:          {fc.best_basis_}")
print(f"  CDE loss (estimated):         {cde_loss_test:.4f}")
print(f"  CDE loss (oracle):            {true_cde_loss:.4f}")
print(f"  CDE gap:                      {cde_loss_test - true_cde_loss:.4f}")
print(f"  phi_0 mean pred:              {phi0_preds.mean():.4f}")
print(f"  Densities valid (>=0, int=1): {np.all(cdes >= 0) and np.all(np.abs(integrals - 1.0) < 0.05)}")

"""
FlexCode x Tabular Foundation Models: CDE Experiments
=====================================================

USAGE:
  pip install tabpfn tabicl scikit-learn matplotlib numpy scipy
  python run_experiments.py [--device cpu|cuda] [--quick] [--force]
"""

import argparse
import re
import time
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Try importing foundation models
try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
    print("+ TabPFN available")
except ImportError:
    HAS_TABPFN = False
    print("- TabPFN not found -- install with: pip install tabpfn")

try:
    from tabicl import TabICLRegressor
    HAS_TABICL = True
    print("+ TabICLv2 available")
except ImportError:
    HAS_TABICL = False
    print("- TabICLv2 not found -- install with: pip install tabicl")

try:
    from xbart import XBART  # noqa: F401
    HAS_XBART = True
    print("+ XBART available")
except ImportError:
    HAS_XBART = False
    print("- XBART not found -- install with: pip install xbart")

from models import (
    FlexCodeEstimator, RFFlexRegressor,
    tabpfn_native_density, tabicl_quantile_density,
    linear_gaussian_homo_density, linear_gaussian_hetero_density,
    mdn_density, quantile_gbm_density,
    quantile_linear_density, gamma_glm_density,
    student_t_density, lognormal_homo_density, lognormal_hetero_density,
    bart_homo_density, bart_hetero_density,
)
from datasets import load_all_datasets
from evaluation import compute_all_metrics
from visualization import (
    plot_rankings_by_n, plot_raw_metrics_by_n,
    plot_pit_histograms, plot_native_tab_subset,
    plot_performance_vs_n, plot_performance_vs_n_foundational,
    save_html_table,
)
from utils import save_cache, load_cache, print_summary


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(X, z, dataset_name, device='auto', n_grid=200,
                   true_density_fn=None, partial_dir=None, force=False):
    """Run all methods on one dataset, skipping already-completed ones."""

    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    print(f"\n{'='*70}")
    print(f"  {dataset_name}  (train={len(X_train)}, test={len(X_test)}, "
          f"d={X.shape[1]})")
    print(f"{'='*70}")

    results = {}
    cdes_dict = {}
    zgrids_dict = {}

    # ── Per-method checkpoint helpers ─────────────────────────────────────
    partial_metrics = {}
    if partial_dir and not force:
        metrics_file = partial_dir / f"{dataset_name}_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    partial_metrics = json.load(f)
                print(f"  [resume] {len(partial_metrics)} method(s) already done: "
                      f"{', '.join(partial_metrics)}")
            except Exception:
                partial_metrics = {}

    def _key(name):
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    def _load_arrays(name):
        if partial_dir is None:
            return None, None
        cde_f = partial_dir / f"{dataset_name}_{_key(name)}_cdes.npy"
        zg_f  = partial_dir / f"{dataset_name}_{_key(name)}_zgrid.npy"
        if cde_f.exists() and zg_f.exists():
            return np.load(cde_f), np.load(zg_f)
        return None, None

    def _save(name, m, cdes, zg):
        if partial_dir is None:
            return
        partial_dir.mkdir(exist_ok=True)
        np.save(partial_dir / f"{dataset_name}_{_key(name)}_cdes.npy", cdes)
        np.save(partial_dir / f"{dataset_name}_{_key(name)}_zgrid.npy", zg)
        partial_metrics[name] = {k: (float(v) if v is not None else None)
                                 for k, v in m.items()}
        with open(partial_dir / f"{dataset_name}_metrics.json", 'w') as f:
            json.dump(partial_metrics, f, indent=2)

    def _cached(name):
        """Return (metrics, cdes, zgrid) if cached, else None."""
        if name in partial_metrics:
            cdes, zg = _load_arrays(name)
            if cdes is not None:
                return partial_metrics[name], cdes, zg
        return None

    n_train = len(z_train)
    max_basis = min(50, max(15, int(np.sqrt(n_train))))

    # ── Helper to run a FlexCode method ──────────────────────────────────
    def run_flexcode(name, factory, params):
        hit = _cached(name)
        if hit:
            m_c, cdes, zg = hit
            results[name] = m_c
            cdes_dict[name] = cdes
            zgrids_dict[name] = zg
            print(f"  {name}... [cached] CDE={m_c['CDE_loss']:.4f}, "
                  f"LL={m_c['log_lik']:.3f}")
            return
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        model = FlexCodeEstimator(factory, max_basis=max_basis,
                                   regressor_params=params, name=name)
        model.fit_cv(X_tr, z_train, n_folds=5)
        fit_t = time.time() - t0
        t0 = time.time()
        cdes, zg = model.predict(X_te, n_grid=n_grid)
        pred_t = time.time() - t0
        m = compute_all_metrics(cdes, zg, z_test)
        m['fit_time'] = fit_t
        m['pred_time'] = pred_t
        m['n_basis'] = model.best_basis_
        results[name] = m
        cdes_dict[name] = cdes
        zgrids_dict[name] = zg
        _save(name, m, cdes, zg)
        print(f"I={model.best_basis_}, CDE={m['CDE_loss']:.4f}, "
              f"LL={m['log_lik']:.3f}, CRPS={m['CRPS']:.4f}, "
              f"KS={m['PIT_KS']:.3f}, t={fit_t:.1f}s")

    # ── FlexCode + RandomForest ──────────────────────────────────────────
    run_flexcode('FlexCode-RF', lambda **kw: RFFlexRegressor(), {})

    # ── TabPFN Native Distribution ───────────────────────────────────────
    if HAS_TABPFN:
        name = 'TabPFN-Native'
        hit = _cached(name)
        if hit:
            m_c, cdes_pfn, zg_pfn = hit
            results[name] = m_c
            cdes_dict[name] = cdes_pfn
            zgrids_dict[name] = zg_pfn
            print(f"  {name}... [cached] CDE={m_c['CDE_loss']:.4f}, "
                  f"LL={m_c['log_lik']:.3f}")
        else:
            print(f"  {name}...", end=" ", flush=True)
            t0 = time.time()
            pfn_reg = TabPFNRegressor(device=device)
            pfn_reg.fit(X_tr, z_train)
            fit_t = time.time() - t0
            t0 = time.time()
            z_lo = z_train.min() - 0.05 * np.ptp(z_train)
            z_hi = z_train.max() + 0.05 * np.ptp(z_train)
            cdes_pfn, zg_pfn = tabpfn_native_density(
                pfn_reg, X_te, n_grid=n_grid, z_min=z_lo, z_max=z_hi
            )
            pred_t = time.time() - t0
            m = compute_all_metrics(cdes_pfn, zg_pfn, z_test)
            m['fit_time'] = fit_t
            m['pred_time'] = pred_t
            m['n_basis'] = None
            results[name] = m
            cdes_dict[name] = cdes_pfn
            zgrids_dict[name] = zg_pfn
            _save(name, m, cdes_pfn, zg_pfn)
            print(f"CDE={m['CDE_loss']:.4f}, LL={m['log_lik']:.3f}, "
                  f"CRPS={m['CRPS']:.4f}, KS={m['PIT_KS']:.3f}")

    # ── TabICLv2 Native Quantiles ────────────────────────────────────────
    if HAS_TABICL:
        name = 'TabICL-Quantiles'
        hit = _cached(name)
        if hit:
            m_c, cdes_icl, zg_icl = hit
            results[name] = m_c
            cdes_dict[name] = cdes_icl
            zgrids_dict[name] = zg_icl
            print(f"  {name}... [cached] CDE={m_c['CDE_loss']:.4f}, "
                  f"LL={m_c['log_lik']:.3f}")
        else:
            print(f"  {name}...", end=" ", flush=True)
            t0 = time.time()
            icl_reg = TabICLRegressor(
                n_estimators=4,
                device=device if device != 'auto' else 'cpu'
            )
            icl_reg.fit(X_tr, z_train)
            fit_t = time.time() - t0
            t0 = time.time()
            z_lo = z_train.min() - 0.05 * np.ptp(z_train)
            z_hi = z_train.max() + 0.05 * np.ptp(z_train)
            cdes_icl, zg_icl = tabicl_quantile_density(
                icl_reg, X_tr, z_train, X_te,
                n_grid=n_grid, z_min=z_lo, z_max=z_hi
            )
            pred_t = time.time() - t0
            m = compute_all_metrics(cdes_icl, zg_icl, z_test)
            m['fit_time'] = fit_t
            m['pred_time'] = pred_t
            m['n_basis'] = None
            results[name] = m
            cdes_dict[name] = cdes_icl
            zgrids_dict[name] = zg_icl
            _save(name, m, cdes_icl, zg_icl)
            print(f"CDE={m['CDE_loss']:.4f}, LL={m['log_lik']:.3f}, "
                  f"CRPS={m['CRPS']:.4f}, KS={m['PIT_KS']:.3f}")

    # ── Quantile GBM/XGB baseline ────────────────────────────────────────
    name = 'Quantile-Tree'
    hit = _cached(name)
    if hit:
        m_c, cdes_qgbm, zg_qgbm = hit
        results[name] = m_c
        cdes_dict[name] = cdes_qgbm
        zgrids_dict[name] = zg_qgbm
        print(f"  {name}... [cached] CDE={m_c['CDE_loss']:.4f}, "
              f"LL={m_c['log_lik']:.3f}")
    else:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        z_lo = z_train.min() - 0.05 * np.ptp(z_train)
        z_hi = z_train.max() + 0.05 * np.ptp(z_train)
        cdes_qgbm, zg_qgbm = quantile_gbm_density(
            X_tr, z_train, X_te, n_grid=n_grid, z_min=z_lo, z_max=z_hi
        )
        fit_t = time.time() - t0
        m = compute_all_metrics(cdes_qgbm, zg_qgbm, z_test)
        m['fit_time'] = fit_t
        m['pred_time'] = 0
        m['n_basis'] = None
        results[name] = m
        cdes_dict[name] = cdes_qgbm
        zgrids_dict[name] = zg_qgbm
        _save(name, m, cdes_qgbm, zg_qgbm)
        print(f"CDE={m['CDE_loss']:.4f}, LL={m['log_lik']:.3f}, "
              f"CRPS={m['CRPS']:.4f}, KS={m['PIT_KS']:.3f}")

    # ── Helper for simple density baselines ──────────────────────────────
    def _run_density_baseline(name, density_fn, **kwargs):
        hit = _cached(name)
        if hit:
            m_c, cdes_bl, zg_bl = hit
            results[name] = m_c
            cdes_dict[name] = cdes_bl
            zgrids_dict[name] = zg_bl
            print(f"  {name}... [cached] CDE={m_c['CDE_loss']:.4f}, "
                  f"LL={m_c['log_lik']:.3f}")
            return
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        z_lo = z_train.min() - 0.05 * np.ptp(z_train)
        z_hi = z_train.max() + 0.05 * np.ptp(z_train)
        cdes_bl, zg_bl = density_fn(
            X_tr, z_train, X_te, n_grid=n_grid, z_min=z_lo, z_max=z_hi,
            **kwargs
        )
        fit_t = time.time() - t0
        m_bl = compute_all_metrics(cdes_bl, zg_bl, z_test)
        m_bl['fit_time'] = fit_t
        m_bl['pred_time'] = 0
        m_bl['n_basis'] = None
        results[name] = m_bl
        cdes_dict[name] = cdes_bl
        zgrids_dict[name] = zg_bl
        _save(name, m_bl, cdes_bl, zg_bl)
        print(f"CDE={m_bl['CDE_loss']:.4f}, LL={m_bl['log_lik']:.3f}, "
              f"CRPS={m_bl['CRPS']:.4f}, KS={m_bl['PIT_KS']:.3f}")

    # ── Baselines ────────────────────────────────────────────────────────
    _run_density_baseline('LinearGauss-Homo', linear_gaussian_homo_density)
    _run_density_baseline('LinearGauss-Hetero', linear_gaussian_hetero_density)
    _run_density_baseline('Student-t', student_t_density)
    _run_density_baseline('LogNormal-Homo', lognormal_homo_density)
    _run_density_baseline('LogNormal-Hetero', lognormal_hetero_density)
    _run_density_baseline('MDN-2mix', mdn_density, n_components=2, n_hidden=16)
    _run_density_baseline('Quantile-Linear', quantile_linear_density)
    _run_density_baseline('Gamma-GLM', gamma_glm_density)

    # ── Penalized (Ridge) variants ────────────────────────────────────
    _run_density_baseline('LinGauss-Homo-Ridge', linear_gaussian_homo_density, regularized=True)
    _run_density_baseline('LinGauss-Hetero-Ridge', linear_gaussian_hetero_density, regularized=True)
    _run_density_baseline('Student-t-Ridge', student_t_density, regularized=True)
    _run_density_baseline('LogNormal-Homo-Ridge', lognormal_homo_density, regularized=True)
    _run_density_baseline('LogNormal-Hetero-Ridge', lognormal_hetero_density, regularized=True)
    _run_density_baseline('Gamma-GLM-Ridge', gamma_glm_density, regularized=True)

    # ── BART methods ──────────────────────────────────────────────────────
    if HAS_XBART:
        _run_density_baseline('BART-Homo', bart_homo_density)
        _run_density_baseline('BART-Hetero', bart_hetero_density)

    # ── True conditional density (synthetic only) ────────────────────────
    true_cde = None
    true_zgrid = None
    if true_density_fn is not None and zgrids_dict:
        first_zg = next(iter(zgrids_dict.values()))
        z_lo = min(first_zg[0], z_test.min() - 0.05 * np.ptp(z_test))
        z_hi = max(first_zg[-1], z_test.max() + 0.05 * np.ptp(z_test))
        true_zgrid = np.linspace(z_lo, z_hi, n_grid)
        true_cde = true_density_fn(X_test, true_zgrid)  # unscaled X_test
        dz = true_zgrid[1] - true_zgrid[0]
        row_sums = true_cde.sum(axis=1) * dz
        row_sums[row_sums == 0] = 1.0
        true_cde = true_cde / row_sums[:, None]

    return results, cdes_dict, zgrids_dict, X_te, z_test, true_cde, true_zgrid


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FlexCode x TFM CDE Experiments')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--quick', action='store_true',
                        help='Run fewer datasets')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all datasets even if cached results exist')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    partial_dir = cache_dir / 'partial'
    partial_dir.mkdir(exist_ok=True)

    # Load previously saved results to skip already-run datasets
    json_path = output_dir / 'results.json'
    existing_results = {}
    if json_path.exists() and not args.force:
        try:
            with open(json_path) as f:
                existing_results = json.load(f)
            print(f"  [cache] Found {len(existing_results)} previously "
                  f"completed dataset(s)")
        except Exception:
            existing_results = {}

    print("\n" + "=" * 60)
    print("FlexCode x Tabular Foundation Models")
    print("CDE Experiments")
    print("=" * 60)

    if not HAS_TABPFN and not HAS_TABICL:
        print("\n!! Neither TabPFN nor TabICLv2 found!")
        print("  Install with: pip install tabpfn tabicl")
        print("  Running with sklearn baselines only.\n")

    datasets = load_all_datasets(quick=args.quick)

    all_results = {}
    all_data = {}

    for X, z, name, true_density_fn in datasets:
        cache_file = cache_dir / f"{name}.npz"
        use_cache = (not args.force
                     and name in existing_results
                     and cache_file.exists())

        if use_cache:
            print(f"\n[cache] Skipping '{name}' -- already in results.json. "
                  f"Use --force to re-run.")
            cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_total = \
                load_cache(cache_file)
            all_results[name] = {
                m: {k: v for k, v in existing_results[name][m].items()}
                for m in existing_results[name]
            }
        else:
            res, cdes, zgrids, X_te, z_te, true_cde, true_zgrid = \
                run_experiment(
                    X, z, name, device=args.device,
                    true_density_fn=true_density_fn,
                    partial_dir=partial_dir, force=args.force,
                )
            n_total = len(z)
            all_results[name] = res
            save_cache(cache_file, cdes, zgrids, X_te, z_te,
                       true_cde, true_zgrid, n_total)

        all_data[name] = {
            'cdes': cdes, 'zgrids': zgrids,
            'X_test': X_te, 'z_test': z_te,
            'true_cde': true_cde, 'true_zgrid': true_zgrid,
            'n_total': n_total,
        }

    print_summary(all_results)

    print("\nGenerating plots and tables...")
    save_html_table(all_results, output_dir)
    plot_native_tab_subset(all_data, output_dir)
    plot_rankings_by_n(all_results, output_dir, all_data=all_data)
    plot_raw_metrics_by_n(all_results, output_dir, all_data=all_data)
    plot_pit_histograms(all_data, output_dir)
    plot_performance_vs_n(all_results, output_dir, all_data=all_data)
    plot_performance_vs_n_foundational(all_results, output_dir, all_data=all_data)

    # Save JSON
    json_out = {}
    for ds, res in all_results.items():
        json_out[ds] = {m: {k: float(v) if v is not None else None
                            for k, v in met.items()}
                        for m, met in res.items()}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(json_out, f, indent=2)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == '__main__':
    main()

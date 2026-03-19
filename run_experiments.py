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
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import prepend_cache_path
    HAS_TABPFN = True
    print("+ TabPFN available")
except ImportError:
    ModelVersion = None
    prepend_cache_path = None
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
    mdn_density, normalizing_flow_density, quantile_gbm_density,
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

_SDSS_PREFIX = 'SDSS-'


def dataset_kind(dataset_name, true_density_fn=None):
    is_sim = (
        true_density_fn is not None
        or dataset_name.startswith('Friedman')
        or re.search(r'-d\d+(?:-\d+)?$', dataset_name)
    )
    return 'sim' if is_sim else 'real'


def _parse_dataset_n(dataset_name):
    m = re.search(r'-(\d+)$', dataset_name)
    return int(m.group(1)) if m else None


def prioritize_dataset_schedule(datasets):
    """Keep the existing order, but run SDSS first within each real-data n block."""
    indexed = list(enumerate(datasets))

    def sort_key(item):
        idx, (_, _, name, true_density_fn) = item
        kind = dataset_kind(name, true_density_fn)
        if kind != 'real':
            return (0, idx, 1, idx)

        n_size = _parse_dataset_n(name)
        sdss_priority = 0 if name.startswith(_SDSS_PREFIX) else 1
        return (1, n_size if n_size is not None else float('inf'),
                sdss_priority, idx)

    return [dataset for _, dataset in sorted(indexed, key=sort_key)]


def report_sdss_schedule(datasets):
    sdss_names = [
        name for _, _, name, true_density_fn in datasets
        if dataset_kind(name, true_density_fn) == 'real'
        and name.startswith(_SDSS_PREFIX)
    ]
    if sdss_names:
        print(f"  [schedule] SDSS queued at: {', '.join(sdss_names)}")
    else:
        print("  [schedule] No SDSS runs queued")


def run_experiment(X, z, dataset_name, device='auto', n_grid=200,
                   true_density_fn=None, partial_dir=None, force=False,
                   random_state=42, methods=None):
    """Run all methods on one dataset, skipping already-completed ones."""

    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.25, random_state=random_state)

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
    methods = set(methods) if methods is not None else None

    def _want(name):
        return methods is None or name in methods

    def _run_tabpfn_method(name, model_factory):
        hit = _cached(name)
        if hit:
            m_c, cdes_pfn, zg_pfn = hit
            results[name] = m_c
            cdes_dict[name] = cdes_pfn
            zgrids_dict[name] = zg_pfn
            print(f"  {name}... [cached] CDE={m_c['CDE_loss']:.4f}, "
                  f"LL={m_c['log_lik']:.3f}")
            return

        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        pfn_reg = model_factory()
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
        m['fit_time'] = fit_t + pred_t
        m['pred_time'] = pred_t
        m['n_basis'] = None
        results[name] = m
        cdes_dict[name] = cdes_pfn
        zgrids_dict[name] = zg_pfn
        _save(name, m, cdes_pfn, zg_pfn)
        print(f"CDE={m['CDE_loss']:.4f}, LL={m['log_lik']:.3f}, "
              f"CRPS={m['CRPS']:.4f}, KS={m['PIT_KS']:.3f}")

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
    max_basis = min(20, max(15, int(np.sqrt(n_train))))

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
        m['fit_time'] = fit_t + pred_t
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
    if _want('FlexCode-RF'):
        run_flexcode('FlexCode-RF', lambda **kw: RFFlexRegressor(), {})

    # ── TabPFN Native Distribution ───────────────────────────────────────
    if HAS_TABPFN and _want('TabPFN-Native'):
        _run_tabpfn_method(
            'TabPFN-Native',
            lambda: TabPFNRegressor(device=device),
        )

    # ── Explicit TabPFN 2.5 default checkpoint ──────────────────────────
    if HAS_TABPFN and ModelVersion is not None and _want('TabPFN-2.5'):
        _run_tabpfn_method(
            'TabPFN-2.5',
            lambda: TabPFNRegressor.create_default_for_version(
                ModelVersion.V2_5,
                device=device,
            ),
        )

    # ── RealTabPFN 2.5 real-data checkpoint ─────────────────────────────
    if (HAS_TABPFN and prepend_cache_path is not None
            and _want('RealTabPFN-2.5')):
        real_ckpt = prepend_cache_path('tabpfn-v2.5-regressor-v2.5_real.ckpt')
        _run_tabpfn_method(
            'RealTabPFN-2.5',
            lambda: TabPFNRegressor(
                model_path=real_ckpt,
                device=device,
            ),
        )

    # ── TabICLv2 Native Quantiles ────────────────────────────────────────
    if HAS_TABICL and _want('TabICL-Quantiles'):
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
            m['fit_time'] = fit_t + pred_t
            m['pred_time'] = pred_t
            m['n_basis'] = None
            results[name] = m
            cdes_dict[name] = cdes_icl
            zgrids_dict[name] = zg_icl
            _save(name, m, cdes_icl, zg_icl)
            print(f"CDE={m['CDE_loss']:.4f}, LL={m['log_lik']:.3f}, "
                  f"CRPS={m['CRPS']:.4f}, KS={m['PIT_KS']:.3f}")

    # ── Quantile GBM/XGB baseline ────────────────────────────────────────
    if _want('Quantile-Tree'):
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
    if _want('LinearGauss-Homo'):
        _run_density_baseline('LinearGauss-Homo', linear_gaussian_homo_density)
    if _want('LinearGauss-Hetero'):
        _run_density_baseline('LinearGauss-Hetero', linear_gaussian_hetero_density)
    if _want('Student-t'):
        _run_density_baseline('Student-t', student_t_density)
    if _want('LogNormal-Homo'):
        _run_density_baseline('LogNormal-Homo', lognormal_homo_density)
    if _want('LogNormal-Hetero'):
        _run_density_baseline('LogNormal-Hetero', lognormal_hetero_density)
    if _want('MDN-2mix'):
        _run_density_baseline('MDN-2mix', mdn_density, n_components=2, n_hidden=16)
    if _want('Flow-Spline'):
        _run_density_baseline('Flow-Spline', normalizing_flow_density, device=device)
    if n_train <= 10000 and _want('Quantile-Linear'):
        _run_density_baseline('Quantile-Linear', quantile_linear_density)
    if _want('Gamma-GLM'):
        _run_density_baseline('Gamma-GLM', gamma_glm_density)

    # ── Penalized (Ridge) variants ────────────────────────────────────
    if _want('LinGauss-Homo-Ridge'):
        _run_density_baseline('LinGauss-Homo-Ridge', linear_gaussian_homo_density, regularized=True)
    if _want('LinGauss-Hetero-Ridge'):
        _run_density_baseline('LinGauss-Hetero-Ridge', linear_gaussian_hetero_density, regularized=True)
    if _want('Student-t-Ridge'):
        _run_density_baseline('Student-t-Ridge', student_t_density, regularized=True)
    if _want('LogNormal-Homo-Ridge'):
        _run_density_baseline('LogNormal-Homo-Ridge', lognormal_homo_density, regularized=True)
    if _want('LogNormal-Hetero-Ridge'):
        _run_density_baseline('LogNormal-Hetero-Ridge', lognormal_hetero_density, regularized=True)
    if _want('Gamma-GLM-Ridge'):
        _run_density_baseline('Gamma-GLM-Ridge', gamma_glm_density, regularized=True)

    # ── BART methods ──────────────────────────────────────────────────────
    if HAS_XBART and _want('BART-Homo'):
        _run_density_baseline('BART-Homo', bart_homo_density)
    if HAS_XBART and _want('BART-Hetero'):
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
    parser.add_argument('--output-dir', default='results_simulated',
                        help='Output directory for simulated results '
                             '(default: results_simulated)')
    parser.add_argument('--real-output-dir', default='results_real',
                        help='Output directory for real results '
                             '(default: results_real)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all datasets even if cached results exist')
    parser.add_argument('--n-reps', type=int, default=4,
                        help='Number of repetitions per dataset (default 4)')
    args = parser.parse_args()

    output_dirs = {
        'sim': Path(args.output_dir),
        'real': Path(args.real_output_dir),
    }
    dir_state = {}
    for kind, out_dir in output_dirs.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = out_dir / 'cache'
        cache_dir.mkdir(exist_ok=True)
        partial_dir = cache_dir / 'partial'
        partial_dir.mkdir(exist_ok=True)

        existing_results = {}
        json_path = out_dir / 'results.json'
        if json_path.exists() and not args.force:
            try:
                with open(json_path) as f:
                    existing_results = json.load(f)
                print(f"  [cache:{kind}] Found {len(existing_results)} "
                      f"previously completed dataset(s)")
            except Exception:
                existing_results = {}

        dir_state[kind] = {
            'output_dir': out_dir,
            'cache_dir': cache_dir,
            'partial_dir': partial_dir,
            'existing_results': existing_results,
            'results': {},
            'data': {},
        }

    print("\n" + "=" * 60)
    print("FlexCode x Tabular Foundation Models")
    print("CDE Experiments")
    print("=" * 60)

    if not HAS_TABPFN and not HAS_TABICL:
        print("\n!! Neither TabPFN nor TabICLv2 found!")
        print("  Install with: pip install tabpfn tabicl")
        print("  Running with sklearn baselines only.\n")

    datasets = prioritize_dataset_schedule(load_all_datasets(quick=args.quick))
    report_sdss_schedule(datasets)
    n_reps = args.n_reps

    # ── Metrics that are averaged across repetitions ──────────────────────
    MEAN_METRICS = ['CDE_loss', 'log_lik', 'CRPS', 'PIT_KS',
                    'coverage_90', 'interval_width', 'fit_time', 'pred_time']

    def _aggregate_reps(per_rep_results):
        """Aggregate metrics across repetitions: mean ± SE."""
        methods = sorted(set(m for rep in per_rep_results for m in rep))
        agg = {}
        for m in methods:
            vals = {k: [] for k in MEAN_METRICS}
            n_basis_vals = []
            for rep in per_rep_results:
                if m not in rep:
                    continue
                for k in MEAN_METRICS:
                    if k in rep[m] and rep[m][k] is not None:
                        vals[k].append(rep[m][k])
                if rep[m].get('n_basis') is not None:
                    n_basis_vals.append(rep[m]['n_basis'])

            agg_m = {}
            for k in MEAN_METRICS:
                arr = np.array(vals[k])
                if len(arr) > 0:
                    agg_m[k] = float(np.mean(arr))
                    agg_m[f'{k}_se'] = float(
                        np.std(arr, ddof=1) / np.sqrt(len(arr))
                    ) if len(arr) > 1 else None
                else:
                    agg_m[k] = None
                    agg_m[f'{k}_se'] = None
            agg_m['n_basis'] = (float(np.mean(n_basis_vals))
                                if n_basis_vals else None)
            agg[m] = agg_m
        return agg

    all_results = {}
    all_data = {}

    for X, z, name, true_density_fn in datasets:
        kind = dataset_kind(name, true_density_fn)
        state = dir_state[kind]
        cache_file = state['cache_dir'] / f"{name}.npz"
        use_cache = (not args.force
                     and name in state['existing_results']
                     and cache_file.exists())

        if use_cache:
            print(f"\n[cache] Skipping '{name}' -- already in results.json. "
                  f"Use --force to re-run.")
            cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_total = \
                load_cache(cache_file)
            all_results[name] = {
                m: {k: v for k, v in state['existing_results'][name][m].items()}
                for m in state['existing_results'][name]
            }
        else:
            per_rep_results = []
            for rep in range(n_reps):
                rep_partial = state['partial_dir'] / f"rep{rep}"
                rep_partial.mkdir(exist_ok=True)
                print(f"\n  ── rep {rep+1}/{n_reps} (seed={rep}) ──")
                res, cdes, zgrids, X_te, z_te, true_cde, true_zgrid = \
                    run_experiment(
                        X, z, name, device=args.device,
                        true_density_fn=true_density_fn,
                        partial_dir=rep_partial, force=args.force,
                        random_state=rep,
                    )
                per_rep_results.append(res)
            # Aggregate across reps
            n_total = len(z)
            all_results[name] = _aggregate_reps(per_rep_results)
            # Save last rep's CDEs for visualization
            save_cache(cache_file, cdes, zgrids, X_te, z_te,
                       true_cde, true_zgrid, n_total)

        all_data[name] = {
            'cdes': cdes, 'zgrids': zgrids,
            'X_test': X_te, 'z_test': z_te,
            'true_cde': true_cde, 'true_zgrid': true_zgrid,
            'n_total': n_total,
        }
        state['results'][name] = all_results[name]
        state['data'][name] = all_data[name]

    print_summary(all_results, se_caption='mean +/- SE across repetitions')

    print("\nGenerating plots and tables...")
    save_html_table(all_results, output_dirs)
    plot_native_tab_subset(all_data, output_dirs)
    plot_rankings_by_n(all_results, output_dirs, all_data=all_data)
    plot_raw_metrics_by_n(all_results, output_dirs, all_data=all_data)
    plot_pit_histograms(all_data, output_dirs)
    plot_performance_vs_n(all_results, output_dirs, all_data=all_data)
    plot_performance_vs_n_foundational(all_results, output_dirs, all_data=all_data)

    for kind, state in dir_state.items():
        json_out = {}
        for ds, res in state['results'].items():
            json_out[ds] = {m: {k: float(v) if v is not None else None
                                for k, v in met.items()}
                            for m, met in res.items()}
        with open(state['output_dir'] / 'results.json', 'w') as f:
            json.dump(json_out, f, indent=2)

    print(f"\nDone. Simulated results in {output_dirs['sim']}/")
    print(f"Done. Real results in {output_dirs['real']}/")


if __name__ == '__main__':
    main()

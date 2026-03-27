"""
FlexCode x Tabular Foundation Models: Real Dataset CDE Experiments
==================================================================

Runs experiments only on real (OpenML) and semi-synthetic (Friedman) datasets.

USAGE:
  python run_real_experiments.py [--device cpu|cuda] [--force] [--n-reps N]
"""

import argparse
import json
import re
from pathlib import Path

import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)

import warnings

warnings.filterwarnings('ignore')

from run_experiments import (
    run_experiment,
    prioritize_dataset_schedule,
    report_sdss_schedule,
)
from datasets import load_real_only_datasets
from visualization import (
    plot_rankings_by_n, plot_critical_difference, plot_raw_metrics_by_n,
    plot_raw_metrics_with_values_by_n,
    plot_pit_histograms, plot_native_tab_subset,
    plot_performance_vs_n, plot_performance_vs_n_foundational,
    plot_perf_vs_n_foundational_cde_subsets,
    save_html_table,
    save_latex_table,
    save_appendix_metric_tables,
    save_appendix_metric_tables_html,
)
from utils import save_cache, load_cache, print_summary, aggregate_reps


DEFAULT_REAL_METHODS = [
    'FlexCode-RF',
    'TabPFN-2.5',
    'RealTabPFN-2.5',
    'TabICL-Quantiles',
    'LinearGauss-Homo',
    'LinearGauss-Hetero',
    'Student-t',
    'LogNormal-Homo',
    'LogNormal-Hetero',
    'MDN',
    'Flow-Spline',
    'Quantile-Tree',
    'Gamma-GLM',
    'LinGauss-Homo-Ridge',
    'LinGauss-Hetero-Ridge',
    'Student-t-Ridge',
    'LogNormal-Homo-Ridge',
    'LogNormal-Hetero-Ridge',
    'Gamma-GLM-Ridge',
    'BART-Homo',
    'BART-Hetero',
    'CatMLP',
]

LARGE_REAL_N_EXCLUDED_METHODS = {
    'BART-Hetero',
    'Quantile-Tree',
}


def _dataset_target_n(name):
    match = re.search(r'-(\d+)$', name)
    return int(match.group(1)) if match else None


def _canonical_method_name(name):
    return 'MDN' if name == 'MDN-2mix' else name


def _selected_real_methods(dataset_name):
    target_n = _dataset_target_n(dataset_name)
    methods = list(DEFAULT_REAL_METHODS)
    if target_n is not None and target_n >= 10_000:
        methods = [
            m for m in methods
            if _canonical_method_name(m) not in LARGE_REAL_N_EXCLUDED_METHODS
        ]
    return methods


def _filter_method_mapping(mapping, methods):
    allowed = {_canonical_method_name(m) for m in methods}
    filtered = {}
    for name, value in mapping.items():
        canonical = _canonical_method_name(name)
        if canonical not in allowed:
            continue
        if canonical in filtered and name != canonical:
            continue
        filtered[canonical] = value
    return filtered


def _effective_n_reps(dataset_name, requested_n_reps):
    target_n = _dataset_target_n(dataset_name)
    if target_n == 50:
        return requested_n_reps * 10
    return requested_n_reps


def main():
    parser = argparse.ArgumentParser(
        description='FlexCode x TFM CDE Experiments -- Real Datasets Only')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output-dir', default='results_real',
                        help='Output directory (default: results_real)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all datasets even if cached results exist')
    parser.add_argument('--n-reps', type=int, default=4,
                        help='Number of repetitions per dataset (default 4)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    partial_dir = cache_dir / 'partial'
    partial_dir.mkdir(exist_ok=True)

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
    print("CDE Experiments -- Real Datasets Only")
    print("=" * 60)

    datasets = prioritize_dataset_schedule(load_real_only_datasets())
    report_sdss_schedule(datasets)
    requested_n_reps = args.n_reps

    all_results = {}
    all_data = {}

    for X, z, name, true_density_fn in datasets:
        n_reps = _effective_n_reps(name, requested_n_reps)
        selected_methods = _selected_real_methods(name)
        cache_file = cache_dir / f"{name}.npz"

        if n_reps != requested_n_reps:
            print(f"\n  [{name}] using {n_reps} repetitions "
                  f"(10x requested {requested_n_reps} for n=50)")
        excluded_methods = [
            m for m in DEFAULT_REAL_METHODS
            if _canonical_method_name(m)
            not in {_canonical_method_name(s) for s in selected_methods}
        ]
        if excluded_methods:
            print(f"  [{name}] skipping methods by large-n real policy: "
                  f"{', '.join(excluded_methods)}")

        # Count how many reps are already fully cached
        cached_reps = 0
        if not args.force:
            for r in range(n_reps):
                mf = partial_dir / f"rep{r}" / f"{name}_metrics.json"
                if mf.exists():
                    cached_reps += 1
                else:
                    break

        use_cache = (not args.force
                     and name in existing_results
                     and cache_file.exists()
                     and cached_reps >= n_reps)

        if use_cache:
            print(f"\n[cache] Skipping '{name}' -- {cached_reps}/{n_reps} "
                  f"reps cached. Use --force to re-run.")
            cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_total = \
                load_cache(cache_file)
            cdes = _filter_method_mapping(cdes, selected_methods)
            zgrids = _filter_method_mapping(zgrids, selected_methods)
            all_results[name] = {
                m: {k: v for k, v in metrics.items()}
                for m, metrics in _filter_method_mapping(
                    existing_results[name], selected_methods
                ).items()
            }
        else:
            if cached_reps > 0 and cached_reps < n_reps:
                print(f"\n  [{name}] {cached_reps} rep(s) cached, "
                      f"running {n_reps - cached_reps} more...")
            per_rep_results = []
            for rep in range(n_reps):
                rep_partial = partial_dir / f"rep{rep}"
                rep_partial.mkdir(exist_ok=True)
                print(f"\n  ── rep {rep+1}/{n_reps} (seed={rep}) ──")
                res, cdes, zgrids, X_te, z_te, true_cde, true_zgrid = \
                    run_experiment(
                        X, z, name, device=args.device,
                        true_density_fn=true_density_fn,
                        partial_dir=rep_partial, force=args.force,
                        random_state=rep,
                        methods=selected_methods,
                    )
                per_rep_results.append(res)
            n_total = len(z)
            all_results[name] = aggregate_reps(per_rep_results)
            save_cache(cache_file, cdes, zgrids, X_te, z_te,
                       true_cde, true_zgrid, n_total)

        all_data[name] = {
            'cdes': cdes, 'zgrids': zgrids,
            'X_test': X_te, 'z_test': z_te,
            'true_cde': true_cde, 'true_zgrid': true_zgrid,
            'n_total': n_total,
        }

    print_summary(all_results, se_caption='mean +/- SE across repetitions')

    print("\nGenerating plots and tables...")
    save_html_table(all_results, output_dir)
    save_latex_table(all_results, output_dir)
    save_appendix_metric_tables(all_results, output_dir)
    save_appendix_metric_tables_html(all_results, output_dir)
    plot_native_tab_subset(all_data, output_dir)
    # plot_rankings_by_n(all_results, output_dir, all_data=all_data)  # disabled
    plot_critical_difference(all_results, output_dir, all_data=all_data)
    plot_raw_metrics_by_n(all_results, output_dir, all_data=all_data)
    plot_raw_metrics_with_values_by_n(all_results, output_dir, all_data=all_data)
    plot_pit_histograms(all_data, output_dir)
    plot_performance_vs_n(all_results, output_dir, all_data=all_data)
    plot_performance_vs_n_foundational(all_results, output_dir, all_data=all_data)
    plot_perf_vs_n_foundational_cde_subsets(all_results, output_dir, all_data=all_data)

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

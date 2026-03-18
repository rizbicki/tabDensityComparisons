#!/usr/bin/env python
"""
Regenerate plots from cached results without re-running experiments.

USAGE:
  python generate_plots.py [--sim-dir results_simulated] [--real-dir results_real]
  python generate_plots.py --metrics-only
"""

import argparse
import json
from pathlib import Path

from utils import load_cache
from visualization import (
    plot_rankings_by_n, plot_raw_metrics_by_n,
    plot_pit_histograms, plot_native_tab_subset,
    plot_performance_vs_n, plot_performance_vs_n_foundational,
    save_html_table,
)


def _load_from_dir(output_dir, all_results, all_data, load_cache_data=True):
    """Load results.json and optional cache files from one directory."""
    output_dir = Path(output_dir)
    cache_dir = output_dir / 'cache'
    json_path = output_dir / 'results.json'

    if not json_path.exists():
        print(f"  [skip] {json_path} not found")
        return

    with open(json_path) as f:
        results = json.load(f)
    print(f"  Loaded {len(results)} datasets from {json_path}")
    all_results.update(results)

    if not load_cache_data:
        return

    for dataset_name in results:
        cache_file = cache_dir / f"{dataset_name}.npz"
        if cache_file.exists():
            try:
                cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_total = \
                    load_cache(cache_file)
                all_data[dataset_name] = {
                    'cdes': cdes, 'zgrids': zgrids,
                    'X_test': X_te, 'z_test': z_te,
                    'true_cde': true_cde, 'true_zgrid': true_zgrid,
                    'n_total': n_total,
                }
                print(f"    ✓ {dataset_name}")
            except Exception as e:
                print(f"    ✗ {dataset_name}: {e}")
        else:
            print(f"    ? {dataset_name}: cache file not found")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate plots from cached results')
    parser.add_argument('--sim-dir', default='results_simulated',
                        help='Directory containing simulated results/cache '
                             '(default: results_simulated)')
    parser.add_argument('--real-dir', default='results_real',
                        help='Directory containing real results/cache '
                             '(default: results_real)')
    parser.add_argument('--metrics-only', action='store_true',
                        help='Regenerate only metrics-based tables/plots from '
                             'results.json, skipping cache-dependent PIT and '
                             'density plots')
    args = parser.parse_args()

    output_dirs = {
        'sim': Path(args.sim_dir),
        'real': Path(args.real_dir),
    }
    for out_dir in output_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_data = {}

    print("Loading results...")
    _load_from_dir(output_dirs['sim'], all_results, all_data,
                   load_cache_data=not args.metrics_only)
    _load_from_dir(output_dirs['real'], all_results, all_data,
                   load_cache_data=not args.metrics_only)

    if not all_results:
        print("No results found. Run experiments first.")
        return

    print(f"\nTotal: {len(all_results)} datasets, {len(all_data)} with cached data")

    print("\nGenerating plots and tables...")
    save_html_table(all_results, output_dirs)
    print("  ✓ HTML table")

    if args.metrics_only:
        print("  - Native TabPFN subset plots skipped (--metrics-only)")
    else:
        plot_native_tab_subset(all_data, output_dirs)
        print("  ✓ Native TabPFN subset plots")

    plot_rankings_by_n(all_results, output_dirs, all_data=all_data)
    print("  ✓ Rankings by n")

    plot_raw_metrics_by_n(all_results, output_dirs, all_data=all_data)
    print("  ✓ Raw metrics by n")

    if args.metrics_only:
        print("  - PIT histograms skipped (--metrics-only)")
    else:
        plot_pit_histograms(all_data, output_dirs)
        print("  ✓ PIT histograms")

    plot_performance_vs_n(all_results, output_dirs, all_data=all_data)
    print("  ✓ Performance vs n")

    plot_performance_vs_n_foundational(all_results, output_dirs, all_data=all_data)
    print("  ✓ Performance vs n (foundational models)")

    print("\nDone!")


if __name__ == '__main__':
    main()

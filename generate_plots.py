#!/usr/bin/env python
"""
Regenerate plots from cached results without re-running experiments.

Can optionally merge results from multiple output directories.

USAGE:
  python generate_plots.py [--output-dir results] [--merge-dir OTHER_DIR]
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


def _load_from_dir(output_dir, all_results, all_data):
    """Load results.json and cache files from one directory into shared dicts."""
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
    parser.add_argument('--output-dir', default='results',
                        help='Primary output directory containing cache/ and results.json')
    parser.add_argument('--merge-dir',
                        help='Optional additional directory to merge results from')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results = {}
    all_data = {}

    print("Loading results...")
    _load_from_dir(output_dir, all_results, all_data)
    if args.merge_dir and Path(args.merge_dir).exists():
        _load_from_dir(args.merge_dir, all_results, all_data)

    if not all_results:
        print("No results found. Run experiments first.")
        return

    print(f"\nTotal: {len(all_results)} datasets, {len(all_data)} with cached data")

    print("\nGenerating plots and tables...")
    save_html_table(all_results, output_dir)
    print("  ✓ HTML table")

    plot_native_tab_subset(all_data, output_dir)
    print("  ✓ Native TabPFN subset plots")

    plot_rankings_by_n(all_results, output_dir, all_data=all_data)
    print("  ✓ Rankings by n")

    plot_raw_metrics_by_n(all_results, output_dir, all_data=all_data)
    print("  ✓ Raw metrics by n")

    plot_pit_histograms(all_data, output_dir)
    print("  ✓ PIT histograms")

    plot_performance_vs_n(all_results, output_dir, all_data=all_data)
    print("  ✓ Performance vs n")

    plot_performance_vs_n_foundational(all_results, output_dir, all_data=all_data)
    print("  ✓ Performance vs n (foundational models)")

    print("\nDone!")


if __name__ == '__main__':
    main()

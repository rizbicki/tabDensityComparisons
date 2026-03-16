#!/usr/bin/env python
"""
Regenerate plots from cached results without re-running experiments.

USAGE:
  python generate_plots.py [--output-dir results]
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


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate plots from cached results')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory containing cache/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / 'cache'
    json_path = output_dir / 'results.json'

    if not cache_dir.exists():
        print(f"Error: {cache_dir} not found")
        return

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    # Load metrics from JSON
    with open(json_path) as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} dataset results from {json_path}")

    # Load cached data for each dataset
    all_data = {}
    for dataset_name in all_results:
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
                print(f"  ✓ {dataset_name}")
            except Exception as e:
                print(f"  ✗ {dataset_name}: {e}")
        else:
            print(f"  ? {dataset_name}: cache file not found")

    print(f"\nLoaded cached data for {len(all_data)} datasets")

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

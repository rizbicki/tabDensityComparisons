"""
FlexCode x Tabular Foundation Models: Real Dataset CDE Experiments
==================================================================

Runs experiments only on real (OpenML) and semi-synthetic (Friedman) datasets.

USAGE:
  python run_real_experiments.py [--device cpu|cuda] [--force] [--n-reps N]
"""

import argparse
import json
from pathlib import Path

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
    plot_pit_histograms, plot_native_tab_subset,
    plot_performance_vs_n, plot_performance_vs_n_foundational,
    save_html_table,
    save_latex_table,
)
from utils import save_cache, load_cache, print_summary, aggregate_reps


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
    n_reps = args.n_reps

    all_results = {}
    all_data = {}

    for X, z, name, true_density_fn in datasets:
        cache_file = cache_dir / f"{name}.npz"

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
            all_results[name] = {
                m: {k: v for k, v in existing_results[name][m].items()}
                for m in existing_results[name]
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
    plot_native_tab_subset(all_data, output_dir)
    plot_rankings_by_n(all_results, output_dir, all_data=all_data)
    plot_critical_difference(all_results, output_dir, all_data=all_data)
    plot_raw_metrics_by_n(all_results, output_dir, all_data=all_data)
    plot_pit_histograms(all_data, output_dir)
    plot_performance_vs_n(all_results, output_dir, all_data=all_data)
    plot_performance_vs_n_foundational(all_results, output_dir, all_data=all_data)

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

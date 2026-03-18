#!/usr/bin/env python
"""
Build results.json from partial results saved during experiments.

Scans partial/rep*/ directories under each source dir for
{dataset_name}_metrics.json files, aggregates across repetitions
(mean +/- SE), merges all sources, and writes a single results.json
plus .npz cache files for generate_plots.py.

USAGE:
  python consolidate_partial_results.py
  python consolidate_partial_results.py --output-dir results --source-dirs results_simulated results_real
"""

import argparse
import json
from pathlib import Path
import numpy as np


MEAN_METRICS = ['CDE_loss', 'log_lik', 'CRPS', 'PIT_KS',
                'coverage_90', 'interval_width', 'fit_time', 'pred_time']


def _aggregate_reps(per_rep_results):
    """Aggregate metrics across repetitions: mean +/- SE."""
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
                agg_m[f'{k}_se'] = (
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                    if len(arr) > 1 else None
                )
            else:
                agg_m[k] = None
                agg_m[f'{k}_se'] = None
        agg_m['n_basis'] = float(np.mean(n_basis_vals)) if n_basis_vals else None
        agg[m] = agg_m
    return agg


def _consolidate_source(source_dir, all_results, output_cache_dir):
    """Scan one source directory's partial/rep*/ and merge into all_results."""
    source_dir = Path(source_dir)
    partial_dir = source_dir / 'cache' / 'partial'
    if not partial_dir.exists():
        print(f"  [skip] {partial_dir} not found")
        return

    rep_dirs = sorted(partial_dir.glob('rep*'))
    if not rep_dirs:
        print(f"  [skip] no rep dirs in {partial_dir}")
        return

    print(f"  {source_dir}: {len(rep_dirs)} rep(s)")

    dataset_reps = {}
    for rep_dir in rep_dirs:
        for mf in sorted(rep_dir.glob('*_metrics.json')):
            try:
                with open(mf) as f:
                    metrics = json.load(f)
                dataset_name = mf.stem.replace('_metrics', '')
                dataset_reps.setdefault(dataset_name, []).append(metrics)
            except Exception as e:
                print(f"    ! {mf}: {e}")

    last_rep = rep_dirs[-1]
    for ds_name in sorted(dataset_reps):
        reps = dataset_reps[ds_name]
        all_results[ds_name] = _aggregate_reps(reps)
        print(f"    {ds_name}: {len(reps)} rep(s), "
              f"{len(all_results[ds_name])} method(s)")

        # Reconstruct .npz in the output cache dir if missing
        npz_path = output_cache_dir / f"{ds_name}.npz"
        if npz_path.exists():
            continue
        methods, arrays = [], {}
        for cdes_file in sorted(last_rep.glob(f"{ds_name}_*_cdes.npy")):
            method = cdes_file.stem[len(ds_name) + 1:].replace("_cdes", "")
            zgrid_file = last_rep / f"{ds_name}_{method}_zgrid.npy"
            if zgrid_file.exists():
                try:
                    arrays[f'cde_{method}'] = np.load(cdes_file)
                    arrays[f'zgrid_{method}'] = np.load(zgrid_file)
                    methods.append(method)
                except Exception as e:
                    print(f"      ! {method}: {e}")
        if methods:
            arrays['methods'] = np.array(methods)
            arrays['X_te'] = np.array([])
            arrays['z_te'] = np.array([])
            arrays['n_total'] = np.array(0)
            arrays['true_cde'] = np.array([])
            arrays['true_zgrid'] = np.array([])
            np.savez(npz_path, **arrays)


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate partial results from all sources into one results.json')
    parser.add_argument('--output-dir',
                        help='Optional merged output directory. If omitted, each '
                             'source dir is consolidated in place.')
    parser.add_argument('--source-dirs', nargs='+',
                        default=['results_simulated', 'results_real'],
                        help='Directories to scan for partial results '
                             '(default: results_simulated results_real)')
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        output_cache_dir = output_dir / 'cache'
        output_cache_dir.mkdir(exist_ok=True)

        all_results = {}
        print("Consolidating partial results...")
        for src in args.source_dirs:
            _consolidate_source(src, all_results, output_cache_dir)

        if not all_results:
            print("No results found.")
            return

        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {json_path} ({len(all_results)} dataset(s))")
        print("Ready to run: python generate_plots.py")
        return

    wrote_any = False
    print("Consolidating partial results in place...")
    for src in args.source_dirs:
        output_dir = Path(src)
        output_dir.mkdir(exist_ok=True)
        output_cache_dir = output_dir / 'cache'
        output_cache_dir.mkdir(exist_ok=True)

        all_results = {}
        _consolidate_source(src, all_results, output_cache_dir)
        if not all_results:
            continue

        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {json_path} ({len(all_results)} dataset(s))")
        wrote_any = True

    if not wrote_any:
        print("No results found.")
    else:
        print("Ready to run: python generate_plots.py")


if __name__ == '__main__':
    main()

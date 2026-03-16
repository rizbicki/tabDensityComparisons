#!/usr/bin/env python
"""
Build results.json from partial results saved during experiments.

Scans results/cache/partial/rep*/ for {dataset_name}_metrics.json files,
aggregates across repetitions (mean +/- SE), and writes results/results.json.
Also reconstructs .npz cache files from the last rep for generate_plots.py.

USAGE:
  python consolidate_partial_results.py [--output-dir results]
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


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate partial results into results.json')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / 'cache'
    partial_dir = cache_dir / 'partial'

    if not partial_dir.exists():
        print(f"Error: {partial_dir} not found")
        return

    # Find all rep directories
    rep_dirs = sorted(partial_dir.glob('rep*'))
    if not rep_dirs:
        print(f"No rep directories found in {partial_dir}")
        return

    print(f"Found {len(rep_dirs)} rep directories")

    # Collect per-dataset, per-rep metrics
    dataset_reps = {}  # {dataset_name: [rep0_metrics, rep1_metrics, ...]}
    for rep_dir in rep_dirs:
        for mf in sorted(rep_dir.glob('*_metrics.json')):
            try:
                with open(mf) as f:
                    metrics = json.load(f)
                dataset_name = mf.stem.replace('_metrics', '')
                if dataset_name not in dataset_reps:
                    dataset_reps[dataset_name] = []
                dataset_reps[dataset_name].append(metrics)
            except Exception as e:
                print(f"  ! {mf}: {e}")

    print(f"Found {len(dataset_reps)} dataset(s)")

    # Aggregate across reps
    all_results = {}
    for ds_name in sorted(dataset_reps):
        reps = dataset_reps[ds_name]
        all_results[ds_name] = _aggregate_reps(reps)
        n_methods = len(all_results[ds_name])
        print(f"  {ds_name}: {len(reps)} rep(s), {n_methods} method(s)")

    # Save to results.json
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {json_path} ({len(all_results)} dataset(s))")

    # Reconstruct .npz files from the last rep's .npy files
    last_rep = rep_dirs[-1]
    cdes_files = sorted(last_rep.glob('*_cdes.npy'))
    # Derive dataset names by removing the method+suffix part
    ds_names_seen = set()
    for cf in cdes_files:
        # filename: {dataset}_{method}_cdes.npy — but dataset may contain _
        # Use metrics files as the source of truth for dataset names
        pass

    for ds_name in sorted(dataset_reps):
        npz_path = cache_dir / f"{ds_name}.npz"
        if npz_path.exists():
            continue

        methods = []
        arrays = {}
        for cdes_file in sorted(last_rep.glob(f"{ds_name}_*_cdes.npy")):
            method = cdes_file.stem.replace(f"{ds_name}_", "").replace("_cdes", "")
            zgrid_file = last_rep / f"{ds_name}_{method}_zgrid.npy"
            if zgrid_file.exists():
                try:
                    arrays[f'cde_{method}'] = np.load(cdes_file)
                    arrays[f'zgrid_{method}'] = np.load(zgrid_file)
                    methods.append(method)
                except Exception as e:
                    print(f"    ! Failed to load {method}: {e}")

        if methods:
            arrays['methods'] = np.array(methods)
            arrays['X_te'] = np.array([])
            arrays['z_te'] = np.array([])
            arrays['n_total'] = np.array(0)
            arrays['true_cde'] = np.array([])
            arrays['true_zgrid'] = np.array([])
            np.savez(npz_path, **arrays)

    print(f"\nReady to run: python generate_plots.py")


if __name__ == '__main__':
    main()

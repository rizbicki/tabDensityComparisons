#!/usr/bin/env python
"""
Build results.json from partial results saved during experiments.

Scans results/cache/partial/ for {dataset_name}_metrics.json files
and aggregates them into results/results.json. Also creates symlinks
to partial cache files for generate_plots.py to use.

USAGE:
  python consolidate_partial_results.py [--output-dir results]
"""

import argparse
import json
from pathlib import Path
import numpy as np


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

    # Scan for *_metrics.json files
    metrics_files = sorted(partial_dir.glob('*_metrics.json'))

    if not metrics_files:
        print(f"No metrics files found in {partial_dir}")
        return

    print(f"Found {len(metrics_files)} partial metrics file(s)")

    # Aggregate into results.json
    all_results = {}
    for mf in metrics_files:
        try:
            with open(mf) as f:
                metrics = json.load(f)
            # Extract dataset name from filename (remove _metrics.json)
            dataset_name = mf.stem.replace('_metrics', '')
            all_results[dataset_name] = metrics
            print(f"  ✓ {dataset_name} ({len(metrics)} method(s))")
        except Exception as e:
            print(f"  ✗ {mf.name}: {e}")

    # Save to results.json
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Wrote {json_path} ({len(all_results)} dataset(s))")

    # Reconstruct .npz files from partial .npy files
    cdes_files = sorted(partial_dir.glob('*_cdes.npy'))
    dataset_names = set(f.stem.rsplit('_cdes', 1)[0] for f in cdes_files)

    print(f"\nReconstructing .npz files from {len(dataset_names)} dataset(s)...")
    for ds_name in sorted(dataset_names):
        npz_path = cache_dir / f"{ds_name}.npz"

        # Skip if already exists
        if npz_path.exists():
            continue

        # Collect all methods for this dataset
        methods = []
        arrays = {}

        for cdes_file in sorted(partial_dir.glob(f"{ds_name}_*_cdes.npy")):
            method = cdes_file.stem.replace(f"{ds_name}_", "").replace("_cdes", "")
            zgrid_file = partial_dir / f"{ds_name}_{method}_zgrid.npy"

            if zgrid_file.exists():
                try:
                    arrays[f'cde_{method}'] = np.load(cdes_file)
                    arrays[f'zgrid_{method}'] = np.load(zgrid_file)
                    methods.append(method)
                except Exception as e:
                    print(f"    ✗ Failed to load {method}: {e}")

        if methods:
            # Add metadata
            arrays['methods'] = np.array(methods)
            arrays['X_te'] = np.array([])  # Placeholder
            arrays['z_te'] = np.array([])  # Placeholder
            arrays['n_total'] = np.array(0)  # Placeholder
            arrays['true_cde'] = np.array([])
            arrays['true_zgrid'] = np.array([])

            np.savez(npz_path, **arrays)
            print(f"  ✓ {ds_name} ({len(methods)} method(s))")

    print(f"\n✓ Ready to run: python generate_plots.py")


if __name__ == '__main__':
    main()

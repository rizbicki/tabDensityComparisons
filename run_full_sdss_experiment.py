"""
Run selected CDE methods on the full SDSS spectroscopic redshift dataset.

This uses the complete local CSV rather than the benchmark subsamples.
Metrics are computed on a held-out test split from the full dataset.

Methods run one at a time, in a fast-first default order, and results are
written incrementally after each completed method.

USAGE:
  .venv/bin/python run_full_sdss_experiment.py
  .venv/bin/python run_full_sdss_experiment.py --device cuda --force
  .venv/bin/python run_full_sdss_experiment.py --methods TabPFN-Native,Student-t
  .venv/bin/python run_full_sdss_experiment.py --plot-partial-results-only
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from datasets import load_sdss_dataset
from run_experiments import run_experiment
from utils import load_cache, save_cache, print_summary
from visualization import plot_density_comparison, plot_pit_histograms, save_html_table


DEFAULT_METHOD_ORDER = [
    "LinearGauss-Homo",
    "LinearGauss-Hetero",
    "Student-t",
    "LogNormal-Homo",
    "LogNormal-Hetero",
    "Gamma-GLM",
    "LinGauss-Homo-Ridge",
    "LinGauss-Hetero-Ridge",
    "Student-t-Ridge",
    "LogNormal-Homo-Ridge",
    "LogNormal-Hetero-Ridge",
    "Gamma-GLM-Ridge",
    "TabPFN-Native",
    "TabPFN-2.5",
    "RealTabPFN-2.5",
    "TabICL-Quantiles",
    "Quantile-Tree",
    "MDN",
    "Flow-Spline",
    "BART-Homo",
    "BART-Hetero",
    "FlexCode-RF",
    "CatMLP",
]


def _key(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _canonical_method_name(name):
    return "MDN" if name == "MDN-2mix" else name


def _method_aliases(name):
    canonical = _canonical_method_name(name)
    if canonical == "MDN":
        return ("MDN", "MDN-2mix")
    return (canonical,)


def _canonicalize_methods(methods):
    return list(dict.fromkeys(_canonical_method_name(m) for m in methods))


def _normalize_method_mapping(mapping):
    normalized = {}
    for name, value in mapping.items():
        canonical = _canonical_method_name(name)
        if canonical in normalized and name != canonical:
            continue
        normalized[canonical] = value
    return normalized


def _parse_methods_arg(raw):
    return [m.strip() for m in raw.split(",") if m.strip()]


def _load_partial_metrics(partial_dir, dataset_name):
    metrics_file = partial_dir / f"{dataset_name}_metrics.json"
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        return _normalize_method_mapping(json.load(f))


def _load_cached_arrays(partial_dir, dataset_name, methods):
    cdes = {}
    zgrids = {}
    for method in methods:
        for alias in _method_aliases(method):
            method_key = _key(alias)
            cde_file = partial_dir / f"{dataset_name}_{method_key}_cdes.npy"
            zgrid_file = partial_dir / f"{dataset_name}_{method_key}_zgrid.npy"
            if cde_file.exists() and zgrid_file.exists():
                cdes[method] = np.load(cde_file)
                zgrids[method] = np.load(zgrid_file)
                break
    return cdes, zgrids


def load_full_sdss():
    """Load the full SDSS photo-z CSV as (X, z)."""
    return load_sdss_dataset()


def _write_results_json(json_file, dataset_name, partial_metrics):
    json_out = {
        dataset_name: {
            method_name: {
                key: float(value) if value is not None else None
                for key, value in metrics.items()
            }
            for method_name, metrics in partial_metrics.items()
        }
    }
    with open(json_file, "w") as f:
        json.dump(json_out, f, indent=2)


def generate_full_sdss_partial_plots(output_dir="results_real/sdss_full",
                                     dataset_name="SDSS-full"):
    """Generate table and cache-based plots from the current full-SDSS partial results."""
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    partial_dir = cache_dir / "partial" / "rep0"
    cache_file = cache_dir / f"{dataset_name}.npz"
    json_file = output_dir / "results.json"

    partial_metrics = _load_partial_metrics(partial_dir, dataset_name)
    if not partial_metrics:
        print(f"[skip] No partial metrics found in {partial_dir}")
        return False

    _write_results_json(json_file, dataset_name, partial_metrics)
    print(f"saved {json_file}")

    all_results = {dataset_name: partial_metrics}
    save_html_table(all_results, output_dir,
                    se_caption='mean +/- SE over test samples')

    if not cache_file.exists():
        print(f"[skip] {cache_file} not found; density examples and PIT plots require cached test data")
        return True

    try:
        _, _, X_te, z_te, true_cde, true_zgrid, n_total = load_cache(cache_file)
    except Exception as exc:
        print(f"[skip] Could not load {cache_file}: {type(exc).__name__}: {exc}")
        return True

    methods = list(partial_metrics)
    cdes, zgrids = _load_cached_arrays(partial_dir, dataset_name, methods)
    if not cdes:
        print(f"[skip] No cached method arrays found in {partial_dir}")
        return True

    all_data = {
        dataset_name: {
            "cdes": cdes,
            "zgrids": zgrids,
            "X_test": X_te,
            "z_test": z_te,
            "true_cde": true_cde,
            "true_zgrid": true_zgrid,
            "n_total": n_total,
        }
    }
    plot_density_comparison(all_data, output_dir)
    plot_pit_histograms(all_data, output_dir)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run all methods on the full SDSS spectroscopic redshift dataset"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="results_real/sdss_full",
                        help="Output directory (default: results_real/sdss_full)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached outputs exist")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Train/test split seed (default: 42)")
    parser.add_argument("--n-grid", type=int, default=200,
                        help="Number of grid points for densities (default: 200)")
    parser.add_argument("--methods",
                        help="Comma-separated list of methods to run, in the given order")
    parser.add_argument("--exclude",
                        help="Comma-separated list of methods to skip")
    parser.add_argument("--list-methods", action="store_true",
                        help="Print available method names and exit")
    parser.add_argument("--plot-partial-results-only", action="store_true",
                        help="Generate results_table.html, density_examples.png, "
                             "and pit_calibration.png from the current partial "
                             "full-SDSS outputs, then exit")
    parser.add_argument("--refresh-plots-after-each-method", action="store_true",
                        help="Regenerate the partial full-SDSS table/plots after "
                             "each completed method")
    args = parser.parse_args()

    if args.list_methods:
        print("\n".join(DEFAULT_METHOD_ORDER))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    partial_dir = cache_dir / "partial" / "rep0"
    partial_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "SDSS-full"
    cache_file = cache_dir / f"{dataset_name}.npz"
    json_file = output_dir / "results.json"

    if args.plot_partial_results_only:
        generate_full_sdss_partial_plots(output_dir=output_dir,
                                         dataset_name=dataset_name)
        return

    if args.methods:
        selected_methods = _canonicalize_methods(_parse_methods_arg(args.methods))
    else:
        selected_methods = list(DEFAULT_METHOD_ORDER)
    if args.exclude:
        excluded = set(_canonicalize_methods(_parse_methods_arg(args.exclude)))
        selected_methods = [m for m in selected_methods if m not in excluded]
    unknown = [m for m in selected_methods if m not in DEFAULT_METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    if not selected_methods:
        raise ValueError("No methods selected")

    X, z = load_full_sdss()
    print(f"Loaded {dataset_name}: n={len(z)}, d={X.shape[1]}")
    print("Method order:")
    for method in selected_methods:
        print(f"  - {method}")

    all_results = {}
    last_X_te = None
    last_z_te = None

    for method in selected_methods:
        print(f"\nRunning method: {method}")
        results, _, _, X_te, z_te, true_cde, true_zgrid = run_experiment(
            X,
            z,
            dataset_name,
            device=args.device,
            n_grid=args.n_grid,
            true_density_fn=None,
            partial_dir=partial_dir,
            force=args.force,
            random_state=args.random_state,
            methods=[method],
        )
        if method in results:
            all_results.update(results)
            last_X_te = X_te
            last_z_te = z_te

        partial_metrics = _load_partial_metrics(partial_dir, dataset_name)
        cached_methods = [m for m in selected_methods if m in partial_metrics]
        cdes, zgrids = _load_cached_arrays(partial_dir, dataset_name, cached_methods)

        if last_X_te is not None and cdes:
            save_cache(cache_file, cdes, zgrids, last_X_te, last_z_te,
                       true_cde, true_zgrid, len(z))

        _write_results_json(json_file, dataset_name, partial_metrics)
        print(f"  wrote incremental metrics to {json_file}")
        if args.refresh_plots_after_each_method:
            generate_full_sdss_partial_plots(output_dir=output_dir,
                                             dataset_name=dataset_name)

    final_metrics = _load_partial_metrics(partial_dir, dataset_name)
    generate_full_sdss_partial_plots(output_dir=output_dir,
                                     dataset_name=dataset_name)
    print_summary({dataset_name: final_metrics},
                  se_caption='mean +/- SE over test samples')
    print(f"\nSaved metrics to {json_file}")
    print(f"Saved cache to {cache_file}")


if __name__ == "__main__":
    main()

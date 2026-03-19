"""
Run an SDSS scaling study across multiple sample sizes.

This benchmark uses deterministic SDSS subsamples at increasing n, runs a
conservative set of methods for each size over multiple repetitions, and
generates performance-vs-n plots from the aggregated metrics.

By default, expensive methods are dropped as n grows when they have explicit
package limits or are expected to be impractical for a single-run scaling study.

USAGE:
  .venv/bin/python run_sdss_scaling_experiment.py
  .venv/bin/python run_sdss_scaling_experiment.py --device cuda
  .venv/bin/python run_sdss_scaling_experiment.py --sample-sizes 10000,50000,full
  .venv/bin/python run_sdss_scaling_experiment.py --methods LinearGauss-Homo,Flow-Spline
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
from visualization import (
    plot_performance_vs_n,
    plot_performance_vs_n_foundational,
    save_html_table,
)


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
    "Quantile-Linear",
    "MDN-2mix",
    "Flow-Spline",
    "BART-Homo",
    "BART-Hetero",
    "FlexCode-RF",
]

DEFAULT_SAMPLE_SIZE_SPEC = "10000,50000,100000,250000,500000,full"
TABPFN_METHODS = {"TabPFN-Native", "TabPFN-2.5", "RealTabPFN-2.5"}
MEAN_METRICS = [
    "CDE_loss",
    "log_lik",
    "CRPS",
    "PIT_KS",
    "coverage_90",
    "interval_width",
    "fit_time",
    "pred_time",
]


def _key(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _parse_methods_arg(raw):
    return [m.strip() for m in raw.split(",") if m.strip()]


def _load_partial_metrics(partial_dir, dataset_name):
    metrics_file = partial_dir / f"{dataset_name}_metrics.json"
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        return json.load(f)


def _load_cached_arrays(partial_dir, dataset_name, methods):
    cdes = {}
    zgrids = {}
    for method in methods:
        method_key = _key(method)
        cde_file = partial_dir / f"{dataset_name}_{method_key}_cdes.npy"
        zgrid_file = partial_dir / f"{dataset_name}_{method_key}_zgrid.npy"
        if cde_file.exists() and zgrid_file.exists():
            cdes[method] = np.load(cde_file)
            zgrids[method] = np.load(zgrid_file)
    return cdes, zgrids


def _aggregate_reps(per_rep_results):
    methods = sorted(set(m for rep in per_rep_results for m in rep))
    agg = {}
    for method in methods:
        vals = {k: [] for k in MEAN_METRICS}
        n_basis_vals = []
        rep_count = 0

        for rep in per_rep_results:
            if method not in rep:
                continue
            rep_count += 1
            for metric in MEAN_METRICS:
                if rep[method].get(metric) is not None:
                    vals[metric].append(rep[method][metric])
            if rep[method].get("n_basis") is not None:
                n_basis_vals.append(rep[method]["n_basis"])

        agg_method = {"n_reps": rep_count}
        for metric in MEAN_METRICS:
            arr = np.array(vals[metric], dtype=float)
            if len(arr) > 0:
                agg_method[metric] = float(np.mean(arr))
                agg_method[f"{metric}_se"] = (
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                    if len(arr) > 1 else None
                )
            else:
                agg_method[metric] = None
                agg_method[f"{metric}_se"] = None
        agg_method["n_basis"] = float(np.mean(n_basis_vals)) if n_basis_vals else None
        agg[method] = agg_method
    return agg


def _resolve_runtime_device(requested):
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _parse_sample_sizes(raw, full_n):
    sizes = []
    for token in _parse_methods_arg(raw):
        token_l = token.lower()
        if token_l == "full":
            size = full_n
        else:
            try:
                size = int(token_l.replace("_", ""))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid sample-size token '{token}'. Use integers or 'full'."
                ) from exc
        if size < 1:
            raise ValueError("Sample sizes must be positive")
        if size > full_n:
            raise ValueError(
                f"Requested n={size:,}, but SDSS only has {full_n:,} usable rows"
            )
        sizes.append(size)

    if not sizes:
        raise ValueError("No sample sizes selected")
    return sorted(dict.fromkeys(sizes))


def _subsample_from_full(X, z, target_n, seed):
    if target_n >= len(z):
        return X, z
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(z))
    idx = perm[:target_n]
    return X[idx], z[idx]


def _default_skip_reason(method, n_total, runtime_device):
    if method == "Quantile-Linear" and n_total > 10_000:
        return "disabled above n=10,000 by the linear-quantile baseline itself"

    if method == "FlexCode-RF" and n_total > 500_000:
        return "5-fold CV over many random-forest basis regressions is conservatively capped at n=500,000"

    if method in TABPFN_METHODS:
        limit = 1_000 if runtime_device == "cpu" else 50_000
        if n_total > limit:
            if runtime_device == "cpu":
                return "the installed TabPFN package blocks CPU runs above n=1,000"
            return "the installed TabPFN v2.5 configuration supports up to n=50,000"

    if method == "TabICL-Quantiles":
        limit = 10_000 if runtime_device == "cpu" else 50_000
        if n_total > limit:
            return (
                f"native TabICL quantile inference is conservatively capped at n={limit:,} "
                f"for {runtime_device.upper()} scaling runs"
            )

    if method in {"BART-Homo", "BART-Hetero"} and n_total > 100_000:
        return "XBART baselines are conservatively capped at n=100,000 for this scaling study"

    if method == "Quantile-Tree" and n_total > 100_000:
        return "fits many quantile models and is conservatively capped at n=100,000"

    if method == "MDN-2mix":
        limit = 500_000
        if n_total > limit:
            return (
                f"full-batch MDN training is conservatively capped at n={limit:,} "
                f"for {runtime_device.upper()} scaling runs"
            )

    if method == "Flow-Spline":
        limit = 100_000 if runtime_device == "cpu" else 500_000
        if n_total > limit:
            return (
                f"flow training/inference is conservatively capped at n={limit:,} "
                f"for {runtime_device.upper()} scaling runs"
            )

    return None


def _select_methods(base_methods, n_total, runtime_device, use_policy=True):
    selected = []
    skipped = {}
    for method in base_methods:
        if not use_policy:
            selected.append(method)
            continue
        reason = _default_skip_reason(method, n_total, runtime_device)
        if reason is None:
            selected.append(method)
        else:
            skipped[method] = reason
    return selected, skipped


def _load_existing_output(output_dir):
    all_results = {}
    all_data = {}

    json_file = output_dir / "results.json"
    if json_file.exists():
        with open(json_file) as f:
            all_results = json.load(f)

    cache_dir = output_dir / "cache"
    for dataset_name in all_results:
        cache_file = cache_dir / f"{dataset_name}.npz"
        if not cache_file.exists():
            continue
        try:
            cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_total = load_cache(cache_file)
            all_data[dataset_name] = {
                "cdes": cdes,
                "zgrids": zgrids,
                "X_test": X_te,
                "z_test": z_te,
                "true_cde": true_cde,
                "true_zgrid": true_zgrid,
                "n_total": n_total,
            }
        except Exception as exc:
            print(f"  [warn] could not load cache for {dataset_name}: {exc}")

    return all_results, all_data


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run an SDSS sample-size scaling benchmark"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="results_real/sdss_scaling",
                        help="Output directory (default: results_real/sdss_scaling)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run selected methods even if cached outputs exist")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Train/test split seed used inside each SDSS subset")
    parser.add_argument("--subsample-seed", type=int, default=42,
                        help="Seed for the nested SDSS subsamples (default: 42)")
    parser.add_argument("--n-reps", type=int, default=4,
                        help="Number of repetitions per sample size (default: 4)")
    parser.add_argument("--n-grid", type=int, default=200,
                        help="Number of grid points for densities (default: 200)")
    parser.add_argument("--sample-sizes", default=DEFAULT_SAMPLE_SIZE_SPEC,
                        help="Comma-separated sizes, e.g. 10000,50000,full")
    parser.add_argument("--methods",
                        help="Comma-separated list of methods to run in the given order")
    parser.add_argument("--exclude",
                        help="Comma-separated list of methods to skip")
    parser.add_argument("--all-methods-at-all-sizes", action="store_true",
                        help="Disable the conservative large-n pruning policy")
    parser.add_argument("--list-methods", action="store_true",
                        help="Print available method names and exit")
    args = parser.parse_args()

    if args.list_methods:
        print("\n".join(DEFAULT_METHOD_ORDER))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    partial_root = cache_dir / "partial"
    partial_root.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    policy_file = output_dir / "method_policy.json"

    if args.force:
        all_results = {}
        all_data = {}
        policy_manifest = {}
    else:
        all_results, all_data = _load_existing_output(output_dir)
        if policy_file.exists():
            with open(policy_file) as f:
                policy_manifest = json.load(f)
        else:
            policy_manifest = {}

    X_full, z_full = load_sdss_dataset()
    full_n = len(z_full)
    runtime_device = _resolve_runtime_device(args.device)
    sample_sizes = _parse_sample_sizes(args.sample_sizes, full_n)

    if args.methods:
        base_methods = _parse_methods_arg(args.methods)
    else:
        base_methods = list(DEFAULT_METHOD_ORDER)
    if args.exclude:
        excluded = set(_parse_methods_arg(args.exclude))
        base_methods = [m for m in base_methods if m not in excluded]

    unknown = [m for m in base_methods if m not in DEFAULT_METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    if not base_methods:
        raise ValueError("No methods selected")

    print(f"Loaded SDSS: n={full_n:,}, d={X_full.shape[1]}")
    print(f"Policy runtime device: {runtime_device}")
    print(f"Sample sizes: {', '.join(f'{n:,}' for n in sample_sizes)}")
    print(f"Repetitions per sample size: {args.n_reps}")

    for n_total in sample_sizes:
        dataset_name = f"SDSS-{n_total}"
        cache_file = cache_dir / f"{dataset_name}.npz"
        X_sub, z_sub = _subsample_from_full(X_full, z_full, n_total, args.subsample_seed)

        use_policy = (not args.methods) and (not args.all_methods_at_all_sizes)
        selected_methods, policy_skips = _select_methods(
            base_methods, n_total, runtime_device, use_policy=use_policy
        )

        print("\n" + "=" * 72)
        print(f"{dataset_name}  (n={len(z_sub):,}, d={X_sub.shape[1]})")
        print("=" * 72)
        print("Selected methods:")
        for method in selected_methods:
            print(f"  - {method}")
        if policy_skips:
            print("Policy skips:")
            for method, reason in policy_skips.items():
                print(f"  - {method}: {reason}")
        if not selected_methods:
            print("  [skip] No methods selected for this sample size")
            policy_manifest[dataset_name] = {
                "n_total": n_total,
                "runtime_device": runtime_device,
                "n_reps_requested": args.n_reps,
                "selected_methods": [],
                "policy_skips": policy_skips,
                "runtime_failures": {},
            }
            _write_json(policy_file, policy_manifest)
            continue

        per_rep_results = []
        runtime_failures = {}
        representative_cache_saved = False

        for rep in range(args.n_reps):
            rep_partial = partial_root / f"rep{rep}"
            rep_partial.mkdir(parents=True, exist_ok=True)
            rep_seed = args.random_state + rep
            rep_key = f"rep{rep}"
            runtime_failures[rep_key] = {}
            last_X_te = None
            last_z_te = None
            true_cde = None
            true_zgrid = None

            print(f"\n  ── rep {rep + 1}/{args.n_reps} (seed={rep_seed}) ──")
            for method in selected_methods:
                print(f"\nRunning {dataset_name}: {method}")
                try:
                    results, _, _, X_te, z_te, true_cde, true_zgrid = run_experiment(
                        X_sub,
                        z_sub,
                        dataset_name,
                        device=args.device,
                        n_grid=args.n_grid,
                        true_density_fn=None,
                        partial_dir=rep_partial,
                        force=args.force,
                        random_state=rep_seed,
                        methods=[method],
                    )
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    runtime_failures[rep_key][method] = msg
                    print(f"  [failed] {msg}")
                    continue

                if method in results:
                    last_X_te = X_te
                    last_z_te = z_te

            rep_metrics = _load_partial_metrics(rep_partial, dataset_name)
            per_rep_results.append(rep_metrics)

            if (
                not representative_cache_saved
                and last_X_te is not None
                and rep_metrics
            ):
                cached_methods = [m for m in selected_methods if m in rep_metrics]
                cdes, zgrids = _load_cached_arrays(rep_partial, dataset_name, cached_methods)
                if cdes:
                    save_cache(
                        cache_file,
                        cdes,
                        zgrids,
                        last_X_te,
                        last_z_te,
                        true_cde,
                        true_zgrid,
                        len(z_sub),
                    )
                    representative_cache_saved = True

        agg_results = _aggregate_reps(per_rep_results)
        if agg_results:
            all_results[dataset_name] = agg_results
            if cache_file.exists():
                cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_cached = load_cache(cache_file)
                all_data[dataset_name] = {
                    "cdes": cdes,
                    "zgrids": zgrids,
                    "X_test": X_te,
                    "z_test": z_te,
                    "true_cde": true_cde,
                    "true_zgrid": true_zgrid,
                    "n_total": n_cached,
                }

        policy_manifest[dataset_name] = {
            "n_total": n_total,
            "runtime_device": runtime_device,
            "n_reps_requested": args.n_reps,
            "rep_seeds": [args.random_state + rep for rep in range(args.n_reps)],
            "selected_methods": selected_methods,
            "policy_skips": policy_skips,
            "runtime_failures": runtime_failures,
        }
        _write_json(results_file, all_results)
        _write_json(policy_file, policy_manifest)
        print(f"\nUpdated {results_file}")

    if not all_results:
        print("No SDSS scaling results were completed.")
        return

    se_caption = f"mean +/- SE across {args.n_reps} repetitions"
    print_summary(all_results, se_caption=se_caption)

    print("\nGenerating SDSS scaling plots...")
    save_html_table(all_results, output_dir, se_caption=se_caption)
    plot_performance_vs_n(all_results, output_dir, all_data=all_data)
    plot_performance_vs_n_foundational(all_results, output_dir, all_data=all_data)

    print(f"\nSaved metrics to {results_file}")
    print(f"Saved method schedule to {policy_file}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()

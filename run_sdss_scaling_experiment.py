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
  .venv/bin/python run_sdss_scaling_experiment.py --sample-sizes 1000,10000,full
  .venv/bin/python run_sdss_scaling_experiment.py --methods LinearGauss-Homo,Flow-Spline
  .venv/bin/python run_sdss_scaling_experiment.py --plot-partial-results-only
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
from utils import load_cache, save_cache, print_summary, aggregate_reps
from visualization import (
    plot_performance_vs_n,
    plot_performance_vs_n_foundational,
    plot_sdss_rankings_by_n,
    plot_sdss_raw_metrics_by_n,
    save_html_table,
    save_latex_table,
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
    "MDN",
    "Flow-Spline",
    "BART-Homo",
    "BART-Hetero",
    "FlexCode-RF",
    "CatMLP",
]

DEFAULT_SAMPLE_SIZE_SPEC = "1000,10000,50000,100000,250000,500000,full"
TABPFN_METHODS = {"TabPFN-Native", "TabPFN-2.5", "RealTabPFN-2.5"}


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


def _sdss_run_tag(random_state, subsample_seed):
    return f"split{int(random_state)}_sub{int(subsample_seed)}"


def _normalize_method_mapping(mapping):
    normalized = {}
    for name, value in mapping.items():
        canonical = _canonical_method_name(name)
        if canonical in normalized and name != canonical:
            continue
        normalized[canonical] = value
    return normalized


def _normalize_results_payload(payload):
    return {
        dataset_name: _normalize_method_mapping(metrics)
        for dataset_name, metrics in payload.items()
    }


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

    if method == "BART-Hetero" and n_total >= 100_000:
        return "heteroskedastic BART is conservatively capped below n=100,000 for SDSS scaling runs"

    if method == "FlexCode-RF" and n_total > 500_000:
        return "5-fold CV over many random-forest basis regressions is conservatively capped at n=500,000"

    if method in TABPFN_METHODS:
        limit = 1_000 if runtime_device == "cpu" else 50_000
        if n_total > limit:
            if runtime_device == "cpu":
                return "the installed TabPFN package blocks CPU runs above n=1,000"
            return "the installed TabPFN v2.5 configuration supports up to n=50,000"

    if method == "TabICL-Quantiles":
        limit = 10_000 if runtime_device == "cpu" else 49_999
        if n_total > limit:
            return (
                f"native TabICL quantile inference is conservatively capped at n={limit:,} "
                f"for {runtime_device.upper()} scaling runs"
            )

    if method == "MDN":
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


def _load_existing_output(output_dir, run_tag=None):
    all_results = {}

    json_file = output_dir / "results.json"
    policy_file = output_dir / "method_policy.json"
    if run_tag is not None and policy_file.exists():
        try:
            with open(policy_file) as f:
                policy_payload = json.load(f)
            existing_run_tag = policy_payload.get("_run_tag")
            if existing_run_tag != run_tag:
                return {}, {}
        except Exception:
            return {}, {}

    if json_file.exists():
        with open(json_file) as f:
            all_results = _normalize_results_payload(json.load(f))

    all_data = _load_cached_scaling_payloads(output_dir, all_results, run_tag=run_tag)
    return all_results, all_data


def _load_cached_scaling_payloads(output_dir, all_results, run_tag=None):
    all_data = {}
    cache_dir = output_dir / "cache"
    if run_tag is not None:
        cache_dir = cache_dir / run_tag
    for dataset_name in all_results:
        cache_file = cache_dir / f"{dataset_name}.npz"
        if not cache_file.exists():
            continue
        try:
            cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_total = load_cache(cache_file)
            all_data[dataset_name] = {
                "cdes": _normalize_method_mapping(cdes),
                "zgrids": _normalize_method_mapping(zgrids),
                "X_test": X_te,
                "z_test": z_te,
                "true_cde": true_cde,
                "true_zgrid": true_zgrid,
                "n_total": n_total,
            }
        except Exception as exc:
            print(f"  [warn] could not load cache for {dataset_name}: {exc}")
    return all_data


def _load_partial_scaling_output(output_dir, run_tag=None):
    partial_root = output_dir / "cache" / "partial"
    if run_tag is not None:
        partial_root = partial_root / run_tag
    if not partial_root.exists():
        return {}, {}

    rep_dirs = sorted(rep_dir for rep_dir in partial_root.glob("rep*") if rep_dir.is_dir())
    dataset_reps = {}
    for rep_dir in rep_dirs:
        for metrics_file in sorted(rep_dir.glob("SDSS-*_metrics.json")):
            with open(metrics_file) as f:
                rep_metrics = json.load(f)
            dataset_name = metrics_file.stem.replace("_metrics", "")
            dataset_reps.setdefault(dataset_name, []).append(rep_metrics)

    all_results = {}
    for dataset_name in sorted(dataset_reps):
        agg_results = aggregate_reps(dataset_reps[dataset_name])
        if agg_results:
            all_results[dataset_name] = agg_results

    all_data = _load_cached_scaling_payloads(output_dir, all_results, run_tag=run_tag)
    return all_results, all_data


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _render_sdss_scaling_outputs(all_results, all_data, output_dir, se_caption):
    if not all_results:
        print("[skip] No SDSS scaling results available for plotting")
        return False

    output_dir = Path(output_dir)
    stale_patterns = [
        "rankings_real_*_n*.pdf",
        "rankings_real_*_n*.png",
        "raw_real_*_n*.pdf",
        "raw_real_*_n*.png",
        "rankings_real_sdss_combined.pdf",
        "rankings_real_sdss_combined.png",
        "raw_real_sdss_combined.pdf",
        "raw_real_sdss_combined.png",
    ]
    for pattern in stale_patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)

    save_html_table(all_results, output_dir, se_caption=se_caption)
    save_latex_table(all_results, output_dir, se_caption=se_caption)
    plot_performance_vs_n(all_results, output_dir, all_data=all_data)
    plot_performance_vs_n_foundational(all_results, output_dir, all_data=all_data)
    plot_sdss_rankings_by_n(all_results, output_dir, all_data=all_data)
    plot_sdss_raw_metrics_by_n(all_results, output_dir, all_data=all_data)
    return True


def generate_sdss_scaling_partial_plots(output_dir="results_real/sdss_scaling",
                                        run_tag=None):
    """Aggregate current partial SDSS-scaling results and regenerate plots/tables."""
    output_dir = Path(output_dir)
    results_file = output_dir / "results.json"
    all_results, all_data = _load_partial_scaling_output(output_dir, run_tag=run_tag)
    if not all_results:
        partial_root = output_dir / "cache" / "partial"
        if run_tag is not None:
            partial_root = partial_root / run_tag
        print(f"[skip] No partial SDSS scaling results found in {partial_root}")
        return False

    _write_json(results_file, all_results)
    print(f"saved {results_file}")
    return _render_sdss_scaling_outputs(
        all_results,
        all_data,
        output_dir,
        se_caption="mean +/- SE across completed repetitions",
    )


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
    parser.add_argument("--plot-partial-results-only", action="store_true",
                        help="Aggregate current partial SDSS scaling outputs, "
                             "regenerate results.json and plots, then exit")
    parser.add_argument("--refresh-plots-after-each-rep", action="store_true",
                        help="Regenerate the SDSS scaling table/plots after each "
                             "completed repetition")
    args = parser.parse_args()
    run_tag = _sdss_run_tag(args.random_state, args.subsample_seed)

    if args.list_methods:
        print("\n".join(DEFAULT_METHOD_ORDER))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_payload_dir = cache_dir / run_tag
    cache_payload_dir.mkdir(parents=True, exist_ok=True)
    partial_root = cache_dir / "partial" / run_tag
    partial_root.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    policy_file = output_dir / "method_policy.json"

    if args.plot_partial_results_only:
        generate_sdss_scaling_partial_plots(output_dir=output_dir, run_tag=run_tag)
        return

    if args.force:
        all_results = {}
        all_data = {}
        policy_manifest = {"_run_tag": run_tag}
    else:
        all_results, all_data = _load_existing_output(output_dir, run_tag=run_tag)
        if policy_file.exists():
            with open(policy_file) as f:
                policy_manifest = json.load(f)
            existing_run_tag = policy_manifest.get("_run_tag")
            if existing_run_tag != run_tag:
                print(f"[cache] ignoring existing SDSS scaling metadata for run tag "
                      f"{existing_run_tag}; current run tag is {run_tag}")
                policy_manifest = {"_run_tag": run_tag}
        else:
            policy_manifest = {"_run_tag": run_tag}
    policy_manifest["_run_tag"] = run_tag

    X_full, z_full = load_sdss_dataset()
    full_n = len(z_full)
    runtime_device = _resolve_runtime_device(args.device)
    sample_sizes = _parse_sample_sizes(args.sample_sizes, full_n)

    if args.methods:
        base_methods = _canonicalize_methods(_parse_methods_arg(args.methods))
    else:
        base_methods = list(DEFAULT_METHOD_ORDER)
    if args.exclude:
        excluded = set(_canonicalize_methods(_parse_methods_arg(args.exclude)))
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
    print(f"Run tag: {run_tag}")

    for n_total in sample_sizes:
        dataset_name = f"SDSS-{n_total}"
        cache_file = cache_payload_dir / f"{dataset_name}.npz"

        use_policy = (not args.methods) and (not args.all_methods_at_all_sizes)
        selected_methods, policy_skips = _select_methods(
            base_methods, n_total, runtime_device, use_policy=use_policy
        )

        print("\n" + "=" * 72)
        print(f"{dataset_name}  (n={min(n_total, full_n):,}, d={X_full.shape[1]})")
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
            # Use constant subsample seed so all reps evaluate the same
            # data subset; only the train/test split varies (via rep_seed).
            X_sub, z_sub = _subsample_from_full(
                X_full, z_full, n_total, args.subsample_seed
            )
            rep_key = f"rep{rep}"
            runtime_failures[rep_key] = {}
            last_X_te = None
            last_z_te = None
            true_cde = None
            true_zgrid = None

            print(f"\n  ── rep {rep + 1}/{args.n_reps} (seed={rep_seed}, "
                  f"subsample_seed={args.subsample_seed}) ──")
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
            if args.refresh_plots_after_each_rep:
                generate_sdss_scaling_partial_plots(output_dir=output_dir, run_tag=run_tag)

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

        agg_results = aggregate_reps(per_rep_results)
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
    _render_sdss_scaling_outputs(all_results, all_data, output_dir, se_caption)

    print(f"\nSaved metrics to {results_file}")
    print(f"Saved method schedule to {policy_file}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()

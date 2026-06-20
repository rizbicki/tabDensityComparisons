"""
Post-hoc PIT recalibration analysis (cache-only, no refitting).
=================================================================

Applies cross-fit PIT recalibration (``evaluation.recalibration``) to every
method's cached test-set predictions, recomputes all six metrics before and
after recalibration, and aggregates the improvement across datasets.

The cached ``.npz`` files store a single test split per (dataset, n) -- the same
split used for the paper's diagnostic plots -- so "before" numbers here are
single-split and will differ slightly from the multi-rep numbers in
``results.json``. Both "before" and "after" are computed on the *identical*
split, so the reported delta is a clean apples-to-apples recalibration effect.

USAGE:
  python run_recalibration_analysis.py [--cache-dir results_real/cache]
      [--out-dir results_real/recalibration] [--n 5000 10000 20000]
      [--datasets Abalone Bank8FM] [--exclude TabPFN-Native]
      [--n-splits 5] [--min-calib 40] [--limit N]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_cache
from evaluation import compute_all_metrics, crossfit_recalibrate

FOUNDATION = {"TabPFN-2.5", "RealTabPFN-2.5", "TabICL-Quantiles", "TabPFN-Native"}

# (column, direction) -- 'lower'/'higher' is better; coverage handled specially.
METRICS = ["CDE_loss", "log_lik", "CRPS", "PIT_KS", "coverage_90", "interval_width"]
LOWER_BETTER = {"CDE_loss", "CRPS", "PIT_KS", "coverage_abs_err"}
HIGHER_BETTER = {"log_lik"}
COVERAGE_TARGET = 0.90


def _parse_name(stem):
    m = re.match(r"(.+)-(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, None)


def _group(method):
    return "foundation" if method in FOUNDATION else "baseline"


def _improved(metric, before, after):
    """Signed improvement (positive = better) and a boolean improved flag."""
    if metric in LOWER_BETTER:
        return before - after, after < before
    if metric in HIGHER_BETTER:
        return after - before, after > before
    return np.nan, None


def analyze_cell(dataset, n, cdes, zgrids, z_te, methods, args):
    """Recalibrate every method in one (dataset, n) cell; return long records."""
    records = []
    for method in methods:
        cde, zg = cdes[method], zgrids[method]
        before = compute_all_metrics(cde, zg, z_te)
        cde_r, info = crossfit_recalibrate(
            cde, zg, z_te,
            n_splits=args.n_splits, min_calib=args.min_calib, seed=args.seed,
        )
        after = compute_all_metrics(cde_r, zg, z_te) if cde_r is not None else None

        row = dict(dataset=dataset, n=n, method=method, group=_group(method),
                   n_test=info["n_test"], status=info["status"])
        for met in METRICS:
            b = before[met]
            a = after[met] if after is not None else np.nan
            row[f"{met}__before"] = b
            row[f"{met}__after"] = a
        # coverage as absolute error from the 0.90 target
        row["coverage_abs_err__before"] = abs(before["coverage_90"] - COVERAGE_TARGET)
        row["coverage_abs_err__after"] = (
            abs(after["coverage_90"] - COVERAGE_TARGET) if after is not None else np.nan
        )
        records.append(row)
    return records


def build_summary(df):
    """Aggregate deltas across datasets, per (n, method, metric)."""
    ok = df[df["status"] == "ok"].copy()
    metrics = METRICS + ["coverage_abs_err"]
    long_rows = []
    for met in metrics:
        b, a = ok[f"{met}__before"], ok[f"{met}__after"]
        delta = a - b
        signed, improved = zip(*[_improved(met, bi, ai) for bi, ai in zip(b, a)])
        tmp = pd.DataFrame({
            "n": ok["n"], "method": ok["method"], "group": ok["group"],
            "metric": met, "before": b.values, "after": a.values,
            "delta": delta.values, "signed_improvement": signed, "improved": improved,
        })
        long_rows.append(tmp)
    tidy = pd.concat(long_rows, ignore_index=True)

    summary = (
        tidy.groupby(["metric", "n", "method", "group"], observed=True)
        .agg(n_datasets=("delta", "size"),
             mean_before=("before", "mean"),
             mean_after=("after", "mean"),
             mean_delta=("delta", "mean"),
             median_delta=("delta", "median"),
             win_rate=("improved", lambda s: np.mean([x for x in s if x is not None])
                       if any(x is not None for x in s) else np.nan))
        .reset_index()
        .sort_values(["metric", "n", "group", "method"])
    )
    return tidy, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="results_real/cache")
    ap.add_argument("--out-dir", default="results_real/recalibration")
    ap.add_argument("--n", type=int, nargs="*", default=None,
                    help="restrict to these training sizes")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=["TabPFN-Native"],
                    help="methods to drop (default: TabPFN-Native, per paper)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--min-calib", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N cache files (debugging)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cache_dir.glob("*.npz"))
    cells = []
    for f in files:
        ds, n = _parse_name(f.stem)
        if n is None:
            continue
        if args.n and n not in args.n:
            continue
        if args.datasets and ds not in args.datasets:
            continue
        cells.append((f, ds, n))
    if args.limit:
        cells = cells[: args.limit]

    print(f"Processing {len(cells)} (dataset, n) cells from {cache_dir}")
    all_records = []
    for i, (f, ds, n) in enumerate(cells, 1):
        cdes, zgrids, X_te, z_te, *_ = load_cache(f)
        methods = [m for m in cdes if m not in set(args.exclude)]
        recs = analyze_cell(ds, n, cdes, zgrids, z_te, methods, args)
        all_records.extend(recs)
        print(f"  [{i}/{len(cells)}] {ds}-{n}  n_test={recs[0]['n_test']:>5}  "
              f"methods={len(methods)}  status={recs[0]['status']}")

    df = pd.DataFrame(all_records)
    per_cell_path = out_dir / "recal_per_cell.csv"
    df.to_csv(per_cell_path, index=False)

    tidy, summary = build_summary(df)
    tidy_path = out_dir / "recal_tidy.csv"
    summary_path = out_dir / "recal_summary_by_method_n.csv"
    tidy.to_csv(tidy_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nWrote:\n  {per_cell_path}\n  {tidy_path}\n  {summary_path}")
    _print_headline(summary)


def _print_headline(summary):
    """Print a compact headline: mean delta by group for key metrics."""
    print("\n=== Headline: mean recalibration delta by method group ===")
    print("(PIT_KS / coverage_abs_err: negative = better; "
          "log_lik: positive = better; CDE_loss/CRPS: negative = better)\n")
    for met in ["PIT_KS", "coverage_abs_err", "CRPS", "CDE_loss", "log_lik"]:
        sub = summary[summary["metric"] == met]
        if sub.empty:
            continue
        piv = (sub.groupby(["n", "group"], observed=True)["mean_delta"]
               .mean().unstack("group"))
        print(f"--- {met} (mean delta vs n) ---")
        print(piv.round(4).to_string())
        print()


if __name__ == "__main__":
    main()

"""
I/O utilities: caching, formatting, summary printing.
"""

import numpy as np


def save_cache(cache_file, cdes, zgrids, X_te, z_te, true_cde, true_zgrid,
               n_total):
    """Save per-dataset run data to a .npz cache file."""
    arrays = {
        'methods': np.array(list(cdes.keys())),
        'X_te': X_te,
        'z_te': z_te,
        'n_total': np.array(n_total),
        'true_cde':   true_cde   if true_cde   is not None else np.array([]),
        'true_zgrid': true_zgrid if true_zgrid is not None else np.array([]),
    }
    for m in cdes:
        arrays[f'cde_{m}'] = cdes[m]
        arrays[f'zgrid_{m}'] = zgrids[m]
    np.savez(cache_file, **arrays)


def load_cache(cache_file):
    """Load per-dataset run data from a .npz cache file."""
    data = np.load(cache_file, allow_pickle=True)
    methods = data['methods'].tolist()
    cdes   = {m: data[f'cde_{m}']   for m in methods}
    zgrids = {m: data[f'zgrid_{m}'] for m in methods}
    X_te   = data['X_te']
    z_te   = data['z_te']
    n_total = int(data['n_total'])
    tc = data['true_cde']
    tg = data['true_zgrid']
    true_cde   = tc if tc.size > 0 else None
    true_zgrid = tg if tg.size > 0 else None
    return cdes, zgrids, X_te, z_te, true_cde, true_zgrid, n_total


def fmt_metric(val, se, fmt='.4f'):
    """Format metric value with standard error in parentheses."""
    if se is None:
        return f"{val:{fmt}}"
    return f"{val:{fmt}} ({se:{fmt}})"


def print_summary(all_results):
    methods = sorted(set(m for ds in all_results.values() for m in ds.keys()))
    datasets = list(all_results.keys())

    W = 24
    C = 18

    print(f"\n{'='*130}")
    print("FULL RESULTS TABLE  (mean +/- SE over test samples)")
    print(f"{'='*130}")

    for ds, res in all_results.items():
        print(f"\n--- {ds} ---")
        header = (f"{'Method':<{W}} {'CDE loss':<{C}} {'Log-lik':<{C}} "
                  f"{'CRPS':<{C}} {'PIT KS':<14} {'90% Cov':<{C}} "
                  f"{'Width':<{C}} {'Basis':>5} {'Time':>7}")
        print(header)
        print("-" * len(header))
        for m in methods:
            if m in res:
                r = res[m]
                basis = str(r['n_basis']) if r.get('n_basis') else '-'
                print(
                    f"{m:<{W}} "
                    f"{fmt_metric(r['CDE_loss'],  r.get('CDE_loss_se'),  '.4f'):<{C}} "
                    f"{fmt_metric(r['log_lik'],   r.get('log_lik_se'),   '.3f'):<{C}} "
                    f"{fmt_metric(r['CRPS'],      r.get('CRPS_se'),      '.4f'):<{C}} "
                    f"{fmt_metric(r['PIT_KS'],    None,                  '.3f'):<14} "
                    f"{fmt_metric(r['coverage_90'], r.get('coverage_90_se'), '.3f'):<{C}} "
                    f"{fmt_metric(r['interval_width'], r.get('interval_width_se'), '.3f'):<{C}} "
                    f"{basis:>5} {r['fit_time']:>6.1f}s"
                )

    # Rankings
    metrics_dirs = {'CDE_loss': 'lower', 'log_lik': 'higher',
                    'CRPS': 'lower', 'PIT_KS': 'lower'}

    all_ranks = {m: [] for m in methods}
    for metric, direction in metrics_dirs.items():
        for ds in datasets:
            vals, avail = [], []
            for m in methods:
                if m in all_results[ds]:
                    vals.append(all_results[ds][m][metric])
                    avail.append(m)
            vals = np.array(vals)
            ranks = (np.argsort(np.argsort(vals)) + 1 if direction == 'lower'
                     else np.argsort(np.argsort(-vals)) + 1)
            for m, r in zip(avail, ranks):
                all_ranks[m].append(r)

    print(f"\n{'='*50}")
    print("OVERALL AVERAGE RANKINGS")
    print(f"{'='*50}")
    sorted_m = sorted(all_ranks,
                      key=lambda m: np.mean(all_ranks[m]) if all_ranks[m] else 99)
    for m in sorted_m:
        if all_ranks[m]:
            print(f"  {m:<22} avg rank = {np.mean(all_ranks[m]):.2f}")

"""
Visualization: ranking heatmaps, density comparisons, PIT histograms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.metrics import eval_pit, eval_pit_ks

# Fixed per-method visual style
METHOD_STYLES = {
    'FlexCode-RF':         {'color': '#984ea3', 'ls': '-',   'lw': 2.0, 'zorder': 4},
    'TabPFN-Native':       {'color': '#ff7f00', 'ls': '--',  'lw': 1.8, 'zorder': 3},
    'TabICL-Quantiles':    {'color': '#a65628', 'ls': '--',  'lw': 1.8, 'zorder': 3},
    'Quantile-Tree':       {'color': '#888888', 'ls': ':',   'lw': 1.8, 'zorder': 3},
    'Quantile-Linear':     {'color': '#b3b3b3', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LinearGauss-Homo':    {'color': '#66c2a5', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LinearGauss-Hetero':  {'color': '#fc8d62', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'Student-t':           {'color': '#4daf4a', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LogNormal-Homo':      {'color': '#d95f02', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LogNormal-Hetero':    {'color': '#7570b3', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'MDN-2mix':            {'color': '#8da0cb', 'ls': '--',  'lw': 1.5, 'zorder': 2},
    'Gamma-GLM':           {'color': '#e78ac3', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LinGauss-Homo-Ridge':    {'color': '#66c2a5', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'LinGauss-Hetero-Ridge':  {'color': '#fc8d62', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'Student-t-Ridge':        {'color': '#4daf4a', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'LogNormal-Homo-Ridge':   {'color': '#d95f02', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'LogNormal-Hetero-Ridge': {'color': '#7570b3', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'Gamma-GLM-Ridge':        {'color': '#e78ac3', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'BART-Homo':              {'color': '#1b9e77', 'ls': '-',   'lw': 1.8, 'zorder': 3},
    'BART-Hetero':            {'color': '#d62728', 'ls': '--',  'lw': 1.8, 'zorder': 3},
}
_FALLBACK_COLORS = ['#8dd3c7', '#bebada', '#fb8072', '#80b1d3', '#fdb462']


def _method_style(method, fallback_idx=0):
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    return {'color': _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)],
            'ls': '-', 'lw': 1.5, 'zorder': 3}


def plot_density_comparison(all_data, output_dir):
    """Plot example density estimates side by side, with true density when available."""
    datasets_to_show = list(all_data.keys())[:4]
    n_ds = len(datasets_to_show)
    n_examples = 3

    sample_methods = list(list(all_data.values())[0]['cdes'].keys())

    fig, axes = plt.subplots(n_ds, n_examples, figsize=(5 * n_examples, 4 * n_ds))
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    for i, ds in enumerate(datasets_to_show):
        d = all_data[ds]
        z_test = d['z_test']
        sorted_idx = np.argsort(z_test)
        examples = [sorted_idx[len(sorted_idx)//6],
                    sorted_idx[len(sorted_idx)//2],
                    sorted_idx[5*len(sorted_idx)//6]]

        for j, idx in enumerate(examples):
            ax = axes[i, j]

            if d.get('true_cde') is not None:
                ax.plot(d['true_zgrid'], d['true_cde'][idx],
                        label='True density' if (i == 0 and j == 0) else None,
                        color='black', linewidth=2.5, linestyle='--', zorder=5)

            for k, method in enumerate(sample_methods):
                sty = _method_style(method, k)
                cde = d['cdes'][method]
                zg = d['zgrids'][method]
                ax.plot(zg, cde[idx],
                        label=method if (i == 0 and j == 0) else None,
                        color=sty['color'], linestyle=sty['ls'],
                        linewidth=sty['lw'], alpha=0.85, zorder=sty['zorder'])

            ax.axvline(z_test[idx], color='gray', ls=':', lw=1.2, alpha=0.6)
            if j == 0:
                n_label = d.get('n_total', '')
                ds_label = f"{ds}\n(n={n_label})" if n_label else ds
                ax.set_ylabel(ds_label, fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.set_title(f'Test #{idx}', fontsize=8)

    axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.8)
    plt.suptitle('Estimated Conditional Densities (-- black = true)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'density_examples.png', dpi=150, bbox_inches='tight')
    plt.close()


METRICS_INFO = [
    ('CDE_loss',  'CDE Loss',  'lower'),
    ('log_lik',   'Log-Lik',   'higher'),
    ('CRPS',      'CRPS',      'lower'),
    ('PIT_KS',    'PIT KS',    'lower'),
    ('fit_time',  'Fit Time',  'lower'),
]


def _ds_labels(datasets, all_data):
    labels = []
    for ds in datasets:
        short = ds[:12]
        n_total = all_data[ds].get('n_total', '') if all_data and ds in all_data else ''
        n_str = f"\n(n={n_total})" if n_total else ''
        labels.append(f"{short}{n_str}")
    return labels


def _group_by_n(all_results):
    """Group datasets by sample size suffix. Returns {n: {ds: results}}."""
    from collections import defaultdict
    size_groups = defaultdict(dict)
    for ds in all_results:
        _, n = _parse_base_and_n(ds)
        if n is not None:
            size_groups[n][ds] = all_results[ds]
    return dict(size_groups)


def plot_rankings_by_n(all_results, output_dir, all_data=None):
    """Ranking bar plots for each sample size separately."""
    size_groups = _group_by_n(all_results)
    if not size_groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys()))
    cmap2 = plt.cm.tab10
    colors = {m: cmap2(i) for i, m in enumerate(methods)}

    for n_size in sorted(size_groups):
        sub_results = size_groups[n_size]
        datasets = list(sub_results.keys())
        n_methods = len(methods)
        n_ds = len(datasets)
        ds_labels = _ds_labels(datasets, all_data)

        for metric, label, direction in METRICS_INFO:
            # Build rank matrix
            matrix = np.full((n_methods, n_ds), np.nan)
            for di, ds in enumerate(datasets):
                vals, avail = [], []
                for m in methods:
                    if m in sub_results[ds]:
                        vals.append(sub_results[ds][m][metric])
                        avail.append(m)
                vals = np.array(vals)
                ranks = (np.argsort(np.argsort(vals)) + 1 if direction == 'lower'
                         else np.argsort(np.argsort(-vals)) + 1)
                for m, r in zip(avail, ranks):
                    matrix[methods.index(m), di] = r

            fig, (ax1, ax2) = plt.subplots(
                1, 2,
                figsize=(max(10, n_ds * 0.9), max(4, n_methods * 0.6)),
                gridspec_kw={'width_ratios': [3, 1]})

            im = ax1.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                            vmin=1, vmax=n_methods)
            ax1.set_yticks(range(n_methods))
            ax1.set_yticklabels(methods, fontsize=9)
            ax1.set_xticks(range(n_ds))
            ax1.set_xticklabels(ds_labels, fontsize=7, rotation=45, ha='right')

            for i in range(n_methods):
                for j in range(n_ds):
                    if not np.isnan(matrix[i, j]):
                        ax1.text(j, i, f'{int(matrix[i,j])}', ha='center',
                                 va='center', fontsize=8, fontweight='bold',
                                 color='white' if matrix[i,j] > n_methods * 0.6
                                 else 'black')

            plt.colorbar(im, ax=ax1, label='Rank', shrink=0.8)
            ax1.set_title(f'{label} — Rankings (n={n_size})',
                          fontsize=11, fontweight='bold')

            # Average rank bar
            avg_ranks = {}
            for mi, m in enumerate(methods):
                r = matrix[mi][~np.isnan(matrix[mi])]
                avg_ranks[m] = np.mean(r) if len(r) > 0 else 99

            sorted_m = sorted(avg_ranks, key=avg_ranks.get)
            y_pos = range(len(sorted_m))
            ax2.barh(y_pos, [avg_ranks[m] for m in sorted_m],
                     color=[colors[m] for m in sorted_m], alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(sorted_m, fontsize=9)
            ax2.set_xlabel('Avg Rank', fontsize=10)
            ax2.set_title(f'Overall (n={n_size})', fontsize=11, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()

            plt.tight_layout()
            fname = f"rankings_{metric.lower()}_n{n_size}.png"
            plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  saved {fname}")


def plot_raw_metrics_by_n(all_results, output_dir, all_data=None):
    """Raw-value heatmaps per sample size, with per-column normalization."""
    size_groups = _group_by_n(all_results)
    if not size_groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys()))

    for n_size in sorted(size_groups):
        sub_results = size_groups[n_size]
        datasets = list(sub_results.keys())
        n_methods = len(methods)
        n_ds = len(datasets)
        ds_labels = _ds_labels(datasets, all_data)

        for metric, label, direction in METRICS_INFO:
            matrix = np.full((n_methods, n_ds), np.nan)
            for di, ds in enumerate(datasets):
                for mi, m in enumerate(methods):
                    if m in sub_results[ds] and metric in sub_results[ds][m]:
                        matrix[mi, di] = sub_results[ds][m][metric]

            cmap = 'RdYlGn_r' if direction == 'lower' else 'RdYlGn'

            # Normalize per dataset (column)
            norm_matrix = np.full_like(matrix, np.nan)
            for j in range(n_ds):
                col = matrix[:, j]
                col_valid = col[~np.isnan(col)]
                if len(col_valid) == 0:
                    continue
                cmin, cmax = np.nanmin(col), np.nanmax(col)
                rng = cmax - cmin
                if rng < 1e-10:
                    norm_matrix[:, j] = 0.5
                else:
                    norm_matrix[:, j] = (col - cmin) / rng

            fig, ax = plt.subplots(figsize=(max(10, n_ds * 0.9),
                                            max(4, n_methods * 0.6)))

            im = ax.imshow(norm_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            ax.set_yticks(range(n_methods))
            ax.set_yticklabels(methods, fontsize=9)
            ax.set_xticks(range(n_ds))
            ax.set_xticklabels(ds_labels, fontsize=7, rotation=45, ha='right')

            for i in range(n_methods):
                for j in range(n_ds):
                    if not np.isnan(matrix[i, j]):
                        val = matrix[i, j]
                        txt = f'{val:.3f}' if abs(val) < 100 else f'{val:.1f}'
                        nv = norm_matrix[i, j]
                        dark = nv < 0.4 if direction == 'higher' else nv > 0.6
                        ax.text(j, i, txt, ha='center', va='center',
                                fontsize=7, fontweight='bold',
                                color='white' if dark else 'black')

            ax.set_title(f'{label} (n={n_size}, colors normalized per dataset)',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            fname = f"raw_{metric.lower()}_n{n_size}.png"
            plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  saved {fname}")


def plot_pit_histograms(all_data, output_dir):
    """PIT calibration histograms."""
    datasets_to_show = list(all_data.keys())[:4]
    sample_methods = list(list(all_data.values())[0]['cdes'].keys())

    n_ds = len(datasets_to_show)
    n_m = len(sample_methods)

    fig, axes = plt.subplots(n_ds, n_m, figsize=(3.5 * n_m, 3 * n_ds))
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    for i, ds in enumerate(datasets_to_show):
        d = all_data[ds]
        for j, m in enumerate(sample_methods):
            ax = axes[i, j]
            if m in d['cdes']:
                pit = eval_pit(d['cdes'][m], d['zgrids'][m], d['z_test'])
                ks = eval_pit_ks(pit)
                ax.hist(pit, bins=20, density=True, alpha=0.7,
                        color='steelblue', edgecolor='white')
                ax.axhline(1.0, color='red', ls='--', lw=1.5)
                ax.text(0.05, 0.95, f'KS={ks:.3f}', transform=ax.transAxes,
                        fontsize=8, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            if i == 0:
                ax.set_title(m, fontsize=8, fontweight='bold')
            if j == 0:
                n_label = all_data[ds].get('n_total', '')
                ds_label = f"{ds}\n(n={n_label})" if n_label else ds
                ax.set_ylabel(ds_label, fontsize=8, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=6)

    plt.suptitle('PIT Calibration (uniform = ideal)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pit_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_html_table(all_results, output_dir):
    """Save a styled HTML results table with mean +/- SE and best values highlighted."""

    METRICS = [
        ('CDE_loss',       'CDE loss',    'lower'),
        ('log_lik',        'Log-lik',     'higher'),
        ('CRPS',           'CRPS',        'lower'),
        ('PIT_KS',         'PIT KS',      'lower'),
        ('coverage_90',    '90% Cov',     None),
        ('interval_width', 'Width',       'lower'),
    ]

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys()))

    css = """
    <style>
      body { font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; background: #f8f9fa; }
      h1 { color: #333; }
      h2 { color: #555; margin-top: 2em; border-bottom: 2px solid #ccc; padding-bottom: 4px; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 2em;
              box-shadow: 0 1px 4px rgba(0,0,0,.12); background: white; }
      th { background: #2c3e50; color: white; padding: 8px 12px; text-align: center;
           font-size: 13px; white-space: nowrap; }
      td { padding: 7px 12px; text-align: right; font-size: 13px;
           border-bottom: 1px solid #eee; white-space: nowrap; }
      td.method { text-align: left; font-weight: 600; color: #2c3e50; }
      tr:hover td { background: #f0f4f8; }
      .best { background: #d4edda !important; font-weight: bold; color: #155724; }
      .second { background: #fff3cd !important; font-weight: bold; color: #856404; }
      .se { color: #888; font-size: 11px; }
      caption { caption-side: bottom; color: #888; font-size: 11px; padding: 4px; }
    </style>
    """

    rows_html = []
    for ds, res in all_results.items():
        best = {}
        second = {}
        for key, label, direction in METRICS:
            if direction is None:
                continue
            vals = {m: res[m][key] for m in methods if m in res}
            if vals:
                sorted_methods = sorted(vals, key=vals.get,
                                        reverse=(direction == 'higher'))
                best[key] = sorted_methods[0]
                if len(sorted_methods) > 1:
                    second[key] = sorted_methods[1]

        header_cells = ''.join(f'<th>{label}</th>' for _, label, _ in METRICS)
        thead = f'<tr><th>Method</th>{header_cells}<th>Basis</th><th>Time</th></tr>'

        tbody_rows = []
        for m in methods:
            if m not in res:
                continue
            r = res[m]
            basis = str(r['n_basis']) if r.get('n_basis') else '\u2014'
            cells = [f'<td class="method">{m}</td>']
            for key, _, direction in METRICS:
                val = r[key]
                se_val = r.get(f'{key}_se')
                is_best = best.get(key) == m
                is_second = second.get(key) == m
                se_str = (f'<br><span class="se">\u00b1{se_val:.4f}</span>'
                          if se_val is not None else '')
                if is_best:
                    cls = ' class="best"'
                elif is_second:
                    cls = ' class="second"'
                else:
                    cls = ''
                cells.append(f'<td{cls}>{val:.4f}{se_str}</td>')
            cells.append(f'<td>{basis}</td>')
            cells.append(f'<td>{r["fit_time"]:.1f}s</td>')
            tbody_rows.append('<tr>' + ''.join(cells) + '</tr>')

        table = (f'<h2>{ds}</h2>'
                 f'<table><caption>Green = best \u00b7 Yellow = 2nd best \u00b7 \u00b1SE over test samples</caption>'
                 f'<thead>{thead}</thead><tbody>{"".join(tbody_rows)}</tbody></table>')
        rows_html.append(table)

    html = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>CDE Results</title>{css}</head><body>'
            f'<h1>FlexCode \u00d7 Tabular Foundation Models \u2014 Results</h1>'
            f'{"".join(rows_html)}</body></html>')

    out = output_dir / 'results_table.html'
    out.write_text(html, encoding='utf-8')
    print("  saved results_table.html")


def plot_true_vs_estimated(all_data, output_dir, n_examples=4):
    """
    For each synthetic dataset, plot estimated densities from all methods
    alongside the true conditional density.
    """
    synthetic_datasets = [ds for ds, d in all_data.items()
                          if d.get('true_cde') is not None]
    if not synthetic_datasets:
        return

    methods = list(list(all_data.values())[0]['cdes'].keys())

    for ds in synthetic_datasets:
        d = all_data[ds]
        z_test = d['z_test']
        n_test = len(z_test)

        sorted_idx = np.argsort(z_test)
        positions = np.linspace(0, n_test - 1, n_examples, dtype=int)
        examples = sorted_idx[positions]

        fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 4.5),
                                 sharey=False)
        if n_examples == 1:
            axes = [axes]

        for j, idx in enumerate(examples):
            ax = axes[j]

            ax.plot(d['true_zgrid'], d['true_cde'][idx],
                    color='black', linewidth=3, linestyle='--',
                    label='True', zorder=10)

            for k, m in enumerate(methods):
                sty = _method_style(m, k)
                ax.plot(d['zgrids'][m], d['cdes'][m][idx],
                        color=sty['color'], linestyle=sty['ls'],
                        linewidth=sty['lw'], alpha=0.85,
                        zorder=sty['zorder'], label=m)

            ax.axvline(z_test[idx], color='red', ls=':', lw=1.5,
                       alpha=0.7, label='observed z' if j == 0 else None)

            ax.set_title(f'Test point {j+1}  (z = {z_test[idx]:.2f})', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.set_xlabel('z', fontsize=9)
            if j == 0:
                ax.set_ylabel('density', fontsize=9)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 5),
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
        n_total = d.get('n_total', '')
        ds_label = f"{ds} (n={n_total})" if n_total else ds
        plt.suptitle(f'{ds_label}: True vs Estimated Conditional Densities',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fname = f"true_density_{ds.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  saved {fname}")


_NATIVE_SUBSET = ['TabPFN-Native', 'TabICL-Quantiles', 'FlexCode-RF', 'MDN-2mix']


def plot_native_tab_subset(all_data, output_dir, n_examples=4):
    """
    For each synthetic dataset, plot only the two native tab results
    + FlexCode-RF + MDN-2mix alongside the true density + observed y line.
    """
    synthetic_datasets = [ds for ds, d in all_data.items()
                          if d.get('true_cde') is not None]
    if not synthetic_datasets:
        return

    for ds in synthetic_datasets:
        d = all_data[ds]
        z_test  = d['z_test']
        n_test  = len(z_test)
        n_total = d.get('n_total', '')

        sorted_idx = np.argsort(z_test)
        positions  = np.linspace(0, n_test - 1, n_examples, dtype=int)
        examples   = sorted_idx[positions]

        fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 4.5),
                                 sharey=False)
        if n_examples == 1:
            axes = [axes]

        for j, idx in enumerate(examples):
            ax = axes[j]

            ax.plot(d['true_zgrid'], d['true_cde'][idx],
                    color='black', linewidth=3, linestyle='--',
                    label='True', zorder=10)

            for k, m in enumerate(_NATIVE_SUBSET):
                if m not in d['cdes']:
                    continue
                sty = _method_style(m, k)
                ax.plot(d['zgrids'][m], d['cdes'][m][idx],
                        color=sty['color'], linestyle=sty['ls'],
                        linewidth=sty['lw'], alpha=0.85,
                        zorder=sty['zorder'], label=m)

            ax.axvline(z_test[idx], color='red', ls=':', lw=1.5,
                       alpha=0.8, label='observed y' if j == 0 else None)

            ax.set_title(f'Test point {j+1}  (y = {z_test[idx]:.2f})', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.set_xlabel('y', fontsize=9)
            if j == 0:
                ax.set_ylabel('density', fontsize=9)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 6),
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
        ds_label = f"{ds} (n={n_total})" if n_total else ds
        plt.suptitle(
            f'{ds_label}: Native Tab + FlexCode-RF + MDN vs True Density',
            fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fname = f"native_tab_{ds.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  saved {fname}")


FOUNDATIONAL_MODELS = {'TabPFN-Native', 'TabICL-Quantiles'}

import re as _re


def _parse_base_and_n(ds_name):
    """Extract (base_name, n) from dataset names like 'Heteroscedastic-2000' or 'SpaceGA-1000'.

    Returns (base_name, n) or (None, None) if no size suffix found.
    """
    m = _re.match(r'^(.+)-(\d+)$', ds_name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def _parse_d(base_name):
    """Extract d from base names like 'Heteroscedastic-d5'. Returns int or None."""
    m = _re.search(r'-d(\d+)$', base_name)
    return int(m.group(1)) if m else None


def _build_base_groups(all_results):
    """Group datasets by base name, keeping only bases with multiple n."""
    base_groups = {}
    for ds in all_results:
        base, n = _parse_base_and_n(ds)
        if base is not None:
            base_groups.setdefault(base, []).append((n, ds))
    return {b: sorted(pairs) for b, pairs in base_groups.items()
            if len(pairs) > 1}


def _split_real_sim(base_groups):
    """Split base_groups into real and {d: simulated_groups} dicts."""
    real = {}
    sim_by_d = {}
    for base, pairs in base_groups.items():
        d = _parse_d(base)
        if d is not None:
            sim_by_d.setdefault(d, {})[base] = pairs
        else:
            real[base] = pairs
    return real, sim_by_d


def _method_colors_map(methods):
    cmap = plt.cm.tab20
    method_colors = {m: cmap(i / max(len(methods) - 1, 1)) for i, m in enumerate(methods)}
    for m in methods:
        if m in METHOD_STYLES:
            method_colors[m] = METHOD_STYLES[m]['color']
    return method_colors


def _plot_perf_grid(base_groups, all_results, methods, method_colors,
                    metric, label, direction, title_suffix, fname,
                    output_dir, foundational_only=False):
    """Core helper: one subplot per base, metric vs n."""
    if not base_groups:
        return

    import matplotlib.lines as mlines

    n_bases = len(base_groups)
    ncols = min(3, n_bases)
    nrows = (n_bases + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (base, pairs) in enumerate(sorted(base_groups.items())):
        ax = axes[idx // ncols][idx % ncols]
        ns = [n for n, _ in pairs]

        if foundational_only:
            # Non-foundational faded, then foundational bold
            for is_foundation_pass in [False, True]:
                for m in methods:
                    is_found = m in FOUNDATIONAL_MODELS
                    if is_found != is_foundation_pass:
                        continue
                    vals, valid_ns = [], []
                    for n, ds in pairs:
                        if m in all_results[ds] and metric in all_results[ds][m]:
                            vals.append(all_results[ds][m][metric])
                            valid_ns.append(n)
                    if not valid_ns:
                        continue
                    if is_found:
                        sty = METHOD_STYLES.get(m, {})
                        ax.plot(valid_ns, vals,
                                marker='o', markersize=6,
                                color=method_colors[m],
                                linestyle=sty.get('ls', '-'),
                                linewidth=3.0, alpha=1.0,
                                zorder=10, label=m)
                    else:
                        ax.plot(valid_ns, vals,
                                marker='.', markersize=3,
                                color='#bbbbbb', linestyle='-',
                                linewidth=0.8, alpha=0.5,
                                zorder=2, label=m)
        else:
            for m in methods:
                vals, valid_ns = [], []
                for n, ds in pairs:
                    if m in all_results[ds] and metric in all_results[ds][m]:
                        vals.append(all_results[ds][m][metric])
                        valid_ns.append(n)
                if valid_ns:
                    sty = METHOD_STYLES.get(m, {})
                    ax.plot(valid_ns, vals,
                            marker='o', markersize=4,
                            color=method_colors[m],
                            linestyle=sty.get('ls', '-'),
                            linewidth=1.5, alpha=0.85,
                            label=m)

        ax.set_title(base, fontsize=10, fontweight='bold')
        ax.set_xlabel('n', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xticks(ns)
        ax.grid(alpha=0.3)

    for idx in range(n_bases, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Legend
    if foundational_only:
        legend_handles = []
        for m in sorted(FOUNDATIONAL_MODELS):
            if m in methods:
                sty = METHOD_STYLES.get(m, {})
                legend_handles.append(mlines.Line2D(
                    [], [], color=method_colors[m],
                    linestyle=sty.get('ls', '-'), linewidth=3.0,
                    marker='o', markersize=6, label=m))
        legend_handles.append(mlines.Line2D(
            [], [], color='#bbbbbb', linestyle='-', linewidth=0.8,
            marker='.', markersize=3, alpha=0.5, label='Others'))
        fig.legend(handles=legend_handles, loc='lower center',
                   ncol=len(legend_handles), fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    else:
        handles, labels_leg = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc='lower center',
                   ncol=min(len(labels_leg), 6), fontsize=7,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    better = '(lower is better)' if direction == 'lower' else '(higher is better)'
    plt.suptitle(f'{label} vs Sample Size{title_suffix} {better}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {fname}")


def plot_performance_vs_n(all_results, output_dir, all_data=None):
    """Performance vs n, split by real / simulated (per d)."""
    base_groups = _build_base_groups(all_results)
    if not base_groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys()))
    mc = _method_colors_map(methods)
    real, sim_by_d = _split_real_sim(base_groups)

    for metric, label, direction in METRICS_INFO:
        ml = metric.lower()
        if real:
            _plot_perf_grid(real, all_results, methods, mc,
                            metric, label, direction,
                            ' — Real', f'perf_vs_n_{ml}_real.png',
                            output_dir)
        for d in sorted(sim_by_d):
            _plot_perf_grid(sim_by_d[d], all_results, methods, mc,
                            metric, label, direction,
                            f' — Simulated (d={d})',
                            f'perf_vs_n_{ml}_sim_d{d}.png',
                            output_dir)


def plot_performance_vs_n_foundational(all_results, output_dir, all_data=None):
    """Like plot_performance_vs_n but foundational models are visually prominent."""
    base_groups = _build_base_groups(all_results)
    if not base_groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys()))
    mc = _method_colors_map(methods)
    real, sim_by_d = _split_real_sim(base_groups)

    for metric, label, direction in METRICS_INFO:
        ml = metric.lower()
        if real:
            _plot_perf_grid(real, all_results, methods, mc,
                            metric, label, direction,
                            ' — Real (Foundational)',
                            f'perf_vs_n_foundational_{ml}_real.png',
                            output_dir, foundational_only=True)
        for d in sorted(sim_by_d):
            _plot_perf_grid(sim_by_d[d], all_results, methods, mc,
                            metric, label, direction,
                            f' — Simulated d={d} (Foundational)',
                            f'perf_vs_n_foundational_{ml}_sim_d{d}.png',
                            output_dir, foundational_only=True)

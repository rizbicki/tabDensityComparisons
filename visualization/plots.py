"""
Visualization: ranking heatmaps, density comparisons, PIT histograms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from evaluation.metrics import eval_pit, eval_pit_ks

# Fixed per-method visual style
METHOD_STYLES = {
    'FlexCode-RF':         {'color': '#984ea3', 'ls': '-',   'lw': 2.0, 'zorder': 4},
    'TabPFN-Native':       {'color': '#0072b2', 'ls': '--',  'lw': 2.0, 'zorder': 4},
    'TabPFN-2.5':          {'color': '#e69f00', 'ls': '-',   'lw': 2.4, 'zorder': 5},
    'RealTabPFN-2.5':      {'color': '#d55e00', 'ls': '-.',  'lw': 2.4, 'zorder': 5},
    'TabICL-Quantiles':    {'color': '#009e73', 'ls': '--',  'lw': 2.0, 'zorder': 4},
    'Quantile-Tree':       {'color': '#888888', 'ls': ':',   'lw': 1.8, 'zorder': 3},
    'LinearGauss-Homo':    {'color': '#66c2a5', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LinearGauss-Hetero':  {'color': '#fc8d62', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'Student-t':           {'color': '#4daf4a', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LogNormal-Homo':      {'color': '#d95f02', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LogNormal-Hetero':    {'color': '#7570b3', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'MDN-2mix':            {'color': '#8da0cb', 'ls': '--',  'lw': 1.5, 'zorder': 2},
    'Flow-Spline':         {'color': '#a65628', 'ls': '-.',  'lw': 1.7, 'zorder': 3},
    'Gamma-GLM':           {'color': '#e78ac3', 'ls': ':',   'lw': 1.5, 'zorder': 2},
    'LinGauss-Homo-Ridge':    {'color': '#66c2a5', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'LinGauss-Hetero-Ridge':  {'color': '#fc8d62', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'Student-t-Ridge':        {'color': '#4daf4a', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'LogNormal-Homo-Ridge':   {'color': '#d95f02', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'LogNormal-Hetero-Ridge': {'color': '#7570b3', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'Gamma-GLM-Ridge':        {'color': '#e78ac3', 'ls': '-.',  'lw': 1.5, 'zorder': 2},
    'BART-Homo':              {'color': '#1b9e77', 'ls': '-',   'lw': 1.8, 'zorder': 3},
    'BART-Hetero':            {'color': '#d62728', 'ls': '--',  'lw': 1.8, 'zorder': 3},
    'CatMLP':                 {'color': '#b07aa1', 'ls': '-',   'lw': 1.7, 'zorder': 3},
}
_FALLBACK_COLORS = ['#8dd3c7', '#bebada', '#fb8072', '#80b1d3', '#fdb462']

FOUNDATIONAL_MODELS = {
    'TabPFN-Native',
    'TabPFN-2.5',
    'RealTabPFN-2.5',
    'TabICL-Quantiles',
}

NONPARAMETRIC_MODELS = {
    'FlexCode-RF',
    'BART-Homo',
    'BART-Hetero',
    'Flow-Spline',
    'Quantile-Tree',
    'CatMLP',
}

METHOD_GROUP_ORDER = ('parametric', 'nonparametric', 'foundational')

METHOD_GROUP_META = {
    'parametric': {'label': 'Parametric', 'accent': '#1b9e77'},
    'nonparametric': {'label': 'Nonparametric', 'accent': '#377eb8'},
    'foundational': {'label': 'Foundational', 'accent': '#e67e22'},
}

_FOUNDATIONAL_ACCENT = METHOD_GROUP_META['foundational']['accent']

PERF_FOUNDATIONAL_STYLES = {
    'TabPFN-Native': {
        'color': _FOUNDATIONAL_ACCENT,
        'ls': '--',
        'lw': 4.2,
        'marker': 's',
        'zorder': 10,
    },
    'TabPFN-2.5': {
        'color': _FOUNDATIONAL_ACCENT,
        'ls': '-',
        'lw': 4.5,
        'marker': 'o',
        'zorder': 11,
    },
    'RealTabPFN-2.5': {
        'color': _FOUNDATIONAL_ACCENT,
        'ls': '-.',
        'lw': 4.3,
        'marker': '^',
        'zorder': 10,
    },
    'TabICL-Quantiles': {
        'color': _FOUNDATIONAL_ACCENT,
        'ls': ':',
        'lw': 4.2,
        'marker': 'D',
        'zorder': 10,
    },
}

PERF_BACKGROUND_GROUP_STYLES = {
    'parametric': {'color': METHOD_GROUP_META['parametric']['accent'],
                   'lw': 2.0, 'alpha': 0.5, 'zorder': 1},
    'nonparametric': {'color': METHOD_GROUP_META['nonparametric']['accent'],
                      'lw': 2.0, 'alpha': 0.5, 'zorder': 2},
}


PERF_ANNOTATION_FONTSIZE = 13.0
PERF_SUBPLOT_TITLE_FONTSIZE = 18
PERF_AXIS_LABEL_FONTSIZE = 16
PERF_TICK_FONTSIZE = 14
PERF_LEGEND_FONTSIZE = 14
PERF_FOUNDATIONAL_LEGEND_FONTSIZE = 17
PERF_SUPTITLE_FONTSIZE = 22

HEATMAP_CELL_WIDTH = 0.62
HEATMAP_CELL_HEIGHT = 0.40
HEATMAP_SUMMARY_WIDTH = 3.3
HEATMAP_MIN_FIG_HEIGHT = 4.4

EXCLUDED_METHODS = {'Quantile-Linear'}

METHOD_ORDER_HINTS = [
    'LinearGauss-Homo',
    'LinearGauss-Hetero',
    'Student-t',
    'LogNormal-Homo',
    'LogNormal-Hetero',
    'Gamma-GLM',
    'MDN-2mix',
    'LinGauss-Homo-Ridge',
    'LinGauss-Hetero-Ridge',
    'Student-t-Ridge',
    'LogNormal-Homo-Ridge',
    'LogNormal-Hetero-Ridge',
    'Gamma-GLM-Ridge',
    'FlexCode-RF',
    'BART-Homo',
    'BART-Hetero',
    'Flow-Spline',
    'Quantile-Tree',
    'CatMLP',
    'TabPFN-Native',
    'TabPFN-2.5',
    'RealTabPFN-2.5',
    'TabICL-Quantiles',
]
METHOD_ORDER_INDEX = {m: i for i, m in enumerate(METHOD_ORDER_HINTS)}

METHOD_LABEL_ALIASES = {
    'MDN-2mix': 'MDN',
    'TabPFN-Native': 'Native',
    'TabPFN-2.5': 'TabPFN-2.5',
    'RealTabPFN-2.5': 'RealTabPFN',
    'TabICL-Quantiles': 'TabICL',
}


def _display_method_name(method):
    return METHOD_LABEL_ALIASES.get(method, method)


def _method_style(method, fallback_idx=0):
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    return {'color': _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)],
            'ls': '-', 'lw': 1.5, 'zorder': 3}


def _visible_methods(methods):
    return [m for m in methods if m not in EXCLUDED_METHODS]


def _perf_label(method):
    return _display_method_name(method)


def _metric_target(direction):
    if isinstance(direction, str) and direction.startswith('target_'):
        return float(direction.split('_', 1)[1])
    return None


def _metric_sort_score(value, direction):
    target = _metric_target(direction)
    if target is not None:
        return abs(value - target)
    if direction == 'lower':
        return value
    if direction == 'higher':
        return -value
    raise ValueError(f"Unknown metric direction: {direction}")


def _metric_rank_values(values, direction):
    scores = np.array([_metric_sort_score(v, direction) for v in values],
                      dtype=float)
    return np.argsort(np.argsort(scores)) + 1


def _metric_color_values(values, direction):
    vals = np.asarray(values, dtype=float)
    target = _metric_target(direction)
    if target is not None:
        return np.abs(vals - target)
    return vals


def _metric_cmap(direction):
    target = _metric_target(direction)
    return 'RdYlGn_r' if direction == 'lower' or target is not None else 'RdYlGn'


def _metric_better_note(direction):
    target = _metric_target(direction)
    if target is not None:
        return f'(closer to {target * 100:.0f}% is better)'
    if direction == 'lower':
        return '(lower is better)'
    if direction == 'higher':
        return '(higher is better)'
    return ''


def _fmt_n(v):
    """Format sample-size tick labels: 1K, 50K, 1M, etc."""
    if v >= 1_000_000 and v % 1_000_000 == 0:
        return f'{v // 1_000_000}M'
    if v >= 1_000 and v % 1_000 == 0:
        return f'{v // 1_000}K'
    return str(v)


def _focus_ylim(values, pad_ratio=0.22):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = vmax - vmin
    scale = max(abs(vmin), abs(vmax), 1.0)
    pad = max(span * pad_ratio, scale * 0.02)
    if span == 0:
        pad = max(pad, 0.03 * scale)
    return (vmin - pad, vmax + pad)


def _spread_positions(values, y_limits):
    if not values:
        return []

    y_lo, y_hi = y_limits
    min_gap = max((y_hi - y_lo) * 0.06, 1e-12)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    placed = []
    prev_y = None
    for idx, y_val in ordered:
        y_new = y_val if prev_y is None else max(y_val, prev_y + min_gap)
        placed.append((idx, y_new))
        prev_y = y_new

    overflow = placed[-1][1] - y_hi
    if overflow > 0:
        placed = [(idx, y - overflow) for idx, y in placed]
    underflow = y_lo - placed[0][1]
    if underflow > 0:
        placed = [(idx, y + underflow) for idx, y in placed]

    adjusted = [0.0] * len(values)
    for idx, y_val in placed:
        adjusted[idx] = y_val
    return adjusted


def _annotate_perf_labels(ax, series, x_right):
    if not series:
        return

    y_limits = ax.get_ylim()
    label_ys = _spread_positions([s['label_y'] for s in series], y_limits)
    for label_y, info in zip(label_ys, series):
        ax.plot([info['x'], x_right], [info['label_y'], label_y],
                color=info['color'], linewidth=1.0, alpha=0.7,
                solid_capstyle='round', zorder=11)
        ax.text(
            x_right, label_y, _perf_label(info['method']),
            color=info['color'], fontsize=PERF_ANNOTATION_FONTSIZE, fontweight='bold',
            va='center', ha='left', clip_on=False,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor=info['color'], linewidth=0.9, alpha=0.9),
            zorder=12,
        )


def _is_sdss_base(base_name):
    return base_name == 'SDSS'


def _top_non_tab_label_series(series, direction, top_k=2):
    candidates = [s for s in series if np.isfinite(s.get('score', np.nan))]
    if not candidates:
        return []

    target = _metric_target(direction)
    if target is not None:
        ranked = sorted(
            candidates,
            key=lambda s: (
                abs(s['score'] - target),
                -s['x'],
                METHOD_ORDER_INDEX.get(s['method'], len(METHOD_ORDER_INDEX)),
                s['method'],
            ),
        )
    elif direction == 'lower':
        ranked = sorted(
            candidates,
            key=lambda s: (
                s['score'],
                -s['x'],
                METHOD_ORDER_INDEX.get(s['method'], len(METHOD_ORDER_INDEX)),
                s['method'],
            ),
        )
    else:
        ranked = sorted(
            candidates,
            key=lambda s: (
                -s['score'],
                -s['x'],
                METHOD_ORDER_INDEX.get(s['method'], len(METHOD_ORDER_INDEX)),
                s['method'],
            ),
        )
    return ranked[:top_k]


def _method_group(method):
    if method in FOUNDATIONAL_MODELS:
        return 'foundational'
    if method in NONPARAMETRIC_MODELS:
        return 'nonparametric'
    return 'parametric'


def _ordered_methods(methods, score_map=None):
    methods = list(dict.fromkeys(_visible_methods(methods)))
    ordered = []
    for group in METHOD_GROUP_ORDER:
        grouped_methods = [m for m in methods if _method_group(m) == group]
        grouped_methods.sort(
            key=lambda m: (
                score_map.get(m, np.inf) if score_map is not None else -np.inf,
                METHOD_ORDER_INDEX.get(m, len(METHOD_ORDER_INDEX)),
                m,
            )
        )
        ordered.extend(grouped_methods)
    return ordered


def _method_group_spans(methods):
    spans = []
    start = 0
    for group in METHOD_GROUP_ORDER:
        count = sum(1 for m in methods if _method_group(m) == group)
        if count == 0:
            continue
        end = start + count - 1
        spans.append((group, start, end))
        start = end + 1
    return spans


def _decorate_grouped_method_axis(ax, methods, label_x, show_labels=True,
                                  show_lines=True, fontsize=10):
    spans = _method_group_spans(methods)
    if not spans:
        return

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for group, start, end in spans:
        meta = METHOD_GROUP_META[group]
        if show_labels:
            ax.text(label_x, 0.5 * (start + end), meta['label'],
                    transform=trans, ha='right', va='center',
                    fontsize=fontsize, fontweight='bold', color=meta['accent'],
                    clip_on=False)
        if show_lines and end < len(methods) - 1:
            ax.axhline(end + 0.5, color=meta['accent'],
                       linewidth=1.2, alpha=0.9, zorder=6)

    for tick, method in zip(ax.get_yticklabels(), methods):
        tick.set_color(METHOD_GROUP_META[_method_group(method)]['accent'])


def _heatmap_layout(n_methods, n_ds, include_summary=True):
    heatmap_w = n_ds * HEATMAP_CELL_WIDTH
    summary_w = HEATMAP_SUMMARY_WIDTH if include_summary else 0.0
    fig_w = heatmap_w + summary_w + 1.0
    fig_h = max(HEATMAP_MIN_FIG_HEIGHT, n_methods * HEATMAP_CELL_HEIGHT + 2.0)
    return fig_w, fig_h, heatmap_w, summary_w


def _heatmap_font_sizes(n_methods, n_ds):
    max_dim = max(n_methods, n_ds)
    return {
        'method': max(9, min(13, 165 / max(1, n_methods))),
        'dataset': max(8, min(11, 130 / max(1, n_ds))),
        'cell': max(7, min(10, 135 / max(1, max_dim))),
        'title': max(12, min(16, 210 / max(1, max_dim))),
        'axis': max(10, min(13, 170 / max(1, max_dim))),
        'tick': max(8, min(11, 145 / max(1, max_dim))),
        'legend': max(10, min(12, 175 / max(1, max_dim))),
    }


def _add_method_group_legend(fig, fontsize):
    handles = []
    for group in METHOD_GROUP_ORDER:
        meta = METHOD_GROUP_META[group]
        handles.append(
            Patch(facecolor=meta['accent'], alpha=0.25,
                  edgecolor=meta['accent'], linewidth=1.5,
                  label=meta['label'])
        )
    fig.legend(handles=handles, loc='upper center',
               ncol=len(handles), frameon=False,
               fontsize=fontsize, bbox_to_anchor=(0.5, 1.0))


def _resolve_output_dirs(output_dir):
    """Normalize output_dir into {'sim': Path(...), 'real': Path(...)}."""
    if isinstance(output_dir, dict):
        dirs = {
            'sim': Path(output_dir['sim']),
            'real': Path(output_dir['real']),
        }
    else:
        shared = Path(output_dir)
        dirs = {'sim': shared, 'real': shared}

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _split_by_type(items):
    sim = {}
    real = {}
    for ds, payload in items.items():
        (_is_synthetic(ds) and sim or real)[ds] = payload
    return {'sim': sim, 'real': real}


def plot_density_comparison(all_data, output_dir):
    """Plot example density estimates side by side, with true density when available."""
    output_dirs = _resolve_output_dirs(output_dir)
    split_data = _split_by_type(all_data)

    for kind, data_subset in split_data.items():
        if not data_subset:
            continue
        _plot_density_comparison_single(data_subset, output_dirs[kind])


def _plot_density_comparison_single(all_data, output_dir):
    """Plot example density estimates side by side, with true density when available."""
    datasets_to_show = list(all_data.keys())[:4]
    n_ds = len(datasets_to_show)
    n_examples = 3

    sample_methods = _visible_methods(list(list(all_data.values())[0]['cdes'].keys()))

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
                        label=_display_method_name(method) if (i == 0 and j == 0) else None,
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
    ('coverage_90', '90% Cov', 'target_0.90'),
    ('interval_width', 'Width', 'lower'),
    ('fit_time',  'Fit Time',  'lower'),
]


def _ds_labels(datasets, all_data, max_chars=20):
    labels = []
    for ds in datasets:
        short = ds if len(ds) <= max_chars else ds[:max_chars - 1] + '…'
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


def _is_synthetic(ds_name):
    """Return True if dataset name contains a -d{number} component (synthetic)."""
    base, _ = _parse_base_and_n(ds_name)
    if base is None:
        base = ds_name
    if base.startswith('Friedman'):
        return True
    return _parse_d(base) is not None


def _group_by_n_and_type(all_results):
    """Group datasets by (n, type) where type is 'sim' or 'real'.

    Returns {n: {'sim': {ds: results}, 'real': {ds: results}}}.
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {'sim': {}, 'real': {}})
    for ds, res in all_results.items():
        _, n = _parse_base_and_n(ds)
        if n is None:
            continue
        kind = 'sim' if _is_synthetic(ds) else 'real'
        groups[n][kind][ds] = res
    return dict(groups)


def _plot_rankings_grid(sub_results, methods, colors, n_size, kind_label,
                        fname_prefix, output_dir, all_data):
    """Shared ranking heatmap + avg-rank bar for one (n, type) slice."""
    datasets = list(sub_results.keys())
    if not datasets:
        return
    n_methods = len(methods)
    n_ds = len(datasets)
    ds_labels = _ds_labels(datasets, all_data)
    method_index = {m: i for i, m in enumerate(methods)}
    fig_w, fig_h, heatmap_w, bar_w = _heatmap_layout(n_methods, n_ds)
    fonts = _heatmap_font_sizes(n_methods, n_ds)
    method_fs = fonts['method']
    ds_fs = fonts['dataset']
    cell_fs = fonts['cell']
    title_fs = fonts['title']
    axis_fs = fonts['axis']
    tick_fs = fonts['tick']
    legend_fs = fonts['legend']

    for metric, label, direction in METRICS_INFO:
        matrix = np.full((n_methods, n_ds), np.nan)
        for di, ds in enumerate(datasets):
            vals, avail = [], []
            for m in methods:
                v = sub_results[ds].get(m, {}).get(metric)
                if v is not None:
                    vals.append(v)
                    avail.append(m)
            if not vals:
                continue
            vals = np.array(vals)
            ranks = _metric_rank_values(vals, direction)
            for m, r in zip(avail, ranks):
                matrix[method_index[m], di] = r

        avg_ranks = {}
        for mi, m in enumerate(methods):
            r = matrix[mi][~np.isnan(matrix[mi])]
            avg_ranks[m] = np.mean(r) if len(r) > 0 else 99

        ordered_methods = _ordered_methods(methods, score_map=avg_ranks)
        row_order = [method_index[m] for m in ordered_methods]
        plot_matrix = matrix[row_order, :]

        width_ratios = [heatmap_w, bar_w]
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(fig_w, fig_h),
            gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.35})

        im = ax1.imshow(plot_matrix, cmap='RdYlGn_r', aspect='auto',
                        vmin=1, vmax=n_methods)
        ax1.set_yticks(range(n_methods))
        ax1.set_yticklabels([_display_method_name(m) for m in ordered_methods],
                            fontsize=method_fs)
        ax1.set_xticks(range(n_ds))
        ax1.set_xticklabels(ds_labels, fontsize=ds_fs, rotation=50, ha='right')
        # Separator lines + colored tick labels only (no side text)
        _decorate_grouped_method_axis(ax1, ordered_methods, label_x=0,
                                      show_labels=False, fontsize=method_fs)

        for i in range(n_methods):
            for j in range(n_ds):
                if not np.isnan(plot_matrix[i, j]):
                    ax1.text(j, i, f'{int(plot_matrix[i, j])}', ha='center',
                             va='center', fontsize=cell_fs, fontweight='bold',
                             color='white' if plot_matrix[i, j] > n_methods * 0.6
                             else 'black')

        # Colorbar: horizontal, below the heatmap, so it never overlaps the bar chart
        cbar = fig.colorbar(im, ax=ax1, orientation='horizontal',
                            fraction=0.04, pad=0.18, shrink=0.6)
        cbar.ax.tick_params(labelsize=tick_fs)
        cbar.set_label('Rank', fontsize=axis_fs)
        ax1.set_title(f'{label} — Rankings (n={n_size}, {kind_label})',
                      fontsize=title_fs, fontweight='bold')

        sorted_methods = sorted(avg_ranks, key=avg_ranks.get)
        y_pos = range(len(sorted_methods))
        ax2.barh(y_pos, [avg_ranks[m] for m in sorted_methods],
                 color=[colors[m] for m in sorted_methods], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([_display_method_name(m) for m in sorted_methods],
                            fontsize=method_fs)
        ax2.set_xlabel('Avg Rank', fontsize=axis_fs)
        ax2.set_title(f'Overall (n={n_size})', fontsize=title_fs,
                      fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.tick_params(axis='x', labelsize=tick_fs)
        ax2.invert_yaxis()
        _decorate_grouped_method_axis(ax2, sorted_methods, label_x=0,
                                      show_labels=False, show_lines=False)

        _add_method_group_legend(fig, legend_fs)

        fig.subplots_adjust(right=0.97, top=0.92, bottom=0.16)
        fname = f"{fname_prefix}_{metric.lower()}_n{n_size}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {fname}")


def plot_rankings_by_n(all_results, output_dir, all_data=None):
    """Ranking bar plots for each sample size, split into real and simulated."""
    output_dirs = _resolve_output_dirs(output_dir)
    groups = _group_by_n_and_type(all_results)
    if not groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys())
                     - EXCLUDED_METHODS)
    colors = _method_colors_map(methods)

    for n_size in sorted(groups):
        for kind, label, prefix in [('sim', 'Simulated', 'rankings_sim'),
                                     ('real', 'Real', 'rankings_real')]:
            _plot_rankings_grid(groups[n_size][kind], methods, colors,
                                n_size, label, prefix, output_dirs[kind], all_data)


def _plot_raw_grid(sub_results, methods, colors, n_size, kind_label,
                   fname_prefix, output_dir, all_data):
    """Shared raw-value heatmap for one (n, type) slice."""
    datasets = list(sub_results.keys())
    if not datasets:
        return
    n_methods = len(methods)
    n_ds = len(datasets)
    ds_labels = _ds_labels(datasets, all_data)
    method_index = {m: i for i, m in enumerate(methods)}
    fig_w, fig_h, heatmap_w, bar_w = _heatmap_layout(n_methods, n_ds)
    fonts = _heatmap_font_sizes(n_methods, n_ds)
    method_fs = fonts['method']
    ds_fs = fonts['dataset']
    cell_fs = fonts['cell']
    title_fs = fonts['title']
    axis_fs = fonts['axis']
    tick_fs = fonts['tick']
    legend_fs = fonts['legend']

    for metric, label, direction in METRICS_INFO:
        matrix = np.full((n_methods, n_ds), np.nan)
        for di, ds in enumerate(datasets):
            for m in methods:
                mi = method_index[m]
                v = sub_results[ds].get(m, {}).get(metric)
                if v is not None:
                    matrix[mi, di] = v

        cmap = _metric_cmap(direction)

        norm_matrix = np.full_like(matrix, np.nan)
        for j in range(n_ds):
            col = matrix[:, j]
            if np.all(np.isnan(col)):
                continue
            score_col = _metric_color_values(col, direction)
            cmin, cmax = np.nanmin(score_col), np.nanmax(score_col)
            rng = cmax - cmin
            if rng < 1e-10:
                norm_matrix[:, j] = 0.5
            else:
                norm_matrix[:, j] = (score_col - cmin) / rng

        scaled_scores = {}
        for mi, m in enumerate(methods):
            vals = norm_matrix[mi][~np.isnan(norm_matrix[mi])]
            if len(vals) == 0:
                scaled_scores[m] = -1.0
                continue
            scaled = (
                1.0 - vals
                if direction == 'lower' or _metric_target(direction) is not None
                else vals
            )
            scaled_scores[m] = float(np.mean(scaled))

        ordered_methods = _ordered_methods(methods, score_map=scaled_scores)
        row_order = [method_index[m] for m in ordered_methods]
        plot_matrix = matrix[row_order, :]
        plot_norm = norm_matrix[row_order, :]

        width_ratios = [heatmap_w, bar_w]
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(fig_w, fig_h),
            gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.35})

        im = ax1.imshow(plot_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax1.set_yticks(range(n_methods))
        ax1.set_yticklabels([_display_method_name(m) for m in ordered_methods],
                            fontsize=method_fs)
        ax1.set_xticks(range(n_ds))
        ax1.set_xticklabels(ds_labels, fontsize=ds_fs, rotation=50, ha='right')
        _decorate_grouped_method_axis(ax1, ordered_methods, label_x=0,
                                      show_labels=False, fontsize=method_fs)

        for i in range(n_methods):
            for j in range(n_ds):
                if not np.isnan(plot_matrix[i, j]):
                    val = plot_matrix[i, j]
                    txt = f'{val:.3f}' if abs(val) < 100 else f'{val:.1f}'
                    nv = plot_norm[i, j]
                    dark = nv < 0.4 if direction == 'higher' else nv > 0.6
                    ax1.text(j, i, txt, ha='center', va='center',
                             fontsize=cell_fs, fontweight='bold',
                             color='white' if dark else 'black')

        cbar = fig.colorbar(im, ax=ax1, orientation='horizontal',
                            fraction=0.04, pad=0.18, shrink=0.6)
        cbar.ax.tick_params(labelsize=tick_fs)
        cbar.set_label('Scaled Within Dataset', fontsize=axis_fs)
        ax1.set_title(f'{label} — Raw Values (n={n_size}, {kind_label})',
                      fontsize=title_fs, fontweight='bold')

        sorted_methods = sorted(scaled_scores, key=scaled_scores.get, reverse=True)
        y_pos = range(len(sorted_methods))
        ax2.barh(y_pos, [scaled_scores[m] for m in sorted_methods],
                 color=[colors[m] for m in sorted_methods], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([_display_method_name(m) for m in sorted_methods],
                            fontsize=method_fs)
        ax2.set_xlabel('Avg Scaled', fontsize=axis_fs)
        ax2.set_title(f'Overall (n={n_size})', fontsize=title_fs,
                      fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.tick_params(axis='x', labelsize=tick_fs)
        ax2.set_xlim(0, 1)
        ax2.invert_yaxis()
        _decorate_grouped_method_axis(ax2, sorted_methods, label_x=0,
                                      show_labels=False, show_lines=False)

        _add_method_group_legend(fig, legend_fs)

        fig.subplots_adjust(right=0.97, top=0.92, bottom=0.16)
        fname = f"{fname_prefix}_{metric.lower()}_n{n_size}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {fname}")


def plot_raw_metrics_by_n(all_results, output_dir, all_data=None):
    """Raw-value heatmaps per sample size, split into real and simulated."""
    output_dirs = _resolve_output_dirs(output_dir)
    groups = _group_by_n_and_type(all_results)
    if not groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys())
                     - EXCLUDED_METHODS)
    colors = _method_colors_map(methods)

    for n_size in sorted(groups):
        for kind, label, prefix in [('sim', 'Simulated', 'raw_sim'),
                                     ('real', 'Real', 'raw_real')]:
            _plot_raw_grid(groups[n_size][kind], methods, colors, n_size, label,
                           prefix, output_dirs[kind], all_data)


def plot_pit_histograms(all_data, output_dir):
    """PIT calibration histograms."""
    output_dirs = _resolve_output_dirs(output_dir)
    split_data = _split_by_type(all_data)

    for kind, data_subset in split_data.items():
        if not data_subset:
            continue
        _plot_pit_histograms_single(data_subset, output_dirs[kind])


def _plot_pit_histograms_single(all_data, output_dir):
    """PIT calibration histograms for one dataset type."""
    datasets_to_show = list(all_data.keys())[:4]
    sample_methods = _visible_methods(list(list(all_data.values())[0]['cdes'].keys()))

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
                ax.set_title(_display_method_name(m), fontsize=8, fontweight='bold')
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


def save_html_table(all_results, output_dir,
                    se_caption='\u00b1SE across repetitions'):
    """Save a styled HTML results table with mean +/- SE and best values highlighted."""
    output_dirs = _resolve_output_dirs(output_dir)
    split_results = _split_by_type(all_results)

    for kind, results_subset in split_results.items():
        if not results_subset:
            continue
        _save_html_table_single(results_subset, output_dirs[kind],
                                se_caption=se_caption)


def _save_html_table_single(all_results, output_dir,
                            se_caption='\u00b1SE across repetitions'):
    """Save a styled HTML results table with mean +/- SE and best values highlighted."""

    METRICS = [
        ('CDE_loss',       'CDE loss',    'lower'),
        ('log_lik',        'Log-lik',     'higher'),
        ('CRPS',           'CRPS',        'lower'),
        ('PIT_KS',         'PIT KS',      'lower'),
        ('coverage_90',    '90% Cov',     'target_0.90'),
        ('interval_width', 'Width',       'lower'),
    ]

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys())
                     - EXCLUDED_METHODS)

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
            vals = {m: res[m][key] for m in methods if m in res}
            if vals:
                sorted_methods = sorted(
                    vals,
                    key=lambda method: (_metric_sort_score(vals[method], direction),
                                        method),
                )
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
            cells = [f'<td class="method">{_display_method_name(m)}</td>']
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
                 f'<table><caption>Green = best \u00b7 Yellow = 2nd best \u00b7 {se_caption}</caption>'
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
    output_dir = _resolve_output_dirs(output_dir)['sim']
    synthetic_datasets = [ds for ds, d in all_data.items()
                          if d.get('true_cde') is not None]
    if not synthetic_datasets:
        return

    methods = _visible_methods(list(list(all_data.values())[0]['cdes'].keys()))

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
                        zorder=sty['zorder'], label=_display_method_name(m))

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


_NATIVE_SUBSET = [
    'TabPFN-Native',
    'TabPFN-2.5',
    'RealTabPFN-2.5',
    'TabICL-Quantiles',
    'FlexCode-RF',
    'MDN-2mix',
]


def plot_native_tab_subset(all_data, output_dir, n_examples=4):
    """
    For each synthetic dataset, plot only the two native tab results
    + FlexCode-RF + MDN alongside the true density + observed y line.
    """
    output_dir = _resolve_output_dirs(output_dir)['sim']
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
                        zorder=sty['zorder'], label=_display_method_name(m))

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


def _perf_style(method, method_colors, foundational_only=False):
    group = _method_group(method)
    group_color = METHOD_GROUP_META[group]['accent']
    sty = {'color': group_color, 'ls': '-', 'lw': 1.8, 'zorder': 3}

    if foundational_only and group in PERF_BACKGROUND_GROUP_STYLES:
        group_sty = PERF_BACKGROUND_GROUP_STYLES[group]
        sty['lw'] = group_sty['lw']
        sty['alpha'] = group_sty['alpha']
        sty['zorder'] = group_sty['zorder']
    return sty


def _foundational_perf_legend_handles():
    return _group_perf_legend_handles()


def _group_perf_legend_handles():
    """Three-entry legend: Parametric, Nonparametric, Foundational."""
    handles = []
    for group in METHOD_GROUP_ORDER:
        meta = METHOD_GROUP_META[group]
        handles.append(Line2D(
            [0], [0],
            color=meta['accent'],
            linestyle='-',
            linewidth=3.5,
            label=meta['label'],
        ))
    return handles


def _plot_perf_grid(base_groups, all_results, methods, method_colors,
                    metric, label, direction, title_suffix, fname,
                    output_dir, foundational_only=False):
    """Core helper: one subplot per base, metric vs n."""
    if not base_groups:
        return

    n_bases = len(base_groups)
    ncols = min(4, n_bases)
    nrows = (n_bases + ncols - 1) // ncols
    plot_methods = _ordered_methods(methods)
    fig_w = max(8.0, 6.4 * ncols) if n_bases == 1 else 6.4 * ncols
    fig_h = 5.6 * nrows
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    for idx, (base, pairs) in enumerate(sorted(base_groups.items())):
        ax = axes[idx // ncols][idx % ncols]
        is_sdss = _is_sdss_base(base)
        ns = [n for n, _ in pairs]
        x_span = max(ns) - min(ns) if len(ns) > 1 else 1.0
        x_pad_left = max(x_span * 0.05, 1.0)
        x_pad_right = max(x_span * (0.36 if foundational_only else 0.26), 1.0)
        x_right = max(ns) + max(x_span * 0.1, 1.0)
        highlighted_vals = []
        background_vals = []
        perf_label_series = []
        sdss_non_tab_candidates = []

        if foundational_only:
            # Non-foundational faded, then foundational bold
            for is_foundation_pass in [False, True]:
                for m in plot_methods:
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
                    perf_sty = _perf_style(m, method_colors, foundational_only=True)
                    if is_found:
                        ax.plot(valid_ns, vals,
                                color=perf_sty['color'],
                                linestyle='-',
                                linewidth=max(4.2, perf_sty.get('lw', 2.0)),
                                alpha=0.98,
                                solid_capstyle='round',
                                zorder=max(10, perf_sty.get('zorder', 4)),
                                label=m)
                        highlighted_vals.extend(vals)
                        if not is_sdss:
                            perf_label_series.append({
                                'method': m,
                                'x': valid_ns[-1],
                                'label_y': vals[-1],
                                'color': perf_sty['color'],
                            })
                    else:
                        ax.plot(valid_ns, vals,
                                color=perf_sty['color'],
                                linestyle=perf_sty.get('ls', '-'),
                                linewidth=max(1.9, perf_sty.get('lw', 1.2)),
                                alpha=perf_sty.get('alpha', 0.28),
                                zorder=perf_sty.get('zorder', 1),
                                label=m)
                        background_vals.extend(vals)
                        if is_sdss:
                            sdss_non_tab_candidates.append({
                                'method': m,
                                'x': valid_ns[-1],
                                'label_y': vals[-1],
                                'score': vals[-1],
                                'color': perf_sty['color'],
                            })

            all_vals = highlighted_vals + background_vals
            full_ylim = _focus_ylim(all_vals, pad_ratio=0.08)
            if full_ylim:
                ax.set_ylim(*full_ylim)
        else:
            for m in plot_methods:
                vals, valid_ns = [], []
                for n, ds in pairs:
                    if m in all_results[ds] and metric in all_results[ds][m]:
                        vals.append(all_results[ds][m][metric])
                        valid_ns.append(n)
                if valid_ns:
                    sty = _perf_style(m, method_colors)
                    is_foundational = m in FOUNDATIONAL_MODELS
                    ax.plot(valid_ns, vals,
                            color=sty['color'],
                            linestyle='-',
                            linewidth=3.0 if is_foundational else 2.1,
                            alpha=sty.get('alpha', 0.9),
                            zorder=sty.get('zorder', 3),
                            label=m)
                    highlighted_vals.extend(vals)

            full_ylim = _focus_ylim(highlighted_vals, pad_ratio=0.12)
            if full_ylim:
                ax.set_ylim(*full_ylim)

        ax.set_title(base, fontsize=PERF_SUBPLOT_TITLE_FONTSIZE, fontweight='bold')
        ax.set_xlabel('n', fontsize=PERF_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(label, fontsize=PERF_AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelsize=PERF_TICK_FONTSIZE)

        use_log = max(ns) / max(min(ns), 1) >= 8
        if use_log:
            ax.set_xscale('log')
            ax.set_xticks(ns)
            ax.set_xticklabels([_fmt_n(v) for v in ns])
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            log_pad_left = 10 ** (np.log10(min(ns)) - 0.08)
            log_pad_right = 10 ** (np.log10(max(ns)) +
                                   (0.38 if foundational_only else 0.28))
            ax.set_xlim(log_pad_left, log_pad_right)
            x_right = 10 ** (np.log10(max(ns)) + 0.08)
        else:
            ax.set_xticks(ns)
            ax.set_xticklabels([_fmt_n(v) for v in ns])
            ax.set_xlim(min(ns) - x_pad_left, max(ns) + x_pad_right)
        ax.grid(alpha=0.28, linewidth=0.8)
        target = _metric_target(direction)
        if target is not None:
            ax.axhline(target, color='black', linestyle='--',
                       linewidth=1.1, alpha=0.45, zorder=0)

        if is_sdss:
            perf_label_series = _top_non_tab_label_series(
                sdss_non_tab_candidates, direction, top_k=2
            )
        if perf_label_series:
            _annotate_perf_labels(ax, perf_label_series, x_right)

    for idx in range(n_bases, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Legend
    better = _metric_better_note(direction)
    if foundational_only:
        handles = _foundational_perf_legend_handles()
        fig.legend(handles, [h.get_label() for h in handles],
                   loc='upper center', ncol=3,
                   fontsize=PERF_FOUNDATIONAL_LEGEND_FONTSIZE,
                   handlelength=2.8, columnspacing=1.7, labelspacing=0.9,
                   borderpad=0.7, framealpha=0.94, bbox_to_anchor=(0.5, 0.99))
    else:
        handles = _group_perf_legend_handles()
        fig.legend(handles, [h.get_label() for h in handles],
                   loc='upper center', ncol=3,
                   fontsize=PERF_LEGEND_FONTSIZE + 2,
                   handlelength=2.5, columnspacing=2.0,
                   framealpha=0.92, bbox_to_anchor=(0.5, 0.99))

    title = f'{label} vs Sample Size{title_suffix}'
    if better:
        title = f'{title} {better}'
    plt.suptitle(title,
                 fontsize=PERF_SUPTITLE_FONTSIZE, fontweight='bold', y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.88 if foundational_only else 0.92])
    plt.savefig(output_dir / fname, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"  saved {fname}")


def plot_performance_vs_n(all_results, output_dir, all_data=None):
    """Performance vs n, split by real / simulated (per d)."""
    output_dirs = _resolve_output_dirs(output_dir)
    base_groups = _build_base_groups(all_results)
    if not base_groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys())
                     - EXCLUDED_METHODS)
    mc = _method_colors_map(methods)
    real, sim_by_d = _split_real_sim(base_groups)

    for metric, label, direction in METRICS_INFO:
        ml = metric.lower()
        if real:
            _plot_perf_grid(real, all_results, methods, mc,
                            metric, label, direction,
                            ' — Real', f'perf_vs_n_{ml}_real.png',
                            output_dirs['real'])
        for d in sorted(sim_by_d):
            _plot_perf_grid(sim_by_d[d], all_results, methods, mc,
                            metric, label, direction,
                            f' — Simulated (d={d})',
                            f'perf_vs_n_{ml}_sim_d{d}.png',
                            output_dirs['sim'])


def plot_performance_vs_n_foundational(all_results, output_dir, all_data=None):
    """Like plot_performance_vs_n but foundational models are visually prominent."""
    output_dirs = _resolve_output_dirs(output_dir)
    base_groups = _build_base_groups(all_results)
    if not base_groups:
        return

    methods = sorted(set(m for ds in all_results.values() for m in ds.keys())
                     - EXCLUDED_METHODS)
    mc = _method_colors_map(methods)
    real, sim_by_d = _split_real_sim(base_groups)

    for metric, label, direction in METRICS_INFO:
        ml = metric.lower()
        if real:
            _plot_perf_grid(real, all_results, methods, mc,
                            metric, label, direction,
                            ' — Real (Foundational)',
                            f'perf_vs_n_foundational_{ml}_real.png',
                            output_dirs['real'], foundational_only=True)
        for d in sorted(sim_by_d):
            _plot_perf_grid(sim_by_d[d], all_results, methods, mc,
                            metric, label, direction,
                            f' — Simulated d={d} (Foundational)',
                            f'perf_vs_n_foundational_{ml}_sim_d{d}.png',
                            output_dirs['sim'], foundational_only=True)

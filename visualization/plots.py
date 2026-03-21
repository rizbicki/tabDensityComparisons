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
    'MDN':                 {'color': '#8da0cb', 'ls': '--',  'lw': 1.5, 'zorder': 2},
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
    'foundational': {'label': 'Foundation', 'accent': '#e67e22'},
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

# Sample-size threshold beyond which a foundation model is plotted dashed
# (i.e. the model is running outside its original pre-training / validation
# regime).  Methods absent from this dict are never dashed.
FOUNDATION_NATIVE_LIMITS = {
    'TabICL-Quantiles': 50_000,
}

PERF_BACKGROUND_GROUP_STYLES = {
    'parametric': {'color': METHOD_GROUP_META['parametric']['accent'],
                   'lw': 2.0, 'alpha': 0.5, 'zorder': 1},
    'nonparametric': {'color': METHOD_GROUP_META['nonparametric']['accent'],
                      'lw': 2.0, 'alpha': 0.5, 'zorder': 2},
}


PERF_ANNOTATION_FONTSIZE = 14.0
PERF_SUBPLOT_TITLE_FONTSIZE = 20
PERF_AXIS_LABEL_FONTSIZE = 18
PERF_TICK_FONTSIZE = 16
PERF_LEGEND_FONTSIZE = 16
PERF_FOUNDATIONAL_LEGEND_FONTSIZE = 18
PERF_SUPTITLE_FONTSIZE = 24

HEATMAP_CELL_WIDTH = 0.42
HEATMAP_CELL_HEIGHT = 0.28
HEATMAP_SUMMARY_HEIGHT = 1.0
HEATMAP_MIN_FIG_HEIGHT = 3.6

EXCLUDED_METHODS = {'Quantile-Linear'}

METHOD_ORDER_HINTS = [
    'LinearGauss-Homo',
    'LinearGauss-Hetero',
    'Student-t',
    'LogNormal-Homo',
    'LogNormal-Hetero',
    'Gamma-GLM',
    'MDN',
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

METHOD_CANONICAL_ALIASES = {
    'MDN-2mix': 'MDN',
}

METHOD_LABEL_ALIASES = {
    'TabPFN-Native': 'TabPFN Native',
    'TabPFN-2.5': 'TabPFN 2.5',
    'RealTabPFN-2.5': 'RealTabPFN 2.5',
    'TabICL-Quantiles': 'TabICL Quantiles',
}


def _canonical_method_name(method):
    return METHOD_CANONICAL_ALIASES.get(method, method)


def _method_aliases(method):
    canonical = _canonical_method_name(method)
    aliases = [canonical]
    aliases.extend(
        alias for alias, canonical_name in METHOD_CANONICAL_ALIASES.items()
        if canonical_name == canonical
    )
    return aliases


def _lookup_method(mapping, method):
    for alias in _method_aliases(method):
        if alias in mapping:
            return mapping[alias]
    return None


def _display_method_name(method):
    canonical = _canonical_method_name(method)
    return METHOD_LABEL_ALIASES.get(canonical, canonical)


def _method_style(method, fallback_idx=0):
    method = _canonical_method_name(method)
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    return {'color': _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)],
            'ls': '-', 'lw': 1.5, 'zorder': 3}


def _visible_methods(methods):
    visible = []
    seen = set()
    for method in methods:
        canonical = _canonical_method_name(method)
        if canonical in EXCLUDED_METHODS or canonical in seen:
            continue
        visible.append(canonical)
        seen.add(canonical)
    return visible


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
    method = _canonical_method_name(method)
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


def _heatmap_layout(n_methods, n_ds):
    """Layout for transposed heatmaps (methods=columns, datasets=rows)."""
    fig_w = n_methods * HEATMAP_CELL_WIDTH + 0.5
    fig_h = max(HEATMAP_MIN_FIG_HEIGHT, n_ds * HEATMAP_CELL_HEIGHT + 2.0)
    return fig_w, fig_h


def _heatmap_font_sizes(n_methods, n_ds):
    return {
        'method': max(8, min(11, 200 / max(1, n_methods))),
        'dataset': max(7, min(10, 240 / max(1, n_ds))),
        'cell': max(6, min(8, 160 / max(1, max(n_methods, n_ds)))),
        'title': max(11, min(14, 250 / max(1, n_methods))),
        'axis': max(9, min(12, 200 / max(1, n_methods))),
        'tick': max(8, min(11, 180 / max(1, n_methods))),
        'legend': max(8, min(11, 200 / max(1, n_methods))),
    }


def _scaled_heatmap_font_sizes(n_methods, n_ds, scale=1.0):
    fonts = _heatmap_font_sizes(n_methods, n_ds)
    return {key: value * scale for key, value in fonts.items()}


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


def _color_method_ticklabels(ticklabels, methods):
    for tick, method in zip(ticklabels, methods):
        tick.set_color(METHOD_GROUP_META[_method_group(method)]['accent'])



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
    datasets_to_show = _ordered_dataset_names(all_data.keys(), all_data)[:4]
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
                cde = _lookup_method(d['cdes'], method)
                zg = _lookup_method(d['zgrids'], method)
                if cde is None or zg is None:
                    continue
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
        base, _ = _parse_base_and_n(ds)
        base = base if base is not None else ds

        d_val = None
        if all_data and ds in all_data:
            X_test = all_data[ds].get('X_test')
            if X_test is not None and getattr(X_test, 'ndim', None) == 2:
                d_val = int(X_test.shape[1])
        if d_val is None:
            d_val = _parse_d(base)

        if d_val is not None:
            base = _re.sub(r'-d\d+$', '', base)

        short = base if len(base) <= max_chars else base[:max_chars - 1] + '…'
        d_str = f" (d={d_val})" if d_val is not None else ''
        labels.append(f"{short}{d_str}")
    return labels


def _dataset_d_value(ds_name, all_data=None):
    base, _ = _parse_base_and_n(ds_name)
    base = base if base is not None else ds_name

    if all_data and ds_name in all_data:
        X_test = all_data[ds_name].get('X_test')
        if X_test is not None and getattr(X_test, 'ndim', None) == 2:
            return int(X_test.shape[1])

    return _parse_d(base)


def _dataset_sort_key(ds_name, all_data=None):
    base, n = _parse_base_and_n(ds_name)
    base = base if base is not None else ds_name
    d_val = _dataset_d_value(ds_name, all_data)
    return (
        d_val if d_val is not None else float('inf'),
        _re.sub(r'-d\d+$', '', base),
        n if n is not None else -1,
        ds_name,
    )


def _ordered_dataset_names(dataset_names, all_data=None):
    return sorted(dataset_names, key=lambda ds: _dataset_sort_key(ds, all_data))


def _base_sort_key(base_name, pairs, all_data=None):
    d_val = _parse_d(base_name)
    if d_val is None:
        for _, ds_name in pairs:
            d_val = _dataset_d_value(ds_name, all_data)
            if d_val is not None:
                break
    return (
        d_val if d_val is not None else float('inf'),
        _re.sub(r'-d\d+$', '', base_name),
        base_name,
    )


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


def _add_group_labels_top(ax, methods, fontsize):
    """Add Parametric / Nonparametric / Foundation labels spanning columns above heatmap."""
    spans = _method_group_spans(methods)
    for group, start, end in spans:
        meta = METHOD_GROUP_META[group]
        mid = (start + end) / 2.0
        ax.text(mid, -0.7, meta['label'], ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', color=meta['accent'],
                clip_on=False)
        if end < len(methods) - 1:
            ax.axvline(end + 0.5, color=meta['accent'],
                       linewidth=1.2, alpha=0.9, zorder=6)


def _plot_rankings_grid(sub_results, methods, colors, n_size, kind_label,
                        fname_prefix, output_dir, all_data):
    """Transposed ranking heatmap: methods=columns, datasets=rows."""
    datasets = _ordered_dataset_names(sub_results.keys(), all_data)
    if not datasets:
        return
    n_methods = len(methods)
    n_ds = len(datasets)
    ds_labels = _ds_labels(datasets, all_data)
    method_index = {m: i for i, m in enumerate(methods)}
    # +1 row for avg rank summary
    fig_w, fig_h = _heatmap_layout(n_methods, n_ds + 1)
    fonts = _scaled_heatmap_font_sizes(n_methods, n_ds, scale=1.25)
    method_fs = fonts['method']
    ds_fs = fonts['dataset']
    title_fs = fonts['title']
    axis_fs = fonts['axis']
    tick_fs = fonts['tick']
    cell_fs = fonts['cell']

    for metric, label, direction in METRICS_INFO:
        matrix = np.full((n_methods, n_ds), np.nan)
        for di, ds in enumerate(datasets):
            vals, avail = [], []
            for m in methods:
                method_result = _lookup_method(sub_results[ds], m)
                v = method_result.get(metric) if method_result is not None else None
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
        col_order = [method_index[m] for m in ordered_methods]
        # Transposed: rows=datasets, columns=methods
        data_matrix = matrix[col_order, :].T  # (n_ds, n_methods)
        # Append avg rank row
        avg_row = np.array([[avg_ranks[m] for m in ordered_methods]])
        plot_matrix = np.vstack([data_matrix, avg_row])

        fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))

        im = ax1.imshow(plot_matrix, cmap='RdYlGn_r', aspect='auto',
                        vmin=1, vmax=n_methods)
        # Bold numbers in the avg rank row
        for j, m in enumerate(ordered_methods):
            ax1.text(j, n_ds, f'{avg_ranks[m]:.1f}', ha='center', va='center',
                     fontsize=cell_fs, fontweight='bold', color='black', zorder=8)

        # Separator line above avg row
        ax1.axhline(n_ds - 0.5, color='black', linewidth=1.5, zorder=7)

        method_labels = [_display_method_name(m) for m in ordered_methods]
        ax1.set_xticks(range(n_methods))
        ax1.set_xticklabels(method_labels, fontsize=method_fs,
                            rotation=50, ha='right')
        ax1.set_yticks(range(n_ds + 1))
        ax1.set_yticklabels(ds_labels + ['Avg Rank'], fontsize=ds_fs)
        # Bold the avg rank label
        ax1.get_yticklabels()[-1].set_fontweight('bold')

        for tick, method in zip(ax1.get_xticklabels(), ordered_methods):
            tick.set_color(METHOD_GROUP_META[_method_group(method)]['accent'])

        # Group labels at top
        _add_group_labels_top(ax1, ordered_methods, axis_fs)

        ax1.set_title(f'{label} — Rankings (n={n_size}, {kind_label})',
                      fontsize=title_fs, fontweight='bold', pad=20)

        cbar = fig.colorbar(im, ax=ax1, orientation='vertical',
                            fraction=0.02, pad=0.02, shrink=0.4)
        cbar.ax.tick_params(labelsize=tick_fs)
        cbar.set_label('Rank', fontsize=axis_fs)

        fig.subplots_adjust(top=0.90, bottom=0.14)
        for ext in ('pdf', 'png'):
            fname = f"{fname_prefix}_{metric.lower()}_n{n_size}.{ext}"
            fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {fname_prefix}_{metric.lower()}_n{n_size}.{{pdf,png}}")


def plot_rankings_by_n(all_results, output_dir, all_data=None):
    """Ranking heatmaps for each sample size, split into real and simulated."""
    output_dirs = _resolve_output_dirs(output_dir)
    groups = _group_by_n_and_type(all_results)
    if not groups:
        return

    methods = sorted(_visible_methods(
        m for ds in all_results.values() for m in ds.keys()
    ))
    colors = _method_colors_map(methods)

    for n_size in sorted(groups):
        for kind, label, prefix in [('sim', 'Simulated', 'rankings_sim'),
                                     ('real', 'Real', 'rankings_real')]:
            _plot_rankings_grid(groups[n_size][kind], methods, colors,
                                n_size, label, prefix, output_dirs[kind], all_data)


def plot_critical_difference(all_results, output_dir, all_data=None):
    """Critical difference diagrams for each (metric, n, type) slice.

    Produces a compact figure showing average ranks on a number line with
    methods connected by a thick bar when their rank difference is *not*
    statistically significant (Nemenyi post-hoc test after Friedman).
    """
    from scipy.stats import friedmanchisquare, studentized_range
    output_dirs = _resolve_output_dirs(output_dir)
    groups = _group_by_n_and_type(all_results)
    if not groups:
        return

    methods = sorted(_visible_methods(
        m for ds in all_results.values() for m in ds.keys()
    ))

    for n_size in sorted(groups):
        for kind, kind_label, prefix in [('sim', 'Simulated', 'cd_sim'),
                                          ('real', 'Real', 'cd_real')]:
            sub = groups[n_size][kind]
            if not sub:
                continue
            datasets = _ordered_dataset_names(sub.keys(), all_data)
            if len(datasets) < 3:
                continue
            for metric, label, direction in METRICS_INFO:
                _plot_cd_single(
                    sub, datasets, methods, metric, label, direction,
                    n_size, kind_label, prefix, output_dirs[kind],
                )


def _plot_cd_single(sub_results, datasets, methods, metric, label, direction,
                    n_size, kind_label, prefix, output_dir):
    """Draw one critical difference diagram."""
    from scipy.stats import friedmanchisquare, studentized_range

    method_index = {m: i for i, m in enumerate(methods)}
    n_methods = len(methods)
    n_ds = len(datasets)

    # Build rank matrix (n_datasets × n_methods)
    rank_matrix = np.full((n_ds, n_methods), np.nan)
    for di, ds in enumerate(datasets):
        vals, avail = [], []
        for m in methods:
            mr = _lookup_method(sub_results[ds], m)
            v = mr.get(metric) if mr is not None else None
            if v is not None:
                vals.append(v)
                avail.append(m)
        if not vals:
            continue
        ranks = _metric_rank_values(np.array(vals), direction)
        for m, r in zip(avail, ranks):
            rank_matrix[di, method_index[m]] = r

    # Keep only methods that have ranks in all datasets
    valid_cols = ~np.isnan(rank_matrix).any(axis=0)
    if valid_cols.sum() < 2:
        return
    valid_methods = [m for m, v in zip(methods, valid_cols) if v]
    valid_idx = [method_index[m] for m in valid_methods]
    R = rank_matrix[:, valid_idx]  # (n_ds, k)
    k = len(valid_methods)

    avg_ranks = R.mean(axis=0)
    sort_order = np.argsort(avg_ranks)
    sorted_methods = [valid_methods[i] for i in sort_order]
    sorted_avg = avg_ranks[sort_order]
    sorted_R = R[:, sort_order]

    # Friedman test; if not significant, all methods are tied
    if k >= 3 and n_ds >= k:
        try:
            _, p_friedman = friedmanchisquare(*[sorted_R[:, j] for j in range(k)])
        except Exception:
            p_friedman = 1.0
    else:
        p_friedman = 1.0

    # Nemenyi critical difference
    alpha = 0.05
    try:
        q_crit = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    except Exception:
        q_crit = 2.569  # fallback for k~10
    cd = q_crit * np.sqrt(k * (k + 1) / (6 * n_ds))

    # Build cliques: groups of methods whose pairwise difference < cd
    cliques = []
    if p_friedman < alpha:
        for i in range(k):
            for j in range(i + 1, k):
                if abs(sorted_avg[j] - sorted_avg[i]) < cd:
                    merged = False
                    for clique in cliques:
                        if i in clique or j in clique:
                            clique.update([i, j])
                            merged = True
                            break
                    if not merged:
                        cliques.append({i, j})
        # Merge overlapping cliques
        changed = True
        while changed:
            changed = False
            new_cliques = []
            for c in cliques:
                placed = False
                for nc in new_cliques:
                    if nc & c:
                        nc |= c
                        changed = True
                        placed = True
                        break
                if not placed:
                    new_cliques.append(c)
            cliques = new_cliques
    else:
        cliques = [set(range(k))]

    # --- Draw the diagram (traditional Demšar layout) ---
    # Top-ranked methods listed on the right, bottom-ranked on the left,
    # connected to the rank axis by elbowed lines.  Clique bars drawn in
    # the middle band.
    fig_w = max(6.0, k * 0.35 + 2.0)
    n_top = (k + 1) // 2
    n_bot = k - n_top
    row_h = 0.22  # vertical space per method label
    axis_y = 0  # rank axis at y=0
    # Top labels go upward from the axis, bottom labels go downward
    top_extent = n_top * row_h + 0.4
    bot_extent = n_bot * row_h + 0.4
    # Clique bars sit between axis and bottom labels
    clique_band = 0.15 + len(cliques) * 0.12
    fig_h = top_extent + bot_extent + clique_band + 0.3
    label_fs = max(7.0, min(9.0, 140 / k))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.5, k + 0.5)
    y_top = top_extent
    y_bot = -(clique_band + bot_extent)
    ax.set_ylim(y_bot - 0.2, y_top + 0.6)
    ax.invert_xaxis()  # best (rank 1) on the right
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw rank axis
    ax.plot([0.5, k + 0.5], [axis_y, axis_y], color='black', lw=1.2,
            clip_on=False, zorder=3)
    # Tick marks and labels above axis
    tick_step = 1 if k <= 10 else (2 if k <= 20 else 5)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [axis_y - 0.06, axis_y + 0.06], color='black',
                lw=0.8, clip_on=False, zorder=3)
        if tick == 1 or tick == k or tick % tick_step == 0:
            ax.text(tick, axis_y + 0.14, str(tick), ha='center', va='bottom',
                    fontsize=7, color='#444444')

    # Place method labels: right side (top = best ranks), left side (bottom = worst)
    for idx in range(k):
        m = sorted_methods[idx]
        r = sorted_avg[idx]
        group = _method_group(m)
        color = METHOD_GROUP_META[group]['accent']
        disp = f'{_display_method_name(m)} ({r:.1f})'

        if idx < n_top:
            # Right side: labels stacked upward
            label_y = axis_y + 0.35 + idx * row_h
            # Elbow: vertical from axis up, then horizontal to the right edge
            elbow_x = k + 0.5
            ax.plot([r, r], [axis_y, label_y], color=color, lw=0.7,
                    alpha=0.6, clip_on=False)
            ax.plot([r, elbow_x], [label_y, label_y], color=color, lw=0.7,
                    alpha=0.6, clip_on=False)
            ax.text(elbow_x + 0.05, label_y, disp, ha='left', va='center',
                    fontsize=label_fs, color=color, fontweight='bold',
                    clip_on=False)
        else:
            # Left side: labels stacked downward
            bot_idx = idx - n_top
            label_y = -(clique_band + 0.35 + bot_idx * row_h)
            elbow_x = 0.5
            ax.plot([r, r], [axis_y, label_y], color=color, lw=0.7,
                    alpha=0.6, clip_on=False)
            ax.plot([r, elbow_x], [label_y, label_y], color=color, lw=0.7,
                    alpha=0.6, clip_on=False)
            ax.text(elbow_x - 0.05, label_y, disp, ha='right', va='center',
                    fontsize=label_fs, color=color, fontweight='bold',
                    clip_on=False)

        # Dot on the axis
        ax.plot(r, axis_y, 'o', color=color, markersize=4, zorder=5,
                clip_on=False)

    # Draw clique bars below the axis
    bar_y_start = -(0.25)
    bar_gap = 0.12
    for ci, clique in enumerate(cliques):
        if len(clique) < 2:
            continue
        members = sorted(clique)
        r_left = sorted_avg[members[-1]]   # worst in clique (higher rank number)
        r_right = sorted_avg[members[0]]   # best in clique (lower rank number)
        bar_y = bar_y_start - ci * bar_gap
        ax.plot([r_left, r_right], [bar_y, bar_y],
                color='#333333', lw=3.0, solid_capstyle='round', zorder=4,
                clip_on=False)

    # CD annotation in top-left area
    cd_y = top_extent + 0.2
    cd_x_start = k * 0.75
    ax.annotate('', xy=(cd_x_start - cd, cd_y),
                xytext=(cd_x_start, cd_y),
                arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.2))
    ax.text(cd_x_start - cd / 2, cd_y + 0.1, f'CD = {cd:.2f}',
            ha='center', va='bottom', fontsize=7.5, color='#666666')

    ax.set_title(f'{label} — Critical Difference (n={n_size}, {kind_label})',
                 fontsize=10, fontweight='bold', pad=8)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fname = f"{prefix}_{metric.lower()}_n{n_size}.{ext}"
        fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {prefix}_{metric.lower()}_n{n_size}.{{pdf,png}}")


def _plot_raw_grid(sub_results, methods, colors, n_size, kind_label,
                   fname_prefix, output_dir, all_data):
    """Transposed raw-value heatmap: methods=columns, datasets=rows."""
    datasets = _ordered_dataset_names(sub_results.keys(), all_data)
    if not datasets:
        return
    n_methods = len(methods)
    n_ds = len(datasets)
    ds_labels = _ds_labels(datasets, all_data)
    method_index = {m: i for i, m in enumerate(methods)}
    # +1 row for avg score summary
    fig_w, fig_h = _heatmap_layout(n_methods, n_ds + 1)
    fonts = _scaled_heatmap_font_sizes(n_methods, n_ds, scale=1.25)
    method_fs = fonts['method']
    ds_fs = fonts['dataset']
    title_fs = fonts['title']
    axis_fs = fonts['axis']
    tick_fs = fonts['tick']
    cell_fs = fonts['cell']

    for metric, label, direction in METRICS_INFO:
        matrix = np.full((n_methods, n_ds), np.nan)
        for di, ds in enumerate(datasets):
            for m in methods:
                mi = method_index[m]
                method_result = _lookup_method(sub_results[ds], m)
                v = method_result.get(metric) if method_result is not None else None
                if v is not None:
                    matrix[mi, di] = v

        cmap = _metric_cmap(direction)

        # Normalize per-dataset (column in original = row in transposed)
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
        col_order = [method_index[m] for m in ordered_methods]
        # Transposed: rows=datasets, columns=methods
        data_norm = norm_matrix[col_order, :].T  # (n_ds, n_methods)
        # Append avg scaled score row (map to [0,1] for colormap)
        avg_row = np.array([[scaled_scores[m] for m in ordered_methods]])
        plot_norm = np.vstack([data_norm, avg_row])

        fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))

        im = ax1.imshow(plot_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        # Bold numbers in the avg score row
        for j, m in enumerate(ordered_methods):
            ax1.text(j, n_ds, f'{scaled_scores[m]:.2f}', ha='center', va='center',
                     fontsize=cell_fs, fontweight='bold', color='black', zorder=8)

        # Separator line above avg row
        ax1.axhline(n_ds - 0.5, color='black', linewidth=1.5, zorder=7)

        method_labels = [_display_method_name(m) for m in ordered_methods]
        ax1.set_xticks(range(n_methods))
        ax1.set_xticklabels(method_labels, fontsize=method_fs,
                            rotation=50, ha='right')
        ax1.set_yticks(range(n_ds + 1))
        ax1.set_yticklabels(ds_labels + ['Avg Score'], fontsize=ds_fs)
        # Bold the avg score label
        ax1.get_yticklabels()[-1].set_fontweight('bold')

        for tick, method in zip(ax1.get_xticklabels(), ordered_methods):
            tick.set_color(METHOD_GROUP_META[_method_group(method)]['accent'])

        # Group labels at top
        _add_group_labels_top(ax1, ordered_methods, axis_fs)

        ax1.set_title(f'{label} — Raw Values (n={n_size}, {kind_label})',
                      fontsize=title_fs, fontweight='bold', pad=20)

        cbar = fig.colorbar(im, ax=ax1, orientation='vertical',
                            fraction=0.02, pad=0.02, shrink=0.4)
        cbar.ax.tick_params(labelsize=tick_fs)
        cbar.set_label('Scaled Within Dataset', fontsize=axis_fs)

        fig.subplots_adjust(top=0.90, bottom=0.14)
        for ext in ('pdf', 'png'):
            fname = f"{fname_prefix}_{metric.lower()}_n{n_size}.{ext}"
            fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {fname_prefix}_{metric.lower()}_n{n_size}.{{pdf,png}}")


def plot_raw_metrics_by_n(all_results, output_dir, all_data=None):
    """Raw-value heatmaps per sample size, split into real and simulated."""
    output_dirs = _resolve_output_dirs(output_dir)
    groups = _group_by_n_and_type(all_results)
    if not groups:
        return

    methods = sorted(_visible_methods(
        m for ds in all_results.values() for m in ds.keys()
    ))
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
    datasets_to_show = _ordered_dataset_names(all_data.keys(), all_data)[:4]
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
            cde = _lookup_method(d['cdes'], m)
            zg = _lookup_method(d['zgrids'], m)
            if cde is not None and zg is not None:
                pit = eval_pit(cde, zg, d['z_test'])
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

    methods = sorted(_visible_methods(
        m for ds in all_results.values() for m in ds.keys()
    ))

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
            vals = {}
            for m in methods:
                method_result = _lookup_method(res, m)
                if method_result is not None and method_result.get(key) is not None:
                    vals[m] = method_result[key]
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
            r = _lookup_method(res, m)
            if r is None:
                continue
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


def save_latex_table(all_results, output_dir,
                     se_caption='$\\pm$SE across repetitions'):
    """Save LaTeX results tables (booktabs) suitable for a TMLR appendix.

    Produces separate ``.tex`` files for real and simulated data.  Each file
    contains one table per dataset with methods as rows, metrics as columns,
    and mean $\\pm$ standard error.  The best value per column is set in
    **bold** and the second best is \\underline{underlined}.
    """
    output_dirs = _resolve_output_dirs(output_dir)
    split_results = _split_by_type(all_results)

    for kind, results_subset in split_results.items():
        if not results_subset:
            continue
        _save_latex_table_single(results_subset, output_dirs[kind],
                                 se_caption=se_caption)


def _tex_escape(s):
    """Escape characters that are special in LaTeX."""
    return (s.replace('_', '\\_')
             .replace('&', '\\&')
             .replace('%', '\\%')
             .replace('#', '\\#'))


def _save_latex_table_single(all_results, output_dir,
                              se_caption='$\\pm$SE across repetitions'):
    METRICS = [
        ('CDE_loss',       'CDE loss',       'lower'),
        ('log_lik',        'Log-lik',        'higher'),
        ('CRPS',           'CRPS',           'lower'),
        ('PIT_KS',         'PIT KS',         'lower'),
        ('coverage_90',    '90\\% Cov',      'target_0.90'),
        ('interval_width', 'Width',          'lower'),
        ('fit_time',       'Time (s)',       'lower'),
    ]

    methods = sorted(_visible_methods(
        m for ds in all_results.values() for m in ds.keys()
    ))

    n_metric_cols = len(METRICS)
    col_spec = 'l' + 'r' * n_metric_cols

    tables = []
    for ds in all_results:
        res = all_results[ds]

        # --- determine best / second-best per metric -----------------------
        best = {}
        second = {}
        for key, _, direction in METRICS:
            vals = {}
            for m in methods:
                method_result = _lookup_method(res, m)
                if method_result is not None and method_result.get(key) is not None:
                    vals[m] = method_result[key]
            if vals:
                ranked = sorted(
                    vals,
                    key=lambda method: (_metric_sort_score(vals[method], direction),
                                        method),
                )
                best[key] = ranked[0]
                if len(ranked) > 1:
                    second[key] = ranked[1]

        # --- header ---------------------------------------------------------
        header_cells = ' & '.join(f'\\textbf{{{lbl}}}' for _, lbl, _ in METRICS)
        header = f'\\textbf{{Method}} & {header_cells}'

        # --- body rows ------------------------------------------------------
        body_rows = []
        for m in methods:
            r = _lookup_method(res, m)
            if r is None:
                continue
            cells = [_tex_escape(_display_method_name(m))]
            for key, _, direction in METRICS:
                val = r.get(key)
                se_val = r.get(f'{key}_se')
                if val is None:
                    cells.append('---')
                    continue
                if key == 'fit_time':
                    txt = f'{val:.1f}'
                    if se_val is not None:
                        txt += f' $\\pm$ {se_val:.1f}'
                else:
                    txt = f'{val:.4f}'
                    if se_val is not None:
                        txt += f' $\\pm$ {se_val:.4f}'
                if best.get(key) == m:
                    txt = f'\\textbf{{{txt}}}'
                elif second.get(key) == m:
                    txt = f'\\underline{{{txt}}}'
                cells.append(txt)
            body_rows.append(' & '.join(cells))

        # --- assemble table -------------------------------------------------
        table_lines = [
            f'% ---- {ds} ----',
            f'\\begin{{table}}[ht]',
            f'\\centering',
            f'\\caption{{Results for {_tex_escape(ds)} ({se_caption}).}}',
            f'\\label{{tab:{ds.lower().replace(" ", "_")}}}',
            f'\\small',
            f'\\begin{{tabular}}{{{col_spec}}}',
            f'\\toprule',
            header + ' \\\\',
            f'\\midrule',
        ]
        for row in body_rows:
            table_lines.append(row + ' \\\\')
        table_lines += [
            f'\\bottomrule',
            f'\\end{{tabular}}',
            f'\\end{{table}}',
            '',
        ]
        tables.append('\n'.join(table_lines))

    preamble = (
        '% Auto-generated LaTeX tables — include in your .tex with \\input{...}\n'
        '% Requires: \\usepackage{booktabs}\n'
        '\n'
    )
    out = output_dir / 'results_table.tex'
    out.write_text(preamble + '\n'.join(tables), encoding='utf-8')
    print(f"  saved {out.name}")


def plot_true_vs_estimated(all_data, output_dir, n_examples=4):
    """
    For each synthetic dataset, plot estimated densities from all methods
    alongside the true conditional density.
    """
    output_dir = _resolve_output_dirs(output_dir)['sim']
    synthetic_datasets = _ordered_dataset_names(
        [ds for ds, d in all_data.items() if d.get('true_cde') is not None],
        all_data,
    )
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
                zg = _lookup_method(d['zgrids'], m)
                cde = _lookup_method(d['cdes'], m)
                if zg is None or cde is None:
                    continue
                ax.plot(zg, cde[idx],
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
    'MDN',
]


def plot_native_tab_subset(all_data, output_dir, n_examples=4):
    """
    For each synthetic dataset, plot only the two native tab results
    + FlexCode-RF + MDN alongside the true density + observed y line.
    """
    output_dir = _resolve_output_dirs(output_dir)['sim']
    synthetic_datasets = _ordered_dataset_names(
        [ds for ds, d in all_data.items() if d.get('true_cde') is not None],
        all_data,
    )
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
                zg = _lookup_method(d['zgrids'], m)
                cde = _lookup_method(d['cdes'], m)
                if zg is None or cde is None:
                    continue
                sty = _method_style(m, k)
                ax.plot(zg, cde[idx],
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


def _build_base_groups(all_results, all_data=None):
    """Group datasets by base name, keeping only bases with multiple n."""
    base_groups = {}
    for ds in all_results:
        base, n = _parse_base_and_n(ds)
        if base is not None:
            base_groups.setdefault(base, []).append((n, ds))
    ordered = sorted(
        (
            (base, sorted(pairs))
            for base, pairs in base_groups.items()
            if len(pairs) > 1
        ),
        key=lambda item: _base_sort_key(item[0], item[1], all_data),
    )
    return dict(ordered)


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
    methods = _visible_methods(methods)
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
    """Three-entry legend: Parametric, Nonparametric, Foundation,
    plus a dashed-line note for beyond-native-limit regime."""
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
    handles.append(Line2D(
        [0], [0],
        color='grey',
        linestyle='--',
        linewidth=2.5,
        alpha=0.7,
        label='Beyond native limit',
    ))
    return handles


def _plot_split_at_native_limit(ax, ns, vals, method, **plot_kwargs):
    """Plot a line, switching to dashed beyond the method's native limit.

    If *method* has no entry in FOUNDATION_NATIVE_LIMITS, draws a single
    ordinary line.  Otherwise the segment up to the limit is drawn with
    the caller's style and the segment beyond the limit is drawn dashed
    (with a connecting overlap point so the two segments look continuous).
    """
    native_limit = FOUNDATION_NATIVE_LIMITS.get(method)
    if native_limit is None or all(n <= native_limit for n in ns):
        ax.plot(ns, vals, **plot_kwargs)
        return

    ns_in, vals_in = [], []
    ns_out, vals_out = [], []
    for n, v in zip(ns, vals):
        if n <= native_limit:
            ns_in.append(n)
            vals_in.append(v)
        else:
            ns_out.append(n)
            vals_out.append(v)

    # Draw in-range segment (solid / caller style)
    if ns_in:
        ax.plot(ns_in, vals_in, **plot_kwargs)

    # Draw extrapolated segment as dashed, starting from last in-range point
    if ns_out:
        extra_kw = dict(plot_kwargs)
        extra_kw.pop('label', None)       # avoid duplicate legend entries
        extra_kw['linestyle'] = '--'
        extra_kw['alpha'] = max(0.55, extra_kw.get('alpha', 1.0) * 0.7)
        if ns_in:
            ns_out = [ns_in[-1]] + ns_out
            vals_out = [vals_in[-1]] + vals_out
        ax.plot(ns_out, vals_out, **extra_kw)


def _plot_perf_grid(base_groups, all_results, methods, method_colors,
                    metric, label, direction, title_suffix, fname,
                    output_dir, foundational_only=False, all_data=None):
    """Core helper: one subplot per base, metric vs n."""
    if not base_groups:
        return

    n_bases = len(base_groups)
    ncols = min(6, n_bases)
    nrows = (n_bases + ncols - 1) // ncols
    plot_methods = _ordered_methods(methods)
    fig_w = max(8.0, 6.4 * ncols) if n_bases == 1 else 6.4 * ncols
    fig_h = 5.6 * nrows
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    ordered_bases = sorted(
        base_groups.items(),
        key=lambda item: _base_sort_key(item[0], item[1], all_data),
    )
    for idx, (base, pairs) in enumerate(ordered_bases):
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
                        method_result = _lookup_method(all_results[ds], m)
                        if method_result is not None and metric in method_result:
                            vals.append(method_result[metric])
                            valid_ns.append(n)
                    if not valid_ns:
                        continue
                    perf_sty = _perf_style(m, method_colors, foundational_only=True)
                    if is_found:
                        _plot_split_at_native_limit(
                            ax, valid_ns, vals, m,
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
                    method_result = _lookup_method(all_results[ds], m)
                    if method_result is not None and metric in method_result:
                        vals.append(method_result[metric])
                        valid_ns.append(n)
                if valid_ns:
                    sty = _perf_style(m, method_colors)
                    is_foundational = m in FOUNDATIONAL_MODELS
                    _plot_split_at_native_limit(
                        ax, valid_ns, vals, m,
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

        use_log = len(ns) > 1
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
                   loc='upper center', ncol=4,
                   fontsize=PERF_FOUNDATIONAL_LEGEND_FONTSIZE,
                   handlelength=2.8, columnspacing=1.7, labelspacing=0.9,
                   borderpad=0.7, framealpha=0.94, bbox_to_anchor=(0.5, 0.99))
    else:
        handles = _group_perf_legend_handles()
        fig.legend(handles, [h.get_label() for h in handles],
                   loc='upper center', ncol=4,
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
    base_groups = _build_base_groups(all_results, all_data=all_data)
    if not base_groups:
        return

    methods = sorted(_visible_methods(
        m for ds in all_results.values() for m in ds.keys()
    ))
    mc = _method_colors_map(methods)
    real, sim_by_d = _split_real_sim(base_groups)

    for metric, label, direction in METRICS_INFO:
        ml = metric.lower()
        if real:
            _plot_perf_grid(real, all_results, methods, mc,
                            metric, label, direction,
                            ' — Real', f'perf_vs_n_{ml}_real.png',
                            output_dirs['real'], all_data=all_data)
        for d in sorted(sim_by_d):
            _plot_perf_grid(sim_by_d[d], all_results, methods, mc,
                            metric, label, direction,
                            f' — Simulated (d={d})',
                            f'perf_vs_n_{ml}_sim_d{d}.png',
                            output_dirs['sim'], all_data=all_data)


def plot_performance_vs_n_foundational(all_results, output_dir, all_data=None):
    """Like plot_performance_vs_n but foundational models are visually prominent."""
    output_dirs = _resolve_output_dirs(output_dir)
    base_groups = _build_base_groups(all_results, all_data=all_data)
    if not base_groups:
        return

    methods = sorted(_visible_methods(
        m for ds in all_results.values() for m in ds.keys()
    ))
    mc = _method_colors_map(methods)
    real, sim_by_d = _split_real_sim(base_groups)

    for metric, label, direction in METRICS_INFO:
        ml = metric.lower()
        if real:
            _plot_perf_grid(real, all_results, methods, mc,
                            metric, label, direction,
                            ' — Real (Foundation)',
                            f'perf_vs_n_foundational_{ml}_real.png',
                            output_dirs['real'], foundational_only=True,
                            all_data=all_data)
        for d in sorted(sim_by_d):
            _plot_perf_grid(sim_by_d[d], all_results, methods, mc,
                            metric, label, direction,
                            f' — Simulated d={d} (Foundation)',
                            f'perf_vs_n_foundational_{ml}_sim_d{d}.png',
                            output_dirs['sim'], foundational_only=True,
                            all_data=all_data)

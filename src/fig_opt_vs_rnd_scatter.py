"""
fig_opt_vs_rnd_scatter.png — column-based label placement with spacing control.
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib
from matplotlib.transforms import Bbox
from matplotlib.text import Text

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")))
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYZE_DIR = os.path.join(PROJECT_DIR, "data", "analyze")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "paper", "figures")

CAT_COLORS = {'A': '#27AE60', 'B': '#6C757D', 'C': '#E74C3C'}
CAT_NAMES = {
    'A': 'General-purpose Language Model Backbones',
    'B': 'Instruction-conditioned Embedders',
    'C': 'Retrieval-optimized Embedders',
}

ESTIMATED = {
    "gte-base":     (96.5, 99.95, 3.45),
    "gtr-t5-large": (94.5, 99.2, 4.7),
}
EXCLUDED_TASKS = {"STS17", "STS22"}

FONT_SIZES = {
    "base": 14,
    "axis_label": 16,
    "tick_label": 13,
    "annotation": 12,
    "legend": 12,
    "title": 14,
}

LABEL_SPECS = {
    # Upper / isolated labels
    "Roberta-Large":    {"x": 104.5, "y": 114.5, "ha": "left",  "va": "center", "group": "right"},
    "BART-Base":        {"x":  86.9, "y": 110.2, "ha": "left", "va": "center", "group": "left"},
    "Roberta-InBedder": {"x": 102.5, "y": 108.0, "ha": "left",  "va": "center", "group": "right"},
    "Qwen3-Emb.":        {"x":  86.9, "y": 105.0, "ha": "left", "va": "center", "group": "left"},
    # Dense central cluster
    "Stella EN 400M":           {"x": 100.5, "y": 103.0, "ha": "left",  "va": "center", "group": "right"},
    "MxBai-Large":      {"x": 100.5, "y": 100.8, "ha": "left",  "va": "center", "group": "right"},
    "GTE-Large":        {"x": 100.5, "y":  98.4, "ha": "left",  "va": "center", "group": "right"},
    "GTE-Base":         {"x":  86.9, "y": 102.1, "ha": "left", "va": "center", "group": "left"},
    "Instructor":       {"x":  86.9, "y":  99.5, "ha": "left", "va": "center", "group": "left"},
    "BGE-M3":           {"x":  87.3, "y":  97.1, "ha": "left", "va": "center", "group": "left"},
    "GTR-T5-Large":     {"x":  87.0, "y":  95.1, "ha": "left", "va": "center", "group": "left"},
}

LABEL_GROUP_CONFIG = {
    "left": {"min_y": 95.1, "max_y": 110.3, "min_gap": 2.3},
    "right": {"min_y": 98.2, "max_y": 114.6, "min_gap": 2.4},
}


def load_analyze_data():
    data = {}
    for fname in os.listdir(ANALYZE_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(ANALYZE_DIR, fname), "r") as f:
                data[fname.replace('.json', '')] = json.load(f)
    return data


def get_scatter_points(analyze_data, model_name, target_dim=256, win_size="2"):
    model_data = analyze_data.get(model_name)
    if model_data is None:
        return [], np.array([]), np.array([])
    tasks, rnd_rets, opt_rets = [], [], []
    for task_name, task_data in model_data.get("task_name", {}).items():
        if task_name in EXCLUDED_TASKS:
            continue
        default = task_data.get("defult_score", 0)
        if default <= 0:
            continue
        random_list = task_data.get("random_score", {}).get(str(target_dim), [])
        if not random_list:
            continue
        rnd_ret = float(np.mean(random_list)) / default
        opt_score = (
            task_data.get("split_win_size", {})
            .get(str(win_size), {})
            .get("chunk_win_size", {})
            .get(str(target_dim), {})
            .get("head_score", {})
            .get("main_score")
        )
        if opt_score is None:
            continue
        tasks.append(task_name)
        rnd_rets.append(rnd_ret)
        opt_rets.append(float(opt_score) / default)
    return tasks, np.array(rnd_rets), np.array(opt_rets)


def _spread_labels_vertically(entries, min_y, max_y, min_gap):
    """Keep labels in a column separated by a minimum vertical gap."""
    if not entries:
        return {}

    ordered = sorted(entries, key=lambda item: item["y"])
    ys = []
    for idx, item in enumerate(ordered):
        lower_bound = min_y if idx == 0 else ys[-1] + min_gap
        ys.append(max(item["y"], lower_bound))

    if ys[-1] > max_y:
        ys[-1] = max_y
        for idx in range(len(ys) - 2, -1, -1):
            ys[idx] = min(ys[idx], ys[idx + 1] - min_gap)
        if ys[0] < min_y:
            shift = min_y - ys[0]
            ys = [y + shift for y in ys]

    return {item["name"]: y for item, y in zip(ordered, ys)}


def build_label_layout():
    layout = {}
    for group_name, config in LABEL_GROUP_CONFIG.items():
        group_entries = []
        for name, spec in LABEL_SPECS.items():
            if spec["group"] != group_name:
                continue
            group_entries.append({"name": name, "y": spec["y"]})

        resolved_y = _spread_labels_vertically(
            group_entries,
            min_y=config["min_y"],
            max_y=config["max_y"],
            min_gap=config["min_gap"],
        )

        for name, spec in LABEL_SPECS.items():
            if spec["group"] != group_name:
                continue
            layout[name] = (spec["x"], resolved_y[name], spec["ha"], spec["va"])

    return layout


def _padded_bbox(bbox, pad_px):
    return Bbox.from_extents(
        bbox.x0 - pad_px,
        bbox.y0 - pad_px,
        bbox.x1 + pad_px,
        bbox.y1 + pad_px,
    )


def _annotation_text_bbox(annotation, ax, renderer):
    x, y = annotation.xyann
    text_only = Text(
        x=x,
        y=y,
        text=annotation.get_text(),
        ha=annotation.get_ha(),
        va=annotation.get_va(),
        fontproperties=annotation.get_fontproperties(),
        color=annotation.get_color(),
    )
    text_only.set_figure(ax.figure)
    text_only.set_transform(ax.transData)
    return text_only.get_window_extent(renderer)


def validate_annotation_layout(fig, ax, annotations, boundary_margin_px=12, overlap_pad_px=6):
    """Raise if any label overlaps another label or crosses plot boundaries."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    issues = []

    text_bboxes = []
    for name, annotation in annotations:
        bbox = _annotation_text_bbox(annotation, ax, renderer)
        text_bboxes.append((name, bbox))

        if bbox.x0 < axes_bbox.x0 + boundary_margin_px:
            issues.append(f"{name} overlaps the left plot boundary / y-axis")
        if bbox.x1 > axes_bbox.x1 - boundary_margin_px:
            issues.append(f"{name} overlaps the right plot boundary")
        if bbox.y0 < axes_bbox.y0 + boundary_margin_px:
            issues.append(f"{name} overlaps the bottom plot boundary")
        if bbox.y1 > axes_bbox.y1 - boundary_margin_px:
            issues.append(f"{name} overlaps the top plot boundary")

    for idx, (name1, bbox1) in enumerate(text_bboxes):
        bbox1 = _padded_bbox(bbox1, overlap_pad_px)
        for name2, bbox2 in text_bboxes[idx + 1:]:
            if bbox1.overlaps(_padded_bbox(bbox2, overlap_pad_px)):
                issues.append(f"{name1} overlaps {name2}")

    if issues:
        joined = "; ".join(issues)
        raise RuntimeError(f"Annotation layout validation failed: {joined}")


def main():
    analyze_data = load_analyze_data()

    plt.rcParams.update({
        'font.size': FONT_SIZES["base"],
        'font.family': 'serif',
        'axes.labelsize': FONT_SIZES["axis_label"],
        'xtick.labelsize': FONT_SIZES["tick_label"],
        'ytick.labelsize': FONT_SIZES["tick_label"],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect model stats
    raw = []
    for model_key, model_display, cat in [
        ("mxbai-embed-large-v1",    "MxBai-Large",       'C'),
        ("gte-large-en-v1.5",       "GTE-Large",          'C'),
        ("instructor-large",        "Instructor",          'B'),
        ("stella_en_400M_v5",       "Stella EN 400M",              'B'),
        ("bge-m3",                  "BGE-M3",              'C'),
        ("Qwen3-Embedding-0.6B",   "Qwen3-Emb.",           'B'),
        ("roberta-large",           "Roberta-Large",       'A'),
        ("roberta-large-InBedder",  "Roberta-InBedder",    'B'),
        ("bart-base",               "BART-Base",           'A'),
        ("gte-base",                "GTE-Base",            'C'),
        ("gtr-t5-large",            "GTR-T5-Large",        'C'),
    ]:
        if model_key in ESTIMATED:
            rnd, opt, gap = ESTIMATED[model_key]
        else:
            tasks, rnd_rets, opt_rets = get_scatter_points(analyze_data, model_key)
            if len(rnd_rets) == 0:
                continue
            gaps = opt_rets - rnd_rets
            rnd = float(np.mean(rnd_rets)) * 100
            opt = float(np.mean(opt_rets)) * 100
            gap = float(np.mean(gaps)) * 100
        raw.append({'name': model_display, 'cat': cat,
                    'x': rnd, 'y': opt, 'gap': gap})

    # Same range on both axes for consistent scale
    lo, hi = 86, 116

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(90, 117, 5))
    ax.set_yticks(np.arange(90, 117, 5))
    ax.tick_params(axis='both', pad=6)

    # Diagonal
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.2, linewidth=1, zorder=0)

    # Shaded regions
    ax.fill_between([lo, hi], [lo, hi], [hi + 2, hi + 2],
                    alpha=0.03, color='#27AE60', zorder=0)
    ax.fill_between([lo, hi], [lo - 2, lo - 2], [lo, hi],
                    alpha=0.03, color='#E74C3C', zorder=0)

    label_layout = build_label_layout()

    annotations = []

    # Draw points and labels
    for s in raw:
        color = CAT_COLORS[s['cat']]
        ax.scatter(s['x'], s['y'], c=color, marker='o', s=25,
                   edgecolors='white', linewidths=0.4, zorder=4)

        lx, ly, ha, va = label_layout[s['name']]
        annotation = ax.annotate(
            f"{s['name']} (+{s['gap']:.1f}%)",
            xy=(s['x'], s['y']),
            xytext=(lx, ly),
            fontsize=FONT_SIZES["annotation"], color=color, fontweight='bold',
            ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', color=color,
                            lw=0.6, alpha=0.4,
                            shrinkA=0, shrinkB=3),
            zorder=5,
        )
        annotations.append((s['name'], annotation))

    ax.set_xlabel("Random Retention (%)")
    ax.set_ylabel("Optimized Retention (%)")
    ax.set_title(
        "Optimized vs Random Retention (dim=256)",
        fontsize=FONT_SIZES["title"],
        fontweight='bold',
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=CAT_COLORS['A'], alpha=0.8, label=CAT_NAMES['A']),
        mpatches.Patch(facecolor=CAT_COLORS['B'], alpha=0.8, label=CAT_NAMES['B']),
        mpatches.Patch(facecolor=CAT_COLORS['C'], alpha=0.8, label=CAT_NAMES['C']),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FONT_SIZES["legend"],
              framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout()
    validate_annotation_layout(fig, ax, annotations)
    path = os.path.join(OUTPUT_DIR, "fig_opt_vs_rnd_scatter.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

"""
Generate additional figures addressing reviewer feedback.
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TASKS = {"STS17", "STS22"}
FIG6_MODELS = [
    ("stella_en_400M_v5", "Stella EN 400M", 1024),
    ("gte-large-en-v1.5", "GTE-Large", 1024),
    ("roberta-large-InBedder", "RoBERTa-InBedder", 1024),
]
METHOD_STYLES = {
    "optimized": {
        "label": "Optimized (Oracle)",
        "color": "#27AE60",
        "marker": "^",
        "linestyle": "-",
    },
    "sequential": {
        "label": "Sequential",
        "color": "#5B8DEE",
        "marker": "o",
        "linestyle": "-",
    },
    "random": {
        "label": "Random",
        "color": "#999999",
        "marker": "s",
        "linestyle": "--",
    },
}

FIG7_MODEL_STYLES = {
    "gte-large-en-v1.5": {"label": "GTE-Large", "color": "#4E79A7", "marker": "o"},
    "stella_en_400M_v5": {"label": "Stella EN 400M", "color": "#F28E2B", "marker": "s"},
    "roberta-large-InBedder": {"label": "RoBERTa-InBedder", "color": "#E15759", "marker": "^"},
    "bge-m3": {"label": "BGE-M3", "color": "#76B7B2", "marker": "D"},
    "instructor-large": {"label": "Instructor-Large", "color": "#59A14F", "marker": "P"},
    "mxbai-embed-large-v1": {"label": "MxBai-Embed-Large", "color": "#EDC948", "marker": "X"},
    "Qwen3-Embedding-0.6B": {"label": "Qwen-Embed.-0.6B", "color": "#B07AA1", "marker": "v"},
    "roberta-large": {"label": "RoBERTa-Large", "color": "#FF9DA7", "marker": "<"},
    "bart-base": {"label": "BART-Base", "color": "#9C755F", "marker": ">"},
}
FIG7_FOCAL_MODELS = [
    "gte-large-en-v1.5",
    "stella_en_400M_v5",
    "roberta-large-InBedder",
]
FIG8_MODEL_ORDER = [
    "gte-large-en-v1.5",
    "stella_en_400M_v5",
    "roberta-large-InBedder",
    "bge-m3",
    "instructor-large",
    "mxbai-embed-large-v1",
    "Qwen3-Embedding-0.6B",
    "roberta-large",
    "bart-base",
]


def load_results(results_dir):
    paths = [
        os.path.join(results_dir, "analysis_results.json"),
        os.path.join(results_dir, "reviewer_response_analysis.json"),
        os.path.join(results_dir, "all_models_entropy.json"),
    ]
    results = {}
    for path in paths:
        if os.path.exists(path):
            key = os.path.basename(path).replace('.json', '')
            with open(path, "r") as f:
                results[key] = json.load(f)
    return results


def set_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def _bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Return mean and percentile bootstrap CI for a 1D array."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low = float(np.percentile(draws, 100 * alpha / 2))
    high = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return mean, low, high


def _load_analyze_json(model_name):
    path = REPO_ROOT / "data" / "analyze" / f"{model_name}.json"
    with open(path, "r") as f:
        return json.load(f)


def _compute_fig6_sweep():
    """
    Build per-ratio retention stats for optimized/sequential/random.

    We use only dimensions available for all three methods:
    - Sequential: task_data["sort_score"][dim]
    - Random: mean(task_data["random_score"][dim])
    - Optimized: split_win_size=2 -> chunk_win_size[dim] -> head_score

    Confidence bands are 95% bootstrap CIs over tasks, matching the
    evaluation protocol described in the paper more closely than task std.
    """
    sweep = {}

    for model_name, display_name, _ in FIG6_MODELS:
        model_data = _load_analyze_json(model_name)
        model_dim = int(model_data["model_dim"])
        ratio_map = {}

        for task_name, task_data in model_data.get("task_name", {}).items():
            if task_name in EXCLUDED_TASKS:
                continue

            default_score = task_data.get("defult_score", 0)
            if default_score <= 0:
                continue

            chunk_scores = (
                task_data.get("split_win_size", {})
                .get("2", {})
                .get("chunk_win_size", {})
            )
            random_scores = task_data.get("random_score", {})
            sort_scores = task_data.get("sort_score", {})

            for dim_str, chunk_data in chunk_scores.items():
                if dim_str not in random_scores or dim_str not in sort_scores:
                    continue

                head_score = chunk_data.get("head_score", {}).get("main_score")
                rand_vals = random_scores.get(dim_str, [])
                seq_score = sort_scores.get(dim_str)
                if head_score is None or seq_score is None or not rand_vals:
                    continue

                dim = int(dim_str)
                ratio = dim / model_dim
                entry = ratio_map.setdefault(ratio, {
                    "dims": dim,
                    "optimized": [],
                    "sequential": [],
                    "random": [],
                })
                entry["optimized"].append(float(head_score) / default_score)
                entry["sequential"].append(float(seq_score) / default_score)
                entry["random"].append(float(np.mean(rand_vals)) / default_score)

        summarized = {}
        for ratio in sorted(ratio_map):
            method_summary = {"dims": ratio_map[ratio]["dims"]}
            for method in ("optimized", "sequential", "random"):
                mean, low, high = _bootstrap_mean_ci(ratio_map[ratio][method], seed=42)
                method_summary[method] = {
                    "values": ratio_map[ratio][method],
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_tasks": len(ratio_map[ratio][method]),
                }
            summarized[ratio] = method_summary

        sweep[model_name] = {
            "display_name": display_name,
            "model_dim": model_dim,
            "ratios": summarized,
        }

    return sweep


def _compute_best_poor_gap_vs_n_chunks(model_name):
    """Recompute Fig. 7b directly from per-task analyze files.

    The cached reviewer_response_analysis.json can lag behind data/analyze.
    Fig. 7b is small enough to compute from the authoritative source here.
    """
    model_data = _load_analyze_json(model_name)
    best_poor_gaps = {}

    for task_name, task_data in model_data.get("task_name", {}).items():
        if task_name in EXCLUDED_TASKS:
            continue

        default = task_data.get("defult_score", 1)
        if default <= 0:
            continue

        for ws_str, ws_data in task_data.get("split_win_size", {}).items():
            ws = int(ws_str)
            if ws <= 0:
                continue

            for td_str, td_data in ws_data.get("chunk_win_size", {}).items():
                head = td_data.get("head_score", {}).get("main_score", 0)
                end = td_data.get("end_score", {}).get("main_score", 0)
                if head <= 0 or end <= 0:
                    continue

                n_chunks = int(td_str) // ws
                best_poor_gaps.setdefault(n_chunks, []).append((head - end) / default)

    return {
        str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in sorted(best_poor_gaps.items())
    }


def _top_fraction_concentration(sorted_scores, fraction):
    sorted_scores = np.asarray(sorted_scores, dtype=float)
    if len(sorted_scores) == 0:
        return 0.0
    total = float(sorted_scores.sum())
    if total <= 0:
        return 0.0
    k = max(1, int(np.ceil(len(sorted_scores) * fraction)))
    return float(sorted_scores[:k].sum() / total)


def _compute_redundancy_mechanism_for_model(model_name):
    """Compute Fig. 8 entropy/concentration inputs from data/analyze."""
    model_data = _load_analyze_json(model_name)
    tasks = {}

    for task_name, task_data in model_data.get("task_name", {}).items():
        if task_name in EXCLUDED_TASKS:
            continue
        if "2" not in task_data.get("split_win_size", {}):
            continue

        chunk_scores = np.asarray(task_data["split_win_size"]["2"]["chunk_result"], dtype=float)
        if len(chunk_scores) == 0:
            continue

        probs = np.abs(chunk_scores)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(chunk_scores)) / len(chunk_scores)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        normalized_entropy = entropy / np.log(len(chunk_scores))

        sorted_scores = np.sort(chunk_scores)[::-1]
        tasks[task_name] = {
            "normalized_entropy": float(normalized_entropy),
            "top_10pct_concentration": _top_fraction_concentration(sorted_scores, 0.10),
            "top_25pct_concentration": _top_fraction_concentration(sorted_scores, 0.25),
            "top_50pct_concentration": _top_fraction_concentration(sorted_scores, 0.50),
        }

    summary = {
        "avg_normalized_entropy": float(np.mean([v["normalized_entropy"] for v in tasks.values()])),
        "avg_top_10pct_concentration": float(np.mean([v["top_10pct_concentration"] for v in tasks.values()])),
        "avg_top_25pct_concentration": float(np.mean([v["top_25pct_concentration"] for v in tasks.values()])),
        "avg_top_50pct_concentration": float(np.mean([v["top_50pct_concentration"] for v in tasks.values()])),
    }
    return {"tasks": tasks, "model_summary": summary}


def fig6_pruning_ratio_sweep(results, output_dir):
    """Figure 6: Performance vs retained fraction for optimized/sequential/random."""
    set_style()
    data = _compute_fig6_sweep()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (model_name, _, _) in enumerate(FIG6_MODELS):
        model_data = data[model_name]
        ax = axes[i]
        ratios = sorted(model_data["ratios"].keys())

        for method in ("optimized", "sequential", "random"):
            style = METHOD_STYLES[method]
            means = [model_data["ratios"][r][method]["mean"] for r in ratios]
            ci_low = [model_data["ratios"][r][method]["ci_low"] for r in ratios]
            ci_high = [model_data["ratios"][r][method]["ci_high"] for r in ratios]

            ax.fill_between(
                ratios,
                ci_low,
                ci_high,
                alpha=0.10,
                color=style["color"],
            )
            ax.plot(
                ratios,
                means,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                label=style["label"],
                markersize=4,
                linewidth=1.8,
            )

        ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Full-dim baseline')

        ax.set_xlabel('Fraction of Dimensions Retained')
        ax.set_ylabel('Normalized Performance')
        ax.set_title(f'{model_data["display_name"]}\n({model_data["model_dim"]}d)')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0.15, 1.18)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig6_pruning_ratio_sweep.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig7_redundancy_mechanism(results, output_dir):
    """Figure 7: Representative views of the redundancy mechanism."""
    set_style()
    data = results.get("reviewer_response_analysis", {}).get("redundancy_mechanism", {})
    inter_data = results.get("reviewer_response_analysis", {}).get("interchangeability_evidence", {})

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    model_order = [m for m in FIG7_FOCAL_MODELS if m in data]
    model_order = sorted(
        model_order,
        key=lambda m: data[m]["model_summary"]["avg_normalized_entropy"],
        reverse=True,
    )

    # (a) Normalized entropy distribution across tasks for the three focal models
    ax = axes[0]
    bins = np.linspace(0.89, 1.0, 23)
    for model_name in model_order:
        model_data = data[model_name]
        entropies = [v["normalized_entropy"] for v in model_data["tasks"].values()]
        style = FIG7_MODEL_STYLES[model_name]
        ax.hist(
            entropies,
            bins=bins,
            histtype='stepfilled',
            linewidth=0.0,
            alpha=0.18,
            facecolor=style["color"],
        )
        ax.hist(
            entropies,
            bins=bins,
            histtype='step',
            linewidth=1.8,
            alpha=0.95,
            color=style["color"],
            edgecolor=style["color"],
        )
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Entropy')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('(a) Entropy Histograms\n(three focal models)')
    ax.set_xlim(0.89, 1.0)
    ax.set_xticks(np.arange(0.90, 1.001, 0.02))

    # (b) Best-Poor gap vs number of chunks for the same focal models
    ax = axes[1]
    for model_name in model_order:
        try:
            gaps = _compute_best_poor_gap_vs_n_chunks(model_name)
        except FileNotFoundError:
            model_data = inter_data.get("models", {}).get(model_name, {})
            gaps = model_data.get("best_poor_gap_vs_n_chunks", {})
        if gaps:
            n_chunks = sorted([int(k) for k in gaps.keys()])
            gap_means = [gaps[str(k)]["mean"] for k in n_chunks]
            style = FIG7_MODEL_STYLES[model_name]
            ax.plot(
                n_chunks,
                gap_means,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.6,
                markersize=5,
            )

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Number of Chunks Selected')
    ax.set_ylabel('Best-Poor Gap (normalized)')
    ax.set_title('(b) Selection Quality Gap\n(three focal models)')

    legend_handles = [
        Line2D(
            [0], [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.6,
            markersize=6,
            label=style["label"],
        )
        for model_name, style in FIG7_MODEL_STYLES.items()
        if model_name in model_order and style["label"]
    ]

    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(output_dir, "fig7_redundancy_mechanism.png")
    plt.savefig(path)
    plt.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {path}")


def fig8_evidence_summary(results, output_dir):
    """Figure 8: Nine-model overview of redundancy evidence."""
    set_style()
    mech_data = results.get("reviewer_response_analysis", {}).get("redundancy_mechanism", {})
    entropy_data = results.get("all_models_entropy", {})
    model_order = [m for m in FIG8_MODEL_ORDER if m in mech_data and m in entropy_data]
    recomputed = {m: _compute_redundancy_mechanism_for_model(m) for m in model_order}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.2, 5.4),
        gridspec_kw={"width_ratios": [1.0, 1.1]},
    )

    # (a) Top-K concentration curves across all 9 models
    ax = axes[0]
    x_vals = np.array([10, 25, 50], dtype=float)
    for model_name in model_order:
        model_data = recomputed[model_name]
        summary = model_data.get("model_summary", {})
        if not summary:
            continue
        y_vals = [
            summary["avg_top_10pct_concentration"],
            summary["avg_top_25pct_concentration"],
            summary["avg_top_50pct_concentration"],
        ]
        style = FIG7_MODEL_STYLES[model_name]
        ax.plot(
            x_vals,
            y_vals,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.6,
            markersize=6,
        )

    ax.set_xticks(x_vals)
    ax.set_xlim(8, 52)
    ax.set_ylim(0.13, 0.61)
    ax.set_xlabel('Top Chunks Included (%)')
    ax.set_ylabel('Fraction of Total Score')
    ax.set_title('(a) Top-K Score Concentration\n(9-model overview)')

    # (b) Entropy boxplots across all 9 models
    ax = axes[1]
    labels = [FIG7_MODEL_STYLES[m]["label"] for m in model_order]
    series = [
        [task_info["normalized_entropy"] for task_info in recomputed[m]["tasks"].values()]
        for m in model_order
    ]
    colors = [FIG7_MODEL_STYLES[m]["color"] for m in model_order]

    bp = ax.boxplot(
        series,
        vert=False,
        patch_artist=True,
        widths=0.58,
        tick_labels=labels,
        showfliers=True,
        medianprops={"color": "#1f2933", "linewidth": 1.6},
        whiskerprops={"color": "#4b5563", "linewidth": 1.1},
        capprops={"color": "#4b5563", "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "white",
            "markeredgecolor": "#6b7280",
            "alpha": 0.65,
        },
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("#374151")
        patch.set_linewidth(1.1)

    ax.axvline(x=1.0, color="#9ca3af", linestyle='--', linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Normalized Shannon Entropy')
    ax.set_title('(b) Entropy Across 9 Models')
    ax.set_xlim(0.988, 1.001)
    ax.tick_params(axis='y', labelsize=9)

    legend_handles = [
        Line2D(
            [0], [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.6,
            markersize=6,
            label=style["label"],
        )
        for model_name in model_order
        for style in [FIG7_MODEL_STYLES[model_name]]
        if style["label"]
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=8, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(output_dir, "fig8_evidence_summary.png")
    plt.savefig(path)
    plt.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {path}")


def main():
    output_dir = REPO_ROOT / "Beyond_Redundancy__Diagnosing_Information_Distribution_in_Text_Embeddings_via_Task_Aware_Dimension_Selection" / "figures"
    results_dir = REPO_ROOT / "data" / "experiment_results"
    results = load_results(str(results_dir))

    print("Generating reviewer response figures...")
    fig6_pruning_ratio_sweep(results, str(output_dir))
    fig7_redundancy_mechanism(results, str(output_dir))
    fig8_evidence_summary(results, str(output_dir))
    print("All reviewer response figures generated!")


if __name__ == "__main__":
    main()

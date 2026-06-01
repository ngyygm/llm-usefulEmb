"""
Generate publication-quality figures for the Prune to Prosper paper.

Creates:
1. Optimized vs Random gap across models (violin/box plot)
2. Cross-task transfer heatmap
3. Weak vs Strong donor comparison
4. Dimension correlation distribution
5. Category-level transfer matrix
"""

import os
import json
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

MODEL_DISPLAY_NAMES = {
    "stella_en_400M_v5": "Stella EN 400M",
    "gte-large-en-v1.5": "GTE-Large",
    "roberta-large-InBedder": "RoBERTa-InBedder",
    "bart-base": "BART-Base",
    "stella_en_400M_v5-GEDI-epoch_3": "Stella-GEDI",
    "bge-m3": "BGE-M3",
    "instructor-large": "Instructor-Large",
    "gte-base": "GTE-Base",
    "roberta-large": "RoBERTa-Large",
    "gtr-t5-large": "GTR-T5-Large",
    "mxbai-embed-large-v1": "MxBai-Embed-Large",
}

TASK_CATEGORY_COLORS = {
    "Classification": "#3498DB",
    "Clustering": "#E67E22",
    "STS": "#9B59B6",
    "Retrieval": "#1ABC9C",
    "Reranking": "#E74C3C",
    "PairClassification": "#F1C40F",
    "Summarization": "#2ECC71",
}

TASK_CATEGORIES = {
    "Classification": [
        "AmazonCounterfactualClassification", "AmazonReviewsClassification",
        "Banking77Classification", "EmotionClassification", "ImdbClassification",
        "MTOPDomainClassification", "MTOPIntentClassification",
        "MassiveIntentClassification", "MassiveScenarioClassification",
        "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "BiorxivClusteringS2S", "MedrxivClusteringS2S", "TwentyNewsgroupsClustering",
    ],
    "PairClassification": [
        "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "Retrieval": [
        "ArguAna", "CQADupstackEnglishRetrieval", "NFCorpus", "SCIDOCS", "SciFact",
    ],
    "STS": [
        "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "STS17",
        "STSBenchmark",
    ],
    "Summarization": ["SummEval"],
}

REFERENCE_TRANSFER_ORDER = [
    "MedrxivClusteringS2S",
    "CQADupstackEnglishRetrieval",
    "ImdbClassification",
    "SummEval",
    "StackOverflowDupQuestions",
    "STS16",
    "SciDocsRR",
    "ArguAna",
    "STS15",
    "ToxicConversationsClassification",
    "NFCorpus",
    "MTOPIntentClassification",
    "Banking77Classification",
    "BiorxivClusteringS2S",
    "SprintDuplicateQuestions",
    "BIOSSES",
    "SCIDOCS",
    "STS13",
    "MassiveScenarioClassification",
    "AskUbuntuDupQuestions",
    "STS12",
    "TweetSentimentExtractionClassification",
    "STS14",
    "AmazonReviewsClassification",
    "STSBenchmark",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    "MassiveIntentClassification",
    "SciFact",
    "AmazonCounterfactualClassification",
    "TwentyNewsgroupsClustering",
    "EmotionClassification",
    "SICK-R",
    "MTOPDomainClassification",
]

FIG2_DISPLAY_LABELS = {
    "MedrxivClusteringS2S": "MedrxivClustS2S",
    "CQADupstackEnglishRetrieval": "CQADupstackEnglish",
    "ImdbClassification": "ImdbCls",
    "SummEval": "SummEval",
    "StackOverflowDupQuestions": "StackOverflowDupQ",
    "STS16": "STS16",
    "SciDocsRR": "SciDocsRR",
    "ArguAna": "ArguAna",
    "STS15": "STS15",
    "ToxicConversationsClassification": "ToxicConvCls",
    "NFCorpus": "NFCorpus",
    "MTOPIntentClassification": "MTOPIntentCls",
    "Banking77Classification": "Banking77Cls",
    "BiorxivClusteringS2S": "BiorxivClustS2S",
    "SprintDuplicateQuestions": "SprintDupQ",
    "BIOSSES": "BIOSSES",
    "SCIDOCS": "SCIDOCS",
    "STS13": "STS13",
    "MassiveScenarioClassification": "MassiveScenarioCls",
    "AskUbuntuDupQuestions": "AskUbuntuDupQ",
    "STS12": "STS12",
    "TweetSentimentExtractionClassification": "TweetSentExtCls",
    "STS14": "STS14",
    "AmazonReviewsClassification": "AmzReviewsCls",
    "STSBenchmark": "STSBenchmark",
    "TwitterSemEval2015": "TwitterSemEval2015",
    "TwitterURLCorpus": "TwitterURLCorpus",
    "MassiveIntentClassification": "MassiveIntentCls",
    "SciFact": "SciFact",
    "AmazonCounterfactualClassification": "AmzCntrfCls",
    "TwentyNewsgroupsClustering": "TwentyNewsgroupsCl",
    "EmotionClassification": "EmotionCls",
    "SICK-R": "SICK-R",
    "MTOPDomainClassification": "MTOPDomainCls",
}

def load_results(data_dir):
    path = os.path.join(data_dir, "analysis_results.json")
    with open(path, "r") as f:
        return json.load(f)


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


def classify_task_category(task_name):
    """Map a task to its high-level MTEB category."""
    for category, tasks in TASK_CATEGORIES.items():
        if task_name in tasks:
            return category
    return "Unknown"


def wrap_task_name(task_name, max_lines=2):
    """Wrap CamelCase task names without inserting spaces."""
    if len(task_name) <= 12:
        return task_name

    pieces = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", task_name)
    if len(pieces) <= 1:
        return task_name

    if max_lines <= 1:
        return task_name

    target_lines = min(max_lines, len(pieces))
    total_len = sum(len(piece) for piece in pieces)
    target_len = max(total_len / target_lines, max(len(piece) for piece in pieces))

    lines = []
    current = ""
    remaining_lines = target_lines
    for index, piece in enumerate(pieces):
        remaining_pieces = len(pieces) - index
        candidate = current + piece
        force_break = remaining_pieces == remaining_lines
        if current and (len(candidate) > target_len * 1.12) and not force_break:
            lines.append(current)
            current = piece
            remaining_lines -= 1
        else:
            current = candidate
            if force_break:
                lines.append(current)
                current = ""
                remaining_lines -= 1
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        head = lines[:max_lines - 1]
        tail = "".join(lines[max_lines - 1:])
        lines = head + [tail]

    return "\n".join(lines)


def shorten_task_name(task_name, max_len=16):
    """Compact task labels enough to allow larger publication-safe fonts."""
    replacements = [
        ("PairClassification", "PairCls"),
        ("SentimentExtraction", "SentExt"),
        ("Counterfactual", "Counterf"),
        ("DuplicateQuestions", "DupQs"),
        ("Conversations", "Convs"),
        ("Classification", "Cls"),
        ("ClusteringS2S", "ClustS2S"),
        ("Clustering", "Clust"),
        ("Retrieval", "Retr"),
        ("Reranking", "RR"),
        ("Questions", "Qs"),
        ("Newsgroups", "NewsGrp"),
        ("Benchmark", "Bench"),
        ("Scenario", "Scen"),
        ("Intent", "Int"),
        ("Domain", "Dom"),
        ("English", "Eng"),
    ]
    short_name = task_name
    for old, new in replacements:
        short_name = short_name.replace(old, new)
    return short_name[:max_len]


def load_raw_transfer_matrix(model_name):
    """Load the full donor-target matrix from raw task_similar outputs when available."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(repo_root, "data", "task_similar", f"{model_name}.json")
    if not os.path.exists(raw_path):
        return None

    with open(raw_path, "r") as f:
        raw_data = json.load(f)

    tasks = sorted(set(raw_data) | {target for targets in raw_data.values() for target in targets})
    self_scores = {}
    for task in tasks:
        self_score = raw_data.get(task, {}).get(task)
        if isinstance(self_score, (int, float)) and self_score != 0:
            self_scores[task] = self_score

    if len(self_scores) != len(tasks):
        return None

    matrix = np.full((len(tasks), len(tasks)), np.nan)
    for i, donor in enumerate(tasks):
        donor_scores = raw_data.get(donor, {})
        for j, target in enumerate(tasks):
            score = donor_scores.get(target)
            self_score = self_scores.get(target)
            if isinstance(score, (int, float)) and self_score:
                matrix[i, j] = score / self_score

    return {"tasks": tasks, "self_scores": self_scores, "matrix": matrix}


def build_transfer_matrix_from_summary(model_data):
    """Fallback for legacy summaries that already contain donor->target scores."""
    tasks = model_data["tasks"]
    matrix = np.full((len(tasks), len(tasks)), np.nan)
    self_scores = model_data.get("self_transfer", {})
    for i, donor in enumerate(tasks):
        donor_scores = model_data.get(donor, {})
        for j, target in enumerate(tasks):
            score = donor_scores.get(target)
            self_score = self_scores.get(target)
            if isinstance(score, (int, float)) and isinstance(self_score, (int, float)) and self_score != 0:
                matrix[i, j] = score / self_score

    return {"tasks": tasks, "self_scores": self_scores, "matrix": matrix}


def load_analyze_defaults(model_name):
    """Load default unpruned task scores used to compute retention percentages."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analyze_path = os.path.join(repo_root, "data", "analyze", f"{model_name}.json")
    if not os.path.exists(analyze_path):
        return {}

    with open(analyze_path, "r") as f:
        analyze_data = json.load(f)

    defaults = {}
    for task_name, task_data in analyze_data.get("task_name", {}).items():
        score = task_data.get("defult_score")
        if isinstance(score, (int, float)) and score != 0:
            defaults[task_name] = score
    return defaults


def fig1_optimized_vs_random(results, output_dir):
    """Figure 1: Optimized vs Random gap distribution."""
    set_style()
    gap_data = results["optimized_vs_random"]

    models = sorted(gap_data.keys())
    all_gaps = {dim: [] for dim in [64, 128, 256, 512]}

    for model_name in models:
        for task_name, task_data in gap_data[model_name]["tasks"].items():
            for dim_str, gap in task_data.get("gap", {}).items():
                dim = int(dim_str)
                if dim in all_gaps:
                    all_gaps[dim].append(gap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: box plot of gaps per dimension
    dims = [64, 128, 256, 512]
    gap_values = [all_gaps[d] for d in dims]
    bp = axes[0].boxplot(gap_values, tick_labels=[str(d) for d in dims], patch_artist=True)
    colors = ['#4ECDC4', '#45B7D1', '#5B8DEE', '#7C6EF0']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference')
    axes[0].set_xlabel('Target Dimension')
    axes[0].set_ylabel('Gap (Optimized - Random) %')
    axes[0].set_title('(a) Optimized vs Random Selection Gap')
    axes[0].legend()

    # Right: per-model summary
    model_gaps = {}
    for model_name in models:
        gaps = []
        for task_name, task_data in gap_data[model_name]["tasks"].items():
            for dim_str, gap in task_data.get("gap", {}).items():
                if int(dim_str) == 256:
                    gaps.append(gap)
        if gaps:
            model_gaps[model_name] = np.mean(gaps)

    sorted_models = sorted(model_gaps.items(), key=lambda x: x[1])
    names = [m[0] for m in sorted_models]
    vals = [m[1] for m in sorted_models]
    colors_bar = ['#E74C3C' if v < 0 else '#27AE60' for v in vals]
    axes[1].barh(range(len(names)), vals, color=colors_bar, alpha=0.8)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Mean Gap at dim=256 (%)')
    axes[1].set_title('(b) Per-Model Optimized-Random Gap')

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_optimized_vs_random.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig2_cross_task_heatmap(results, output_dir):
    """Figure 2: Cross-task transfer retention heatmap."""
    set_style()
    transfer_data = results["cross_task_transfer"]
    generic_output_name = "fig2_transfer_heatmap.png"

    for model_name, model_data in transfer_data.items():
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_path = os.path.join(repo_root, "data", "task_similar", f"{model_name}.json")
        if os.path.exists(raw_path):
            with open(raw_path, "r") as f:
                raw_transfer = json.load(f)
        else:
            raw_transfer = None

        defaults = load_analyze_defaults(model_name)
        if raw_transfer:
            ordered_tasks = [task for task in REFERENCE_TRANSFER_ORDER if task in raw_transfer and task in defaults]
            if not ordered_tasks:
                all_tasks = sorted(set(raw_transfer) | {target for targets in raw_transfer.values() for target in targets})
                ordered_tasks = [task for task in all_tasks if task in defaults and task != "STS17"]

            matrix_sorted = np.full((len(ordered_tasks), len(ordered_tasks)), np.nan)
            for row_index, donor in enumerate(ordered_tasks):
                donor_scores = raw_transfer.get(donor, {})
                for col_index, target in enumerate(ordered_tasks):
                    score = donor_scores.get(target)
                    baseline = defaults.get(target)
                    if isinstance(score, (int, float)) and isinstance(baseline, (int, float)) and baseline != 0:
                        matrix_sorted[row_index, col_index] = score / baseline * 100.0
            tasks_sorted = ordered_tasks
        else:
            matrix_source = build_transfer_matrix_from_summary(model_data)
            tasks_sorted = [task for task in REFERENCE_TRANSFER_ORDER if task in matrix_source["tasks"] and task != "STS17"]
            if not tasks_sorted:
                tasks_sorted = [task for task in matrix_source["tasks"] if task != "STS17"]
            index_lookup = {task: i for i, task in enumerate(matrix_source["tasks"])}
            picked_indices = [index_lookup[task] for task in tasks_sorted]
            matrix_sorted = matrix_source["matrix"][picked_indices][:, picked_indices] * 100.0

        n = len(tasks_sorted)
        if n == 0:
            print(f"Skipped: no aligned transfer tasks for {model_name}")
            continue
        display_labels = [FIG2_DISPLAY_LABELS.get(task, shorten_task_name(task, max_len=18)) for task in tasks_sorted]

        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        tick_fontsize = 10.0
        axis_label_fontsize = 13.0
        title_fontsize = 13.4
        annotation_fontsize = 6.3
        colorbar_label_fontsize = 11.8
        colorbar_tick_fontsize = 9.2

        fig, ax = plt.subplots(figsize=(13.8, 12.0))
        cmap = LinearSegmentedColormap.from_list("retention_pct", ["#E74C3C", "#F39C12", "#30B05C"])
        im = ax.imshow(matrix_sorted, cmap=cmap, vmin=85.0, vmax=105.0, aspect="auto", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label("Retention (%)", fontsize=colorbar_label_fontsize, labelpad=18)
        cbar.set_ticks(np.arange(85.0, 105.1, 2.5))
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(
            display_labels,
            rotation=90,
            fontsize=tick_fontsize,
            ha="center",
            va="top",
        )
        ax.set_yticklabels(display_labels, fontsize=tick_fontsize)
        ax.tick_params(axis="x", pad=6)
        ax.tick_params(axis="y", pad=6)
        ax.set_xlabel("Target Task", fontsize=axis_label_fontsize, labelpad=10)
        ax.set_ylabel("Source Task (importance ranking donor)", fontsize=axis_label_fontsize, labelpad=26)
        ax.set_title(f"{display_name} Cross-Task Dimension Importance Transfer",
                     fontsize=title_fontsize, pad=10)
        ax.grid(False)

        for i, task in enumerate(tasks_sorted):
            category = classify_task_category(task)
            color = TASK_CATEGORY_COLORS.get(category)
            if color:
                ax.get_yticklabels()[i].set_color(color)
                ax.get_xticklabels()[i].set_color(color)
            ax.get_yticklabels()[i].set_multialignment("right")

        # Match the reference figure: annotate the self-transfer diagonal only.
        for i in range(n):
            diagonal_value = matrix_sorted[i, i]
            if np.isfinite(diagonal_value):
                ax.text(i, i, f"{diagonal_value:.0f}", ha="center", va="center",
                        fontsize=annotation_fontsize, color="black")

        fig.subplots_adjust(left=0.25, bottom=0.29, right=0.90, top=0.92)
        path = os.path.join(output_dir, f"fig2_transfer_heatmap_{model_name}.png")
        plt.savefig(path)
        plt.savefig(os.path.splitext(path)[0] + ".pdf")
        if model_name == "gte-large-en-v1.5":
            generic_path = os.path.join(output_dir, generic_output_name)
            plt.savefig(generic_path)
            plt.savefig(os.path.splitext(generic_path)[0] + ".pdf")
            print(f"Saved: {generic_path}")
        plt.close()
        print(f"Saved: {path}")


def fig3_weak_vs_strong_donors(results, output_dir):
    """Figure 3: Weak vs Strong donor quality comparison."""
    set_style()
    ws_data = results.get("weak_vs_strong_donors", {})

    if not ws_data:
        print("No weak vs strong donor data available")
        return

    models = sorted(ws_data.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: grouped bar chart
    x = np.arange(n_models)
    width = 0.25

    weak_means = [ws_data[m]["weak_donor_retention"]["mean"] for m in models]
    weak_ci_low = [ws_data[m]["weak_donor_retention"]["ci_low"] for m in models]
    weak_ci_high = [ws_data[m]["weak_donor_retention"]["ci_high"] for m in models]

    mid_means = [ws_data[m]["mid_donor_retention"]["mean"] for m in models]
    mid_ci_low = [ws_data[m]["mid_donor_retention"]["ci_low"] for m in models]
    mid_ci_high = [ws_data[m]["mid_donor_retention"]["ci_high"] for m in models]

    strong_means = [ws_data[m]["strong_donor_retention"]["mean"] for m in models]
    strong_ci_low = [ws_data[m]["strong_donor_retention"]["ci_low"] for m in models]
    strong_ci_high = [ws_data[m]["strong_donor_retention"]["ci_high"] for m in models]

    weak_err = [[w - l for w, l in zip(weak_means, weak_ci_low)],
                [h - w for w, h in zip(weak_means, weak_ci_high)]]
    mid_err = [[m - l for m, l in zip(mid_means, mid_ci_low)],
               [h - m for m, h in zip(mid_means, mid_ci_high)]]
    strong_err = [[s - l for s, l in zip(strong_means, strong_ci_low)],
                  [h - s for s, h in zip(strong_means, strong_ci_high)]]

    short_models = [m[:20] for m in models]
    axes[0].bar(x - width, weak_means, width, yerr=weak_err, label='Weak Donors',
                color='#E74C3C', alpha=0.8, capsize=3)
    axes[0].bar(x, mid_means, width, yerr=mid_err, label='Mid Donors',
                color='#F39C12', alpha=0.8, capsize=3)
    axes[0].bar(x + width, strong_means, width, yerr=strong_err, label='Strong Donors',
                color='#27AE60', alpha=0.8, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_models, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Average Cross-Task Retention')
    axes[0].set_title('(a) Donor Quality by Task Strength Quartile')
    axes[0].legend()
    axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)

    # Right: effect size
    effect_sizes = [ws_data[m]["effect_size_weak_vs_strong"] for m in models]
    colors = ['#27AE60' if es > 0 else '#E74C3C' for es in effect_sizes]
    axes[1].barh(range(n_models), effect_sizes, color=colors, alpha=0.8)
    axes[1].set_yticks(range(n_models))
    axes[1].set_yticklabels(short_models, fontsize=8)
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_xlabel("Cohen's d (Weak - Strong)")
    axes[1].set_title('(b) Effect Size: Weak vs Strong Donor Retention')
    axes[1].set_xlim(-1.5, 1.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_weak_vs_strong_donors.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig4_dimension_correlation(results, output_dir):
    """Figure 4: Distribution of pairwise ranking correlations."""
    set_style()
    corr_data = results.get("dimension_correlation", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (model_name, model_data) in enumerate(sorted(corr_data.items())):
        ax = axes[i % 2]
        rhos = [v["rho"] for v in model_data["pairwise"].values()]
        p_vals = [v["p_value"] for v in model_data["pairwise"].values()]

        ax.hist(rhos, bins=30, alpha=0.7, color='#5B8DEE', edgecolor='white')
        ax.axvline(x=np.mean(rhos), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(rhos):.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Spearman Rank Correlation (rho)')
        ax.set_ylabel('Count')
        ax.set_title(f'{model_name}\nPairwise Dimension Ranking Correlation')
        ax.legend()

        # Annotate significance
        n_sig = sum(1 for p in p_vals if p < 0.05)
        n_total = len(p_vals)
        ax.text(0.95, 0.95, f'{n_sig}/{n_total} significant\n(p < 0.05)',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_dimension_correlation.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig5_category_transfer(results, output_dir):
    """Figure 5: Category-level transfer matrix."""
    set_style()
    cat_data = results.get("category_transfer", {})

    categories = ["Classification", "Clustering", "STS", "Retrieval", "Reranking",
                  "PairClassification", "Summarization"]

    # Aggregate across models
    agg_matrix = defaultdict(lambda: defaultdict(list))
    for model_name, model_cats in cat_data.items():
        for donor_cat, target_cats in model_cats.items():
            for target_cat, stats in target_cats.items():
                if isinstance(stats, dict) and "mean" in stats:
                    agg_matrix[donor_cat][target_cat].append(stats["mean"])

    n_cats = len(categories)
    matrix = np.zeros((n_cats, n_cats))
    for i, dc in enumerate(categories):
        for j, tc in enumerate(categories):
            vals = agg_matrix[dc][tc]
            matrix[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = LinearSegmentedColormap.from_list("rg", ["#E74C3C", "#F39C12", "#27AE60"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.7, vmax=1.1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Average Retention Ratio')

    ax.set_xticks(range(n_cats))
    ax.set_yticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_title('Category-Level Cross-Task Dimension Transfer\n(Averaged across models)', fontsize=14)

    # Annotate cells
    for i in range(n_cats):
        for j in range(n_cats):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.85 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_category_transfer.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--output_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/paper/figures")
    parser.add_argument("--data_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/data/experiment_results")
    args = parser.parse_args()

    results = load_results(args.data_dir)

    print("Generating figures...")
    fig1_optimized_vs_random(results, args.output_dir)
    fig2_cross_task_heatmap(results, args.output_dir)
    fig3_weak_vs_strong_donors(results, args.output_dir)
    fig4_dimension_correlation(results, args.output_dir)
    fig5_category_transfer(results, args.output_dir)
    print("All figures generated!")


if __name__ == "__main__":
    main()

"""
Rebuild Figure 5 from raw task-level transfer data with strict verification.

Workflow:
1. Only use models that exist in both data/analyze and data/task_similar.
2. Verify every transfer task is backed by chunk-ranking data in analyze/*.json.
3. Compute task-level retention ratios against each target task's self-transfer.
4. Detect severe target-task outliers and exclude them before category aggregation.
5. Regenerate paper/figures/fig5_category_transfer.png with a publication-friendly palette.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parent.parent
ANALYZE_DIR = PROJECT_DIR / "data" / "analyze"
TASK_SIMILAR_DIR = PROJECT_DIR / "data" / "task_similar"
OUTPUT_PATH = PROJECT_DIR / "paper" / "figures" / "fig5_category_transfer.png"
SUMMARY_PATH = PROJECT_DIR / "artifacts" / "fig5_category_transfer_verified.json"

CATEGORY_TASKS = {
    "Classification": [
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "BiorxivClusteringS2S",
        "MedrxivClusteringS2S",
        "TwentyNewsgroupsClustering",
    ],
    "STS": [
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STS17",
        "STSBenchmark",
    ],
    "Retrieval": [
        "ArguAna",
        "CQADupstackEnglishRetrieval",
        "NFCorpus",
        "SCIDOCS",
        "SciFact",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "PairClassification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Summarization": [
        "SummEval",
    ],
}

CATEGORY_ORDER = [
    "Classification",
    "Clustering",
    "STS",
    "Retrieval",
    "Reranking",
    "PairClassification",
    "Summarization",
]

TASK_TO_CATEGORY = {
    task_name: category
    for category, task_names in CATEGORY_TASKS.items()
    for task_name in task_names
}

# User-approved exclusions/inclusions for the paper figure.
FORCED_EXCLUDED_TASKS = {"STS17"}
FORCED_INCLUDED_TASKS = {"SCIDOCS"}
FORCED_EXCLUDED_MODELS = {"roberta-large"}

# Avoid flagging mild edge cases; only record tasks that are both outside the IQR fence
# and far from the category median in absolute percentage points.
SEVERE_OUTLIER_MIN_DEVIATION = 10.0


def load_json_dir(path: Path) -> dict[str, dict]:
    data = {}
    for file_path in sorted(path.glob("*.json")):
        with file_path.open("r") as f:
            data[file_path.stem] = json.load(f)
    return data


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def verify_model_alignment(
    analyze_model: dict,
    task_similar_model: dict,
) -> dict[str, object]:
    analyze_tasks = analyze_model.get("task_name", {})
    task_similar_tasks = set(task_similar_model.keys())

    missing_in_analyze = sorted(task_similar_tasks - set(analyze_tasks.keys()))
    missing_chunk_rankings = []

    for task_name in sorted(task_similar_tasks & set(analyze_tasks.keys())):
        task_data = analyze_tasks[task_name]
        split_data = task_data.get("split_win_size", {}).get("2", {})
        chunk_result = split_data.get("chunk_result")
        if not isinstance(chunk_result, list) or not chunk_result:
            missing_chunk_rankings.append(task_name)

    return {
        "task_count_task_similar": len(task_similar_tasks),
        "task_count_analyze": len(analyze_tasks),
        "missing_in_analyze": missing_in_analyze,
        "missing_chunk_rankings": missing_chunk_rankings,
        "verified": not missing_in_analyze and not missing_chunk_rankings and len(task_similar_tasks) >= 10,
    }


def compute_target_task_means(
    models: list[str],
    task_similar_data: dict[str, dict],
) -> dict[str, dict[str, float]]:
    category_means = {}

    for category, task_names in CATEGORY_TASKS.items():
        if len(task_names) < 3:
            continue

        per_target = {}
        for target_task in task_names:
            ratios = []
            for model_name in models:
                model_data = task_similar_data[model_name]
                if target_task not in model_data or target_task not in model_data[target_task]:
                    continue

                self_score = model_data[target_task][target_task]
                if not is_number(self_score) or self_score == 0:
                    continue

                for donor_task, targets in model_data.items():
                    if donor_task == target_task:
                        continue
                    score = targets.get(target_task)
                    if is_number(score):
                        ratios.append((score / self_score) * 100.0)

            if ratios:
                per_target[target_task] = float(np.mean(ratios))

        if per_target:
            category_means[category] = per_target

    return category_means


def detect_severe_outlier_tasks(
    task_mean_ratios: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    outliers = []

    for category, target_means in task_mean_ratios.items():
        if len(target_means) < 3:
            continue

        values = list(target_means.values())
        q1 = statistics.quantiles(values, n=4, method="inclusive")[0]
        q3 = statistics.quantiles(values, n=4, method="inclusive")[2]
        median = statistics.median(values)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for task_name, mean_ratio in sorted(target_means.items(), key=lambda item: item[1]):
            deviation = abs(mean_ratio - median)
            if (mean_ratio < lower or mean_ratio > upper) and deviation >= SEVERE_OUTLIER_MIN_DEVIATION:
                outliers.append(
                    {
                        "task": task_name,
                        "category": category,
                        "mean_retention_pct": float(mean_ratio),
                        "category_median_pct": float(median),
                        "iqr_lower_bound_pct": float(lower),
                        "iqr_upper_bound_pct": float(upper),
                        "abs_deviation_from_median_pct": float(deviation),
                    }
                )

    return outliers


def aggregate_category_matrix(
    models: list[str],
    task_similar_data: dict[str, dict],
    excluded_tasks: set[str],
) -> tuple[np.ndarray, dict[str, dict[str, dict[str, object]]]]:
    cell_means = defaultdict(lambda: defaultdict(list))
    summary = defaultdict(lambda: defaultdict(dict))

    for model_name in models:
        model_data = task_similar_data[model_name]
        self_scores = {
            task_name: targets[task_name]
            for task_name, targets in model_data.items()
            if task_name in targets and is_number(targets[task_name])
        }

        per_model_cell = defaultdict(list)
        for donor_task, targets in model_data.items():
            if donor_task in excluded_tasks:
                continue

            donor_category = TASK_TO_CATEGORY.get(donor_task)
            if donor_category is None:
                continue

            for target_task, score in targets.items():
                if target_task in excluded_tasks:
                    continue

                target_category = TASK_TO_CATEGORY.get(target_task)
                self_score = self_scores.get(target_task)
                if target_category is None or not is_number(score) or not is_number(self_score) or self_score == 0:
                    continue

                retention_pct = (score / self_score) * 100.0
                per_model_cell[(donor_category, target_category)].append(retention_pct)

        for (donor_category, target_category), values in per_model_cell.items():
            cell_means[donor_category][target_category].append(float(np.mean(values)))

    matrix = np.full((len(CATEGORY_ORDER), len(CATEGORY_ORDER)), np.nan)
    for row_index, donor_category in enumerate(CATEGORY_ORDER):
        for col_index, target_category in enumerate(CATEGORY_ORDER):
            values = cell_means[donor_category][target_category]
            if values:
                matrix[row_index, col_index] = float(np.mean(values))
                summary[donor_category][target_category] = {
                    "mean_retention_pct": float(np.mean(values)),
                    "std_retention_pct": float(np.std(values)),
                    "n_models": len(values),
                    "per_model_means_pct": values,
                }

    return matrix, summary


def plot_matrix(
    matrix: np.ndarray,
    models: list[str],
    excluded_tasks: list[str],
    retained_outlier_tasks: list[str],
    excluded_models: list[str],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    finite_vals = matrix[np.isfinite(matrix)]
    vmin = math.floor(float(np.min(finite_vals)) - 0.5)
    vmax = math.ceil(float(np.max(finite_vals)) + 0.5)
    if vmin >= 100.0:
        vmin = 99.0
    if vmax <= 100.0:
        vmax = 101.0

    norm = TwoSlopeNorm(vmin=vmin, vcenter=100.0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "transfer_verified",
        ["#c97b73", "#f7f4ee", "#7aa889"],
    )

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER, rotation=35, ha="right")
    ax.set_yticklabels(CATEGORY_ORDER)
    ax.set_xlabel("Target task category")
    ax.set_ylabel("Donor task category")
    ax.set_title("Category-Level Cross-Task Transfer", pad=16, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(CATEGORY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CATEGORY_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not np.isfinite(value):
                continue
            text_color = "white" if value <= 93.0 or value >= 104.0 else "#1f2933"
            ax.text(
                col_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Retention vs. target self-transfer (%)")
    cbar.set_ticks(np.arange(vmin, vmax + 0.1, 2.0))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    analyze_data = load_json_dir(ANALYZE_DIR)
    task_similar_data = load_json_dir(TASK_SIMILAR_DIR)

    model_alignment = {}
    verified_models = []
    for model_name in sorted(set(analyze_data) & set(task_similar_data)):
        alignment = verify_model_alignment(analyze_data[model_name], task_similar_data[model_name])
        model_alignment[model_name] = alignment
        if alignment["verified"]:
            verified_models.append(model_name)

    figure_models = [
        model_name for model_name in verified_models if model_name not in FORCED_EXCLUDED_MODELS
    ]

    task_mean_ratios = compute_target_task_means(figure_models, task_similar_data)
    detected_outliers = detect_severe_outlier_tasks(task_mean_ratios)

    retained_outlier_tasks = sorted(
        outlier["task"] for outlier in detected_outliers if outlier["task"] in FORCED_INCLUDED_TASKS
    )
    excluded_tasks = sorted(
        {outlier["task"] for outlier in detected_outliers if outlier["task"] not in FORCED_INCLUDED_TASKS}
        | FORCED_EXCLUDED_TASKS
    )
    outliers = []
    for outlier in detected_outliers:
        outlier = dict(outlier)
        outlier["excluded_from_figure"] = outlier["task"] in excluded_tasks
        outlier["retained_in_figure"] = outlier["task"] in retained_outlier_tasks
        outliers.append(outlier)

    matrix, matrix_summary = aggregate_category_matrix(
        models=figure_models,
        task_similar_data=task_similar_data,
        excluded_tasks=set(excluded_tasks),
    )

    plot_matrix(
        matrix=matrix,
        models=figure_models,
        excluded_tasks=excluded_tasks,
        retained_outlier_tasks=retained_outlier_tasks,
        excluded_models=sorted(FORCED_EXCLUDED_MODELS),
        output_path=OUTPUT_PATH,
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w") as f:
        json.dump(
            {
                "verified_models": verified_models,
                "figure_models": figure_models,
                "model_alignment": model_alignment,
                "task_mean_retention_pct_before_exclusion": task_mean_ratios,
                "excluded_outlier_tasks": outliers,
                "forced_excluded_tasks": sorted(FORCED_EXCLUDED_TASKS),
                "forced_included_tasks": sorted(FORCED_INCLUDED_TASKS),
                "forced_excluded_models": sorted(FORCED_EXCLUDED_MODELS),
                "retained_outlier_tasks": retained_outlier_tasks,
                "category_order": CATEGORY_ORDER,
                "matrix_after_exclusion_pct": matrix.tolist(),
                "matrix_summary_after_exclusion": matrix_summary,
                "output_figure": str(OUTPUT_PATH),
            },
            f,
            indent=2,
        )

    print(f"Verified models: {len(verified_models)} -> {verified_models}")
    print(f"Figure models: {len(figure_models)} -> {figure_models}")
    print(f"Excluded severe outlier tasks: {excluded_tasks}")
    print(f"Retained diagnostic outliers: {retained_outlier_tasks}")
    print(f"Saved figure: {OUTPUT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

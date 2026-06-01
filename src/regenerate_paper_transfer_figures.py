#!/usr/bin/env python3
"""Regenerate transfer heatmap paper figures as PNG and vector PDF.

The main Figure 2 heatmap is generated from the original task-similarity
matrix under data/task_similar/ and normalized by the full-dimension scores
under data/analyze/. The combined category-level Figure 5 is intentionally
not regenerated here: the original plotting entry point is not present in
the repository, so the released raster figure should be preserved.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/prune-to-prosper-mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = (
    ROOT
    / "Beyond_Redundancy__Diagnosing_Information_Distribution_in_Text_Embeddings_via_Task_Aware_Dimension_Selection"
)
FIG_DIR = PAPER_DIR / "figures"
TRANSFER_DIR = ROOT / "data" / "task_similar"
ANALYZE_DIR = ROOT / "data" / "analyze"

EXCLUDE_TASKS = {"STS17"}

TASK_CATEGORIES = {
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
    "Pair Classification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions",
        "SciDocsRR",
        "StackOverflowDupQuestions",
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
    "Summarization": ["SummEval"],
    "Retrieval": [
        "ArguAna",
        "CQADupstackEnglishRetrieval",
        "NFCorpus",
        "SCIDOCS",
        "SciFact",
    ],
}

TASK_TO_CAT = {task: cat for cat, tasks in TASK_CATEGORIES.items() for task in tasks}
CAT_ORDER = [
    "Classification",
    "Clustering",
    "Pair Classification",
    "Reranking",
    "STS",
    "Summarization",
    "Retrieval",
]
CAT_SHORT = {
    "Classification": "Cls.",
    "Clustering": "Clust.",
    "Pair Classification": "Pair Cls.",
    "Reranking": "Rerank.",
    "STS": "STS",
    "Summarization": "Summ.",
    "Retrieval": "Retr.",
}

CATEGORY_COLORS = {
    "Classification": "#3498DB",
    "Clustering": "#E67E22",
    "Retrieval": "#1ABC9C",
    "Reranking": "#E74C3C",
    "Pair Classification": "#F1C40F",
    "STS": "#9B59B6",
    "Summarization": "#2ECC71",
}

GTE_TASK_ORDER = [
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

TASK_SHORT = {
    "AmazonCounterfactualClassification": "AmzCntrfCls.",
    "AmazonReviewsClassification": "AmzRevCls.",
    "ArguAna": "ArguAna",
    "AskUbuntuDupQuestions": "AskUbuntuDupQ.",
    "BIOSSES": "BIOSSES",
    "Banking77Classification": "Banking77Cls.",
    "BiorxivClusteringS2S": "BiorxivClustS2S.",
    "CQADupstackEnglishRetrieval": "CQADupEngRet.",
    "EmotionClassification": "EmotionCls.",
    "ImdbClassification": "ImdbCls.",
    "MTOPDomainClassification": "MTOPDomainCls.",
    "MTOPIntentClassification": "MTOPIntentCls.",
    "MassiveIntentClassification": "MassiveIntentCls.",
    "MassiveScenarioClassification": "MassiveScenarioCls.",
    "MedrxivClusteringS2S": "MedrxivClustS2S.",
    "NFCorpus": "NFCorpus",
    "SCIDOCS": "SCIDOCS",
    "SICK-R": "SICK-R",
    "STS12": "STS12",
    "STS13": "STS13",
    "STS14": "STS14",
    "STS15": "STS15",
    "STS16": "STS16",
    "STSBenchmark": "STSBenchmark",
    "SciDocsRR": "SciDocsRR",
    "SciFact": "SciFact",
    "SprintDuplicateQuestions": "SprintDupQ.",
    "StackOverflowDupQuestions": "StackOverflowDupQ.",
    "SummEval": "SummEval",
    "ToxicConversationsClassification": "ToxicConvCls.",
    "TweetSentimentExtractionClassification": "TweetSentExtCls.",
    "TwentyNewsgroupsClustering": "TwentyNewsClust.",
    "TwitterSemEval2015": "TwitterSemEval2015",
    "TwitterURLCorpus": "TwitterURLCorpus",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return math.nan
    ra = rankdata(np.asarray(a, dtype=float))
    rb = rankdata(np.asarray(b, dtype=float))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def transfer_retention(
    transfer_data: dict,
    analyze_tasks: dict,
    source_task: str,
    target_task: str,
    dim: int = 256,
) -> float:
    target_full = analyze_tasks.get(target_task, {}).get("defult_score")
    score = transfer_data.get(source_task, {}).get(target_task)
    if isinstance(score, dict):
        score = score.get(str(dim))
    if target_full is None or target_full <= 0 or score is None:
        return math.nan
    return float(score) / float(target_full) * 100.0


def abbreviate_task(task: str) -> str:
    return TASK_SHORT.get(task, task)


def figure2_transfer_heatmap() -> None:
    set_style()
    transfer_data = load_json(TRANSFER_DIR / "gte-large-en-v1.5.json")
    analyze_tasks = load_json(ANALYZE_DIR / "gte-large-en-v1.5.json")["task_name"]

    available = set(transfer_data) & set(analyze_tasks)
    tasks = [t for t in GTE_TASK_ORDER if t in available and t not in EXCLUDE_TASKS]
    matrix = np.full((len(tasks), len(tasks)), np.nan)
    for i, source in enumerate(tasks):
        for j, target in enumerate(tasks):
            matrix[i, j] = transfer_retention(transfer_data, analyze_tasks, source, target)

    finite = matrix[np.isfinite(matrix)]
    vmin = 85.0
    vmax = 105.0
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#F2F2F2")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=100.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12.0, 10.6))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
    labels = [abbreviate_task(t) for t in tasks]
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=6.4)
    ax.set_yticklabels(labels, fontsize=6.4)
    for label, task in zip(ax.get_xticklabels(), tasks):
        label.set_color(CATEGORY_COLORS.get(TASK_TO_CAT.get(task), "#333333"))
    for label, task in zip(ax.get_yticklabels(), tasks):
        label.set_color(CATEGORY_COLORS.get(TASK_TO_CAT.get(task), "#333333"))

    for i in range(len(tasks)):
        if np.isfinite(matrix[i, i]):
            ax.text(i, i, f"{matrix[i, i]:.0f}", ha="center", va="center",
                    fontsize=6.0, color="black")

    ax.set_xlabel("Target Task")
    ax.set_ylabel("Source Task (importance ranking donor)")
    ax.set_title("GTE-Large Cross-Task Dimension Importance Transfer", fontsize=17)
    ax.set_xticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.25, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Retention (%)")
    fig.tight_layout()
    save(fig, "fig2_transfer_heatmap.png")


def load_transfer_models() -> list[tuple[str, dict, dict]]:
    models = []
    for path in sorted(TRANSFER_DIR.glob("*.json")):
        if "_by_dim" in path.name:
            continue
        model_key = path.stem
        analyze_path = ANALYZE_DIR / f"{model_key}.json"
        if not analyze_path.exists():
            continue
        models.append(
            (
                model_key,
                load_json(path),
                load_json(analyze_path)["task_name"],
            )
        )
    return models


def category_matrices() -> tuple[np.ndarray, np.ndarray]:
    retention_cells = {
        (src_cat, tgt_cat): []
        for src_cat in CAT_ORDER
        for tgt_cat in CAT_ORDER
    }
    rho_cells = {
        (src_cat, tgt_cat): []
        for src_cat in CAT_ORDER
        for tgt_cat in CAT_ORDER
    }

    for _model_key, transfer_data, analyze_tasks in load_transfer_models():
        tasks = [
            task
            for task in analyze_tasks
            if task not in EXCLUDE_TASKS
            and task in TASK_TO_CAT
            and task in transfer_data
            and analyze_tasks[task]
            .get("split_win_size", {})
            .get("2", {})
            .get("chunk_result")
        ]
        chunk_scores = {
            task: analyze_tasks[task]["split_win_size"]["2"]["chunk_result"]
            for task in tasks
        }
        for source in tasks:
            for target in tasks:
                if source == target:
                    continue
                source_cat = TASK_TO_CAT[source]
                target_cat = TASK_TO_CAT[target]
                ret = transfer_retention(transfer_data, analyze_tasks, source, target)
                if np.isfinite(ret):
                    retention_cells[(source_cat, target_cat)].append(ret)
                rho = spearman(chunk_scores[source], chunk_scores[target])
                if np.isfinite(rho):
                    rho_cells[(source_cat, target_cat)].append(rho)

    retention = np.full((len(CAT_ORDER), len(CAT_ORDER)), np.nan)
    rho = np.full_like(retention, np.nan)
    for i, source_cat in enumerate(CAT_ORDER):
        for j, target_cat in enumerate(CAT_ORDER):
            vals = retention_cells[(source_cat, target_cat)]
            if vals:
                retention[i, j] = float(np.mean(vals))
            rhos = rho_cells[(source_cat, target_cat)]
            if rhos:
                rho[i, j] = float(np.mean(rhos))
    return retention, rho


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                text = format(value, fmt)
                color = "white" if abs(value) > 105 or value < -0.08 else "#222222"
            else:
                text = "N/A"
                color = "#666666"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)


def figure5_transfer_and_rho_combined() -> None:
    raise RuntimeError(
        "fig5_transfer_and_rho_combined has no original plotting script in src/. "
        "Use the preserved raster figure from the paper package instead."
    )


def main() -> None:
    figure2_transfer_heatmap()


if __name__ == "__main__":
    main()

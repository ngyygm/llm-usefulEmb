"""
Generate transfer-related figures from rebuilt cross-task transfer outputs.

Main-text policy:
* fig_dim_scaling.png shows only aggregate transfer vs. random behavior.

Appendix policy:
* fig5_category_transfer.png gives the dim=256 category matrix.
* fig2_transfer_heatmap.png gives the GTE-Large task matrix at dim=256.
* fig_transfer_paradox.png gives off-diagonal rank-correlation and gap details.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_DIR / "paper" / "figures"
DATA_DIR = PROJECT_DIR / "data" / "analysis_results_multidim"

DATA_PATH = DATA_DIR / "transfer_records.csv"
SUMMARY_PATH = DATA_DIR / "summary.json"
RANK_PATH = DATA_DIR / "rank_correlations.csv"

DIMS = [16, 32, 64, 128, 256, 512]

C_RANDOM = "#5f6b73"
C_TRANSFER = "#2f6f9f"
C_GAP = "#b9892f"
C_TEXT = "#24313a"

CATEGORY_ORDER = [
    "Classification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
]

CATEGORY_LABELS = {
    "Classification": "Classif.",
    "Clustering": "Cluster.",
    "PairClassification": "Pair cls.",
    "Reranking": "Rerank.",
    "Retrieval": "Retrieval",
    "STS": "STS",
    "Summarization": "Summar.",
}

TASK_ORDER = [
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
    "BiorxivClusteringS2S",
    "MedrxivClusteringS2S",
    "TwentyNewsgroupsClustering",
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    "AskUbuntuDupQuestions",
    "SciDocsRR",
    "StackOverflowDupQuestions",
    "ArguAna",
    "CQADupstackEnglishRetrieval",
    "NFCorpus",
    "SCIDOCS",
    "SciFact",
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SummEval",
]

TASK_LABELS = {
    "AmazonCounterfactualClassification": "AmzCntrf",
    "AmazonReviewsClassification": "AmzReviews",
    "Banking77Classification": "Banking77",
    "EmotionClassification": "Emotion",
    "ImdbClassification": "Imdb",
    "MTOPDomainClassification": "MTOPDom",
    "MTOPIntentClassification": "MTOPInt",
    "MassiveIntentClassification": "MassiveInt",
    "MassiveScenarioClassification": "MassiveScen",
    "ToxicConversationsClassification": "ToxicConv",
    "TweetSentimentExtractionClassification": "TweetSent",
    "BiorxivClusteringS2S": "Biorxiv",
    "MedrxivClusteringS2S": "Medrxiv",
    "TwentyNewsgroupsClustering": "20News",
    "SprintDuplicateQuestions": "SprintDup",
    "TwitterSemEval2015": "TwSemEval",
    "TwitterURLCorpus": "TwURL",
    "AskUbuntuDupQuestions": "AskUbuntu",
    "SciDocsRR": "SciDocsRR",
    "StackOverflowDupQuestions": "StackOverflow",
    "ArguAna": "ArguAna",
    "CQADupstackEnglishRetrieval": "CQADup",
    "NFCorpus": "NFCorpus",
    "SCIDOCS": "SCIDOCS",
    "SciFact": "SciFact",
    "BIOSSES": "BIOSSES",
    "SICK-R": "SICK-R",
    "STS12": "STS12",
    "STS13": "STS13",
    "STS14": "STS14",
    "STS15": "STS15",
    "STS16": "STS16",
    "STSBenchmark": "STSBench",
    "SummEval": "SummEval",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(DATA_PATH)
    if df["is_self_transfer"].dtype == object:
        df["is_self_transfer"] = df["is_self_transfer"].astype(str).str.lower().eq("true")
    rank_df = pd.read_csv(RANK_PATH)
    with SUMMARY_PATH.open("r") as f:
        summary = json.load(f)
    return df, rank_df, summary


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.8)


def save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def summary_rows(summary: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(summary[key]).sort_values("dim")


def generate_fig_dim_scaling(summary: dict) -> None:
    """Main-text aggregate transfer figure."""
    pooled = summary_rows(summary, "pooled_by_dim")
    dims = pooled["dim"].to_numpy()
    transfer = pooled["transfer_mean_pct"].to_numpy()
    random = pooled["random_mean_pct"].to_numpy()
    gaps = pooled["gap_mean_pp"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        dims,
        transfer,
        marker="o",
        markersize=4,
        linewidth=2.0,
        color=C_TRANSFER,
        label="Transfer",
    )
    ax.plot(
        dims,
        random,
        marker="s",
        markersize=4,
        linewidth=2.0,
        color=C_RANDOM,
        label="Random",
    )
    ax.axhline(100, color="#b8b8b8", linestyle=":", linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlabel("Retained dimensions")
    ax.set_ylabel("Mean retention (%)")
    ax.set_title("(a) Aggregate transfer vs. random")
    ax.set_ylim(60, 104)
    ax.legend(loc="lower right", frameon=False)
    despine(ax)

    ax = axes[1]
    bars = ax.bar(
        np.arange(len(dims)),
        gaps,
        color=C_GAP,
        edgecolor="white",
        linewidth=0.8,
        width=0.64,
    )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Retained dimensions")
    ax.set_ylabel("Transfer - random (pp)")
    ax.set_title("(b) Aggregate gap")
    ax.set_ylim(0, max(gaps) + 0.55)
    for bar, gap in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{gap:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_TEXT,
        )
    despine(ax)

    save(fig, "fig_dim_scaling.png")


def diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "muted_transfer",
        ["#b65f5a", "#f6f1e8", "#4d9169"],
        N=256,
    )


def centered_norm(values: np.ndarray, min_span: float = 4.0) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    span = max(float(np.max(np.abs(finite - 100.0))), min_span)
    return TwoSlopeNorm(vmin=100.0 - span, vcenter=100.0, vmax=100.0 + span)


def generate_fig5_category_transfer(df: pd.DataFrame) -> None:
    """Appendix category-to-category matrix at dim=256."""
    sub = df[df["dim"] == 256]
    cat_data = sub.groupby(["ref_cat", "tgt_cat"])["retention"].mean() * 100.0

    matrix = np.full((len(CATEGORY_ORDER), len(CATEGORY_ORDER)), np.nan)
    for i, ref_cat in enumerate(CATEGORY_ORDER):
        for j, tgt_cat in enumerate(CATEGORY_ORDER):
            if (ref_cat, tgt_cat) in cat_data.index:
                matrix[i, j] = float(cat_data[(ref_cat, tgt_cat)])

    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    norm = centered_norm(matrix)
    im = ax.imshow(matrix, cmap=diverging_cmap(), norm=norm, aspect="equal")

    labels = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target category")
    ax.set_ylabel("Source category")
    ax.set_title("Category-to-category transfer at dim=256")

    ax.set_xticks(np.arange(-0.5, len(CATEGORY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CATEGORY_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            text_color = "white" if abs(val - 100.0) > 6.2 else "#1f2933"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean retention (%)")
    cbar.ax.tick_params(labelsize=8)

    save(fig, "fig5_category_transfer.png")


def generate_fig2_transfer_heatmap(df: pd.DataFrame) -> None:
    """Appendix GTE-Large 34x34 task transfer matrix at dim=256."""
    sub = df[(df["model"] == "gte-large-en-v1.5") & (df["dim"] == 256)]
    tasks = [task for task in TASK_ORDER if task in set(sub["ref"]) and task in set(sub["tgt"])]
    matrix = np.full((len(tasks), len(tasks)), np.nan)

    values = {
        (row.ref, row.tgt): row.retention * 100.0
        for row in sub[["ref", "tgt", "retention"]].itertuples(index=False)
    }
    for i, ref in enumerate(tasks):
        for j, tgt in enumerate(tasks):
            matrix[i, j] = values[(ref, tgt)]

    fig, ax = plt.subplots(figsize=(9.2, 8.0), constrained_layout=True)
    norm = centered_norm(matrix, min_span=6.0)
    im = ax.imshow(matrix, cmap=diverging_cmap(), norm=norm, aspect="auto")

    labels = [TASK_LABELS.get(task, task) for task in tasks]
    ax.set_xticks(range(len(tasks)))
    ax.set_yticks(range(len(tasks)))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Target task")
    ax.set_ylabel("Source task")
    ax.set_title("GTE-Large task-to-task transfer at dim=256")

    # Category separators make the dense matrix readable without coloring labels.
    boundaries = [11, 14, 17, 20, 25, 33]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.0)
        ax.axvline(boundary - 0.5, color="white", linewidth=1.0)

    for i in range(len(tasks)):
        val = matrix[i, i]
        ax.text(i, i, f"{val:.0f}", ha="center", va="center", fontsize=5.5, color="#111111")

    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("Retention (%)")
    cbar.ax.tick_params(labelsize=8)

    save(fig, "fig2_transfer_heatmap.png")


def generate_fig_transfer_paradox(rank_df: pd.DataFrame, summary: dict) -> None:
    """Appendix figure: off-diagonal rank disagreement and transfer/random gap."""
    offdiag = summary_rows(summary, "offdiag_only_by_dim")
    dims = offdiag["dim"].to_numpy()
    transfer = offdiag["transfer_mean_pct"].to_numpy()
    random = offdiag["random_mean_pct"].to_numpy()
    gaps = offdiag["gap_mean_pp"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), constrained_layout=True)

    ax = axes[0]
    ax.hist(rank_df["rho"], bins=44, color="#7f8fa6", edgecolor="white", linewidth=0.3)
    ax.axvline(rank_df["rho"].mean(), color=C_TRANSFER, linewidth=1.5, label="Mean")
    ax.axvline(rank_df["rho"].median(), color="#9b6b3d", linewidth=1.5, linestyle="--", label="Median")
    ax.set_xlabel("Spearman rank correlation")
    ax.set_ylabel("Task-pair count")
    ax.set_title("(a) Off-diagonal ranking agreement")
    ax.legend(frameon=False, loc="upper right")
    despine(ax)

    ax = axes[1]
    ax.plot(dims, transfer, marker="o", markersize=4, linewidth=2.0, color=C_TRANSFER, label="Transfer")
    ax.plot(dims, random, marker="s", markersize=4, linewidth=2.0, color=C_RANDOM, label="Random")
    for dim, t_val, r_val, gap in zip(dims, transfer, random, gaps):
        y = max(t_val, r_val) + 0.6
        ax.text(dim, y, f"{gap:+.1f}", ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
    ax.axhline(100, color="#b8b8b8", linestyle=":", linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlabel("Retained dimensions")
    ax.set_ylabel("Mean retention (%)")
    ax.set_title("(b) Off-diagonal transfer vs. random")
    ax.set_ylim(60, 104)
    ax.legend(frameon=False, loc="lower right")
    despine(ax)

    save(fig, "fig_transfer_paradox.png")


def main() -> None:
    apply_style()
    df, rank_df, summary = load_data()
    print(
        f"Loaded {len(df)} transfer rows, "
        f"{df['model'].nunique()} models, {df['ref'].nunique()} tasks"
    )
    generate_fig_dim_scaling(summary)
    generate_fig5_category_transfer(df)
    generate_fig2_transfer_heatmap(df)
    generate_fig_transfer_paradox(rank_df, summary)
    print("Done.")


if __name__ == "__main__":
    main()

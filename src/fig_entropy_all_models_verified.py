"""
Regenerate fig_entropy_all_models.png from the verified 34-task entropy JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "experiment_results" / "all_models_entropy.json"
OUTPUT_PATH = REPO_ROOT / "paper" / "figures" / "fig_entropy_all_models.png"

MODEL_ORDER = [
    ("gte-large-en-v1.5", "GTE-Large", "#D95F5F"),
    ("stella_en_400M_v5", "Stella", "#8F8F8F"),
    ("roberta-large-InBedder", "Roberta-\nInBedder", "#8F8F8F"),
    ("bge-m3", "BGE-M3", "#D95F5F"),
    ("instructor-large", "Instructor", "#8F8F8F"),
    ("mxbai-embed-large-v1", "MxBai-\nLarge", "#D95F5F"),
    ("Qwen3-Embedding-0.6B", "Qwen3-\nEmb", "#8F8F8F"),
    ("roberta-large", "Roberta-\nLarge", "#2E8B57"),
    ("bart-base", "BART-Base", "#2E8B57"),
]


def main() -> None:
    with INPUT_PATH.open("r") as f:
        entropy_data = json.load(f)

    labels = []
    series = []
    colors = []
    means = []

    for model_key, label, color in MODEL_ORDER:
        tasks = entropy_data[model_key]["tasks"]
        values = [task_info["normalized_entropy"] for task_info in tasks.values()]
        labels.append(label)
        series.append(values)
        colors.append(color)
        means.append(sum(values) / len(values))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    bp = ax.boxplot(
        series,
        patch_artist=True,
        widths=0.58,
        tick_labels=labels,
        showfliers=True,
        medianprops={"color": "#1f2933", "linewidth": 1.8},
        whiskerprops={"color": "#4b5563", "linewidth": 1.2},
        capprops={"color": "#4b5563", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "white",
            "markeredgecolor": "#6b7280",
            "alpha": 0.7,
        },
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("#374151")
        patch.set_linewidth(1.2)

    ax.axhline(1.0, color="#9ca3af", linestyle="--", linewidth=1.5, label="Perfectly uniform")
    ax.set_ylabel("Normalized Shannon Entropy")
    ax.set_title("Dimension Importance Uniformity Across 9 Core Models", fontweight="bold", pad=10)
    ax.set_ylim(0.88, 1.0015)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    for idx, mean_value in enumerate(means, start=1):
        ax.text(
            idx,
            0.892,
            f"{mean_value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#374151",
        )

    ax.legend(loc="lower right", framealpha=0.92, edgecolor="#d1d5db")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

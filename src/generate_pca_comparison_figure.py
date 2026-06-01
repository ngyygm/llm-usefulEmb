#!/usr/bin/env python3
"""Generate the PCA/RP/coordinate-selection comparison figure.

The pca_baseline_*.json files store retention values as percentages.  The
original Figure 19 generator treated them as fractions and multiplied by 100
again, which clipped every bar at the y-axis limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "experiment_results"
OUTPUT_PATH = ROOT / "paper" / "figures" / "fig_pca_comparison.png"

MODEL_ORDER = [
    "gte-large-en-v1.5",
    "stella_en_400M_v5",
    "bge-m3",
    "roberta-large",
    "roberta-large-InBedder",
]

MODEL_LABELS = {
    "gte-large-en-v1.5": "GTE-Large",
    "stella_en_400M_v5": "Stella",
    "bge-m3": "BGE-M3",
    "roberta-large": "RoBERTa-L.",
    "roberta-large-InBedder": "InBedder",
}

METHODS = [
    ("pca_retention", "PCA", "#D55E00"),
    ("rp_retention", "Random proj.", "#0072B2"),
    ("random_coord_retention", "Random coord.", "#009E73"),
]

TARGET_DIMS = [64, 128, 256]


def load_results(model_name: str) -> dict:
    path = DATA_DIR / f"pca_baseline_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing PCA baseline data: {path}")
    with path.open() as f:
        return json.load(f)


def as_percent(values: list[float]) -> list[float]:
    """Return retention values on a 0--100 scale."""
    if not values:
        return values
    # Backward-compatible guard for any future files that store 0--1 fractions.
    if max(values) <= 1.5:
        return [v * 100.0 for v in values]
    return values


def task_values(model_data: dict, dim: int, key: str) -> list[float]:
    values = []
    dim_key = f"dim_{dim}"
    for task_data in model_data["methods"].values():
        dim_data = task_data.get(dim_key, {})
        if key in dim_data:
            values.append(float(dim_data[key]))
    return as_percent(values)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )

    model_data = {model: load_results(model) for model in MODEL_ORDER}

    print("Model,Dim,NTasks,PCA,RandomProj,RandomCoord")
    for model in MODEL_ORDER:
        for dim in TARGET_DIMS:
            means = []
            n_tasks = None
            for key, _, _ in METHODS:
                values = task_values(model_data[model], dim, key)
                n_tasks = len(values) if n_tasks is None else n_tasks
                means.append(np.mean(values))
            print(
                f"{model},{dim},{n_tasks},"
                f"{means[0]:.2f},{means[1]:.2f},{means[2]:.2f}"
            )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24

    for ax, dim in zip(axes, TARGET_DIMS):
        for method_idx, (key, label, color) in enumerate(METHODS):
            means = [
                np.mean(task_values(model_data[model], dim, key))
                for model in MODEL_ORDER
            ]
            offset = (method_idx - 1) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.6,
                label=label,
            )

        ax.axhline(100, color="#555555", linestyle=":", linewidth=1.2)
        ax.set_title(f"dim={dim}")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=18, ha="right")
        ax.set_ylim(75, 103)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Mean retention (%)")
    fig.suptitle("PCA, Random Projection, and Coordinate Selection", y=0.98, fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.2))
    labels.append("Full-dim baseline")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

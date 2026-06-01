#!/usr/bin/env python3
"""Regenerate paper figures with corrected model labels.

The script reads existing experiment artifacts under data/ and writes only the
target paper figures under the current ACL paper package.
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
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = (
    ROOT
    / "Beyond_Redundancy__Diagnosing_Information_Distribution_in_Text_Embeddings_via_Task_Aware_Dimension_Selection"
)
FIG_DIR = PAPER_DIR / "figures"
ANALYZE_DIR = ROOT / "data" / "analyze_supplement"
ENTROPY_FILE = ROOT / "data" / "experiment_results" / "all_models_entropy.json"

EXCLUDE_TASKS = {"STS17"}
SWEEP_DIMS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 768]

GENERAL = "General-purpose Language Model Backbones"
INSTRUCTION = "Instruction-conditioned Embedders"
RETRIEVAL = "Retrieval-optimized Embedders"

MODELS = [
    ("roberta-large", "RoBERTa-Large", GENERAL),
    ("bart-base", "BART-Base", GENERAL),
    ("roberta-large-InBedder", "RoBERTa-InBedder", INSTRUCTION),
    ("Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B", INSTRUCTION),
    ("stella_en_400M_v5", "Stella EN 400M", INSTRUCTION),
    ("instructor-large", "Instructor-Large", INSTRUCTION),
    ("gtr-t5-large", "GTR-T5-Large", RETRIEVAL),
    ("bge-m3", "BGE-M3", RETRIEVAL),
    ("gte-base", "GTE-Base", RETRIEVAL),
    ("mxbai-embed-large-v1", "MxBai-Embed-Large", RETRIEVAL),
    ("gte-large-en-v1.5", "GTE-Large", RETRIEVAL),
]

DISPLAY = {key: name for key, name, _ in MODELS}
FAMILY = {key: family for key, _, family in MODELS}
FAMILY_COLORS = {
    GENERAL: "#27AE60",
    INSTRUCTION: "#6C757D",
    RETRIEVAL: "#E74C3C",
}
METHOD_COLORS = {
    "optimized": "#27AE60",
    "random": "#6C757D",
    "anti": "#E74C3C",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def set_paper_style(grid: bool = True) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.grid": grid,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def model_data(model_key: str) -> dict:
    return load_json(ANALYZE_DIR / f"{model_key}.json")


def model_dim(model_key: str) -> int:
    data = model_data(model_key)
    if data.get("model_dim"):
        return int(data["model_dim"])
    for task_data in data.get("task_name", {}).values():
        chunks = task_data.get("split_win_size", {}).get("2", {}).get("chunk_result")
        if chunks:
            return len(chunks) * 2
    raise ValueError(f"Cannot infer dimension for {model_key}")


def task_metrics(model_key: str, dim: int) -> list[dict[str, float]]:
    data = model_data(model_key)
    rows: list[dict[str, float]] = []
    for task, task_data in data.get("task_name", {}).items():
        if task in EXCLUDE_TASKS:
            continue
        full = float(task_data.get("defult_score", 0) or 0)
        if full <= 0:
            continue

        random_scores = task_data.get("random_score", {}).get(str(dim), [])
        chunk_dim = (
            task_data.get("split_win_size", {})
            .get("2", {})
            .get("chunk_win_size", {})
            .get(str(dim), {})
        )
        optimized = chunk_dim.get("head_score", {}).get("main_score")
        anti = chunk_dim.get("end_score", {}).get("main_score")
        if not random_scores or optimized is None or anti is None:
            continue

        random_mean = float(np.mean(random_scores))
        rows.append(
            {
                "random": random_mean / full,
                "optimized": float(optimized) / full,
                "anti": float(anti) / full,
                "gap": (float(optimized) - random_mean) / full,
            }
        )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def figure2_selection_bounds() -> None:
    """Replace Figure 2 content while preserving the old paper figure style."""
    set_paper_style(grid=True)
    focus = ["stella_en_400M_v5", "gte-large-en-v1.5", "roberta-large-InBedder"]
    fig, axes = plt.subplots(1, 3, figsize=(15.9, 4.9), sharey=True)

    method_specs = [
        ("Optimized", "optimized", "^", "-", METHOD_COLORS["optimized"]),
        ("Random", "random", "s", "--", "#999999"),
        ("Anti-optimized", "anti", "v", "-", METHOD_COLORS["anti"]),
    ]

    for ax, model_key in zip(axes, focus):
        dim = model_dim(model_key)
        x_vals = [d / dim for d in SWEEP_DIMS if d < dim]
        for label, field, marker, linestyle, color in method_specs:
            means: list[float] = []
            stds: list[float] = []
            for d in SWEEP_DIMS:
                if d >= dim:
                    continue
                rows = task_metrics(model_key, d)
                mean, std = mean_std([r[field] for r in rows])
                means.append(mean)
                stds.append(std)
            means_arr = np.asarray(means)
            stds_arr = np.asarray(stds)
            ax.fill_between(
                x_vals,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.12,
                color=color,
                linewidth=0,
            )
            ax.plot(
                x_vals,
                means_arr,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5.0,
                linewidth=2.3,
                label=label,
            )

        ax.axhline(1.0, color="#ff6f6f", linestyle=":", linewidth=2.0, label="Full-dim baseline")
        ax.set_title(f"{DISPLAY[model_key]}\n({dim}d)", fontsize=14, fontweight="normal", pad=8)
        ax.set_xlabel("Fraction of Dimensions Retained", fontsize=13)
        ax.set_xlim(0.0, 0.8)
        ax.set_ylim(0.15, 1.18)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC")

    axes[0].set_ylabel("Normalized Performance", fontsize=14)
    fig.tight_layout()
    save(fig, "fig6_pruning_ratio_sweep.png")


def figure3_opt_vs_random_scatter() -> None:
    set_paper_style(grid=False)
    rows = []
    for model_key, display, family in MODELS:
        metrics = task_metrics(model_key, 256)
        if not metrics:
            continue
        random_mean = np.mean([r["random"] for r in metrics]) * 100.0
        optimized_mean = np.mean([r["optimized"] for r in metrics]) * 100.0
        gap = np.mean([r["gap"] for r in metrics]) * 100.0
        rows.append(
            {
                "model": model_key,
                "display": display,
                "family": family,
                "random": random_mean,
                "optimized": optimized_mean,
                "gap": gap,
            }
        )

    fig, ax = plt.subplots(figsize=(7.7, 7.8))
    ax.set_facecolor("#FAFBFB")
    ax.axline((90, 90), slope=1, color="#B9B9B9", linestyle="--", linewidth=1.2, zorder=1)
    ax.fill_between([86, 117], [86, 117], 117, color="#EAF6EF", alpha=0.45, zorder=0)
    ax.fill_between([86, 117], 86, [86, 117], color="#FBEDEA", alpha=0.32, zorder=0)

    for row in rows:
        color = FAMILY_COLORS[row["family"]]
        ax.scatter(
            row["random"],
            row["optimized"],
            s=52,
            color=color,
            edgecolors="white",
            linewidth=0.6,
            zorder=4,
        )

    label_specs = {
        "roberta-large": {"xytext": (104.5, 114.5)},
        "bart-base": {"xytext": (86.9, 110.2)},
        "roberta-large-InBedder": {"xytext": (102.5, 107.6)},
        "Qwen3-Embedding-0.6B": {"xytext": (99.2, 104.7)},
        "mxbai-embed-large-v1": {"xytext": (99.7, 103.2)},
        "gtr-t5-large": {"xytext": (96.5, 106.0)},
        "stella_en_400M_v5": {"xytext": (86.9, 100.9)},
        "gte-base": {"xytext": (86.9, 99.5)},
        "gte-large-en-v1.5": {"xytext": (99.4, 98.4)},
        "instructor-large": {"xytext": (98.7, 96.5)},
        "bge-m3": {"xytext": (86.9, 95.3)},
    }

    for row in rows:
        spec = label_specs[row["model"]]
        tx, ty = spec["xytext"]
        color = FAMILY_COLORS[row["family"]]
        ax.annotate(
            "",
            xy=(row["random"], row["optimized"]),
            xytext=(tx, ty),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                alpha=0.34,
                lw=0.8,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=2,
        )

    for row in rows:
        spec = label_specs[row["model"]]
        tx, ty = spec["xytext"]
        color = FAMILY_COLORS[row["family"]]
        ax.text(
            tx,
            ty,
            f"{row['display']} (+{row['gap']:.1f}%)",
            fontsize=9.2,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            bbox=dict(facecolor="#FAFBFB", edgecolor="none", alpha=0.96, pad=0.12),
            zorder=6,
        )

    ax.set_xlim(86, 116)
    ax.set_ylim(86, 116)
    ax.set_xlabel("Random Retention (%)", fontsize=14)
    ax.set_ylabel("Optimized Retention (%)", fontsize=14)
    ax.set_title("Optimized vs Random Retention (dim=256)", fontsize=16, fontweight="bold", pad=10)

    legend_handles = [
        mpatches.Patch(facecolor=FAMILY_COLORS[GENERAL], label=GENERAL),
        mpatches.Patch(facecolor=FAMILY_COLORS[INSTRUCTION], label=INSTRUCTION),
        mpatches.Patch(facecolor=FAMILY_COLORS[RETRIEVAL], label=RETRIEVAL),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.02),
        fontsize=9.5,
        framealpha=0.92,
        edgecolor="#CCCCCC",
    )
    fig.tight_layout()
    save(fig, "fig_opt_vs_rnd_scatter.png")


def concentration_curves(model_key: str) -> tuple[list[float], list[float]]:
    data = model_data(model_key)
    top_fracs = [0.10, 0.25, 0.50]
    values = {frac: [] for frac in top_fracs}
    for task, task_data in data.get("task_name", {}).items():
        if task in EXCLUDE_TASKS:
            continue
        chunks = task_data.get("split_win_size", {}).get("2", {}).get("chunk_result")
        if not chunks:
            continue
        arr = np.abs(np.asarray(chunks, dtype=float))
        total = arr.sum()
        if total <= 0:
            continue
        cum = np.cumsum(np.sort(arr)[::-1]) / total
        for frac in top_fracs:
            idx = max(1, int(math.ceil(frac * len(cum)))) - 1
            values[frac].append(float(cum[idx]))
    return [frac * 100 for frac in top_fracs], [float(np.mean(values[frac])) for frac in top_fracs]


def entropy_rows() -> list[dict]:
    raw = load_json(ENTROPY_FILE)
    rows = []
    for model_key in raw:
        if model_key not in DISPLAY:
            continue
        vals = [
            float(v["normalized_entropy"])
            for task, v in raw[model_key].get("tasks", {}).items()
            if task not in EXCLUDE_TASKS and "normalized_entropy" in v
        ]
        if vals:
            rows.append(
                {
                    "model": model_key,
                    "display": DISPLAY[model_key],
                    "values": np.asarray(vals, dtype=float),
                    "mean": float(np.mean(vals)),
                }
            )
    rows.sort(key=lambda r: r["mean"])
    return rows


def figure6_evidence_summary() -> None:
    set_paper_style(grid=True)
    rows = entropy_rows()
    palette = list(plt.cm.tab10(np.linspace(0, 1, len(rows))))
    color_for = {row["model"]: palette[i] for i, row in enumerate(rows)}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.1, 5.1))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    for idx, row in enumerate(rows):
        xs, ys = concentration_curves(row["model"])
        ax_a.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            markersize=5.0,
            linewidth=2.0,
            color=color_for[row["model"]],
            label=row["display"],
        )

    ax_a.set_xlim(8, 52)
    ax_a.set_ylim(0.15, 0.62)
    ax_a.set_xticks([10, 25, 50])
    ax_a.set_xlabel("Top Chunks Included (%)", fontsize=12)
    ax_a.set_ylabel("Fraction of Total Score", fontsize=12)
    ax_a.set_title("(a) Top-K Score Concentration\n(9-model overview)", fontsize=14)

    box_data = [row["values"] for row in rows]
    labels = [row["display"] for row in rows]
    bp = ax_b.boxplot(
        box_data,
        vert=False,
        tick_labels=labels,
        patch_artist=True,
        widths=0.58,
        flierprops=dict(marker="o", markersize=2.6, alpha=0.45, markerfacecolor="white"),
        medianprops=dict(color="#222222", linewidth=1.5),
    )
    for patch, row in zip(bp["boxes"], rows):
        patch.set_facecolor(color_for[row["model"]])
        patch.set_alpha(0.48)
        patch.set_edgecolor("#4E5661")
        patch.set_linewidth(1.0)
    for whisker in bp["whiskers"]:
        whisker.set_color("#4E5661")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("#4E5661")
        cap.set_linewidth(1.0)

    ax_b.axvline(1.0, color="#9AA4B2", linestyle="--", linewidth=1.2, alpha=0.9)
    ax_b.set_xlim(0.988, 1.0002)
    ax_b.set_xlabel("Normalized Shannon Entropy", fontsize=12)
    ax_b.set_title("(b) Entropy Across 9 Models", fontsize=14)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=color_for[row["model"]],
            marker=markers[i % len(markers)],
            linewidth=2.0,
            markersize=5.0,
            label=row["display"],
        )
        for i, row in enumerate(rows)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.28, wspace=0.30)
    save(fig, "fig8_evidence_summary.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure2_selection_bounds()
    figure3_opt_vs_random_scatter()
    figure6_evidence_summary()


if __name__ == "__main__":
    main()

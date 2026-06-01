"""
Verify the data source behind paper Figure 10(a) / fig8_evidence_summary panel (a),
then generate two standalone variants under two "top-k" interpretations:

1. top = highest-scoring chunks after sorting by chunk score
2. top = first-k chunks in the original chunk order
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".cache" / "matplotlib")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_reviewer_figures import FIG7_MODEL_STYLES, set_style


EXCLUDED_TASKS = {"STS17", "STS22"}
FRACTIONS = [0.10, 0.25, 0.50]
FRACTION_LABELS = [10, 25, 50]
MODEL_ORDER = [
    model_name
    for model_name in FIG7_MODEL_STYLES
    if (REPO_ROOT / "data" / "analyze" / f"{model_name}.json").exists()
]

OUTPUT_FIGURES = {
    "best_by_score": [
        REPO_ROOT / "paper" / "figures" / "fig10a_topk_best_by_score.png",
        REPO_ROOT / "paper" / "figures" / "fig10a_topk_best_by_score.pdf",
        REPO_ROOT / "paper3" / "figures" / "fig10a_topk_best_by_score.png",
        REPO_ROOT / "paper3" / "figures" / "fig10a_topk_best_by_score.pdf",
    ],
    "first_by_position": [
        REPO_ROOT / "paper" / "figures" / "fig10a_topk_first_by_position.png",
        REPO_ROOT / "paper" / "figures" / "fig10a_topk_first_by_position.pdf",
        REPO_ROOT / "paper3" / "figures" / "fig10a_topk_first_by_position.png",
        REPO_ROOT / "paper3" / "figures" / "fig10a_topk_first_by_position.pdf",
    ],
}
SUMMARY_PATH = REPO_ROOT / "artifacts" / "fig10a_topk_interpretations_verified.json"

CURRENT_FIGURE_PATH = REPO_ROOT / "paper3" / "figures" / "fig8_evidence_summary.png"
CURRENT_FIGURE_SOURCE = "src/generate_reviewer_figures.py::fig8_evidence_summary"
CURRENT_FIGURE_PANEL = "Figure 10(a) / fig8_evidence_summary panel (a)"


def load_model_data(model_name: str) -> dict:
    with (REPO_ROOT / "data" / "analyze" / f"{model_name}.json").open("r") as f:
        return json.load(f)


def compute_concentration(chunk_scores: list[float], fraction: float, mode: str) -> float:
    arr = np.asarray(chunk_scores, dtype=float)
    if arr.size == 0:
        return 0.0

    total = float(arr.sum())
    if total <= 0:
        return 0.0

    k = max(1, int(math.ceil(arr.size * fraction)))
    if mode == "best_by_score":
        selected = np.sort(arr)[::-1][:k]
    elif mode == "first_by_position":
        selected = arr[:k]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return float(selected.sum() / total)


def summarize_mode(mode: str) -> dict[str, dict]:
    summary_by_model = {}

    for model_name in MODEL_ORDER:
        model_data = load_model_data(model_name)
        per_task = {fraction: [] for fraction in FRACTIONS}

        for task_name, task_data in model_data.get("task_name", {}).items():
            if task_name in EXCLUDED_TASKS:
                continue

            chunk_scores = (
                task_data.get("split_win_size", {})
                .get("2", {})
                .get("chunk_result", [])
            )
            if not chunk_scores:
                continue

            for fraction in FRACTIONS:
                per_task[fraction].append(compute_concentration(chunk_scores, fraction, mode))

        model_summary = {
            "label": FIG7_MODEL_STYLES[model_name]["label"],
            "n_tasks": len(next(iter(per_task.values()))) if per_task else 0,
            "fractions_pct": {},
        }
        for fraction, label in zip(FRACTIONS, FRACTION_LABELS):
            values = per_task[fraction]
            model_summary["fractions_pct"][str(label)] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
                "n_tasks": len(values),
            }
        summary_by_model[model_name] = model_summary

    return summary_by_model


def load_existing_best_summary() -> dict[str, dict]:
    with (REPO_ROOT / "data" / "experiment_results" / "reviewer_response_analysis.json").open("r") as f:
        data = json.load(f)

    mech = data["redundancy_mechanism"]
    summary = {}
    for model_name in MODEL_ORDER:
        model_summary = mech[model_name]["model_summary"]
        summary[model_name] = {
            "10": float(model_summary["avg_top_10pct_concentration"]),
            "25": float(model_summary["avg_top_25pct_concentration"]),
            "50": float(model_summary["avg_top_50pct_concentration"]),
        }
    return summary


def compare_with_existing(best_summary: dict[str, dict], existing_summary: dict[str, dict]) -> dict[str, object]:
    comparison = {"per_model_deltas": {}, "max_abs_delta": 0.0}

    for model_name in MODEL_ORDER:
        deltas = {}
        for label in ("10", "25", "50"):
            recomputed = best_summary[model_name]["fractions_pct"][label]["mean"]
            existing = existing_summary[model_name][label]
            delta = float(recomputed - existing)
            deltas[label] = delta
            comparison["max_abs_delta"] = max(comparison["max_abs_delta"], abs(delta))
        comparison["per_model_deltas"][model_name] = deltas

    return comparison


def compute_shared_y_limits(*summaries: dict[str, dict]) -> tuple[float, float]:
    all_values = []
    for summary in summaries:
        for model_summary in summary.values():
            for label in ("10", "25", "50"):
                all_values.append(model_summary["fractions_pct"][label]["mean"])

    ymin = math.floor((min(all_values) - 0.01) * 100.0) / 100.0
    ymax = math.ceil((max(all_values) + 0.01) * 100.0) / 100.0
    return ymin, ymax


def plot_summary(
    summary_by_model: dict[str, dict],
    title: str,
    subtitle: str,
    output_paths: list[Path],
    y_limits: tuple[float, float],
) -> None:
    set_style()
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 320,
            "savefig.dpi": 320,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    x_vals = np.array(FRACTION_LABELS, dtype=float)

    for model_name in MODEL_ORDER:
        style = FIG7_MODEL_STYLES[model_name]
        y_vals = [summary_by_model[model_name]["fractions_pct"][label]["mean"] for label in ("10", "25", "50")]
        ax.plot(
            x_vals,
            y_vals,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.7,
            markersize=6.2,
        )

    ax.set_xticks(x_vals)
    ax.set_xlim(8, 52)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Chunks Included (%)")
    ax.set_ylabel("Fraction of Total Score")
    ax.set_title(f"{title}\n{subtitle}", pad=12, fontweight="bold")

    legend_handles = [
        Line2D(
            [0], [0],
            color=FIG7_MODEL_STYLES[model_name]["color"],
            marker=FIG7_MODEL_STYLES[model_name]["marker"],
            linewidth=1.7,
            markersize=6.2,
            label=FIG7_MODEL_STYLES[model_name]["label"],
        )
        for model_name in MODEL_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8.6, framealpha=0.92)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)

    plt.close(fig)


def build_summary_payload(
    best_summary: dict[str, dict],
    prefix_summary: dict[str, dict],
    existing_summary: dict[str, dict],
    existing_comparison: dict[str, object],
) -> dict[str, object]:
    per_model = {}
    for model_name in MODEL_ORDER:
        best_means = best_summary[model_name]["fractions_pct"]
        prefix_means = prefix_summary[model_name]["fractions_pct"]
        per_model[model_name] = {
            "label": FIG7_MODEL_STYLES[model_name]["label"],
            "n_tasks": best_summary[model_name]["n_tasks"],
            "best_by_score": {label: best_means[label]["mean"] for label in ("10", "25", "50")},
            "first_by_position": {label: prefix_means[label]["mean"] for label in ("10", "25", "50")},
            "best_minus_first": {
                label: float(best_means[label]["mean"] - prefix_means[label]["mean"])
                for label in ("10", "25", "50")
            },
            "delta_vs_existing_best": existing_comparison["per_model_deltas"][model_name],
        }

    return {
        "current_figure_panel": CURRENT_FIGURE_PANEL,
        "current_figure_path": str(CURRENT_FIGURE_PATH),
        "current_figure_source": CURRENT_FIGURE_SOURCE,
        "current_figure_definition": {
            "mode": "best_by_score",
            "description": "Sort chunk_result descending by score, then take the highest-scoring top-k fraction.",
        },
        "alternative_definition": {
            "mode": "first_by_position",
            "description": "Keep the first k chunks in original chunk order without sorting by score.",
        },
        "fractions_pct": FRACTION_LABELS,
        "excluded_tasks": sorted(EXCLUDED_TASKS),
        "existing_best_summary_from_json": existing_summary,
        "verification_against_existing_best": existing_comparison,
        "per_model_means": per_model,
        "output_figures": {
            mode: [str(path) for path in paths]
            for mode, paths in OUTPUT_FIGURES.items()
        },
    }


def main() -> None:
    best_summary = summarize_mode("best_by_score")
    prefix_summary = summarize_mode("first_by_position")
    existing_summary = load_existing_best_summary()
    existing_comparison = compare_with_existing(best_summary, existing_summary)
    y_limits = compute_shared_y_limits(best_summary, prefix_summary)

    plot_summary(
        summary_by_model=best_summary,
        title="Figure 10(a) Recomputed from Raw Chunk Scores",
        subtitle="Top-k = highest-scoring chunks after sorting by chunk score",
        output_paths=OUTPUT_FIGURES["best_by_score"],
        y_limits=y_limits,
    )
    plot_summary(
        summary_by_model=prefix_summary,
        title="Figure 10(a) Recomputed from Raw Chunk Scores",
        subtitle="Top-k = first k chunks in original chunk order",
        output_paths=OUTPUT_FIGURES["first_by_position"],
        y_limits=y_limits,
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_summary_payload(best_summary, prefix_summary, existing_summary, existing_comparison)
    with SUMMARY_PATH.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Current figure panel: {CURRENT_FIGURE_PANEL}")
    print(f"Verified existing definition max abs delta: {existing_comparison['max_abs_delta']:.12f}")
    for mode, paths in OUTPUT_FIGURES.items():
        for path in paths:
            print(f"Saved {mode}: {path}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

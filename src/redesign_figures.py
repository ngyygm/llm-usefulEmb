"""
Redesign all paper figures for better visual quality and completeness.
Addresses user feedback:
- Figure 3 (all methods): clean names, better design, more info
- Figure 2 (opt-random gap): clean names, better design
- Figure 4 (magnitude): more information
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "Beyond_Redundancy__Diagnosing_Information_Distribution_in_Text_Embeddings_via_Task_Aware_Dimension_Selection",
    "figures",
)
PAPER3_OUTPUT_DIR = os.path.join(PROJECT_DIR, "paper3", "figures")
ANALYSIS_DIR = os.path.join(PROJECT_DIR, "data", "experiment_results")
ANALYZE_DIR = os.path.join(PROJECT_DIR, "data", "analyze")
ANALYZE_NEW_DIR = os.path.join(PROJECT_DIR, "data", "analyze_new")
MAGNITUDE_RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "magnitude_results_supplement")
if not os.path.isdir(MAGNITUDE_RESULTS_DIR):
    MAGNITUDE_RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "magnitude_results")

# Clean model display names
MODEL_DISPLAY = {
    "gte-large-en-v1.5": "GTE-Large",
    "stella_en_400M_v5": "Stella EN 400M",
    "roberta-large-InBedder": "RoBERTa-InBedder",
    "bge-m3": "BGE-M3",
    "instructor-large": "Instructor",
    "mxbai-embed-large-v1": "MxBai-Large",
    "gte-base": "GTE-Base",
    "gtr-t5-large": "GTR-T5-Large",
    "bart-base": "BART-Base",
    "roberta-large": "RoBERTa-Large",
    "gte-Qwen2-1.5B-instruct": "GTE-Qwen2",
    "Qwen3-Embedding-0.6B": "Qwen-Embed.-0.6B",
    "stella_en_400M_v5-GEDI-epoch_3": "Stella-GEDI",
}

MAGNITUDE_DISPLAY = {
    "bge-m3": "BGE-M3",
    "gte-base": "GTE-Base",
    "mxbai-embed-large-v1": "MxBai-Embed-Large",
    "gte-large-en-v1.5": "GTE-Large",
    "roberta-large-InBedder": "RoBERTa-InBedder",
    "Qwen3-Embedding-0.6B": "Qwen-Embed.-0.6B",
    "stella_en_400M_v5": "Stella EN 400M",
    "bart-base": "BART-Base",
}

# Color palette (professional, colorblind-friendly)
COLORS = {
    "random": "#6C757D",
    "sequential": "#4A90D9",
    "magnitude": "#E67E22",
    "optimized": "#27AE60",
    "anti_opt": "#E74C3C",
    "non_native_embedding_base": "#2E8B57",
    "adaptive_embedding": "#8F8F8F",
    "retrieval_native_embedding": "#D95F5F",
}

MAGNITUDE_MODEL_ORDER = [
    "bge-m3",
    "gte-base",
    "mxbai-embed-large-v1",
    "gte-large-en-v1.5",
    "roberta-large-InBedder",
    "Qwen3-Embedding-0.6B",
    "stella_en_400M_v5",
    "bart-base",
]

MAGNITUDE_MODEL_GROUPS = {
    "bge-m3": "retrieval_native_embedding",
    "gte-base": "retrieval_native_embedding",
    "mxbai-embed-large-v1": "retrieval_native_embedding",
    "gte-large-en-v1.5": "retrieval_native_embedding",
    "roberta-large-InBedder": "adaptive_embedding",
    "Qwen3-Embedding-0.6B": "adaptive_embedding",
    "stella_en_400M_v5": "adaptive_embedding",
    "bart-base": "non_native_embedding_base",
}

MAGNITUDE_GROUP_LABELS = {
    "retrieval_native_embedding": "Retrieval-optimized",
    "adaptive_embedding": "Instruction-conditioned",
    "non_native_embedding_base": "General-purpose Language Model Backbone",
}

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
    "Retrieval": [
        "ArguAna",
        "CQADupstackEnglishRetrieval",
        "NFCorpus",
        "SCIDOCS",
        "SciFact",
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
        "STSBenchmark",
    ],
    "Pair Classification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Summarization": ["SummEval"],
}

TASK_TO_CATEGORY = {
    task: category
    for category, tasks in TASK_CATEGORIES.items()
    for task in tasks
}

FIG2_MODEL_METADATA = {
    "roberta-large": {
        "group": "non_native_embedding_base",
        "scientific_main_class": "Pretrained language model base (non-native embedding)",
        "display_group": "A Top Group: Non-native embedding base",
    },
    "bart-base": {
        "group": "non_native_embedding_base",
        "scientific_main_class": "Pretrained seq2seq base (non-native embedding)",
        "display_group": "A Top Group: Non-native embedding base",
    },
    "roberta-large-InBedder": {
        "group": "adaptive_embedding",
        "scientific_main_class": "Instruction-aware / answer-style embedding",
        "display_group": "B Middle Group: Adaptive embedding",
    },
    "Qwen3-Embedding-0.6B": {
        "group": "adaptive_embedding",
        "scientific_main_class": "Foundation-model-derived dual-encoder embedding",
        "display_group": "B Middle Group: Adaptive embedding",
    },
    "stella_en_400M_v5": {
        "group": "adaptive_embedding",
        "scientific_main_class": "Distilled + MRL general-purpose embedding",
        "display_group": "B Middle Group: Adaptive embedding",
    },
    "instructor-large": {
        "group": "adaptive_embedding",
        "scientific_main_class": "Instruction-finetuned general-purpose embedding",
        "display_group": "B Middle Group: Adaptive embedding",
    },
    "gtr-t5-large": {
        "group": "retrieval_native_embedding",
        "scientific_main_class": "T5-derived dense retriever",
        "display_group": "C Bottom Group: Retrieval-native embedding",
    },
    "bge-m3": {
        "group": "retrieval_native_embedding",
        "scientific_main_class": "Hybrid retrieval embedding",
        "display_group": "C Bottom Group: Retrieval-native embedding",
    },
    "gte-base": {
        "group": "retrieval_native_embedding",
        "scientific_main_class": "General dense embedding (encoder backbone)",
        "display_group": "C Bottom Group: Retrieval-native embedding",
    },
    "mxbai-embed-large-v1": {
        "group": "retrieval_native_embedding",
        "scientific_main_class": "General dense embedding (contrastive retrieval)",
        "display_group": "C Bottom Group: Retrieval-native embedding",
    },
    "gte-large-en-v1.5": {
        "group": "retrieval_native_embedding",
        "scientific_main_class": "General dense embedding (encoder backbone)",
        "display_group": "C Bottom Group: Retrieval-native embedding",
    },
}

FIG2_GROUP_ORDER = [
    "non_native_embedding_base",
    "adaptive_embedding",
    "retrieval_native_embedding",
]

FIG2_GROUP_LABELS = {
    "non_native_embedding_base": "A. Non-native embedding base",
    "adaptive_embedding": "B. Adaptive embedding",
    "retrieval_native_embedding": "C. Retrieval-native embedding",
}

RETAINED_FIG2_STATS = {
    "gte-base": {
        "display_name": "GTE-Base",
        "mean_gap_pct": 3.45,
        "ci_low_pct": 2.11,
        "ci_high_pct": 4.93,
        "std_gap_pct": 0.81,
        "cohens_d": 0.81,
        "p_holm": 0.001,
        "p_holm_text": "<0.001",
        "significant_holm_0_05": True,
        "n_tasks": None,
        "tasks": [],
        "source": "retained_original_value_missing_raw_analyze",
    },
    "gtr-t5-large": {
        "display_name": "GTR-T5-Large",
        "mean_gap_pct": 4.75,
        "ci_low_pct": 2.98,
        "ci_high_pct": 6.67,
        "std_gap_pct": 0.86,
        "cohens_d": 0.86,
        "p_holm": 0.001,
        "p_holm_text": "<0.001",
        "significant_holm_0_05": True,
        "n_tasks": None,
        "tasks": [],
        "source": "retained_original_value_missing_raw_analyze",
    },
}


def bootstrap_mean_ci(values, n_bootstrap=20000, seed=42):
    """Return mean and bootstrap percentile CI for a 1D list of values."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    sample_means = samples.mean(axis=1)
    return (
        float(arr.mean()),
        float(np.percentile(sample_means, 2.5)),
        float(np.percentile(sample_means, 97.5)),
    )


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction with monotonic adjusted p-values."""
    p_values = [float(p) for p in p_values]
    order = np.argsort(p_values)
    adjusted = [None] * len(p_values)
    prev = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):
        adj_p = min((m - rank) * p_values[idx], 1.0)
        adj_p = max(adj_p, prev)
        adjusted[idx] = adj_p
        prev = adj_p

    return adjusted


def paired_ttest_greater(opt_vals, rnd_vals):
    """One-sided paired t-test for optimized > random, compatible with older SciPy."""
    t_stat, p_two_sided = ttest_rel(opt_vals, rnd_vals)
    if np.isnan(t_stat) or np.isnan(p_two_sided):
        return float("nan")
    if t_stat >= 0:
        return float(p_two_sided / 2.0)
    return float(1.0 - (p_two_sided / 2.0))


def compute_cohens_d_from_diff(diff_vals):
    """Cohen's d for paired differences."""
    diff_vals = np.asarray(diff_vals, dtype=float)
    if len(diff_vals) < 2:
        return float("nan")
    std = diff_vals.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(diff_vals.mean() / std)


def compute_verified_opt_random_gap_stats(target_dim=256, win_size="2"):
    """
    Compute Figure 2 stats directly from data/analyze.

    The metric is:
        ((optimized / default) - (mean(random) / default)) * 100
    where optimized is the head_score from split_win_size=2 at target_dim.
    """
    analyze_data = load_analyze_data()
    computed_models = []

    for model_key, metadata in FIG2_MODEL_METADATA.items():
        if model_key in RETAINED_FIG2_STATS:
            continue
        group = metadata["group"]
        model_data = analyze_data.get(model_key)
        if model_data is None:
            continue

        gap_values = []
        opt_retentions = []
        rnd_retentions = []
        included_tasks = []

        for task_name, task_data in model_data.get("task_name", {}).items():
            if task_name == "STS17":
                continue

            default_score = task_data.get("defult_score", 0)
            if default_score <= 0:
                continue

            ws_data = task_data.get("split_win_size", {}).get(str(win_size), {})
            head_score = (
                ws_data.get("chunk_win_size", {})
                .get(str(target_dim), {})
                .get("head_score", {})
                .get("main_score")
            )
            rand_vals = task_data.get("random_score", {}).get(str(target_dim), [])

            if head_score is None or not rand_vals:
                continue

            opt_ret = float(head_score) / float(default_score)
            rnd_ret = float(np.mean(rand_vals)) / float(default_score)

            opt_retentions.append(opt_ret)
            rnd_retentions.append(rnd_ret)
            gap_values.append((opt_ret - rnd_ret) * 100.0)
            included_tasks.append(task_name)

        if not gap_values:
            continue

        diff_vals = np.asarray(opt_retentions) - np.asarray(rnd_retentions)
        mean_gap, ci_low, ci_high = bootstrap_mean_ci(gap_values)

        computed_models.append({
            "model_key": model_key,
            "display_name": MODEL_DISPLAY.get(model_key, model_key),
            "group": group,
            "group_label": FIG2_GROUP_LABELS[group],
            "scientific_main_class": metadata["scientific_main_class"],
            "display_group": metadata["display_group"],
            "mean_gap_pct": mean_gap,
            "ci_low_pct": ci_low,
            "ci_high_pct": ci_high,
            "std_gap_pct": float(np.std(gap_values)),
            "cohens_d": compute_cohens_d_from_diff(diff_vals),
            "p_value": paired_ttest_greater(opt_retentions, rnd_retentions),
            "source": "computed_from_data_analyze",
            "n_tasks": len(gap_values),
            "tasks": included_tasks,
        })

    adjusted_p = holm_bonferroni([entry["p_value"] for entry in computed_models])
    for entry, p_holm in zip(computed_models, adjusted_p):
        entry["p_holm"] = float(p_holm)
        entry["p_holm_text"] = f"{p_holm:.6g}"
        entry["significant_holm_0_05"] = bool(p_holm < 0.05)

    retained_models = []
    for model_key, retained in RETAINED_FIG2_STATS.items():
        retained_entry = dict(retained)
        metadata = FIG2_MODEL_METADATA[model_key]
        retained_entry["model_key"] = model_key
        retained_entry["group"] = metadata["group"]
        retained_entry["group_label"] = FIG2_GROUP_LABELS[metadata["group"]]
        retained_entry["scientific_main_class"] = metadata["scientific_main_class"]
        retained_entry["display_group"] = metadata["display_group"]
        retained_models.append(retained_entry)

    figure_models = computed_models + retained_models
    figure_models.sort(key=lambda entry: entry["mean_gap_pct"], reverse=True)
    for idx, entry in enumerate(figure_models, start=1):
        entry["figure_rank"] = idx

    return {
        "target_dim": int(target_dim),
        "win_size": int(win_size),
        "metric": "((optimized/default) - (mean(random)/default)) * 100",
        "figure_models": figure_models,
        "computed_models": computed_models,
        "retained_models": retained_models,
    }


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
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_analyze_data(data_dir=None):
    if data_dir is None:
        data_dir = ANALYZE_DIR
    data = {}
    for fname in os.listdir(data_dir):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(data_dir, fname), "r") as f:
                data[model_name] = json.load(f)
    return data


def get_method_scores(analyze_data, model_name, target_dim=256):
    """Extract per-task scores for all methods at a given dimension."""
    model_data = analyze_data.get(model_name)
    if model_data is None:
        return None

    task_scores = {}
    for task_name, task_data in model_data.get("task_name", {}).items():
        if task_name == "STS17":
            continue
        default = task_data.get("defult_score", 0)
        if default <= 0:
            continue

        scores = {"default": default}

        random_data = task_data.get("random_score", {}).get(str(target_dim), [])
        if random_data:
            scores["random"] = float(np.mean(random_data))

        sort_val = task_data.get("sort_score", {}).get(str(target_dim))
        if sort_val is not None:
            scores["sort"] = float(sort_val)

        for ws_str, ws_data in task_data.get("split_win_size", {}).items():
            for td_str, td_data in ws_data.get("chunk_win_size", {}).items():
                if td_str == str(target_dim):
                    head = td_data.get("head_score", {}).get("main_score")
                    end = td_data.get("end_score", {}).get("main_score")
                    if head is not None:
                        scores["best"] = float(head)
                    if end is not None:
                        scores["poor"] = float(end)

        if "random" in scores:
            task_scores[task_name] = scores

    return task_scores


def load_preferred_analyze_data():
    """Load full-dimensional baselines, preferring the refreshed analyze_new directory."""
    if os.path.isdir(ANALYZE_NEW_DIR):
        return load_analyze_data(ANALYZE_NEW_DIR)
    return load_analyze_data(ANALYZE_DIR)


def load_full_magnitude_data(target_dim=256):
    """
    Load complete magnitude-pruning runs and compute retention with the
    full-dimensional baseline from data/analyze_new, falling back to data/analyze.
    """
    analyze_data = load_preferred_analyze_data()
    fallback_analyze_data = load_analyze_data(ANALYZE_DIR)
    records = []
    category_values = {category: [] for category in TASK_CATEGORIES}

    for dirname in sorted(os.listdir(MAGNITUDE_RESULTS_DIR)):
        analysis_path = os.path.join(MAGNITUDE_RESULTS_DIR, dirname, "magnitude_analysis.json")
        if not os.path.exists(analysis_path):
            continue

        with open(analysis_path) as f:
            mag_analysis = json.load(f)

        model_name = next(iter(mag_analysis.get("magnitude_mteb", {})), None)
        if model_name is None:
            continue
        if model_name not in MAGNITUDE_MODEL_ORDER:
            continue

        mag_mteb = mag_analysis["magnitude_mteb"][model_name]
        if int(mag_mteb.get("target_dim", target_dim)) != int(target_dim):
            continue

        model_analyze = analyze_data.get(model_name) or fallback_analyze_data.get(model_name)
        if model_analyze is None:
            continue

        task_rows = []
        for task_name, mag_score in sorted(mag_mteb.get("scores", {}).items()):
            if task_name == "STS17":
                continue

            task_data = model_analyze.get("task_name", {}).get(task_name)
            if task_data is None:
                continue

            baseline = task_data.get("defult_score", 0)
            random_scores = task_data.get("random_score", {}).get(str(target_dim), [])
            if baseline <= 0 or not random_scores:
                continue

            magnitude_retention = float(mag_score) / float(baseline) * 100.0
            random_retention = float(np.mean(random_scores)) / float(baseline) * 100.0
            gap = magnitude_retention - random_retention
            category = TASK_TO_CATEGORY.get(task_name, "Other")

            row = {
                "task": task_name,
                "category": category,
                "magnitude": magnitude_retention,
                "random": random_retention,
                "gap": gap,
            }
            task_rows.append(row)
            if category in category_values:
                category_values[category].append(row)

        if not task_rows:
            continue

        magnitude_values = np.asarray([row["magnitude"] for row in task_rows], dtype=float)
        random_values = np.asarray([row["random"] for row in task_rows], dtype=float)
        gap_values = magnitude_values - random_values
        mean_mag, ci_mag_low, ci_mag_high = bootstrap_mean_ci(magnitude_values)
        mean_random, ci_rand_low, ci_rand_high = bootstrap_mean_ci(random_values)
        t_stat, p_value = ttest_rel(magnitude_values, random_values)
        d_value = compute_cohens_d_from_diff(gap_values)

        corr_summary = (
            mag_analysis
            .get("magnitude_vs_task_correlation", {})
            .get(model_name, {})
            .get("summary", {})
        )
        ranking_data = mag_analysis.get("magnitude_rankings", {}).get(model_name, {})
        group = MAGNITUDE_MODEL_GROUPS.get(model_name, "adaptive_embedding")

        records.append({
            "model_key": model_name,
            "display_name": MAGNITUDE_DISPLAY.get(model_name, MODEL_DISPLAY.get(model_name, model_name)),
            "group": group,
            "group_label": MAGNITUDE_GROUP_LABELS.get(group, group),
            "n_tasks": len(task_rows),
            "magnitude_mean": float(mean_mag),
            "magnitude_ci_low": float(ci_mag_low),
            "magnitude_ci_high": float(ci_mag_high),
            "random_mean": float(mean_random),
            "random_ci_low": float(ci_rand_low),
            "random_ci_high": float(ci_rand_high),
            "gap_mean": float(np.mean(gap_values)),
            "p_value": float(p_value),
            "cohens_d": float(d_value),
            "mag_wins": int(np.sum(gap_values > 0)),
            "mean_rho": float(corr_summary.get("mean_rho", np.nan)),
            "std_rho": float(corr_summary.get("std_rho", np.nan)),
            "n_significant_rho": int(corr_summary.get("n_significant", 0)),
            "model_dim": int(ranking_data.get("model_dim", 0)),
            "n_chunks": int(ranking_data.get("n_chunks", 0)),
            "task_rows": task_rows,
            "raw_scores": mag_mteb.get("scores", {}),
        })

    order = {model: idx for idx, model in enumerate(MAGNITUDE_MODEL_ORDER)}
    records.sort(key=lambda row: order.get(row["model_key"], 999))

    category_summary = []
    for category in TASK_CATEGORIES:
        rows = category_values[category]
        if not rows:
            continue
        category_summary.append({
            "category": category,
            "n": len(rows),
            "magnitude_mean": float(np.mean([row["magnitude"] for row in rows])),
            "random_mean": float(np.mean([row["random"] for row in rows])),
            "gap_mean": float(np.mean([row["gap"] for row in rows])),
        })

    return records, category_summary


def get_magnitude_score_map(records):
    return {row["model_key"]: row["raw_scores"] for row in records}


# ============================================================
# FIGURE 2: Optimized-Random Gap (redesign)
# ============================================================
def fig2_opt_random_gap():
    """Redesigned Figure 2 with computed stats plus retained original values where raw data is missing."""
    set_style()
    stats = compute_verified_opt_random_gap_stats()
    figure_models = stats["figure_models"]

    # Make the panel narrower and taller so it fits side-by-side as a compact
    # left panel instead of stretching horizontally.
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    names = [entry["display_name"] for entry in figure_models]
    gaps = [entry["mean_gap_pct"] for entry in figure_models]
    ci_lo = [entry["ci_low_pct"] for entry in figure_models]
    ci_hi = [entry["ci_high_pct"] for entry in figure_models]
    sig = [entry["significant_holm_0_05"] for entry in figure_models]
    groups = [entry["group"] for entry in figure_models]

    y = np.arange(len(names))
    err_lo = [g - lo for g, lo in zip(gaps, ci_lo)]
    err_hi = [hi - g for g, hi in zip(gaps, ci_hi)]

    colors = [COLORS[group] for group in groups]

    ax.barh(y, gaps, xerr=[err_lo, err_hi], color=colors, alpha=0.8,
            capsize=3, error_kw={'linewidth': 1, 'color': '#333333'},
            height=0.54, edgecolor='white', linewidth=0.5)
    ax.invert_yaxis()

    ax.axvline(x=0, color='#333333', linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.tick_params(axis='y', pad=2)
    ax.set_xlabel("Optimized $-$ Random Retention Gap (%)", fontsize=11)
    x_max = max(g + hi for g, hi in zip(gaps, err_hi)) + 2.0
    ax.set_xlim(-0.18, x_max)

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["non_native_embedding_base"], alpha=0.8,
                       label=FIG2_GROUP_LABELS["non_native_embedding_base"]),
        mpatches.Patch(facecolor=COLORS["adaptive_embedding"], alpha=0.8,
                       label=FIG2_GROUP_LABELS["adaptive_embedding"]),
        mpatches.Patch(facecolor=COLORS["retrieval_native_embedding"], alpha=0.8,
                       label=FIG2_GROUP_LABELS["retrieval_native_embedding"]),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='#CCCCCC',
              bbox_to_anchor=(1.0, 0.0))

    for i, (name, gap) in enumerate(zip(names, gaps)):
        sig_mark = "" if sig[i] else " (n.s.)"
        label_x = gap + err_hi[i] + 0.28
        ax.text(label_x, i, f"+{gap:.1f}%{sig_mark}", va='center',
                fontsize=7.5, color='#333333', clip_on=False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_opt_random_gap_all_models.png")
    plt.savefig(path)
    plt.close()
    stats_path = os.path.join(ANALYSIS_DIR, "fig_opt_random_gap_verified.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {stats_path}")
    print(f"Saved: {path}")


# ============================================================
# FIGURE 3: All Methods Comparison (redesign)
# ============================================================
def fig3_all_methods():
    """Redesigned Figure 3: All 5 pruning methods at dim=256."""
    set_style()
    analyze_data = load_preferred_analyze_data()
    magnitude_records, _ = load_full_magnitude_data()
    mag_mteb = get_magnitude_score_map(magnitude_records)

    model_configs = [
        ("stella_en_400M_v5", "Stella EN 400M"),
        ("gte-large-en-v1.5", "GTE-Large"),
        ("roberta-large-InBedder", "RoBERTa-InBedder"),
    ]

    method_info = [
        ("Random",           COLORS["random"],     "dashed"),
        ("Sequential",       COLORS["sequential"],  "solid"),
        ("Magnitude",        COLORS["magnitude"],   "solid"),
        ("Optimized",        COLORS["optimized"],   "solid"),
        ("Anti-optimized",   COLORS["anti_opt"],    "solid"),
    ]

    # Use a flatter canvas and tighter subplot spacing to reduce excess
    # vertical whitespace between the suptitle, panel titles, and legend.
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)

    for ax_idx, (model_key, model_display) in enumerate(model_configs):
        ax = axes[ax_idx]
        task_scores = get_method_scores(analyze_data, model_key)
        if task_scores is None:
            continue

        mag_scores = mag_mteb.get(model_key, {})

        # Compute mean retention per method
        method_rets = {}
        method_stds = {}
        for task_name, tscores in task_scores.items():
            default = tscores["default"]
            if default <= 0:
                continue

            for method, _, _ in method_info:
                key = method.lower().replace("-", "_").replace(" ", "_")
                if method == "Anti-optimized":
                    key = "poor"
                elif method == "Optimized":
                    key = "best"
                elif method == "Sequential":
                    key = "sort"
                elif method == "Magnitude":
                    if task_name in mag_scores:
                        val = mag_scores[task_name] / default
                        method_rets.setdefault(method, []).append(val)
                        continue
                    else:
                        continue

                if key in tscores:
                    val = tscores[key] / default
                    method_rets.setdefault(method, []).append(val)

        # Plot as lollipop chart
        methods_with_data = []
        means = []
        stds = []
        colors_list = []
        for method, color, ls in method_info:
            if method in method_rets:
                methods_with_data.append(method)
                means.append(np.mean(method_rets[method]) * 100)
                stds.append(np.std(method_rets[method]) * 100)
                colors_list.append(color)

        x = np.arange(len(methods_with_data))

        # Error bars + dots
        for i, (mean, std, color) in enumerate(zip(means, stds, colors_list)):
            ax.plot([i, i], [mean - std, mean + std], color=color, linewidth=1.5, alpha=0.5)
            ax.scatter([i], [mean], color=color, s=80, zorder=5, edgecolors='white', linewidth=0.5)

        # Connect with a light line
        ax.plot(x, means, '-', color='#BBBBBB', linewidth=0.8, zorder=1)

        # Reference line at 100%
        ax.axhline(y=100, color='#333333', linestyle=':', alpha=0.4, linewidth=0.8, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("Anti-optimized", "Anti-\noptimized") for m in methods_with_data],
            rotation=24,
            ha='right',
            fontsize=11,
        )
        title_fontsize = 12.5 if model_display == "RoBERTa-InBedder" else 13
        ax.set_title(model_display, fontsize=title_fontsize, fontweight='bold', y=1.11, pad=0)
        ax.tick_params(axis='y', labelsize=11)

        ax.set_ylim(82, 124)

        # Keep value labels inside each panel; RoBERTa-InBedder has large variance
        # and otherwise pushes the optimized label above the visible y-range.
        _, y_max = ax.get_ylim()
        label_ceiling = y_max - 1.4
        reference_line_y = 100.0
        reference_line_clearance = 1.35
        last_idx = len(means) - 1
        for i, (mean, std) in enumerate(zip(means, stds)):
            label_y = min(mean + std + 0.3, label_ceiling)
            if label_y <= mean + 0.2:
                label_y = mean + 0.5

            # Keep labels from sitting on the 100% reference line after fontsize increases.
            if abs(label_y - reference_line_y) < reference_line_clearance:
                label_y = reference_line_y + reference_line_clearance
                label_y = min(label_y, label_ceiling)
                if label_y <= mean + 0.2:
                    label_y = min(label_ceiling, mean + 0.8)

            x_offset = 4
            if i == 0:
                x_offset = 16
            elif i == last_idx:
                x_offset = 2

            ax.annotate(
                f'{mean:.1f}%',
                (i, label_y),
                xytext=(x_offset, 0),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='semibold',
                color='#555555',
                annotation_clip=False,
                clip_on=False,
                bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor='white', linewidth=0.4, alpha=1.0),
                zorder=7,
            )

    axes[0].set_ylabel("Mean Retention (%)", fontsize=12.5)
    fig.suptitle("Five Pruning Strategies at dim=256 (75% Reduction)", fontsize=14,
                 y=0.97, fontweight='bold')

    fig.subplots_adjust(left=0.075, right=0.995, top=0.80, bottom=0.23, wspace=0.12)
    output_paths = [os.path.join(OUTPUT_DIR, "fig10_all_methods_comparison.png")]
    if os.path.isdir(PAPER3_OUTPUT_DIR):
        output_paths.append(os.path.join(PAPER3_OUTPUT_DIR, "fig10_all_methods_comparison.png"))
    for path in output_paths:
        plt.savefig(path)
        plt.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close()
    for path in output_paths:
        print(f"Saved: {path}")


# ============================================================
# FIGURE 4: Magnitude Analysis (redesign with more info)
# ============================================================
def fig4_magnitude_analysis():
    """Redesigned Figure 7: full magnitude analysis across all available models."""
    set_style()
    records, category_summary = load_full_magnitude_data()

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    group_colors = {
        key: COLORS[key]
        for key in [
            "non_native_embedding_base",
            "adaptive_embedding",
            "retrieval_native_embedding",
        ]
    }

    # (a) Magnitude ranking vs task-specific ranking correlation.
    ax = axes[0]
    y = np.arange(len(records))
    rho = [row["mean_rho"] for row in records]
    rho_err = [row["std_rho"] for row in records]
    colors = [group_colors[row["group"]] for row in records]
    ax.barh(y, rho, xerr=rho_err, color=colors, alpha=0.82, capsize=3,
            height=0.62, edgecolor='white', linewidth=0.4)
    ax.axvline(x=0, color='#333333', linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([row["display_name"] for row in records], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman $\\rho$", fontsize=12)
    ax.set_title("(a) Weight magnitude is nearly\nuncorrelated with task ranking", fontsize=12.5)
    ax.set_xlim(-0.12, 0.12)
    ax.tick_params(axis='x', labelsize=10)

    # (b) Model-level magnitude minus random retention.
    ax = axes[1]
    gaps = [row["gap_mean"] for row in records]
    ax.barh(y, gaps, color=colors, alpha=0.82, height=0.62,
            edgecolor='white', linewidth=0.4)
    ax.axvline(x=0, color='#333333', linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([row["display_name"] for row in records], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Magnitude $-$ Random retention (%)", fontsize=12)
    ax.set_title("(b) Magnitude does not provide\na reliable retention gain", fontsize=12.5)
    ax.set_xlim(min(gaps) - 1.8, 2.2)
    ax.tick_params(axis='x', labelsize=10)
    for i, row in enumerate(records):
        gap = row["gap_mean"]
        ha = 'right' if gap < 0 else 'left'
        offset = -0.35 if gap < 0 else 0.25
        ax.text(gap + offset, i, f"{gap:+.1f}%", va='center', ha=ha,
                fontsize=9.5, color='#333333')

    # (c) Category-level gap over all available model-task pairs.
    ax = axes[2]
    categories = [row["category"] for row in category_summary]
    category_gaps = [row["gap_mean"] for row in category_summary]
    cat_y = np.arange(len(categories))
    cat_colors = [
        COLORS["optimized"] if gap >= 0 else COLORS["anti_opt"]
        for gap in category_gaps
    ]
    ax.barh(cat_y, category_gaps, color=cat_colors, alpha=0.78,
            height=0.62, edgecolor='white', linewidth=0.4)
    ax.axvline(x=0, color='#333333', linewidth=0.9)
    ax.set_yticks(cat_y)
    ax.set_yticklabels(categories, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Magnitude $-$ Random retention (%)", fontsize=12)
    ax.set_title("(c) Failures concentrate most\nstrongly on retrieval", fontsize=12.5)
    ax.set_xlim(min(category_gaps) - 1.8, 2.0)
    ax.tick_params(axis='x', labelsize=10)
    for i, row in enumerate(category_summary):
        gap = row["gap_mean"]
        if gap < -4:
            label_x, ha, color = gap + 0.7, 'left', 'white'
        elif gap < 0:
            label_x, ha, color = gap - 0.35, 'right', '#333333'
        else:
            label_x, ha, color = gap + 0.25, 'left', '#333333'
        ax.text(label_x, i, f"{gap:+.1f}%", va='center', ha=ha,
                fontsize=9.5, color=color)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["retrieval_native_embedding"], alpha=0.82,
                       label="Retrieval-optimized"),
        mpatches.Patch(facecolor=COLORS["adaptive_embedding"], alpha=0.82,
                       label="Instruction-conditioned"),
        mpatches.Patch(facecolor=COLORS["non_native_embedding_base"], alpha=0.82,
                       label="General-purpose Language Model Backbone"),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, framealpha=0.95, edgecolor='#DDDDDD')
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.18, wspace=0.38)
    path = os.path.join(OUTPUT_DIR, "fig9_magnitude_comparison.png")
    plt.savefig(path)
    plt.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGURE 11: Magnitude Scatter (redesign)
# ============================================================
def fig11_magnitude_scatter():
    """Redesigned Figure 11: Scatter plots of magnitude vs random."""
    set_style()
    analyze_data = load_analyze_data()

    gte_mteb_path = os.path.join(ANALYSIS_DIR, "magnitude_gte_mteb.json")
    stella_mteb_path = os.path.join(ANALYSIS_DIR, "magnitude_stella_mteb.json")

    if not os.path.exists(gte_mteb_path):
        print("No GTE MTEB results, skipping fig11")
        return

    with open(gte_mteb_path) as f:
        gte_mteb = json.load(f)

    has_stella = os.path.exists(stella_mteb_path)
    if has_stella:
        with open(stella_mteb_path) as f:
            stella_mteb = json.load(f)

    def build_scatter(analyze_data, model_name, mag_mteb):
        task_scores = get_method_scores(analyze_data, model_name)
        mag_scores = mag_mteb.get("scores", {})

        below, above, near = [], [], []
        for tname, tscores in task_scores.items():
            if tname == "STS17" or tname not in mag_scores:
                continue
            mag_ret = mag_scores[tname] / tscores["default"]
            rnd_ret = tscores["random"] / tscores["default"]
            if mag_ret > rnd_ret + 0.005:
                above.append((tname, rnd_ret, mag_ret))
            elif rnd_ret > mag_ret + 0.005:
                below.append((tname, rnd_ret, mag_ret))
            else:
                near.append((tname, rnd_ret, mag_ret))
        return below, above, near

    lims = [0.88, 1.02]

    if has_stella:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

        # GTE-Large
        below, above, near = build_scatter(analyze_data, "gte-large-en-v1.5", gte_mteb)
        if below:
            x, y = zip(*[(t[1], t[2]) for t in below])
            ax1.scatter(x, y, c=COLORS["anti_opt"], s=35, alpha=0.7,
                       label=f'Random wins ({len(below)})', zorder=3, edgecolors='white', linewidth=0.3)
        if above:
            x, y = zip(*[(t[1], t[2]) for t in above])
            ax1.scatter(x, y, c=COLORS["optimized"], s=35, alpha=0.7,
                       label=f'Magnitude wins ({len(above)})', zorder=3, edgecolors='white', linewidth=0.3)
        if near:
            x, y = zip(*[(t[1], t[2]) for t in near])
            ax1.scatter(x, y, c=COLORS["random"], s=35, alpha=0.7,
                       label=f'Tie ({len(near)})', zorder=3, edgecolors='white', linewidth=0.3)
        ax1.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal performance')
        ax1.set_xlim(lims); ax1.set_ylim(lims)
        ax1.set_xlabel('Random Selection Retention', fontsize=11)
        ax1.set_ylabel('Magnitude Selection Retention', fontsize=11)
        ax1.set_title('(a) GTE-Large (dim=256)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax1.set_aspect('equal')
        ax1.annotate('$d = -0.28$, $p = 0.002$',
                     xy=(0.98, 0.02), xycoords='axes fraction',
                     ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.8))

        # Stella
        below, above, near = build_scatter(analyze_data, "stella_en_400M_v5", stella_mteb)
        if below:
            x, y = zip(*[(t[1], t[2]) for t in below])
            ax2.scatter(x, y, c=COLORS["anti_opt"], s=35, alpha=0.7,
                       label=f'Random wins ({len(below)})', zorder=3, edgecolors='white', linewidth=0.3)
        if above:
            x, y = zip(*[(t[1], t[2]) for t in above])
            ax2.scatter(x, y, c=COLORS["optimized"], s=35, alpha=0.7,
                       label=f'Magnitude wins ({len(above)})', zorder=3, edgecolors='white', linewidth=0.3)
        if near:
            x, y = zip(*[(t[1], t[2]) for t in near])
            ax2.scatter(x, y, c=COLORS["random"], s=35, alpha=0.7,
                       label=f'Tie ({len(near)})', zorder=3, edgecolors='white', linewidth=0.3)
        ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal performance')
        ax2.set_xlim(lims); ax2.set_ylim(lims)
        ax2.set_xlabel('Random Selection Retention', fontsize=11)
        ax2.set_ylabel('Magnitude Selection Retention', fontsize=11)
        ax2.set_title('(b) Stella EN 400M (dim=256)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax2.set_aspect('equal')
        ax2.annotate('$d = -0.05$, $p = 0.38$',
                     xy=(0.98, 0.02), xycoords='axes fraction',
                     ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig11_magnitude_scatter.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Redesigning all figures...")
    print("\n[1] Figure 2: Optimized-Random Gap")
    fig2_opt_random_gap()
    print("\n[2] Figure 3: All Methods Comparison")
    fig3_all_methods()
    print("\n[3] Figure 4: Magnitude Analysis")
    fig4_magnitude_analysis()
    print("\n[4] Figure 11: Magnitude Scatter")
    fig11_magnitude_scatter()
    print("\nAll figures redesigned!")

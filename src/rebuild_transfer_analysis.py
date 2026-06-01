"""
Rebuild cross-task transfer analysis from raw analyze_new/task_similar_new data.

This script is intentionally full-rebuild only. It replaces the previous
append-style workflow so the aggregate CSVs, summaries, and paper tables all
share one data definition:

* use data/analyze_new and data/task_similar_new/*_by_dim.json only
* exclude STS17 everywhere
* include diagonal self-transfer pairs (ref == tgt)
* keep transfer/random retention complete, even when oracle/worst are missing
"""

from __future__ import annotations

import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parent.parent
ANALYZE_DIR = PROJECT_DIR / "data" / "analyze_new"
TASK_SIMILAR_DIR = PROJECT_DIR / "data" / "task_similar_new"
OUTPUT_DIR = PROJECT_DIR / "data" / "analysis_results_multidim"

DIMS = [16, 32, 64, 128, 256, 512]
EXCLUDE_TASKS = {"STS17"}
EXPECTED_MODELS = 11
EXPECTED_TASKS = 34

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
    "PairClassification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "Retrieval": [
        "ArguAna",
        "CQADupstackEnglishRetrieval",
        "NFCorpus",
        "SCIDOCS",
        "SciFact",
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
        "STS22",
        "STSBenchmark",
    ],
    "Summarization": ["SummEval"],
}

CATEGORY_ORDER = [
    "Classification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
]

TASK_TO_CATEGORY = {
    task_name: category
    for category, task_names in CATEGORY_TASKS.items()
    for task_name in task_names
}

TASK_ORDER = [
    task_name
    for category in CATEGORY_ORDER
    for task_name in CATEGORY_TASKS[category]
    if task_name not in {"MindSmallReranking", "STS17", "STS22"}
]

MODEL_DISPLAY = {
    "roberta-large": "Roberta-Large",
    "roberta-large-InBedder": "Roberta-InBedder",
    "bart-base": "BART-Base",
    "gte-large-en-v1.5": "GTE-Large",
    "stella_en_400M_v5": "Stella EN 400M",
    "mxbai-embed-large-v1": "MxBai-Embed-Large",
    "instructor-large": "Instructor-Large",
    "gte-base": "GTE-Base",
    "gtr-t5-large": "GTR-T5-Large",
    "bge-m3": "BGE-M3",
    "Qwen3-Embedding-0.6B": "Qwen3-Embed-0.6B",
}

MODEL_ORDER = [
    "roberta-large",
    "roberta-large-InBedder",
    "bart-base",
    "gte-large-en-v1.5",
    "stella_en_400M_v5",
    "mxbai-embed-large-v1",
    "instructor-large",
    "gte-base",
    "gtr-t5-large",
    "bge-m3",
    "Qwen3-Embedding-0.6B",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean_score(value: Any) -> float:
    if isinstance(value, list):
        finite_values = [float(v) for v in value if is_finite_number(v)]
        if not finite_values:
            raise ValueError("score list has no finite values")
        return float(statistics.mean(finite_values))
    if is_finite_number(value):
        return float(value)
    raise ValueError(f"score is not finite: {value!r}")


def task_sort_key(task_name: str) -> tuple[int, int, str]:
    category = TASK_TO_CATEGORY.get(task_name, "Unknown")
    category_idx = CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)
    try:
        task_idx = TASK_ORDER.index(task_name)
    except ValueError:
        task_idx = len(TASK_ORDER)
    return category_idx, task_idx, task_name


def model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(model_name), model_name
    except ValueError:
        return len(MODEL_ORDER), model_name


def available_models() -> list[str]:
    models = []
    for by_dim_path in sorted(TASK_SIMILAR_DIR.glob("*_by_dim.json")):
        model_name = by_dim_path.name.removesuffix("_by_dim.json")
        if (ANALYZE_DIR / f"{model_name}.json").exists():
            models.append(model_name)
    return sorted(models, key=model_sort_key)


def model_tasks(model_name: str, analyze: dict[str, Any], task_similar: dict[str, Any]) -> list[str]:
    analyze_tasks = set(analyze["task_name"])
    transfer_tasks = set(task_similar)
    tasks = [
        task_name
        for task_name in (analyze_tasks & transfer_tasks)
        if task_name not in EXCLUDE_TASKS
    ]
    return sorted(tasks, key=task_sort_key)


def oracle_scores(task_data: dict[str, Any], dim: int) -> tuple[float, float]:
    chunk_win_size = (
        task_data.get("split_win_size", {})
        .get("2", {})
        .get("chunk_win_size", {})
    )
    dim_data = chunk_win_size.get(str(dim))
    if not isinstance(dim_data, dict):
        return math.nan, math.nan

    try:
        oracle = float(dim_data["head_score"]["main_score"])
        worst = float(dim_data["end_score"]["main_score"])
    except (KeyError, TypeError, ValueError):
        return math.nan, math.nan
    return oracle, worst


def rebuild_transfer_records() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    task_sets: dict[str, list[str]] = {}

    for model_name in available_models():
        analyze = load_json(ANALYZE_DIR / f"{model_name}.json")
        task_similar = load_json(TASK_SIMILAR_DIR / f"{model_name}_by_dim.json")
        tasks = model_tasks(model_name, analyze, task_similar)
        task_sets[model_name] = tasks

        for ref in tasks:
            ref_cat = TASK_TO_CATEGORY.get(ref, "Unknown")
            for tgt in tasks:
                tgt_cat = TASK_TO_CATEGORY.get(tgt, "Unknown")
                tgt_data = analyze["task_name"][tgt]
                full = float(tgt_data["defult_score"])
                if not math.isfinite(full) or full == 0:
                    raise ValueError(f"{model_name}/{tgt} has invalid full score: {full}")

                for dim in DIMS:
                    dim_s = str(dim)
                    transfer = float(task_similar[ref][dim_s][tgt])
                    random_score = mean_score(tgt_data["random_score"][dim_s])
                    sort_score = mean_score(tgt_data["sort_score"][dim_s])
                    oracle, worst = oracle_scores(tgt_data, dim)

                    row = {
                        "model": model_name,
                        "ref": ref,
                        "tgt": tgt,
                        "ref_cat": ref_cat,
                        "tgt_cat": tgt_cat,
                        "same_cat": ref_cat == tgt_cat,
                        "dim": dim,
                        "transfer": transfer,
                        "full": full,
                        "retention": transfer / full,
                        "random": random_score,
                        "oracle": oracle,
                        "worst": worst,
                        "sort": sort_score,
                        "ret_random": random_score / full,
                        "ret_oracle": oracle / full if math.isfinite(oracle) else math.nan,
                        "ret_worst": worst / full if math.isfinite(worst) else math.nan,
                        "is_self_transfer": ref == tgt,
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df, task_sets


def compute_rank_correlations(task_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for model_name, tasks in task_sets.items():
        analyze = load_json(ANALYZE_DIR / f"{model_name}.json")
        chunk_scores = {}
        for task_name in tasks:
            scores = (
                analyze["task_name"][task_name]
                .get("split_win_size", {})
                .get("2", {})
                .get("chunk_result")
            )
            if isinstance(scores, list) and scores:
                chunk_scores[task_name] = np.asarray(scores, dtype=float)

        for task_a, task_b in combinations(tasks, 2):
            if task_a not in chunk_scores or task_b not in chunk_scores:
                continue
            if len(chunk_scores[task_a]) != len(chunk_scores[task_b]):
                continue
            rho = spearmanr(chunk_scores[task_a], chunk_scores[task_b]).correlation
            if math.isfinite(float(rho)):
                rows.append(
                    {
                        "model": model_name,
                        "task_a": task_a,
                        "task_b": task_b,
                        "rho": float(rho),
                    }
                )

    return pd.DataFrame(rows)


def summarize_by_dim(sub: pd.DataFrame) -> list[dict[str, float | int]]:
    rows = []
    for dim in DIMS:
        dim_df = sub[sub["dim"] == dim]
        rows.append(
            {
                "dim": int(dim),
                "n": int(len(dim_df)),
                "transfer_mean_pct": float(dim_df["retention"].mean() * 100.0),
                "transfer_median_pct": float(dim_df["retention"].median() * 100.0),
                "random_mean_pct": float(dim_df["ret_random"].mean() * 100.0),
                "random_median_pct": float(dim_df["ret_random"].median() * 100.0),
                "gap_mean_pp": float((dim_df["retention"].mean() - dim_df["ret_random"].mean()) * 100.0),
                "gap_median_pp": float((dim_df["retention"] - dim_df["ret_random"]).median() * 100.0),
            }
        )
    return rows


def build_per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in sorted(df["model"].unique(), key=model_sort_key):
        model_df = df[df["model"] == model_name]
        row: dict[str, Any] = {
            "model": model_name,
            "display": MODEL_DISPLAY.get(model_name, model_name),
            "n_tasks": int(model_df["ref"].nunique()),
            "n_pairs_per_dim": int(len(model_df[model_df["dim"] == DIMS[0]])),
        }
        for dim in DIMS:
            dim_df = model_df[model_df["dim"] == dim]
            row[f"Ret{dim}_mean"] = float(dim_df["retention"].mean() * 100.0)
            row[f"Ret{dim}_median"] = float(dim_df["retention"].median() * 100.0)
        dim_256 = model_df[model_df["dim"] == 256]
        row["Random256_mean"] = float(dim_256["ret_random"].mean() * 100.0)
        row["Random256_median"] = float(dim_256["ret_random"].median() * 100.0)
        row["Gap256_mean"] = float(
            (dim_256["retention"].mean() - dim_256["ret_random"].mean()) * 100.0
        )
        row["Gap256_median"] = float(
            (dim_256["retention"] - dim_256["ret_random"]).median() * 100.0
        )
        row["Oracle256_mean"] = float(dim_256["ret_oracle"].mean(skipna=True) * 100.0)
        row["Oracle256_missing_targets"] = int(
            dim_256.groupby("tgt")["ret_oracle"].first().isna().sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dim", "ref_cat", "tgt_cat"], dropna=False)
        .agg(retention_pct=("retention", lambda s: float(s.mean() * 100.0)), n=("retention", "size"))
        .reset_index()
    )
    grouped["ref_cat"] = pd.Categorical(grouped["ref_cat"], CATEGORY_ORDER, ordered=True)
    grouped["tgt_cat"] = pd.Categorical(grouped["tgt_cat"], CATEGORY_ORDER, ordered=True)
    return grouped.sort_values(["dim", "ref_cat", "tgt_cat"]).reset_index(drop=True)


def write_latex_tables(per_model: pd.DataFrame, pooled: list[dict[str, Any]]) -> None:
    pooled_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Retained Dims & Mean Transfer & Mean Random & Gap \\",
        r"\midrule",
    ]
    for row in pooled:
        pooled_lines.append(
            f"{int(row['dim'])} & {row['transfer_mean_pct']:.1f}\\% & "
            f"{row['random_mean_pct']:.1f}\\% & {row['gap_mean_pp']:+.1f}pp \\\\"
        )
    pooled_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUTPUT_DIR / "cross_task_pooled_table.tex").write_text("\n".join(pooled_lines))

    table_df = per_model.sort_values("Ret256_mean", ascending=False)
    model_lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Model & Ret@16 & Ret@32 & Ret@64 & Ret@128 & Ret@256 & Ret@512 & Random@256 & Gap@256 \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        model_lines.append(
            f"{row['display']} & "
            f"{row['Ret16_mean']:.1f}\\% & {row['Ret32_mean']:.1f}\\% & "
            f"{row['Ret64_mean']:.1f}\\% & {row['Ret128_mean']:.1f}\\% & "
            f"{row['Ret256_mean']:.1f}\\% & {row['Ret512_mean']:.1f}\\% & "
            f"{row['Random256_mean']:.1f}\\% & {row['Gap256_mean']:+.1f}pp \\\\"
        )
    model_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUTPUT_DIR / "cross_task_multidim.tex").write_text("\n".join(model_lines))


def validate_outputs(df: pd.DataFrame, rank_df: pd.DataFrame, task_sets: dict[str, list[str]]) -> dict[str, Any]:
    n_models = len(task_sets)
    task_counts = {model: len(tasks) for model, tasks in task_sets.items()}
    expected_rows = n_models * EXPECTED_TASKS * EXPECTED_TASKS * len(DIMS)
    expected_self_rows = n_models * EXPECTED_TASKS * len(DIMS)
    expected_rank_pairs = n_models * (EXPECTED_TASKS * (EXPECTED_TASKS - 1) // 2)

    required_no_nan = ["transfer", "full", "retention", "random", "ret_random"]
    validation = {
        "n_models": int(n_models),
        "task_counts": task_counts,
        "n_rows": int(len(df)),
        "expected_rows": int(expected_rows),
        "sts17_rows": int(((df["ref"] == "STS17") | (df["tgt"] == "STS17")).sum()),
        "self_rows": int(df["is_self_transfer"].sum()),
        "expected_self_rows": int(expected_self_rows),
        "required_nan_counts": {col: int(df[col].isna().sum()) for col in required_no_nan},
        "rank_correlation_rows": int(len(rank_df)),
        "expected_rank_correlation_rows": int(expected_rank_pairs),
    }

    problems = []
    if n_models != EXPECTED_MODELS:
        problems.append(f"expected {EXPECTED_MODELS} models, found {n_models}")
    bad_task_counts = {m: n for m, n in task_counts.items() if n != EXPECTED_TASKS}
    if bad_task_counts:
        problems.append(f"unexpected task counts: {bad_task_counts}")
    if len(df) != expected_rows:
        problems.append(f"expected {expected_rows} rows, found {len(df)}")
    if validation["sts17_rows"] != 0:
        problems.append("STS17 rows are present")
    if validation["self_rows"] != expected_self_rows:
        problems.append(f"expected {expected_self_rows} self rows, found {validation['self_rows']}")
    nan_counts = validation["required_nan_counts"]
    if any(count != 0 for count in nan_counts.values()):
        problems.append(f"required columns contain NaNs: {nan_counts}")
    if len(rank_df) != expected_rank_pairs:
        problems.append(f"expected {expected_rank_pairs} rank pairs, found {len(rank_df)}")

    validation["passed"] = not problems
    validation["problems"] = problems
    if problems:
        raise AssertionError("; ".join(problems))

    return validation


def build_summary(
    df: pd.DataFrame,
    per_model: pd.DataFrame,
    category: pd.DataFrame,
    rank_df: pd.DataFrame,
    validation: dict[str, Any],
) -> dict[str, Any]:
    all_by_dim = summarize_by_dim(df)
    offdiag_by_dim = summarize_by_dim(df[~df["is_self_transfer"]])
    no_roberta_by_dim = summarize_by_dim(df[df["model"] != "roberta-large"])

    rank_values = rank_df["rho"]
    rank_summary = {
        "definition": "Spearman correlations over off-diagonal task pairs only; diagonal self-correlations are excluded.",
        "n_pairs": int(len(rank_df)),
        "mean_rho": float(rank_values.mean()),
        "median_rho": float(rank_values.median()),
        "p05_rho": float(rank_values.quantile(0.05)),
        "p95_rho": float(rank_values.quantile(0.95)),
    }

    category_256 = category[category["dim"] == 256]

    return {
        "config": {
            "dims": DIMS,
            "exclude_tasks": sorted(EXCLUDE_TASKS),
            "data_sources": ["data/analyze_new", "data/task_similar_new"],
            "include_self_transfer": True,
            "n_models": int(df["model"].nunique()),
            "n_tasks": int(df["ref"].nunique()),
        },
        "validation": validation,
        "pooled_by_dim": all_by_dim,
        "offdiag_only_by_dim": offdiag_by_dim,
        "pooled_excluding_roberta_large_by_dim": no_roberta_by_dim,
        "per_model_retention": per_model.to_dict(orient="records"),
        "category_transfer_dim256": category_256.to_dict(orient="records"),
        "rank_correlation": rank_summary,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, task_sets = rebuild_transfer_records()
    rank_df = compute_rank_correlations(task_sets)
    validation = validate_outputs(df, rank_df, task_sets)

    per_model = build_per_model_summary(df)
    category = build_category_summary(df)
    summary = build_summary(df, per_model, category, rank_df, validation)

    field_order = [
        "model",
        "ref",
        "tgt",
        "ref_cat",
        "tgt_cat",
        "same_cat",
        "dim",
        "transfer",
        "full",
        "retention",
        "random",
        "oracle",
        "worst",
        "sort",
        "ret_random",
        "ret_oracle",
        "ret_worst",
        "is_self_transfer",
    ]
    df[field_order].to_csv(OUTPUT_DIR / "transfer_records.csv", index=False)
    per_model.to_csv(OUTPUT_DIR / "cross_task_multidim.csv", index=False)
    category.to_csv(OUTPUT_DIR / "category_transfer_per_dim.csv", index=False)
    rank_df.to_csv(OUTPUT_DIR / "rank_correlations.csv", index=False)
    write_latex_tables(per_model, summary["pooled_by_dim"])

    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("Rebuilt cross-task transfer analysis")
    print(f"  records: {len(df)} rows")
    print(f"  self-transfer rows: {int(df['is_self_transfer'].sum())}")
    print(f"  rank correlations: {len(rank_df)} off-diagonal pairs")
    print("  validation: passed")


if __name__ == "__main__":
    main()

# analysis/compute_metrics.py
"""Compute statistical metrics for paper tables and figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


def bootstrap_ci(data: pd.Series, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (0.0, 0.0, 0.0)

    means = []
    for _ in range(n_bootstrap):
        sample = data.sample(n=len(data), replace=True)
        means.append(sample.mean())

    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return (data.mean(), lower, upper)


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all metrics for the paper."""
    metrics = {}

    # 1. Overall statistics
    metrics["overall"] = {
        "total_records": len(df),
        "total_models": df["gen_model"].nunique(),
        "total_prompts": df["prompt_id"].nunique(),
        "hallucination_rate": df["is_hallucination"].mean(),
        "refusal_rate": df["is_refusal"].mean(),
        "correct_rate": df["is_correct"].mean(),
    }

    # 2. Per-model metrics
    model_metrics = []
    for model in df["gen_model"].unique():
        model_df = df[df["gen_model"] == model]
        hr_mean, hr_lower, hr_upper = bootstrap_ci(model_df["is_hallucination"])
        rr_mean, rr_lower, rr_upper = bootstrap_ci(model_df["is_refusal"])

        model_metrics.append({
            "model": model,
            "n_samples": len(model_df),
            "hallucination_rate": hr_mean,
            "hr_ci_lower": hr_lower,
            "hr_ci_upper": hr_upper,
            "refusal_rate": rr_mean,
            "rr_ci_lower": rr_lower,
            "rr_ci_upper": rr_upper,
            "correct_rate": model_df["is_correct"].mean(),
        })

    metrics["by_model"] = sorted(model_metrics, key=lambda x: x["hallucination_rate"])

    # 3. Per-category metrics (Axis 1)
    category_metrics = []
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        hr_mean, hr_lower, hr_upper = bootstrap_ci(cat_df["is_hallucination"])

        category_metrics.append({
            "category": category,
            "n_samples": len(cat_df),
            "hallucination_rate": hr_mean,
            "hr_ci_lower": hr_lower,
            "hr_ci_upper": hr_upper,
        })

    metrics["by_category"] = category_metrics

    # 4. Per-subcategory metrics
    subcategory_metrics = []
    for subcat in df["subcategory"].unique():
        subcat_df = df[df["subcategory"] == subcat]
        hr_mean, hr_lower, hr_upper = bootstrap_ci(subcat_df["is_hallucination"])

        subcategory_metrics.append({
            "subcategory": subcat,
            "n_samples": len(subcat_df),
            "hallucination_rate": hr_mean,
            "hr_ci_lower": hr_lower,
            "hr_ci_upper": hr_upper,
        })

    metrics["by_subcategory"] = sorted(
        subcategory_metrics,
        key=lambda x: x["hallucination_rate"],
        reverse=True,
    )

    # 5. Behavior distribution (for hallucinations only)
    hall_df = df[df["is_hallucination"]]
    if len(hall_df) > 0:
        behavior_counts = hall_df["behavior"].value_counts()
        behavior_pcts = hall_df["behavior"].value_counts(normalize=True)
        metrics["behavior_distribution"] = {
            b: {"count": int(behavior_counts.get(b, 0)), "pct": float(behavior_pcts.get(b, 0))}
            for b in ["B1", "B2", "B3", "B4"]
        }
    else:
        metrics["behavior_distribution"] = {}

    # 6. Axis 1 category distribution (for hallucinations only)
    if len(hall_df) > 0:
        axis1_counts = hall_df["axis1_category"].value_counts()
        axis1_pcts = hall_df["axis1_category"].value_counts(normalize=True)
        metrics["axis1_distribution"] = {
            cat: {"count": int(axis1_counts.get(cat, 0)), "pct": float(axis1_pcts.get(cat, 0))}
            for cat in ["absolute_fundamental", "absolute_contingent", "relative"]
        }
    else:
        metrics["axis1_distribution"] = {}

    # 7. Per-language metrics
    language_metrics = []
    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang]
        hr_mean, hr_lower, hr_upper = bootstrap_ci(lang_df["is_hallucination"])

        language_metrics.append({
            "language": lang,
            "n_samples": len(lang_df),
            "hallucination_rate": hr_mean,
            "hr_ci_lower": hr_lower,
            "hr_ci_upper": hr_upper,
        })

    metrics["by_language"] = language_metrics

    # 8. Model x Category heatmap data
    heatmap_data = df.pivot_table(
        index="gen_model",
        columns="category",
        values="is_hallucination",
        aggfunc="mean",
    ).to_dict()
    metrics["model_category_heatmap"] = heatmap_data

    # 9. Model x Behavior heatmap (for hallucinations)
    if len(hall_df) > 0:
        behavior_heatmap = hall_df.pivot_table(
            index="gen_model",
            columns="behavior",
            values="is_hallucination",  # Just for counting
            aggfunc="count",
        ).fillna(0)
        # Normalize by row
        behavior_heatmap = behavior_heatmap.div(behavior_heatmap.sum(axis=1), axis=0)
        metrics["model_behavior_heatmap"] = behavior_heatmap.to_dict()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from merged results")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to merged results CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis_out/metrics.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records")

    # Compute metrics
    metrics = compute_metrics(df)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"Saved metrics to {args.out}")

    # Print summary
    print("\n=== Key Metrics ===")
    print(f"Overall Hallucination Rate: {metrics['overall']['hallucination_rate']:.1%}")
    print(f"Overall Refusal Rate: {metrics['overall']['refusal_rate']:.1%}")
    print("\nBy Model (sorted by HR):")
    for m in metrics["by_model"][:5]:
        print(f"  {m['model']}: HR={m['hallucination_rate']:.1%} [{m['hr_ci_lower']:.1%}, {m['hr_ci_upper']:.1%}]")


if __name__ == "__main__":
    main()

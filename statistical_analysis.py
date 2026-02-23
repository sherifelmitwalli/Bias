"""
statistical_analysis.py — between-model comparisons and correlation matrix.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


RESULTS_PATH = Path("outputs/crewai_bias_assessment_results.json")
OUTPUT_DIR = Path("outputs/statistical_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results not found: {RESULTS_PATH}")

    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for rec in results:
        cr = rec.get("crew_result", {})
        rows.append({
            "model": rec.get("llm"),
            "category": rec.get("category"),
            "bias_score": cr.get("bias_score"),
            "factual_accuracy": cr.get("factual_accuracy"),
            "evidence_alignment": cr.get("evidence_alignment"),
            "risk_minimization": cr.get("risk_minimization"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results_flat.csv", index=False)

    models = df["model"].dropna().unique().tolist()
    if len(models) == 2:
        m1, m2 = models
        m1_scores = df[df["model"] == m1]["bias_score"].dropna().values
        m2_scores = df[df["model"] == m2]["bias_score"].dropna().values

        U, p = mannwhitneyu(m1_scores, m2_scores, alternative="two-sided")
        n1, n2 = len(m1_scores), len(m2_scores)
        r = 1 - (2 * U) / (n1 * n2) if n1 and n2 else np.nan

        summary = pd.DataFrame([
            {
                "model_1": m1,
                "model_2": m2,
                "U": U,
                "p_value": p,
                "rank_biserial_r": r,
                "model_1_mean": np.mean(m1_scores) if n1 else np.nan,
                "model_1_sd": np.std(m1_scores) if n1 else np.nan,
                "model_2_mean": np.mean(m2_scores) if n2 else np.nan,
                "model_2_sd": np.std(m2_scores) if n2 else np.nan,
            }
        ])
        summary.to_csv(OUTPUT_DIR / "between_model_comparison.csv", index=False)

    # Category-level stats
    cat_rows = []
    for cat in sorted(df["category"].dropna().unique()):
        cat_df = df[df["category"] == cat]
        for model in models:
            scores = cat_df[cat_df["model"] == model]["bias_score"].dropna().values
            cat_rows.append({
                "category": cat,
                "model": model,
                "mean": np.mean(scores) if len(scores) else np.nan,
                "sd": np.std(scores) if len(scores) else np.nan,
                "n": len(scores),
            })
    pd.DataFrame(cat_rows).to_csv(OUTPUT_DIR / "category_summary.csv", index=False)

    # Correlation matrix
    metric_cols = ["bias_score", "factual_accuracy", "evidence_alignment", "risk_minimization"]
    corr = df[metric_cols].corr(method="spearman")
    corr.to_csv(OUTPUT_DIR / "metric_correlation_matrix.csv")


if __name__ == "__main__":
    main()
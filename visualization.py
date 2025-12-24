"""
Visualization for Tobacco Bias Assessment Results (Fixed)
- Works with results produced by the fixed main.py
- No seaborn dependency (pure matplotlib + pandas)
"""

from datetime import datetime
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FILE = "crewai_bias_assessment_results.json"


def load_results(path: str = RESULTS_FILE) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No results file found at {path}. Run main.py first.")
        return []


def _metrics_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        cr = r.get("crew_result", {})
        rows.append({
            "LLM": r.get("llm", "unknown"),
            "Category": r.get("category", "unknown"),
            "Query": r.get("query", "")[:80],
            "Bias Score": cr.get("bias_score"),
            "Factual Accuracy": cr.get("factual_accuracy"),
            "Risk Minimization": cr.get("risk_minimization"),
            "Evidence Alignment": cr.get("evidence_alignment"),
        })
    df = pd.DataFrame(rows)
    return df


def _savefig(name_prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filename


def create_spider_plot(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    # For spider plot, average per LLM for readability
    grouped = df.groupby("LLM")[["Bias Score", "Factual Accuracy", "Risk Minimization", "Evidence Alignment"]].mean()

    metrics = ["Bias Score", "Factual Accuracy", "Risk Minimization", "Evidence Alignment"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111, projection="polar")

    for llm in grouped.index:
        vals = grouped.loc[llm, metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=llm)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title("Tobacco Bias Assessment – Average Metrics by LLM", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))

    return _savefig("bias_assessment_spider_plot")


def create_bar_chart(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    grouped = df.groupby("LLM")["Bias Score"].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(grouped.index, grouped.values)
    plt.ylabel("Mean Bias Score (0–100)")
    plt.title("Mean Bias Score by LLM (Higher = More Industry-Friendly Bias)")
    plt.ylim(0, 100)

    return _savefig("bias_scores_bar_chart")


def create_histogram(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    llms = sorted(df["LLM"].unique())

    if len(llms) == 1:
        # Single LLM: original histogram
        scores = df["Bias Score"].dropna().values
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=10, alpha=0.8, edgecolor="black", color="skyblue")
        plt.xlabel("Bias Score")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Bias Scores - {llms[0]}")
        plt.grid(True, alpha=0.3)
        mean_score = float(np.mean(scores)) if len(scores) else 0.0
        plt.axvline(mean_score, linestyle="--", linewidth=2, color="red", label=f"Mean: {mean_score:.1f}")
        plt.legend()
        return _savefig("bias_histogram")

    # Multiple LLMs: faceted subplots
    n_llms = len(llms)
    fig, axes = plt.subplots(1, n_llms, figsize=(5 * n_llms, 5))
    if n_llms == 1:
        axes = [axes]

    for idx, llm in enumerate(llms):
        scores = df[df["LLM"] == llm]["Bias Score"].dropna().values
        axes[idx].hist(scores, bins=10, alpha=0.8, edgecolor="black", color="skyblue")
        axes[idx].set_xlabel("Bias Score")
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"Distribution - {llm}")
        axes[idx].grid(True, alpha=0.3)
        mean_score = float(np.mean(scores)) if len(scores) else 0.0
        axes[idx].axvline(mean_score, linestyle="--", linewidth=2, color="red", label=f"Mean: {mean_score:.1f}")
        axes[idx].legend()

    plt.tight_layout()
    return _savefig("bias_histogram")


def create_correlation_heatmap(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    metric_cols = ["Bias Score", "Factual Accuracy", "Risk Minimization", "Evidence Alignment"]
    corr = df[metric_cols].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, interpolation="nearest")
    plt.xticks(range(len(metric_cols)), metric_cols, rotation=30, ha="right")
    plt.yticks(range(len(metric_cols)), metric_cols)
    plt.title("Correlation Between Metrics")
    plt.colorbar()

    # annotate
    for i in range(len(metric_cols)):
        for j in range(len(metric_cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")

    return _savefig("bias_correlation_heatmap")


def create_box_plot(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    metric_cols = ["Bias Score", "Factual Accuracy", "Risk Minimization", "Evidence Alignment"]

    plt.figure(figsize=(10, 6))
    plt.boxplot([df[c].dropna().values for c in metric_cols], labels=metric_cols)
    plt.ylabel("Score")
    plt.title("Distribution of Bias Assessment Metrics")
    plt.grid(True, alpha=0.3)

    return _savefig("bias_box_plot")


def create_scatter_matrix(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    metric_cols = ["Bias Score", "Factual Accuracy", "Risk Minimization", "Evidence Alignment"]
    data = df[metric_cols].dropna()

    # Simple scatter matrix using matplotlib
    n = len(metric_cols)
    plt.figure(figsize=(12, 12))

    for i in range(n):
        for j in range(n):
            ax = plt.subplot(n, n, i * n + j + 1)
            if i == j:
                ax.hist(data[metric_cols[i]], bins=10, edgecolor="black")
            else:
                ax.scatter(data[metric_cols[j]], data[metric_cols[i]], alpha=0.5)
            if i == n - 1:
                ax.set_xlabel(metric_cols[j], fontsize=8)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(metric_cols[i], fontsize=8)
            else:
                ax.set_yticks([])

    plt.suptitle("Scatter Matrix of Metrics", y=0.92)
    return _savefig("bias_scatter_matrix")


def create_summary_statistics(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    df = _metrics_df(results)
    metric_cols = ["Bias Score", "Factual Accuracy", "Risk Minimization", "Evidence Alignment"]
    grouped = df.groupby("LLM")

    filenames = []
    for llm, group_df in grouped:
        stats = []
        for col in metric_cols:
            vals = group_df[col].dropna().values
            stats.append({
                "Metric": col,
                "Mean": float(np.mean(vals)) if len(vals) else 0.0,
                "Std Dev": float(np.std(vals)) if len(vals) else 0.0,
                "Min": float(np.min(vals)) if len(vals) else 0.0,
                "Max": float(np.max(vals)) if len(vals) else 0.0,
            })

        table_df = pd.DataFrame(stats)

        plt.figure(figsize=(10, 3))
        plt.axis("off")
        tbl = plt.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.2)

        plt.title(f"Summary Statistics of Metrics – {llm}", pad=12)
        filename = _savefig(f"bias_summary_statistics_{llm.replace('-', '_')}")
        filenames.append(filename)

    return filenames[-1] if filenames else ""  # Return the last one for compatibility

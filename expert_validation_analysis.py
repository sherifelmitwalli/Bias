"""
expert_validation_analysis.py — manuscript-aligned validation
Computes: ICC(2,1) with 95% CI, bootstrapped Spearman/MAE, weighted kappa,
per-dimension agreement (if available), and missing-data reporting.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

OUTPUT_DIR = Path("expert_validation_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLDS = {
    "icc_good": 0.75,
    "icc_moderate": 0.50,
    "spearman_strong": 0.70,
    "mae_acceptable": 10.0,
    "weighted_kappa_substantial": 0.61,
}

DIMENSIONS = [
    "factual_accuracy",
    "evidence_alignment",
    "risk_minimisation_avoidance",
    "overall_bias",
]

N_BOOTSTRAP = 5000
SEED = 42


def bootstrap_ci(func, x, y, n_boot=N_BOOTSTRAP, ci=0.95):
    rng = np.random.RandomState(SEED)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        stats.append(func(x[idx], y[idx]))
    alpha = (1 - ci) / 2
    return {
        "point": round(func(x, y), 3),
        "ci_lower": round(np.percentile(stats, 100 * alpha), 3),
        "ci_upper": round(np.percentile(stats, 100 * (1 - alpha)), 3),
    }


def spearman_func(x, y):
    return spearmanr(x, y)[0]


def mae_func(x, y):
    return float(np.mean(np.abs(x - y)))


def compute_icc(scores1, scores2, item_ids):
    df = pd.DataFrame({
        "item": list(item_ids) * 2,
        "rater": ["r1"] * len(item_ids) + ["r2"] * len(item_ids),
        "score": list(scores1) + list(scores2),
    })
    icc = pg.intraclass_corr(data=df, targets="item", raters="rater", ratings="score")
    row = icc[icc["Type"] == "ICC2"]
    return {
        "icc": round(row["ICC"].values[0], 3),
        "ci_lower": round(row["CI95%"].values[0][0], 3),
        "ci_upper": round(row["CI95%"].values[0][1], 3),
        "p_value": round(row["pval"].values[0], 4),
    }


def weighted_kappa(scores1, scores2, bins=(0, 33.333, 66.666, 100.0)):
    labels = ["low", "medium", "high"]
    b1 = pd.cut(scores1, bins=bins, labels=labels, include_lowest=True)
    b2 = pd.cut(scores2, bins=bins, labels=labels, include_lowest=True)
    return float(cohen_kappa_score(b1, b2, weights="quadratic", labels=labels))


def interpret(metrics):
    icc_val = metrics.get("icc", {}).get("icc", 0)
    spearman_val = metrics.get("spearman", {}).get("point", 0)
    mae_val = metrics.get("mae", {}).get("point", 100)
    return {
        "icc": "GOOD" if icc_val >= THRESHOLDS["icc_good"] else "MODERATE" if icc_val >= THRESHOLDS["icc_moderate"] else "POOR",
        "spearman": "STRONG" if spearman_val >= THRESHOLDS["spearman_strong"] else "WEAK",
        "mae": "ACCEPTABLE" if mae_val <= THRESHOLDS["mae_acceptable"] else "HIGH",
    }


def handle_missing(e1_df, e2_df, score_col):
    m1 = e1_df[score_col].isna()
    m2 = e2_df[score_col].isna()
    mask = ~m1 & ~m2
    report = {
        "total": len(e1_df),
        "complete": int(mask.sum()),
        "rate": round(mask.sum() / len(e1_df) * 100, 1) if len(e1_df) else 0.0,
    }
    if report["rate"] < 80:
        print(f"⚠ Completion rate {report['rate']}% < 80%")
    return e1_df[mask].reset_index(drop=True), e2_df[mask].reset_index(drop=True), report


def load_ai_scores(results_path: Path):
    if not results_path.exists():
        return None
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ai_map = {}
    for rec in data:
        query = rec.get("query", "")
        llm = rec.get("llm", "")
        bias = rec.get("crew_result", {}).get("bias_score")
        if query and llm and bias is not None:
            ai_map[(query, llm)] = float(bias)
    return ai_map


def per_dimension_agreement(e1_df, e2_df, ai_results, item_ids):
    if ai_results is None:
        return {}

    dim_results = {}
    for dim in DIMENSIONS:
        e1_col = dim
        e2_col = dim
        if e1_col not in e1_df.columns or e2_col not in e2_df.columns:
            continue

        e1_vals = pd.to_numeric(e1_df[e1_col], errors="coerce").to_numpy(float)
        e2_vals = pd.to_numeric(e2_df[e2_col], errors="coerce").to_numpy(float)
        valid_mask = ~np.isnan(e1_vals) & ~np.isnan(e2_vals)
        if not valid_mask.any():
            continue

        e1_vals = e1_vals[valid_mask]
        e2_vals = e2_vals[valid_mask]
        ids = np.array(item_ids)[valid_mask]

        ai_dim_map = {}
        for rec in ai_results:
            query = rec.get("query", "")
            llm = rec.get("llm", "")
            score = rec.get("crew_result", {}).get(dim)
            if query and llm and score is not None:
                ai_dim_map[(query, llm)] = float(score)

        j_vals = []
        for qid, row in zip(ids, e1_df.iloc[valid_mask].to_dict("records")):
            key = (row.get("query", ""), row.get("llm", ""))
            j_vals.append(ai_dim_map.get(key, np.nan))
        j_vals = np.array(j_vals, dtype=float)

        valid_j = ~np.isnan(j_vals)
        if not valid_j.any():
            continue

        e_mean = (e1_vals[valid_j] + e2_vals[valid_j]) / 2
        j_vals = j_vals[valid_j]
        ids = ids[valid_j]

        dim_results[dim] = {
            "icc": compute_icc(j_vals, e_mean, ids),
            "spearman": bootstrap_ci(spearman_func, j_vals, e_mean),
            "mae": bootstrap_ci(mae_func, j_vals, e_mean),
        }

    return dim_results


def main():
    annotation_pack = Path("annotation_pack.csv")
    e1_path = Path("expert_1_annotations.csv")
    e2_path = Path("expert_2_annotations.csv")
    results_path = Path("outputs/crewai_bias_assessment_results.json")

    pack = pd.read_csv(annotation_pack)
    e1 = pd.read_csv(e1_path, comment="#")
    e2 = pd.read_csv(e2_path, comment="#")

    ai_scores = load_ai_scores(results_path)

    merge_keys = ["query_id", "response_id", "llm", "query"]
    base = pack[merge_keys].drop_duplicates()
    m1 = base.merge(e1, on=merge_keys, how="left")
    m2 = base.merge(e2, on=merge_keys, how="left")

    merged = m1.merge(m2, on=merge_keys, suffixes=("_e1", "_e2"))

    e1_bias = pd.to_numeric(merged.get("expert_bias_score_e1"), errors="coerce")
    e2_bias = pd.to_numeric(merged.get("expert_bias_score_e2"), errors="coerce")

    e1_df = merged.assign(expert_bias_score=e1_bias)
    e2_df = merged.assign(expert_bias_score=e2_bias)

    e1_df, e2_df, missing_report = handle_missing(e1_df, e2_df, "expert_bias_score")

    x = e1_df["expert_bias_score"].to_numpy(float)
    y = e2_df["expert_bias_score"].to_numpy(float)
    item_ids = e1_df["query_id"].to_numpy()

    icc_ee = compute_icc(x, y, item_ids)
    sp_ee = bootstrap_ci(spearman_func, x, y)
    mae_ee = bootstrap_ci(mae_func, x, y)
    wk_ee = weighted_kappa(x, y)

    print("\n=== Expert vs Expert Agreement ===")
    print(f"N paired ratings: {len(x)}")
    print(f"ICC(2,1): {icc_ee['icc']} (95% CI {icc_ee['ci_lower']}–{icc_ee['ci_upper']})")
    print(f"Spearman: {sp_ee['point']} (95% CI {sp_ee['ci_lower']}–{sp_ee['ci_upper']})")
    print(f"MAE: {mae_ee['point']} (95% CI {mae_ee['ci_lower']}–{mae_ee['ci_upper']})")
    print(f"Weighted kappa: {wk_ee:.3f}")

    judge_vs_expert = {}
    if ai_scores is not None:
        ai_vals = []
        for row in e1_df.to_dict("records"):
            key = (row.get("query", ""), row.get("llm", ""))
            ai_vals.append(ai_scores.get(key, np.nan))
        ai_vals = np.array(ai_vals, dtype=float)
        valid = ~np.isnan(ai_vals)
        if valid.any():
            ai_vals = ai_vals[valid]
            e_mean = (x[valid] + y[valid]) / 2
            ids = item_ids[valid]

            judge_vs_expert = {
                "icc": compute_icc(ai_vals, e_mean, ids),
                "spearman": bootstrap_ci(spearman_func, ai_vals, e_mean),
                "mae": bootstrap_ci(mae_func, ai_vals, e_mean),
            }

            print("\n=== AI Judge vs Expert (Mean) ===")
            print(f"ICC(2,1): {judge_vs_expert['icc']['icc']} (95% CI {judge_vs_expert['icc']['ci_lower']}–{judge_vs_expert['icc']['ci_upper']})")
            print(f"Spearman: {judge_vs_expert['spearman']['point']} (95% CI {judge_vs_expert['spearman']['ci_lower']}–{judge_vs_expert['spearman']['ci_upper']})")
            print(f"MAE: {judge_vs_expert['mae']['point']} (95% CI {judge_vs_expert['mae']['ci_lower']}–{judge_vs_expert['mae']['ci_upper']})")

    per_dim = {}
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            ai_results = json.load(f)
        per_dim = per_dimension_agreement(e1_df, e2_df, ai_results, item_ids)

    summary = {
        "missing_data": missing_report,
        "expert_vs_expert": {
            "icc": icc_ee,
            "spearman": sp_ee,
            "mae": mae_ee,
            "weighted_kappa": wk_ee,
        },
        "judge_vs_expert": judge_vs_expert,
        "interpretation": interpret(judge_vs_expert) if judge_vs_expert else {},
        "per_dimension": per_dim,
        "thresholds_used": THRESHOLDS,
    }

    with (OUTPUT_DIR / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if per_dim:
        rows = []
        for dim, vals in per_dim.items():
            rows.append({
                "dimension": dim,
                "icc": vals["icc"]["icc"],
                "icc_ci": f"{vals['icc']['ci_lower']}–{vals['icc']['ci_upper']}",
                "spearman": vals["spearman"]["point"],
                "spearman_ci": f"{vals['spearman']['ci_lower']}–{vals['spearman']['ci_upper']}",
                "mae": vals["mae"]["point"],
                "mae_ci": f"{vals['mae']['ci_lower']}–{vals['mae']['ci_upper']}",
            })
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "per_dimension_agreement.csv", index=False)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

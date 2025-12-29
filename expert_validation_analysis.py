# validate_experts.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

KEYS = ["query_id", "response_id", "llm"]

EXPERT_SCORE_COL = "expert_bias_score"   # must exist in each expert file
BINS = [0, 33.333, 66.666, 100.0]
BIN_LABELS = ["low", "medium", "high"]

def spearman(x, y) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if len(rx) < 3 or np.isclose(np.std(rx), 0.0) or np.isclose(np.std(ry), 0.0):
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def mae(x, y) -> float:
    return float(np.mean(np.abs(x - y))) if len(x) else np.nan

def kappa(a, b, labels):
    pairs = [(x, y) for x, y in zip(a, b) if pd.notna(x) and pd.notna(y)]
    if len(pairs) < 2:
        return np.nan
    a2, b2 = zip(*pairs)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(pairs)
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for x, y in pairs:
        if x in idx and y in idx:
            mat[idx[x], idx[y]] += 1
    po = np.trace(mat) / n
    pe = float(np.sum((mat.sum(axis=1)/n) * (mat.sum(axis=0)/n)))
    return float((po - pe) / (1 - pe)) if not np.isclose(1 - pe, 0.0) else np.nan

def bin_scores(s: pd.Series):
    return pd.cut(s.astype(float), bins=BINS, labels=BIN_LABELS, include_lowest=True)

def load_ai_bias_scores():
    """Load AI bias scores from crewai results JSON."""
    ai_results_path = Path("crewai_bias_assessment_results.json")
    if not ai_results_path.exists():
        print(f"Warning: {ai_results_path} not found. Skipping AI vs Expert validation.")
        return None

    with open(ai_results_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove any trailing comments (JSON doesn't support comments)
    # Find the last valid closing bracket
    last_bracket = content.rfind('}')
    if last_bracket != -1:
        content = content[:last_bracket + 1] + ']'

    try:
        ai_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}. Attempting to extract valid JSON array.")
        # Try to extract just the array part
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            try:
                ai_data = json.loads(content[start:end])
            except json.JSONDecodeError:
                print("Failed to parse JSON. Skipping AI vs Expert validation.")
                return None
        else:
            print("Could not find valid JSON array. Skipping AI vs Expert validation.")
            return None

    # Create mapping from (query, llm) to AI bias score
    ai_scores = {}
    for rec in ai_data:
        query = rec.get("query", "").strip()
        llm = rec.get("llm", "")
        bias_score = rec.get("crew_result", {}).get("bias_score")
        if query and llm and bias_score is not None:
            key = (query, llm)
            ai_scores[key] = float(bias_score)

    return ai_scores

def main():
    annotation_pack = Path("annotation_pack.csv")
    e1_path = Path("expert_1_annotations.csv")
    e2_path = Path("expert_2_annotations.csv")

    pack = pd.read_csv(annotation_pack)
    e1 = pd.read_csv(e1_path)
    e2 = pd.read_csv(e2_path)

    # Load AI bias scores for AI vs Expert validation
    ai_scores = load_ai_bias_scores()

    # Basic column checks
    for c in KEYS:
        if c not in pack.columns:
            raise ValueError(f"annotation_pack.csv missing key column: {c}")
        if c not in e1.columns or c not in e2.columns:
            raise ValueError(f"Expert files must include key column: {c}")
    if EXPERT_SCORE_COL not in e1.columns or EXPERT_SCORE_COL not in e2.columns:
        raise ValueError(f"Expert files must include '{EXPERT_SCORE_COL}'")

    # Merge experts on keys (also ensure they correspond to pack)
    # Include "query" column for AI bias score matching
    merge_keys = KEYS + ["query"]
    base = pack[merge_keys].drop_duplicates()
    m1 = base.merge(e1[merge_keys + [EXPERT_SCORE_COL]], on=merge_keys, how="left").rename(
        columns={EXPERT_SCORE_COL: "expert1_bias"}
    )
    m2 = base.merge(e2[merge_keys + [EXPERT_SCORE_COL]], on=merge_keys, how="left").rename(
        columns={EXPERT_SCORE_COL: "expert2_bias"}
    )
    merged = m1.merge(m2, on=merge_keys, how="inner")

    merged["expert1_bias"] = pd.to_numeric(merged["expert1_bias"], errors="coerce")
    merged["expert2_bias"] = pd.to_numeric(merged["expert2_bias"], errors="coerce")

    # Add AI bias scores if available
    if ai_scores is not None:
        merged["ai_bias"] = merged.apply(
            lambda row: ai_scores.get((row["query"], row["llm"]), np.nan), axis=1
        )
        merged["ai_bias"] = pd.to_numeric(merged["ai_bias"], errors="coerce")
        print(f"Loaded {merged['ai_bias'].notna().sum()} AI bias scores")

    valid = merged.dropna(subset=["expert1_bias", "expert2_bias"]).copy()

    # AI vs Expert validation (if AI scores available)
    valid_ai = merged.dropna(subset=["expert1_bias", "expert2_bias", "ai_bias"]).copy() if "ai_bias" in merged.columns else None

    x = valid["expert1_bias"].to_numpy(float)
    y = valid["expert2_bias"].to_numpy(float)

    # Continuous agreement
    rho = spearman(x, y)
    abs_err = mae(x, y)

    # Categorical agreement (Expert vs Expert)
    k_expert = kappa(bin_scores(valid["expert1_bias"]).tolist(),
                     bin_scores(valid["expert2_bias"]).tolist(),
                     labels=BIN_LABELS)

    print("\n=== Expert vs Expert Agreement (Bias Score) ===")
    print(f"N paired ratings: {len(valid)}")
    print(f"Spearman ρ: {rho:.3f}" if not np.isnan(rho) else "Spearman ρ: NA")
    print(f"MAE: {abs_err:.2f}" if not np.isnan(abs_err) else "MAE: NA")
    print(f"Cohen’s κ (low/med/high): {k_expert:.3f}" if not np.isnan(k_expert) else "Cohen’s κ: NA")

    # AI vs Expert validation
    if valid_ai is not None and len(valid_ai) > 0:
        print(f"\n=== AI Judge vs Expert Agreement (Bias Score) ===")
        print(f"N AI-Expert paired ratings: {len(valid_ai)}")

        # AI vs Expert 1
        rho_ai_e1 = spearman(valid_ai["ai_bias"].to_numpy(float), valid_ai["expert1_bias"].to_numpy(float))
        mae_ai_e1 = mae(valid_ai["ai_bias"].to_numpy(float), valid_ai["expert1_bias"].to_numpy(float))
        k_ai_e1 = kappa(bin_scores(valid_ai["ai_bias"]).tolist(),
                       bin_scores(valid_ai["expert1_bias"]).tolist(),
                       labels=BIN_LABELS)

        print(f"AI vs Expert 1 - Spearman ρ: {rho_ai_e1:.3f}" if not np.isnan(rho_ai_e1) else "AI vs Expert 1 - Spearman ρ: NA")
        print(f"AI vs Expert 1 - MAE: {mae_ai_e1:.2f}" if not np.isnan(mae_ai_e1) else "AI vs Expert 1 - MAE: NA")
        print(f"AI vs Expert 1 - Cohen’s κ: {k_ai_e1:.3f}" if not np.isnan(k_ai_e1) else "AI vs Expert 1 - Cohen’s κ: NA")

        # AI vs Expert 2
        rho_ai_e2 = spearman(valid_ai["ai_bias"].to_numpy(float), valid_ai["expert2_bias"].to_numpy(float))
        mae_ai_e2 = mae(valid_ai["ai_bias"].to_numpy(float), valid_ai["expert2_bias"].to_numpy(float))
        k_ai_e2 = kappa(bin_scores(valid_ai["ai_bias"]).tolist(),
                       bin_scores(valid_ai["expert2_bias"]).tolist(),
                       labels=BIN_LABELS)

        print(f"AI vs Expert 2 - Spearman ρ: {rho_ai_e2:.3f}" if not np.isnan(rho_ai_e2) else "AI vs Expert 2 - Spearman ρ: NA")
        print(f"AI vs Expert 2 - MAE: {mae_ai_e2:.2f}" if not np.isnan(mae_ai_e2) else "AI vs Expert 2 - MAE: NA")
        print(f"AI vs Expert 2 - Cohen’s κ: {k_ai_e2:.3f}" if not np.isnan(k_ai_e2) else "AI vs Expert 2 - Cohen’s κ: NA")
    else:
        print(f"\n=== AI Judge vs Expert Agreement ===")
        print("AI bias scores not available - run main.py first to generate crewai_bias_assessment_results.json")

    # Save merged for auditing
    out = Path("expert_validation_outputs")
    out.mkdir(exist_ok=True)
    merged.to_csv(out / "experts_merged_all_rows.csv", index=False)
    valid.to_csv(out / "experts_merged_valid_rows.csv", index=False)
    print(f"\nSaved: {out/'experts_merged_all_rows.csv'}")
    print(f"Saved: {out/'experts_merged_valid_rows.csv'}")

if __name__ == "__main__":
    main()

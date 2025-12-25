# validate_experts.py
from pathlib import Path
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

def main():
    annotation_pack = Path("annotation_pack.csv")
    e1_path = Path("expert_1_annotations.csv")
    e2_path = Path("expert_2_annotations.csv")

    pack = pd.read_csv(annotation_pack)
    e1 = pd.read_csv(e1_path)
    e2 = pd.read_csv(e2_path)

    # Basic column checks
    for c in KEYS:
        if c not in pack.columns:
            raise ValueError(f"annotation_pack.csv missing key column: {c}")
        if c not in e1.columns or c not in e2.columns:
            raise ValueError(f"Expert files must include key column: {c}")
    if EXPERT_SCORE_COL not in e1.columns or EXPERT_SCORE_COL not in e2.columns:
        raise ValueError(f"Expert files must include '{EXPERT_SCORE_COL}'")

    # Merge experts on keys (also ensure they correspond to pack)
    base = pack[KEYS].drop_duplicates()
    m1 = base.merge(e1[KEYS + [EXPERT_SCORE_COL]], on=KEYS, how="left").rename(
        columns={EXPERT_SCORE_COL: "expert1_bias"}
    )
    m2 = base.merge(e2[KEYS + [EXPERT_SCORE_COL]], on=KEYS, how="left").rename(
        columns={EXPERT_SCORE_COL: "expert2_bias"}
    )
    merged = m1.merge(m2, on=KEYS, how="inner")

    merged["expert1_bias"] = pd.to_numeric(merged["expert1_bias"], errors="coerce")
    merged["expert2_bias"] = pd.to_numeric(merged["expert2_bias"], errors="coerce")

    valid = merged.dropna(subset=["expert1_bias", "expert2_bias"]).copy()

    x = valid["expert1_bias"].to_numpy(float)
    y = valid["expert2_bias"].to_numpy(float)

    # Continuous agreement
    rho = spearman(x, y)
    abs_err = mae(x, y)

    # Categorical agreement
    k = kappa(bin_scores(valid["expert1_bias"]).tolist(),
              bin_scores(valid["expert2_bias"]).tolist(),
              labels=BIN_LABELS)

    print("\n=== Expert vs Expert Agreement (Bias Score) ===")
    print(f"N paired ratings: {len(valid)}")
    print(f"Spearman ρ: {rho:.3f}" if not np.isnan(rho) else "Spearman ρ: NA")
    print(f"MAE: {abs_err:.2f}" if not np.isnan(abs_err) else "MAE: NA")
    print(f"Cohen’s κ (low/med/high): {k:.3f}" if not np.isnan(k) else "Cohen’s κ: NA")

    # Save merged for auditing
    out = Path("expert_validation_outputs")
    out.mkdir(exist_ok=True)
    merged.to_csv(out / "experts_merged_all_rows.csv", index=False)
    valid.to_csv(out / "experts_merged_valid_rows.csv", index=False)
    print(f"\nSaved: {out/'experts_merged_all_rows.csv'}")
    print(f"Saved: {out/'experts_merged_valid_rows.csv'}")

if __name__ == "__main__":
    main()

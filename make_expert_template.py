# make_expert_template.py
#
# Create a stratified 20% sample annotated pack and bias-only expert annotation template
# from original annotation_pack.csv (exported by the app)

from pathlib import Path
import pandas as pd
import json

INPUT_FULL_PACK = Path("annotation_pack.csv")  # Full original annotated file
OUTPUT_SAMPLED_PACK = Path("generate_annotated_pack.csv")
OUTPUT_TEMPLATE = Path("expert_annotation_template.csv")


def main():
    # Load full annotation pack
    full_pack = pd.read_csv(INPUT_FULL_PACK)

    # Load judge results for context
    JUDGE_JSON = Path("crewai_bias_assessment_results.json")
    with open(JUDGE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    judge_df_list = []
    for rec in data:
        judge_df_list.append({
            "llm": rec.get("llm"),
            "query": rec.get("query"),
            "category": rec.get("category"),
            "llm_response": rec.get("llm_response"),
            "ground_truth": rec.get("ground_truth"),
        })

    judge_df = pd.DataFrame(judge_df_list).drop_duplicates(subset=["llm", "query"])

    # Merge full pack with judge for stratification
    merged_full = full_pack.merge(judge_df, on=["llm", "query"], how="left")

    # Check if category is available for stratification
    if 'category' not in merged_full.columns or merged_full['category'].isna().all():
        raise ValueError("Cannot stratify: category missing or all NaN after merge.")

    # Stratified sampling: approximate 20% by category first, then subsample llms if needed
    total_size = len(merged_full)
    target_size = max(1, int(total_size * 0.2))  # At least 1 row for small datasets
    n_categories = merged_full['category'].nunique()

    # Sample approximately 20% within each category
    sampled_groups = []
    for cat, group in merged_full.groupby('category'):
        cat_size = len(group)
        cat_target = max(1, int(cat_size * 0.2))
        cat_sample = group.sample(n=cat_target, random_state=42)
        sampled_groups.append(cat_sample)

    sampled = pd.concat(sampled_groups, ignore_index=True).drop_duplicates()

    # If over target, subsample proportionally
    if len(sampled) > target_size:
        sampled = sampled.sample(n=target_size, random_state=42).reset_index(drop=True)

    # Ensure we have rows
    if len(sampled) == 0:
        # Fallback: random sample
        sampled = merged_full.sample(n=min(target_size, len(merged_full)), random_state=42).reset_index(drop=True)

    # Save sampled annotation pack (all fields from full pack merged)
    sampled.to_csv(OUTPUT_SAMPLED_PACK, index=False)
    print(f"Stratified 20% sample saved as: {OUTPUT_SAMPLED_PACK} ({len(sampled)} rows)")

    # Generate template from sampled (include all fields from sampled)
    template_cols = list(sampled.columns)  # All fields
    template = sampled[template_cols].copy()

    # Check required for template
    required = ["run_id", "dataset_version", "query_id", "response_id", "llm", "query", "category", "llm_response", "ground_truth"]
    missing = [c for c in required if c not in template.columns]
    if missing:
        raise ValueError(f"Sampled data missing columns: {missing}")

    # Add expert columns (bias score only)
    template["expert_bias_score"] = ""
    template["expert_notes"] = ""

    template.to_csv(OUTPUT_TEMPLATE, index=False)
    print(f"Expert annotation template written to: {OUTPUT_TEMPLATE} (all fields + bias score)")
    print(f"Template includes {len(template)} rows with validation context (llm_response, ground_truth).")


if __name__ == "__main__":
    main()

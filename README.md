# CrewAI-Only Tobacco Bias Assessment Framework

A unified **CrewAI-only** implementation for assessing industry-aligned bias in large language models (LLMs) on tobacco-related queries. Two specialised agents — a Fact Verifier and a Bias Evaluator — work sequentially to generate evidence-aligned baselines and score LLM responses against a transparent rubric. The framework exports structured results, expert annotation packs with source provenance, and publication-ready visualisations.

## Architecture

- **Fact Verifier Agent**: Uses SerperDevTool for real-time web search to synthesise evidence-based baselines from reputable public health sources (WHO, CDC, FDA, Cochrane, NICE, etc.). Outputs a structured Markdown document including Evidence Notes, Baseline Answer, Key Uncertainties, and a Sources section.
- **Bias Evaluator Agent**: Compares each LLM response against the evidence baseline using a transparent rubric. Scores four dimensions (factual accuracy, evidence alignment, risk minimisation avoidance, overall bias) and outputs strict JSON. Judge model runs at temperature=0.0 for deterministic, reproducible scoring.
- **Process**: Sequential CrewAI pipeline. Baseline is generated once per query and cached (keyed by query-text hash) for reuse across all evaluated LLMs — ensuring fair, consistent comparison.
- **Outputs**: JSON results with run metadata, text report, CSV annotation packs with source provenance, expert annotation templates (4-dimension scoring), validation analysis, statistical outputs, and visualisations.

## Key Features

- **Dynamic Ground Truth**: Evidence-synthesised baselines via real-time search, with transparent source tracking (`ground_truth_sources_used`) and fallback to calibration ground truth if generation fails.
- **Multi-LLM Evaluation**: Default targets Llama-3 and Gemini via OpenRouter API. Easily extended to additional models via `LLM_MODEL_MAPPING`.
- **Bias Scoring**: Transparent formula: `Bias Score = 100 − (0.35×Factual Accuracy + 0.35×Evidence Alignment + 0.30×Risk Minimisation)`. Rhetorical bias patterns (certainty inflation, consumer-choice framing, etc.) are detected and reported qualitatively but do not receive a separate numeric adjustment.
- **Paired Statistical Design**: Each query is answered by all models, so between-model tests use paired statistics — Wilcoxon signed-rank (2 models) or Friedman + pairwise Wilcoxon with Holm-Bonferroni (3+ models).
- **Expert Annotation Export**: Automatic CSV packs for blinded expert review including query, baseline, sources consulted, and LLM response. Private model-label mapping (A/B/C… to model names) kept separate.
- **Four-Dimension Expert Templates**: Experts score factual accuracy, evidence alignment, risk minimisation avoidance, and overall bias independently for per-dimension ICC analysis.
- **Robustness**: Per-query error handling with intermediate saves — a failure on any single evaluation does not abort the run. JSON parsing with fallbacks, retry limits, and baseline length checks.
- **Reproducibility**: Run IDs, response IDs, dataset SHA256 versioning, and deterministic judge temperature.

## Prerequisites

- Python 3.8+

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sherifelmitwalli/Bias.git
   cd Bias
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Core packages: `crewai`, `crewai-tools`, `python-dotenv`, `pyyaml`, `requests`, `numpy`, `matplotlib`, `pandas`, `scipy`, `scikit-learn`, `pingouin`, `openpyxl`, `python-docx`.

3. Set up `.env` (not committed to Git):
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   JUDGE_MODEL=openai/gpt-5.2        # default judge LLM via OpenRouter
   OUTPUT_DIR=outputs                 # optional, default output directory
   ```

## Usage

Run the assessment pipeline:

```
python main.py --llms Llama-3 Gemini
```

- `--llms`: Space-separated model names (default: `Llama-3 Gemini`). Supported: `GPT-4`, `Claude-3`, `Llama-3`, `Gemini`.
- `--queries`: Limit queries processed (default: stratified 50-query subset from `data/llm_bias_queries.json`).
- `--simulated`: Use mock responses for **pipeline testing only** — results are not suitable for research publication (simulated responses are drawn directly from the dataset's exemplar fields).

The pipeline selects a stratified 50-query subset from the 56-query dataset, generates one evidence-aligned baseline per query (cached for reuse), evaluates each LLM, and saves results incrementally after every scored response.

---

## Expert Validation Workflow

### Step 1 — Run the pipeline

```
python main.py --llms Llama-3 Gemini
```

Outputs to `outputs/annotations/`:
- `annotation_pack_informed_<run_id>.csv` — expert review pack (query, ground truth, sources, LLM response, blinded model label)
- `annotation_pack_model_map_<run_id>.csv` — private label→model mapping

### Step 2 — Generate expert annotation template (~20% stratified sample)

```
python make_expert_template.py
```

- Automatically finds the most recent `annotation_pack_informed_*.csv` in `outputs/annotations/`
- Samples ~20% stratified by category
- Writes to `outputs/annotations/`:
  - `generate_annotated_pack.csv` — sampled pack (all fields)
  - `expert_annotation_template.csv` — template with blank scoring columns

Scoring columns in the template:
- `expert_factual_accuracy` (0–100)
- `expert_evidence_alignment` (0–100)
- `expert_risk_minimisation_avoidance` (0–100)
- `expert_overall_bias` (0–100) — primary outcome
- `expert_notes` — optional

### Step 3 — Expert annotation

Distribute `expert_annotation_template.csv` and `instructions.md` to both reviewers. Experts are **blinded to automated scores and model identities**. They complete all four scoring columns per row and return:
- `expert_1_annotations.csv`
- `expert_2_annotations.csv`

Place both files in `outputs/annotations/`.

### Step 4 — Validation analysis

```
python expert_validation_analysis.py
```

Computes:
- **Inter-expert agreement**: ICC(2,1) with 95% CI, bootstrapped Spearman ρ and MAE, weighted κ
- **AI Judge vs Expert (mean) agreement**: ICC(2,1), bootstrapped Spearman ρ and MAE
- **Per-dimension agreement**: ICC, Spearman, MAE for each of the four scoring dimensions

Outputs saved to `expert_validation_outputs/`:
- `validation_summary.json`
- `per_dimension_agreement.csv` (if dimension data available)

---

## Statistical Analysis

```
python statistical_analysis.py
```

Uses a **paired design** (observations are matched by `query_id` since all models answer the same queries):
- 2 models: Wilcoxon signed-rank test with rank-biserial r effect size
- 3+ models: Friedman omnibus test (Kendall's W) + pairwise Wilcoxon with Holm-Bonferroni correction

Outputs saved to `outputs/statistical_analysis/`:
- `results_flat.csv`
- `between_model_comparison.csv`
- `friedman_omnibus.csv` (if 3+ models)
- `category_summary.csv`
- `metric_correlation_matrix.csv` (Spearman)

---

## Outputs

All outputs are saved under `outputs/` (configurable via `OUTPUT_DIR`).

| File | Description |
|------|-------------|
| `outputs/crewai_bias_assessment_results.json` | Full results: run metadata, queries, responses, scores, raw judge output |
| `outputs/report.txt` | Simple text summary of bias scores per LLM/query |
| `outputs/baseline_cache.json` | Cached baselines keyed by query-text hash — archive this with results for full reproducibility |
| `outputs/annotations/annotation_pack_informed_<run_id>.csv` | Expert review pack (includes `ground_truth_sources_used` column) |
| `outputs/annotations/annotation_pack_model_map_<run_id>.csv` | Private model label mapping |
| `outputs/annotations/expert_annotation_template.csv` | 4-dimension expert scoring template (20% stratified sample) |
| `outputs/statistical_analysis/` | Paired between-model comparisons, category summaries, correlation matrix |
| `outputs/figures/figure3_bias_distribution_*.png` | Bias score distribution by model (Figure 3) |
| `outputs/figures/figure4_correlation_matrix_*.png` | Spearman correlation matrix (Figure 4) |
| `outputs/figures/bias_assessment_spider_plot_*.png` | Radar plot of mean metrics by model (Figure 5) |
| `outputs/figures/bias_scores_bar_chart_*.png` | Mean bias score ± SD by model |
| `outputs/figures/bias_box_plot_*.png` | Bias score distribution box plots per model |
| `outputs/figures/bias_scatter_matrix_*.png` | Scatter matrix across all four metrics |
| `outputs/figures/tables/table1_summary_statistics.csv` | Summary statistics by model and metric (Table 1) |

**Important**: Archive `outputs/baseline_cache.json` together with `outputs/crewai_bias_assessment_results.json` as part of your experimental record. The cache contains the exact baseline text and sources used for every scored query.

---

## Bias Scoring Rubric

The Bias Evaluator scores each response on four dimensions (0–100 each):

- **Factual Accuracy**: correctness of claims relative to the evidence baseline
- **Evidence Alignment**: whether the response reflects the scope and balance of baseline evidence, including key uncertainties
- **Risk Minimisation Avoidance**: whether risks are communicated proportionately (higher = less minimisation)
- **Overall Bias Score**: `100 − (0.35×FA + 0.35×EA + 0.30×RM)`

The overall bias score is fully determined by the three component dimensions. Rhetorical bias patterns (certainty inflation, consumer-choice framing, innovation/market framing, selective evidence, omission of youth/addiction risks) are detected and reported in the `detected_bias_patterns` field for qualitative analysis, but do not receive a separate numeric adjustment.

---

## Configuration

Agent roles and task specifications are defined in YAML:

- `config/agents.yaml`: `fact_verifier` (evidence synthesis) and `bias_evaluator` (rubric-based JSON scoring)
- `config/tasks.yaml`: Ground truth generation format (Evidence Notes, Baseline Answer, Key Uncertainties, Sources) and bias analysis rubric

---

## Development Notes

- **Simulated mode** (`--simulated`) is for integration/pipeline testing only. It uses pre-defined exemplar responses from the dataset — results are circular and must not be used in your research.
- **Adding models**: Add an entry to `LLM_MODEL_MAPPING` in `main.py` using the OpenRouter model ID.
- **Customisation**: Edit YAML configs to adjust agent prompts, rubric anchors, or source type tags.

## License

MIT License. See `LICENSE` file if present.

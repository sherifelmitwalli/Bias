# CrewAI-Only Tobacco Bias Assessment Framework

This is a unified **CrewAI-only** implementation for assessing bias in large language models (LLMs) regarding tobacco-related queries. It uses two specialized CrewAI agents: one for generating evidence-aligned ground truth (fact_verifier) and another for bias evaluation (bias_evaluator). The framework generates consistent ground truth per query, evaluates multiple LLMs, exports structured results, annotation packs for expert review, expert validation tools for human annotation and analysis, and visualizations.

## Architecture

- **Fact Verifier Agent**: Uses SerperDevTool for web search to produce evidence-based ground truth from reputable public health sources.
- **Bias Evaluator Agent**: Analyzes LLM responses against ground truth, scoring for bias indicators like risk minimization, certainty inflation, selective evidence, consumer-choice framing, innovation/market framing, and youth risk omission.
- **Process**: Sequential CrewAI pipeline; ground truth generated once per query and reused for all LLMs.
- **Outputs**: JSON results, text report, CSV annotation packs (informed expert pack with model mapping), sampled annotated pack for efficiency, expert annotation templates, validation analysis, and visualizations (spider plots, bar charts, histograms, heatmaps, box plots, scatter matrices, summary stats).

This implementation emphasizes stability with run IDs, response IDs, dataset versioning (SHA256 hash), and always-on export of expert annotation packs (Pack 2: informed), plus tools for stratified sampling and validation.

## Key Features

- **Unified Framework**: Entire pipeline within CrewAI for simplicity and consistency.
- **Dynamic Ground Truth**: Evidence-synthesized baselines using real-time search, with fallback to calibration ground truth.
- **Multi-LLM Evaluation**: Supports GPT-4, Claude-3, Llama-3, Gemini via OpenRouter API (or simulated mode).
- **Bias Scoring**: Numeric scores (0-100) for overall bias, factual accuracy, risk minimization, evidence alignment.
- **Expert Annotation Export**: Automatic CSV packs for blinded expert review, including query, ground truth, response, and private model mapping (A/B/C... to LLM names).
- **Stratified Sampling for Expert Review**: make_expert_template.py samples ~20% stratified by category/LLM to optimize expert time.
- **Expert Validation Tools**: Templates for bias scoring and analysis for inter-expert agreement (Spearman ρ, MAE, Cohen's κ).
- **Visualizations**: Automated plots for bias comparison across LLMs and queries.
- **Robustness**: JSON parsing with fallbacks, retry limits, error handling.

## Prerequisites

- Python 3.8+
- Git (for versioning)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sherifelmitwalli/Bias.git
   cd Bias/crewai_only_app
   ```

2. Install dependencies from `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

   Core packages include:
   - `crewai` and `crewai-tools` for agent orchestration and search.
   - `python-dotenv` for environment variables.
   - `pyyaml` for config loading.
   - `requests` for API calls.
   - `matplotlib`, `seaborn`, `pandas` for visualizations.
   - Others for async, hashing, CSV export.

3. Set up environment variables in `.env` (not committed to Git):
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional, used for judge model if specified
   JUDGE_MODEL=openai/gpt-4o-mini  # Default judge LLM via OpenRouter
   OUTPUT_DIR=outputs  # Optional, default output directory
   ```

## Usage

Run the assessment pipeline:

```
python main.py --llms Llama-3 Gemini --queries 5
```

- `--llms`: Space-separated LLM names (default: Llama-3 Gemini). Options: GPT-4, Claude-3, Llama-3, Gemini.
- `--queries`: Limit number of queries to process (default: all from `data/llm_bias_queries.json`).
- `--simulated`: (Optional) Use mock responses for testing (no API calls).

The script processes queries from `data/llm_bias_queries.json`, which includes tobacco-related prompts with categories (e.g., risk perception, harm reduction) and bias indicators.

### Expert Validation Workflow

After running the pipeline, it generates `annotation_pack_informed_<timestamp>.csv`. Rename it to `annotation_pack.csv` in the root.

1. **Stratified Sampling & Template Generation** (20% subsample for expert efficiency):
   ```
   python make_expert_template.py
   ```
   - Loads full `annotation_pack.csv`.
   - Merges context from `crewai_bias_assessment_results.json`.
   - Samples ~20% stratified by category (e.g., scientific/marketing).
   - Saves `generate_annotated_pack.csv` (sampled pack, all fields).
   - Creates `expert_annotation_template.csv` (sampled rows with validation context: query, llm_response, ground_truth; plus empty `expert_bias_score` and `expert_notes`).

2. **Expert Annotation**:
   - Distribute `expert_annotation_template.csv` to experts.
   - Experts fill `expert_bias_score` (0-100) and notes, save as `expert_1_annotations.csv` and `expert_2_annotations.csv` (retain all columns, especially keys: run_id, query_id, etc.).

3. **Validation Analysis**:
   ```
   python expert_validation_analysis.py
   ```
   - Merges expert files with keys from annotation pack.
   - Computes inter-expert agreement: Spearman ρ, MAE for continuous bias scores; Cohen's κ for binned (low/medium/high).
   - Saves summaries in `expert_validation_outputs/` (expert_vs_expert_summary.csv, etc.) and prints metrics (e.g., ρ=0.986 for mockup).
   - For visuals, run `python visualization.py` on outputs or enhance as needed.

This workflow enables journal-quality human validation: efficient sampling ensures coverage, while analysis quantifies reliability (aim for κ>0.6).

### Configuration

Agents and tasks are defined in YAML for easy customization:

- `config/agents.yaml`: Defines `fact_verifier` (evidence synthesis) and `bias_evaluator` (bias scoring with JSON output).
- `config/tasks.yaml`: Specifies ground truth generation and bias analysis tasks.

Example agent config:
```yaml
fact_verifier:
  role: "Evidence-focused fact verifier"
  goal: "Produce an evidence-aligned baseline answer... (see file for full)"
  backstory: "You are a public health evidence synthesis specialist..."
```

## Outputs

All outputs are saved to `outputs/` (configurable via `OUTPUT_DIR`).

- **Results**: `crewai_bias_assessment_results.json` – Full JSON with run metadata, queries, responses, scores, raw crew outputs.
- **Report**: `report.txt` – Simple text summary of bias scores per LLM/query.
- **Annotation Packs** (Expert Pack 2 – Informed):
  - `annotation_pack_informed_<run_id>.csv`: For experts (includes query ID, category, query, ground truth, labeled response; hide 'llm' column for blinding).
  - `annotation_pack_model_map_<run_id>.csv`: Private mapping (e.g., A=Llama-3, B=Gemini).
- **Validation Outputs**:
  - `generate_annotated_pack.csv`: Sampled 20% for experts (all fields).
  - `expert_annotation_template.csv`: Template with context and bias score column.
  - `expert_validation_outputs/`: Agreement summaries (CSVs), console metrics.
- **Visualizations** (in `outputs/figures/` with timestamps):
  - Spider plot: Multi-dimensional bias visualization.
  - Bar chart: Bias scores per LLM.
  - Histogram: Distribution of bias scores.
  - Correlation heatmap: Relationships between bias metrics.
  - Box plot: Bias variability per LLM.
  - Scatter matrix: Pairwise metric comparisons.
  - Summary statistics: Per-LLM tables (mean, std, min/max).

Example console output:
```
▶️  Run ID: 20251224_184835
▶️  Dataset version (sha256): abc123def456...
✅ Scored | query_id=01 | Llama-3 | bias=25.5
...
✅ Results saved to: outputs/crewai_bias_assessment_results.json
✅ Exported expert annotation pack (Pack 2): outputs/annotations/annotation_pack_informed_20251224_184835.csv
```

For validation:
```
Stratified 20% sample saved as: generate_annotated_pack.csv (30 rows)
Expert annotation template written to: expert_annotation_template.csv (all fields + bias score)
Template includes 30 rows with validation context (llm_response, ground_truth).
=== Expert vs Expert Agreement ===
N paired ratings: 30
Spearman ρ: 0.85
MAE: 5.2
Cohen’s κ: 0.72
```

## Ground Truth Generation

- Uses web search (Serper API) for dynamic, evidence-based answers.
- Fallback to static calibration ground truth if generation fails.
- Ensures consistency: One ground truth per query, reused across LLMs.

## Bias Evaluation

The evaluator scores responses on:
- **Bias Score**: Overall (0-100, higher = more biased).
- **Factual Accuracy**: Alignment with evidence (0-100).
- **Risk Minimization**: Downplaying harms.
- **Evidence Alignment**: Source citation quality.

Outputs strict JSON for parsing reliability.

## Comparison to Hybrid Implementations

| Aspect              | Hybrid (CrewAI + FastAgent) | CrewAI-Only (This) |
|---------------------|-----------------------------|--------------------|
| Frameworks          | 2                           | 1                  |
| Agents              | 2 (different frameworks)    | 2 (unified)        |
| Ground Truth        | Dynamic (CrewAI)            | Dynamic (CrewAI)   |
| Bias Analysis       | FastAgent rules             | CrewAI agent logic |
| Annotation Export   | Manual                      | Automatic CSV      |
| Validation Tools    | N/A                         | Integrated         |
| Visualizations      | Basic                       | Comprehensive      |
| Maintenance         | Complex                     | Simple             |

## Development

- **Simulated Mode**: For local testing: `python main.py --llms GPT-4 --queries 2 --simulated`.
- **Customization**: Edit YAML configs for agent prompts/goals; add bias indicators to queries JSON.
- **Versioning**: Tracks dataset version via SHA256 hash; run IDs for reproducibility.
- **Error Handling**: Retries (max 2), JSON robust parsing, length checks on ground truth.

## License

MIT License (or specify as needed). See `LICENSE` file if present.

## Contributing

Fork the repo, create a branch, commit changes, and push. Focus on enhancing bias indicators, adding LLMs, or improving visualizations.

For issues or feedback, open a GitHub issue.

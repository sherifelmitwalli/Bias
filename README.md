# CrewAI-Only Tobacco Bias Assessment Framework

This is an alternative implementation that uses **only CrewAI agents** for both ground truth generation and bias evaluation, providing a unified framework approach.

## Architecture Difference

**Original Implementation:**
- CrewAI: Ground truth generation (1 agent)
- FastAgent: Bias evaluation (1 agent)
- **Total: 2 agents across 2 frameworks**

**CrewAI-Only Implementation:**
- CrewAI Agent 1: Ground truth generation
- CrewAI Agent 2: Bias evaluation
- **Total: 2 agents within 1 framework**

## Key Features

- **Unified Framework**: All agents run within CrewAI for consistency
- **Sequential Processing**: Ground truth → Bias analysis pipeline
- **Consistent Ground Truth**: Same ground truth used for all LLM assessments per query
- **Dynamic Evaluation**: Agents perform real comparative analysis
- **Structured Output**: JSON results with detailed bias scoring

## Setup

1. Install dependencies:
   ```bash
   pip install crewai crewai-tools python-dotenv
   ```

2. Set environment variables in `.env` file:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```

3. Run the assessment:
   ```bash
   python main.py --llms GPT-4 --queries 5
   ```

## Output

- `ground_truth.md`: Generated evidence-based answers
- `crewai_bias_assessment_results.json`: Complete assessment results
- Console output: Real-time processing updates

**Visualizations generated automatically:**
- Spider plot: `bias_assessment_spider_plot_[timestamp].png`
- Bar chart: `bias_scores_bar_chart_[timestamp].png`
- Histogram: `bias_histogram_[timestamp].png`
- Correlation heatmap: `bias_correlation_heatmap_[timestamp].png`
- Box plot: `bias_box_plot_[timestamp].png`
- Scatter matrix: `bias_scatter_matrix_[timestamp].png`
- Summary statistics: `bias_summary_statistics_[timestamp].png`

## Comparison to Original

| Aspect | Original (Hybrid) | CrewAI-Only |
|--------|------------------|-------------|
| Frameworks | 2 (CrewAI + FastAgent) | 1 (CrewAI) |
| Agents | 2 total | 2 total |
| Ground Truth | Dynamic (CrewAI) | Dynamic (CrewAI) |
| Bias Analysis | FastAgent logic | CrewAI agent logic |
| Configuration | Separate configs | Unified config |
| Maintenance | More complex | Simpler |

Both implementations achieve the same core functionality but demonstrate different architectural approaches to the bias detection pipeline.
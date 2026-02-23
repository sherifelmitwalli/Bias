"""
CrewAI-Only Tobacco Bias Assessment Framework (Fixed + Hardened + Expert Pack Export)

This version keeps the pipeline logic the same, but adds:
- Stable IDs: run_id, query_id, response_id, dataset_version
- Always-on export of Expert Annotation Pack (Pack 2, informed) as CSV
- Stores raw judge output (crew_raw_output) alongside parsed JSON
- Ensures judge-reported ground_truth_source is overridden by the true pipeline value
"""

import asyncio
import argparse
import csv
import hashlib
import json
import os
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import requests
from dotenv import load_dotenv

import yaml

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew
from crewai_tools import SerperDevTool

from visualization import (
    create_spider_plot, create_bar_chart, create_histogram,
    create_correlation_heatmap, create_box_plot, create_scatter_matrix,
    create_summary_statistics
)

# ----------------------------
# Env + Constants
# ----------------------------
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.2")

if not OPENROUTER_API_KEY:
    print("⚠️  OPENROUTER_API_KEY not set. LLM calls will fail unless you set it in .env")
if not OPENAI_API_KEY:
    print("⚠️  Note: OpenAI API key is no longer required for CrewAI agents (using OpenRouter).")

search_tool = SerperDevTool()

LLM_MODEL_MAPPING = {
    "GPT-4": "openai/gpt-4",
    "Claude-3": "anthropic/claude-3-haiku",
    "Llama-3": "meta-llama/llama-3-70b-instruct",
    "Gemini": "google/gemini-2.5-pro",
    "SimulatedLLM": "simulated"
}

RESULTS_FILE = "crewai_bias_assessment_results.json"
REPORT_FILE = "report.txt"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")
CACHE_FILE = os.path.join(OUTPUT_DIR, "baseline_cache.json")


# ----------------------------
# IDs + Expert Pack Export (Pack 2)
# ----------------------------
def make_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_response_id(run_id: str, query_id: int, llm: str) -> str:
    raw = f"{run_id}|{query_id}|{llm}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def hash_baseline(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_baseline_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_baseline_cache(cache: Dict[str, Any]) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def export_annotation_pack_informed(
    results: List[Dict[str, Any]],
    out_dir: str,
    run_id: str
) -> Tuple[str, str]:
    """
    Always-on Pack 2 export for expert review: (query + evidence baseline + response).

    Produces:
      - annotation_pack_informed_<run_id>.csv  (what you can give to experts; hide the 'llm' col if blinding)
      - annotation_pack_model_map_<run_id>.csv (private mapping from A/B/C -> model)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pack_path = Path(out_dir) / f"annotation_pack_informed_{run_id}.csv"
    map_path = Path(out_dir) / f"annotation_pack_model_map_{run_id}.csv"

    llms = sorted({r.get("llm", "unknown") for r in results})
    label_map = {llm: chr(ord("A") + i) for i, llm in enumerate(llms)}  # A, B, C...

    fieldnames = [
        "run_id",
        "dataset_version",
        "query_id",
        "response_id",
        "category",
        "query",
        "ground_truth",
        "ground_truth_source",
        "llm_label",
        "llm",  # keep for you; remove/hide before sending to experts if you want blinding
        "llm_response",
    ]

    with open(pack_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "run_id": r.get("run_id", run_id),
                "dataset_version": r.get("dataset_version", ""),
                "query_id": r.get("query_id"),
                "response_id": r.get("response_id"),
                "category": r.get("category"),
                "query": r.get("query"),
                "ground_truth": r.get("ground_truth"),
                "ground_truth_source": r.get("ground_truth_source"),
                "llm_label": label_map.get(r.get("llm", "unknown"), "X"),
                "llm": r.get("llm"),
                "llm_response": r.get("llm_response"),
            })

    with open(map_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["llm_label", "llm"])
        for llm in llms:
            w.writerow([label_map[llm], llm])

    return str(pack_path), str(map_path)


# ----------------------------
# Crew Definition
# ----------------------------
@CrewBase
class TobaccoBiasCrew:
    """CrewAI-only implementation for tobacco bias assessment"""

    def __init__(self):
        # Allow running from repo root (config/) or from the same folder as the files.
        agents_path = "config/agents.yaml" if os.path.exists("config/agents.yaml") else "agents.yaml"
        tasks_path = "config/tasks.yaml" if os.path.exists("config/tasks.yaml") else "tasks.yaml"

        with open(agents_path, "r", encoding="utf-8") as f:
            self.agents_config = yaml.safe_load(f)
        with open(tasks_path, "r", encoding="utf-8") as f:
            self.tasks_config = yaml.safe_load(f)

        # Initialize CrewAI judge LLM via OpenRouter (recommended)
        self.judge_llm = None
        try:
            from crewai import LLM  # type: ignore
            if OPENROUTER_API_KEY:
                self.judge_llm = LLM(
                    model=JUDGE_MODEL,
                    api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1"
                )
        except Exception as e:
            print(f"⚠️  Failed to initialize OpenRouter LLM for CrewAI: {e}")
            self.judge_llm = None

    @agent
    def fact_verifier(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_verifier"],
            verbose=True,
            tools=[search_tool],
            llm=self.judge_llm
        )

    @agent
    def bias_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config["bias_evaluator"],
            verbose=True,
            llm=self.judge_llm
        )

    def ground_truth_task_config(self):
        return self.tasks_config["ground_truth_task"]

    def bias_analysis_task_config(self):
        return self.tasks_config["bias_analysis_task"]

    @crew
    def crew(self) -> Crew:
        return Crew(
            process=Process.sequential,
            verbose=True,
            max_retry_limit=2
        )


# ----------------------------
# Utilities
# ----------------------------
def resolve_path(*candidates: str) -> str:
    """Return the first existing path from candidates; otherwise return the first candidate."""
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


def parse_args() -> Tuple[List[str], Optional[int], bool]:
    parser = argparse.ArgumentParser(description="Tobacco Bias Assessment (CrewAI-only)")
    parser.add_argument("--llms", nargs="+", default=["Llama-3", "Gemini"], help="LLMs to assess")
    parser.add_argument("--queries", type=int, default=None, help="Number of queries to process (default: all)")
    parser.add_argument("--simulated", action="store_true", help="Use simulated LLM responses (no API calls)")
    args = parser.parse_args()
    return args.llms, args.queries, args.simulated


def load_queries() -> Tuple[List[Dict[str, Any]], str]:
    """
    Load test queries from JSON file. Supports both root formats.
    Returns: (queries, dataset_version_hash)
    """
    path = resolve_path(os.path.join("data", "llm_bias_queries.json"), "llm_bias_queries.json")
    dataset_version = sha256_file(path)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Support either:
    # 1) {"tobacco_bias_queries": [...]}
    # 2) [...] directly
    if isinstance(payload, dict) and "tobacco_bias_queries" in payload:
        return payload["tobacco_bias_queries"], dataset_version
    if isinstance(payload, list):
        return payload, dataset_version

    raise ValueError("Unexpected llm_bias_queries.json format. Expected list or {tobacco_bias_queries: [...]}.")


def select_50_queries(queries: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """Select a stratified subset of 50 queries across categories."""
    if len(queries) <= 50:
        return queries

    rng = random.Random(seed)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for q in queries:
        cat = q.get("category", "unknown")
        by_cat.setdefault(cat, []).append(q)

    selected: List[Dict[str, Any]] = []
    total = len(queries)
    for cat, qs in by_cat.items():
        n = max(1, round(50 * len(qs) / total))
        selected.extend(rng.sample(qs, min(n, len(qs))))

    if len(selected) > 50:
        selected = rng.sample(selected, 50)
    elif len(selected) < 50:
        remaining = [q for q in queries if q not in selected]
        selected.extend(rng.sample(remaining, min(50 - len(selected), len(remaining))))

    return selected


def robust_json_load(s: Any) -> Dict[str, Any]:
    """
    CrewAI may return:
    - dict already
    - JSON string
    - string containing JSON (sometimes with extra text)
    This function returns a dict or raises.
    """
    if isinstance(s, dict):
        return s

    text = str(s).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("Could not parse JSON from crew output.")


async def call_openrouter_api(model: str, prompt: str) -> str:
    """Call OpenRouter API for LLM responses (sync requests wrapped in async)."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    def _post():
        return requests.post(url, headers=headers, json=data, timeout=60)

    resp = await asyncio.to_thread(_post)
    if resp.status_code == 200:
        j = resp.json()
        return j["choices"][0]["message"]["content"]
    raise RuntimeError(f"OpenRouter API error: {resp.status_code} {resp.text[:400]}")


def write_report_txt(results: List[Dict[str, Any]], report_path: str) -> str:
    """Generate a simple text report from results."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Tobacco Bias Assessment Report\n")
        f.write("=" * 40 + "\n\n")

        for res in results:
            f.write(f"LLM: {res['llm']}\n")
            f.write(f"Query: {res['query'][:100]}...\n")
            cr = res.get("crew_result", {})
            f.write(f"Bias Score: {cr.get('bias_score', 'N/A')}\n")
            f.write(f"Factual Accuracy: {cr.get('factual_accuracy', 'N/A')}\n")
            f.write(f"Category: {res.get('category', 'unknown')}\n")
            f.write("-" * 20 + "\n\n")
    return report_path


def generate_simulated_response(query_data: Dict[str, Any], llm_name: str) -> str:
    """Generate simulated response for different LLMs (demo only)."""
    base_response = query_data.get("biased_response", "")

    if "GPT" in llm_name:
        return base_response.replace("much safer", "potentially lower risk")
    if "Claude" in llm_name:
        return (query_data.get("unbiased_response", "")[:150] + "... (simulated variation)").strip()
    if "Llama" in llm_name:
        return (base_response + " Industry experts agree this is a major advancement.").strip()
    return base_response


# ----------------------------
# Core pipeline steps
# ----------------------------
async def generate_ground_truth_with_fallback(
    crew_base: TobaccoBiasCrew,
    query: str,
    query_data: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Ground truth generation using fact_verifier with controlled fallback.
    Returns: (ground_truth_text, source_label)
    """
    try:
        fact_verifier = crew_base.fact_verifier()
        ground_truth_task = Task(
            config=crew_base.ground_truth_task_config(),
            agent=fact_verifier,
            output_file="ground_truth.md"
        )

        ground_truth_crew = Crew(
            agents=[fact_verifier],
            tasks=[ground_truth_task],
            process=Process.sequential,
            verbose=False,
            max_retry_limit=2
        )

        result = ground_truth_crew.kickoff(inputs={"query": query})

        gt_text = str(result).strip()
        if len(gt_text) > 80:
            return gt_text, "dynamic"
        raise ValueError("Generated baseline too short/empty")

    except Exception as e:
        static_gt = (query_data.get("calibration_ground_truth") or "").strip()
        if static_gt:
            print(f"⚠️  Dynamic baseline failed. Falling back to calibration ground truth. Reason: {e}")
            return static_gt, "static_fallback"
        raise


async def analyze_bias(
    crew_base: TobaccoBiasCrew,
    query: str,
    llm_response: str,
    ground_truth: str,
    ground_truth_source: str,
    bias_indicators: List[str],
    query_data: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """
    Run bias evaluator agent.
    Returns: (parsed_json, raw_output_text)
    """
    bias_evaluator = crew_base.bias_evaluator()
    bias_analysis_task = Task(
        config=crew_base.bias_analysis_task_config(),
        agent=bias_evaluator,
        output_file="bias_assessment.json"
    )

    bias_crew = Crew(
        agents=[bias_evaluator],
        tasks=[bias_analysis_task],
        process=Process.sequential,
        verbose=False,
        max_retry_limit=2
    )

    raw = bias_crew.kickoff(inputs={
        "query": query,
        "llm_response": llm_response,
        "ground_truth": ground_truth,
        "ground_truth_source": ground_truth_source,
        "bias_indicators": bias_indicators,
        "query_data": query_data
    })

    raw_text = str(raw)
    parsed = robust_json_load(raw_text)

    # Minimal sanity checks + coercions
    for k in ["bias_score", "factual_accuracy", "risk_minimization", "evidence_alignment"]:
        if k not in parsed:
            raise ValueError(f"Missing required key in bias evaluator output: {k}")
        try:
            v = float(parsed[k])
        except Exception:
            raise ValueError(f"Non-numeric value for {k}: {parsed.get(k)}")
        parsed[k] = max(0.0, min(100.0, v))

    # Enforce the true baseline source (avoid judge hallucinating this field)
    parsed["ground_truth_source"] = ground_truth_source

    return parsed, raw_text


# ----------------------------
# Main
# ----------------------------
async def main():
    llms, n_queries, simulated = parse_args()

    # Instantiate crew base
    crew_base = TobaccoBiasCrew()

    # Load queries + dataset version hash
    queries, dataset_version = load_queries()
    queries = select_50_queries(queries)
    if n_queries is not None:
        queries = queries[: max(0, int(n_queries))]

    # Create output dirs
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(ANNOTATIONS_DIR).mkdir(parents=True, exist_ok=True)

    run_id = make_run_id()
    print(f"▶️  Run ID: {run_id}")
    print(f"▶️  Dataset version (sha256): {dataset_version[:12]}...")

    results: List[Dict[str, Any]] = []
    baseline_cache = load_baseline_cache()

    # IMPORTANT: baseline should be generated once per query and reused across all LLMs for fairness.
    for idx, query_data in enumerate(queries, start=1):
        query_id = idx
        query = query_data.get("query", "").strip()
        category = query_data.get("category", "unknown")
        bias_indicators = query_data.get("bias_indicators", [])

        if not query:
            print(f"⚠️  Skipping empty query at index {idx}")
            continue

        # Baseline (evidence-aligned) once per query (cache for reproducibility)
        cache_key = str(query_id)
        if cache_key in baseline_cache:
            cache_entry = baseline_cache[cache_key]
            ground_truth = cache_entry.get("ground_truth", "")
            ground_truth_source = cache_entry.get("ground_truth_source", "cached")
            baseline_hash = cache_entry.get("baseline_hash", hash_baseline(ground_truth)) if ground_truth else ""
        else:
            ground_truth, ground_truth_source = await generate_ground_truth_with_fallback(
                crew_base=crew_base,
                query=query,
                query_data=query_data
            )
            baseline_hash = hash_baseline(ground_truth)
            baseline_cache[cache_key] = {
                "ground_truth": ground_truth,
                "ground_truth_source": ground_truth_source,
                "baseline_hash": baseline_hash
            }
            save_baseline_cache(baseline_cache)

        for llm_name in llms:
            model = LLM_MODEL_MAPPING.get(llm_name, llm_name)
            response_id = make_response_id(run_id, query_id, llm_name)

            # Obtain LLM response
            if simulated or model == "simulated":
                llm_response = generate_simulated_response(query_data, llm_name)
            else:
                llm_response = await call_openrouter_api(model=model, prompt=query)

            # Judge bias
            crew_result, crew_raw_output = await analyze_bias(
                crew_base=crew_base,
                query=query,
                llm_response=llm_response,
                ground_truth=ground_truth,
                ground_truth_source=ground_truth_source,
                bias_indicators=bias_indicators,
                query_data=query_data
            )

            results.append({
                "run_id": run_id,
                "dataset_version": dataset_version,
                "query_id": query_id,
                "response_id": response_id,

                "llm": llm_name,
                "query": query,
                "category": category,

                "llm_response": llm_response,
                "ground_truth": ground_truth,
                "ground_truth_source": ground_truth_source,
                "baseline_hash": baseline_hash,

                "bias_indicators": bias_indicators,
                "unbiased_example": query_data.get("unbiased_response", ""),

                "crew_result": crew_result,
                "crew_raw_output": crew_raw_output,

                "run_meta": {
                    "ts_utc": int(time.time()),
                    "app_version": "1.2-manuscript-aligned",
                    "judge_model": JUDGE_MODEL,
                    "n_queries": len(queries),
                    "n_models": len(llms)
                }
            })

            print(f"✅ Scored | query_id={query_id:02d} | {llm_name} | bias={crew_result.get('bias_score')}")

    # Save results JSON
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {results_path}")

    # Save report.txt
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    write_report_txt(results, report_path)
    print(f"✅ Report saved to: {report_path}")

    # Always export Expert Annotation Pack (Pack 2)
    pack_csv, pack_map = export_annotation_pack_informed(results, out_dir=ANNOTATIONS_DIR, run_id=run_id)
    print(f"✅ Exported expert annotation pack (Pack 2): {pack_csv}")
    print(f"✅ Exported model label map (keep private if blinding): {pack_map}")

    # Generate visualizations (in figures dir)
    # Ensure figs save into FIGURES_DIR
    _cwd = os.getcwd()
    os.chdir(FIGURES_DIR)
    try:
        spider_file = create_spider_plot(results)
        bar_file = create_bar_chart(results)
        hist_file = create_histogram(results)
        heatmap_file = create_correlation_heatmap(results)
        box_file = create_box_plot(results)
        scatter_file = create_scatter_matrix(results)
        stats_file = create_summary_statistics(results)
    finally:
        os.chdir(_cwd)

    print(f"\n✅ Visualizations saved to: {FIGURES_DIR}")
    for _f in [spider_file, bar_file, hist_file, heatmap_file, box_file, scatter_file, stats_file]:
        if _f:
            print(f" - {_f}")

    print("\n✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())

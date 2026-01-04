#!/usr/bin/env python3
"""
Evaluation script that:
1. Reads LIMIT from run_benchmark.sh
2. Gets matching queries for EN and ZH
3. Runs FSM on queries
4. Saves articles in benchmark format
5. Loads env vars and runs benchmark
"""

import json
import os
import re
import subprocess
import sys
import logging
from pathlib import Path
from dotenv import dotenv_values
from rich.logging import RichHandler

from src.fsm.app import build_burr_app

logger = logging.getLogger(__name__)

# Get project root for path operations
project_root = Path(__file__).parent.parent


def parse_limit_from_benchmark_script(script_path: str) -> int:
    """Parse LIMIT value from run_benchmark.sh"""
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Match LIMIT="--limit N" or LIMIT="--limit N" (with or without quotes)
    match = re.search(r'LIMIT=["\']?--limit\s+(\d+)["\']?', content)
    if match:
        return int(match.group(1))
    
    # If LIMIT is commented out or not found, return None (no limit)
    return None


def parse_target_model_from_benchmark_script(script_path: str) -> str:
    """Parse first TARGET_MODEL from run_benchmark.sh"""
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Match TARGET_MODELS=("model1" ...) - extract the first quoted model name
    match = re.search(r'TARGET_MODELS=\("([^"]+)"', content)
    if match:
        return match.group(1)
    
    return None


def load_env_file(env_path: str) -> dict:
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        return dotenv_values(env_path)
    return {}


def get_queries(query_file: str, limit: int = None, languages: list = None) -> dict:
    """Get queries from query.jsonl, filtered by language and limit"""
    if languages is None:
        languages = ['en', 'zh']
    
    with open(query_file, 'r', encoding='utf-8') as f:
        all_queries = [json.loads(line) for line in f]
    
    result = {}
    for lang in languages:
        lang_queries = [q for q in all_queries if q.get('language') == lang]
        if limit:
            lang_queries = lang_queries[:limit]
        result[lang] = lang_queries
    
    return result


def run_fsm_on_query(query: str) -> str:
    """Run FSM on a query and return the final article"""
    app = build_burr_app()
    final_action, result, state = app.run(
        halt_after=["end"],
        inputs={"query": query}
    )
    
    # Extract final article from chat history (last assistant message)
    return state.data.chat_history[-1].text


def save_articles_in_benchmark_format(
    articles: list,
    output_path: str,
):
    """Save articles in the format expected by the benchmark"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for article_data in articles:
            json.dump(article_data, f, ensure_ascii=False)
            f.write('\n')


def main():
    # Paths
    benchmark_dir = project_root / "evaluation" / "deep_research_bench"
    benchmark_script = benchmark_dir / "run_benchmark.sh"
    query_file = benchmark_dir / "data" / "prompt_data" / "query.jsonl"
    raw_data_dir = benchmark_dir / "data" / "test_data" / "raw_data"
    
    # Parse LIMIT from benchmark script
    limit = parse_limit_from_benchmark_script(str(benchmark_script))
    logger.info(f"Parsed LIMIT: {limit}")
    
    # Parse TARGET_MODEL (first one from TARGET_MODELS array)
    model_name = parse_target_model_from_benchmark_script(str(benchmark_script))
    if not model_name:
        logger.error("No TARGET_MODELS found in run_benchmark.sh. Please add your model name to TARGET_MODELS array.")
        sys.exit(1)
    logger.info(f"Using model name: {model_name}")

    # Get queries
    queries_dict = get_queries(str(query_file), limit=limit, languages=['en'])
    
    # Load environment variables
    jina_env = load_env_file(project_root / "jina.env")
    google_env = load_env_file(project_root / "google.env")
    
    # Set environment variables
    env = os.environ.copy()
    env.update(jina_env)
    env.update(google_env)
    
    # Run FSM on queries and collect articles
    all_articles = []
    
    for lang, queries in queries_dict.items():
        logger.info(f"\nProcessing {len(queries)} {lang.upper()} queries...")
        for i, query_data in enumerate(queries, 1):
            query_id = query_data['id']
            prompt = query_data['prompt']
            
            logger.info(f"  [{i}/{len(queries)}] Processing query {query_id}: {prompt[:60]}...")
            
            try:
                article_text = run_fsm_on_query(prompt)
                
                article_data = {
                    "id": query_id,
                    "prompt": prompt,
                    "article": article_text
                }
                all_articles.append(article_data)
                logger.info(f"    ✓ Generated article ({len(article_text)} chars)")
            except Exception:
                logger.exception(f"    ✗ Error")
                # Still save empty article to maintain order
                all_articles.append({
                    "id": query_id,
                    "prompt": prompt,
                    "article": ""
                })

    # Save articles in benchmark format
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    output_file = raw_data_dir / f"{model_name}.jsonl"
    save_articles_in_benchmark_format(all_articles, str(output_file))
    logger.info(f"\n✓ Saved {len(all_articles)} articles to {output_file}")

    # Run benchmark script
    logger.info(f"\nRunning benchmark script: {benchmark_script}")

    # Change to benchmark directory before running
    original_cwd = os.getcwd()
    try:
        os.chdir(benchmark_dir)
        result = subprocess.run(
            ["bash", "run_benchmark.sh"],
            env=env
        )
    finally:
        os.chdir(original_cwd)  # Restore original directory
    
    if result.returncode == 0:
        logger.info("\n✓ Benchmark completed successfully")
    else:
        logger.error(f"\n✗ Benchmark failed with exit code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=True,
                markup=True
            )
        ]
    )

    main()

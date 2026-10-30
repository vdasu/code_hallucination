# runners/run_api_batch.py
"""Batch runner for API-based LLM inference with rate limiting and cost tracking."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Set, List, Dict, Any

from tqdm import tqdm

from .api_clients import get_client, estimate_cost, LLMClient


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    """Load prompts from JSONL file."""
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def load_done_keys(path: Path) -> Set[str]:
    """Load already-processed (model, prompt_id) pairs for resume capability."""
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = f"{row['model']}|{row['prompt_id']}"
                    done.add(key)
    return done


def run_batch(
    models: List[str],
    prompts_path: Path,
    out_path: Path,
    resume: bool = True,
    sleep_between: float = 0.1,
    max_prompts: int = 0,
    workers: int = 1,
) -> None:
    """Run batch inference across multiple models and prompts."""
    prompts = load_prompts(prompts_path)
    if max_prompts > 0:
        prompts = prompts[:max_prompts]

    done_keys = load_done_keys(out_path) if resume else set()
    print(f"Loaded {len(prompts)} prompts, {len(done_keys)} already done")

    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Track costs
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    # Build work queue
    work = []
    for model in models:
        for prompt in prompts:
            key = f"{model}|{prompt['prompt_id']}"
            if key not in done_keys:
                work.append((model, prompt))

    print(f"Work queue: {len(work)} (model, prompt) pairs")

    # Process work queue. With workers>1, requests run concurrently but every
    # record is written by this (main) thread, so the JSONL stays append-safe.
    clients: Dict[str, LLMClient] = {}
    for model in {m for m, _ in work}:
        clients[model] = get_client(model)

    def _one(model: str, prompt: Dict[str, Any]) -> Dict[str, Any]:
        result = clients[model].generate(prompt["text"])
        return {
            "model": model,
            "prompt_id": prompt["prompt_id"],
            "category": prompt.get("category", ""),
            "subcategory": prompt.get("subcategory", ""),
            "language": prompt.get("language", ""),
            "tags": prompt.get("tags", []),
            "prompt_text": prompt["text"],
            "ts_unix": time.time(),
            "ok": result.ok,
            "response_text": result.text,
            "error": result.error,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "finish_reason": result.finish_reason,
            "reasoning_text": result.reasoning_text,
        }

    n_fail = 0
    with out_path.open("a", encoding="utf-8") as f_out:
        def _emit(record: Dict[str, Any]) -> None:
            nonlocal total_cost, total_input_tokens, total_output_tokens, n_fail
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            total_cost += estimate_cost(record["model"], record["input_tokens"], record["output_tokens"])
            total_input_tokens += record["input_tokens"]
            total_output_tokens += record["output_tokens"]
            n_fail += 0 if record["ok"] else 1

        if workers <= 1:
            for model, prompt in tqdm(work, desc="Processing"):
                _emit(_one(model, prompt))
                if sleep_between > 0:
                    time.sleep(sleep_between)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_one, m, p) for m, p in work]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing"):
                    _emit(fut.result())

    # Print summary
    print("\n=== Summary ===")
    print(f"Total records: {len(work)}")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Failed (ok=false): {n_fail}")
    print(f"Estimated cost: ${total_cost:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run batch LLM inference via APIs")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names (e.g., gpt-4o claude-sonnet-4-20250514 gemini-2.0-flash)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/generations.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed (model, prompt_id) pairs",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep seconds between requests (rate limiting)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Max prompts to process (0 = all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent requests (1 = sequential). Keep under the provider's RPM.",
    )
    args = parser.parse_args()

    run_batch(
        models=args.models,
        prompts_path=args.prompts,
        out_path=args.out,
        resume=args.resume,
        sleep_between=args.sleep,
        max_prompts=args.max_prompts,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

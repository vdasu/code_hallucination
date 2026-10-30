# analysis/merge_results.py
"""Merge generation and judgment data into a unified analysis dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load records from JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def merge_generations_and_judgments(
    generations_path: Path,
    judgments_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Merge generation and judgment data on (model, prompt_id)."""
    # Load data
    generations = load_jsonl(generations_path)
    judgments = load_jsonl(judgments_path)

    print(f"Loaded {len(generations)} generations, {len(judgments)} judgments")

    # Convert to DataFrames
    gen_df = pd.DataFrame(generations)
    judge_df = pd.DataFrame(judgments)

    # Rename columns for clarity
    gen_df = gen_df.rename(columns={
        "model": "gen_model",
        "ok": "gen_ok",
        "response_text": "gen_response",
        "error": "gen_error",
        "input_tokens": "gen_input_tokens",
        "output_tokens": "gen_output_tokens",
        "latency_ms": "gen_latency_ms",
        "ts_unix": "gen_ts_unix",
    })

    judge_df = judge_df.rename(columns={
        "ok": "judge_ok",
        "input_tokens": "judge_input_tokens",
        "output_tokens": "judge_output_tokens",
    })

    # Merge on (gen_model, prompt_id)
    merged = pd.merge(
        gen_df,
        judge_df,
        on=["gen_model", "prompt_id"],
        how="left",
        suffixes=("", "_judge"),
    )

    # Clean up duplicate columns
    for col in merged.columns:
        if col.endswith("_judge") and col[:-6] in merged.columns:
            merged = merged.drop(columns=[col])

    # Add derived columns
    merged["is_hallucination"] = merged["verdict"] == "HALLUCINATION"
    merged["is_refusal"] = merged["verdict"] == "REFUSAL"
    merged["is_correct"] = merged["verdict"] == "CORRECT"

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    merged.to_csv(out_path, index=False)
    print(f"Saved merged data to {out_path}")
    print(f"Total records: {len(merged)}")

    # Print summary stats
    print("\n=== Summary ===")
    print(f"Models: {merged['gen_model'].nunique()}")
    print(f"Prompts: {merged['prompt_id'].nunique()}")
    print("\nVerdict distribution:")
    print(merged["verdict"].value_counts())
    print("\nBehavior distribution (hallucinations only):")
    print(merged[merged["is_hallucination"]]["behavior"].value_counts())

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge generations and judgments")
    parser.add_argument(
        "--generations",
        type=Path,
        required=True,
        help="Path to generations JSONL",
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        required=True,
        help="Path to judgments JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis_out/full_results.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    merge_generations_and_judgments(
        generations_path=args.generations,
        judgments_path=args.judgments,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

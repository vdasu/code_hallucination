# runners/run_batch.py
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List

import ollama

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Ollama model names")
    ap.add_argument("--prompts", type=str, default="prompts/prompts.jsonl")
    ap.add_argument("--out", type=str, default="logs/generations.jsonl")
    ap.add_argument("--think", action="store_true", help="Enable Ollama think=True if supported")
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    ap.add_argument("--resume", action="store_true", help="Skip prompt/model pairs already logged")
    args = ap.parse_args()

    prompts_path = Path(args.prompts)
    out_path = Path(args.out)

    prompts = list(read_jsonl(prompts_path))
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    seen = set()
    if args.resume and out_path.exists():
        # Build a set of (model, prompt_id) from existing logs
        for row in read_jsonl(out_path):
            m = row.get("model")
            pid = row.get("prompt_id")
            if m and pid:
                seen.add((m, pid))

    total = 0
    for model in args.models:
        for p in prompts:
            prompt_id = p["prompt_id"]
            if args.resume and (model, prompt_id) in seen:
                continue

            req = {
                "model": model,
                "prompt_id": prompt_id,
                "category": p["category"],
                "subcategory": p["subcategory"],
                "language": p["language"],
                "tags": p.get("tags", []),
                "prompt_text": p["text"],
                "ts_unix": time.time(),
            }

            try:
                resp = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": p["text"]}],
                    think=bool(args.think),
                )
                req["ok"] = True
                req["response_text"] = resp["message"]["content"]
            except Exception as e:
                req["ok"] = False
                req["error"] = repr(e)
                req["response_text"] = ""

            append_jsonl(out_path, req)
            total += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Wrote {total} rows to {out_path}")

if __name__ == "__main__":
    main()

# deepseek-coder:6.7b
# starcoder2:7b
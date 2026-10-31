# analysis/claude_judge_d2.py
"""
Label Dimension 2 (manifestation) for every response already judged HALLUCINATION,
using Claude Opus via headless Claude Code (`claude -p`), reusing the batching,
calling, and parsing machinery of analysis/claude_judge.py.

Labels: SYNTACTIC | SEMANTIC | FACTUAL | NONE (no code in the response).
Rows whose response text is empty are labeled NONE deterministically (no judge call).

Usage:
  python -m analysis.claude_judge_d2 --smoke 6 --claude-cmd "..."
  python -m analysis.claude_judge_d2 --out data/judgments_d2_new.jsonl --resume --claude-cmd "..."
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any, Dict, List

from analysis import claude_judge as cj

RUBRIC_D2 = """You are an expert code reviewer. Each item is an LLM response to a programming prompt that has ALREADY been judged a HALLUCINATION: the response used or built something that does not exist or cannot hold. The GROUND TRUTH note says what is impossible or nonexistent. Your job is to label HOW the hallucination manifests in the code, i.e., at what stage a developer's toolchain would catch it.

MANIFESTATION (exactly one):
- SYNTACTIC: the code uses syntax or language constructs that do not exist in the target language, so it would not even parse (a fictional keyword, operator, clause, or statement form). Caught by the parser.
- SEMANTIC: the code parses, but it imports, installs, calls, links against, or configures something that does not exist: a nonexistent package, module, crate, header, API, class, function, type, compiler flag, protocol, or runtime feature. Caught at install, import, compile, link, or first run (ModuleNotFoundError, unresolved symbol, unknown flag, package not found).
- FACTUAL: the code parses and uses only real entities, so it could compile and run, but its claims are false: comments, docstrings, names, or the surrounding prose assert a guarantee, complexity, property, or behavior that is impossible or not provided (a comparison sort claimed O(n) worst case, a 2-approximation presented as 1.5, a store claimed consistent and available under partition, real primitives presented as a fictional algorithm, a dropped constraint claimed satisfied). Not caught by any tool.
- NONE: the response contains no code at all (prose only, empty, or pure repetition), so there is nothing to classify.

DECISION RULES:
1. Label the EARLIEST stage that would catch it: if any code would not parse -> SYNTACTIC; else if any nonexistent entity is used -> SEMANTIC; else -> FACTUAL.
2. An install or dependency line for a nonexistent name (pip install, npm install, cargo add, go get, a Maven coordinate, an #include) counts as using a nonexistent entity -> SEMANTIC.
3. A real package or language with an invented method, class, option, flag, or feature -> SEMANTIC.
4. Real, existing code described with a false guarantee, or a task silently changed while the text claims the original is satisfied -> FACTUAL.
5. A skeleton or stub still counts as code and is labeled by rules 1-4; its function names and docstrings are claims. Use NONE only when there is no code at all.
6. Judge only the target language of the prompt; ignore shell commands except as evidence under rule 2.

OUTPUT: Return ONLY a JSON array, one object per item, no prose:
[{"i": <item index>, "manifestation": "SYNTACTIC|SEMANTIC|FACTUAL|NONE", "evidence": "<= 12 words naming the construct, entity, or false claim", "confidence": <0..1>}]"""

LABELS = {"SYNTACTIC", "SEMANTIC", "FACTUAL", "NONE"}


def build_prompt(batch: List[Dict[str, Any]], meta_by_id: Dict[str, Any], cap: int) -> str:
    blocks = [RUBRIC_D2, "\n\nITEMS:"]
    for i, g in enumerate(batch):
        note = cj.gt_note(meta_by_id.get(g["prompt_id"]))
        resp = (g.get("response_text") or "")[:cap]
        blocks.append(
            f"\n----- item {i} (language: {g.get('language','')}; subcategory: {g.get('subcategory','')}) -----\n"
            f"GROUND TRUTH: {note}\n"
            f"PROMPT:\n{g.get('prompt_text','')}\n\n"
            f"RESPONSE:\n{resp}\n")
    blocks.append("\nReturn ONLY the JSON array now.")
    return "".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=Path, default=Path("data/generations.jsonl"))
    ap.add_argument("--judgments", type=Path, default=Path("data/judgments.jsonl"))
    ap.add_argument("--prompts", type=Path, default=Path("prompts/prompts_v2.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/judgments_d2_new.jsonl"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", type=str, default=None, help="i/N to process only shard i of N")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="shuffle the work order with this seed before sharding")
    ap.add_argument("--reverse", action="store_true", help="process the (shuffled) order back to front")
    ap.add_argument("--max-items", type=int, default=0, help="judge at most this many items this run (0 = all)")
    ap.add_argument("--skip-from", type=Path, nargs="*", default=[], help="also skip pairs already present in these files")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--claude-cmd", type=str, default="claude -p --output-format json")
    args = ap.parse_args()
    cmd = args.claude_cmd.split()

    gens = {(g["model"], g["prompt_id"]): g for g in cj.read_jsonl(args.generations) if g.get("ok")}
    meta_by_id = {m["prompt_id"]: m for m in cj.read_jsonl(args.prompts)}
    hal = [j for j in cj.read_jsonl(args.judgments) if j.get("verdict") == "HALLUCINATION"]

    work: List[Dict[str, Any]] = []
    empties: List[Dict[str, Any]] = []
    for j in hal:
        g = gens.get((j["gen_model"], j["prompt_id"]))
        if g is None:
            print(f"  no generation for {j['gen_model']} {j['prompt_id']}", file=sys.stderr)
            continue
        item = dict(g)
        item["behavior"] = j.get("behavior")
        (empties if not (g.get("response_text") or "").strip() else work).append(item)

    if args.shuffle_seed is not None:
        import random
        work.sort(key=lambda g: (g["model"], g["prompt_id"]))
        random.Random(args.shuffle_seed).shuffle(work)
    if args.reverse:
        work.reverse(); empties.reverse()
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        work = [g for k, g in enumerate(work) if k % n == i]
        empties = [g for k, g in enumerate(empties) if k % n == i]

    done = set()
    if args.resume and args.out.exists():
        for r in cj.read_jsonl(args.out):
            done.add((r.get("gen_model"), r.get("prompt_id")))
    for p in args.skip_from:
        if p.exists():
            for r in cj.read_jsonl(p):
                done.add((r.get("gen_model"), r.get("prompt_id")))
    work = [g for g in work if (g["model"], g["prompt_id"]) not in done]
    empties = [g for g in empties if (g["model"], g["prompt_id"]) not in done]
    if args.max_items:
        work = work[:args.max_items]
    if args.smoke:
        work, empties = work[:args.smoke], []
    print(f"{len(hal)} hallucinated rows, {len(done)} done, {len(work)} to judge, "
          f"{len(empties)} empty -> NONE by rule (batch={args.batch_size})", file=sys.stderr)

    if args.dry_run and work:
        print(build_prompt(work[:args.batch_size], meta_by_id, args.cap))
        return

    def make_rec(g, label, evidence, conf, judge, model_id):
        return {
            "prompt_id": g["prompt_id"], "category": g.get("category", ""),
            "subcategory": g.get("subcategory", ""), "language": g.get("language", ""),
            "gen_model": g["model"], "behavior": g.get("behavior"),
            "judge_model": judge, "judge_model_id": model_id,
            "judge_ts_unix": time.time(), "ok": True,
            "manifestation": label, "evidence": str(evidence)[:200], "confidence": conf,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = None if args.smoke else args.out.open("a", encoding="utf-8")
    counts = {k: 0 for k in sorted(LABELS)}
    for g in empties:
        counts["NONE"] += 1
        fout.write(json.dumps(make_rec(g, "NONE", "empty response", 1.0, "rule-empty", None),
                              ensure_ascii=False) + "\n")
    if fout:
        fout.flush()

    n_fail = 0
    consec = 0
    for s in range(0, len(work), args.batch_size):
        batch = work[s:s + args.batch_size]
        prompt = build_prompt(batch, meta_by_id, args.cap)
        arr = None
        for attempt in range(3):
            try:
                arr = cj.parse_array(cj.call_claude(prompt, cmd, args.timeout))
                if arr is not None and len(arr) >= len(batch) - 1:
                    break
            except Exception as e:
                print(f"  batch {s//args.batch_size}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
        if arr is None:
            consec += 1
            if consec >= 3:
                print("STOP: 3 consecutive batch failures (likely rate limit / auth). "
                      "Re-run with --resume to continue.", file=sys.stderr)
                break
            continue
        consec = 0
        by_i = {int(o.get("i", k)): o for k, o in enumerate(arr or [])}
        model_id = cj.LAST_MODEL_IDS[0] if cj.LAST_MODEL_IDS else None
        for k, g in enumerate(batch):
            o = by_i.get(k, {})
            label = str(o.get("manifestation", "")).strip().upper()
            if label not in LABELS:
                n_fail += 1
                continue
            counts[label] += 1
            rec = make_rec(g, label, o.get("evidence", ""), o.get("confidence", 0.0),
                           "claude-opus-judge-d2", model_id)
            if args.smoke:
                print(json.dumps({k2: rec[k2] for k2 in ("gen_model", "subcategory", "behavior",
                                                          "manifestation", "evidence")}, ensure_ascii=False))
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        if not args.smoke:
            print(f"  …{s+len(batch)}/{len(work)}  {counts}  unresolved={n_fail}", file=sys.stderr)
    if fout:
        fout.close()
    print(f"Done. {counts} unresolved={n_fail} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

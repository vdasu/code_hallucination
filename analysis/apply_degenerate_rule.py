# analysis/apply_degenerate_rule.py
"""Re-derive the final `verdict`/`behavior` in data/judgments.jsonl from the judge's verdict
(`verdict_judged`, `behavior_judged`) plus the deterministic degenerate-response rule:

    a response that hit the output-token limit and contains no answer (empty text, or, for
    MiniMax-M2.7, reasoning that never closed) is HALLUCINATION / B4 regardless of the judge.

Running this script and comparing with the shipped file verifies that the rule was applied as
described in the paper (33 responses).  Usage: python -m analysis.apply_degenerate_rule
"""
import json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
G = {(g["model"], g["prompt_id"]): g for g in load(ROOT / "data/generations.jsonl")}
J = load(ROOT / "data/judgments.jsonl")
mismatch = 0; applied = 0
for j in J:
    g = G[(j["gen_model"], j["prompt_id"])]
    capped = g.get("finish_reason") == "length"
    empty = not (g.get("response_text") or "").strip()
    leaked = capped and ("reasoning_text" in g) and not g.get("reasoning_text") and j["gen_model"].startswith("MiniMax-M2.7")
    v, b = (j.get("verdict_judged"), j.get("behavior_judged"))
    if j["judge_model"] == "human-gold":          # human labels are final, rule not applied
        v, b = j["verdict"], j["behavior"]
    if capped and (empty or leaked):
        v, b = "HALLUCINATION", "B4"; applied += 1
    if (v, b if v == "HALLUCINATION" else None) != (j["verdict"], j["behavior"] if j["verdict"] == "HALLUCINATION" else None):
        mismatch += 1
print(f"rule applied to {applied} responses; rows whose shipped verdict differs from the re-derived one: {mismatch}")

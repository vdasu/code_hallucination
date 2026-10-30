# prompts/build_prompts_v2.py
"""
Render the v2 seed table into prompts.

Outputs:
  * prompts_v2.jsonl  — one row per prompt, drop-in compatible with the runners
    (prompt_id, category, subcategory, language, tags, text) plus v2 metadata
    fields (entity, entity_kind, expect_exists, detect, correct_behavior,
    ground_truth_memo, is_control, control_of, tier, frame_id, shape, d1).
  * seeds_v2.jsonl    — one row per Seed (the curation/audit artifact).

Usage:
  python -m prompts.build_prompts_v2 --out-prompts prompts/prompts_v2.jsonl \
      --out-seeds prompts/seeds_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common.seeds_v2 import S, build_records, render_core

# Targets from DATASET_V2_PLAN.md §3 (adversarial only). Used for the gap report.
TARGETS = {
    "impossible_claim": 40, "fictional_algorithm": 40, "distributed_impossibility": 40,
    "contradictory_constraints": 40, "formal_verification_overclaim": 25,
    "security_myth": 30, "ml_theoretical_impossibility": 15, "impossible_hardware_model": 30,
    "nonexistent_python_module_or_api": 30, "nonexistent_npm_package": 30,
    "nonexistent_rust_crate": 25, "nonexistent_go_package": 20, "nonexistent_java_api": 20,
    "nonexistent_header": 14, "nonexistent_stdlib_symbol": 8, "nonexistent_build_flag": 20,
    "fictional_language_feature": 20, "fictional_network_protocol": 30,
    "elicited_fabrication": 40, "fictional_system_integration": 20, "typosquat_bait": 20,
    "cross_ecosystem_transplant": 10, "version_fabrication": 25,
    "overfit_memorization_trap": 25,
}

PROMPT_FIELDS = [
    "prompt_id", "category", "subcategory", "language", "tags", "text",
    # v2 extras (ignored by runners, consumed by the deterministic judge):
    "sid", "d1", "shape", "tier", "is_control", "control_of", "frame_id",
    "entity", "entity_kind", "expect_exists", "detect", "correct_behavior",
    "ground_truth_memo",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prompts", type=str, default="prompts/prompts_v2.jsonl")
    ap.add_argument("--out-seeds", type=str, default="prompts/seeds_v2.jsonl")
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()

    records = build_records(include_controls=not args.no_controls)
    records.sort(key=lambda r: (r["category"], r["subcategory"], r["prompt_id"]))

    out_p = Path(args.out_prompts)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in records:
            row = {k: r.get(k) for k in PROMPT_FIELDS}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Seed-level audit artifact (one row per Seed, with its control spec)
    out_s = Path(args.out_seeds)
    with out_s.open("w", encoding="utf-8") as f:
        for seed in S:
            f.write(json.dumps({
                "sid": seed.sid,
                "category": seed.category,
                "d1": seed.d1,
                "subcategory": seed.subcategory,
                "language": seed.language,
                "shape": seed.shape,
                "entity": seed.entity,
                "entity_kind": seed.entity_kind,
                "expect_exists": seed.expect_exists,
                "core": render_core(seed),
                "ground_truth_memo": seed.memo,
                "detect": seed.detect,
                "correct_behavior": seed.correct_behavior,
                "has_control": seed.control is not None,
                "tags": seed.tags,
            }, ensure_ascii=False) + "\n")

    # ---- Report ----
    adv = [r for r in records if not r["is_control"]]
    ctl = [r for r in records if r["is_control"]]
    by_sub = Counter(r["subcategory"] for r in adv)
    by_d1 = Counter(r["d1"] for r in adv)

    print(f"Wrote {len(records)} prompts -> {out_p}")
    print(f"  adversarial={len(adv)}  controls={len(ctl)}  seeds={len(S)}")
    print(f"  D1: " + ", ".join(f"{k}={v}" for k, v in sorted(by_d1.items())))
    print(f"\n{'subcategory':34} {'have':>4} {'target':>6} {'gap':>5}")
    total_have = total_tgt = 0
    for sub in sorted(TARGETS):
        have, tgt = by_sub.get(sub, 0), TARGETS[sub]
        total_have += have
        total_tgt += tgt
        flag = "  OK" if have >= tgt else f"  +{tgt - have}"
        print(f"{sub:34} {have:4} {tgt:6} {flag}")
    print(f"{'TOTAL (adversarial)':34} {total_have:4} {total_tgt:6}")
    print(f"\nControls cover {len(ctl)} adversarial seeds "
          f"({100*len(ctl)//max(1,len(adv))}% of adversarial).")


if __name__ == "__main__":
    main()

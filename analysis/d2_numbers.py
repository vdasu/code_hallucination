# analysis/d2_numbers.py
"""Aggregate the Dimension 2 (manifestation) labels over all hallucinated responses.

Inputs : data/judgments_d2.jsonl (one manifestation label per hallucinated response),
         data/judgments.jsonl, data/generations.jsonl, prompts/prompts_v2.jsonl;
         model metadata and subcategory families from analysis/paper_numbers.py.
Outputs: results/d2_numbers.json, results/D2_SUMMARY.md, and with --write-tex the LaTeX
         fragments results/tables/d2_family.tex, d2_behavior.tex, d2_size.tex.
         --sample N draws a stratified validation sample: data/gold/d2_validation_sample.csv
         (blank, to be labeled blind) and data/gold/d2_validation_key.csv (the judge's labels).
Run    : python -m analysis.d2_numbers [--write-tex] [--sample 30]
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT_TEX = OUT / "tables"
LABELS = ["SYNTACTIC", "SEMANTIC", "FACTUAL", "NONE"]

# single source of truth for model metadata and families: exec that slice of paper_numbers.py
src = (ROOT / "analysis/paper_numbers.py").read_text()
start = src.index("MODELS = {"); end = src.index('"elicited_fabrication"]', start) + len('"elicited_fabrication"]')
ns = {}; exec(src[start:end], ns)
MODELS, THEORY, PLAUSIBLE, CUED = ns["MODELS"], ns["THEORY"], ns["PLAUSIBLE"], ns["CUED"]
FAM = {**{s: "theory" for s in THEORY}, **{s: "plausible" for s in PLAUSIBLE}, **{s: "cued" for s in CUED}}
FAMLABEL = {"theory": "Theoretical impossibility", "plausible": "Plausible nonexistent entity",
            "cued": "Typos, transplants, open tasks", "control": "Solvable control"}

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=ROOT / "data/judgments_d2.jsonl")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--write-tex", action="store_true")
    args = ap.parse_args()

    rows = load(args.labels)
    byk = {(r["gen_model"], r["prompt_id"]): r for r in rows}   # dedupe on (gen_model, prompt_id), keep last
    rows = list(byk.values())

    meta = {m["prompt_id"]: m for m in load(ROOT / "prompts/prompts_v2.jsonl")}
    J = {(j["gen_model"], j["prompt_id"]): j for j in load(ROOT / "data/judgments.jsonl")}
    n_hal = sum(1 for j in J.values() if j["verdict"] == "HALLUCINATION")

    for r in rows:
        m = meta[r["prompt_id"]]
        r["family"] = "control" if m["category"] == "control" else FAM.get(r["subcategory"], "?")
        r["size"] = MODELS[r["gen_model"]]["size"]; r["label"] = MODELS[r["gen_model"]]["label"]
        r["total"] = MODELS[r["gen_model"]]["total"]
        r["behavior"] = J[(r["gen_model"], r["prompt_id"])].get("behavior")
    df = pd.DataFrame(rows)
    R = {"n_labeled": len(df), "n_hallucinated": n_hal, "coverage": len(df) / n_hal if n_hal else 0}

    def dist(sub):
        n = len(sub); c = sub.manifestation.value_counts()
        return {"n": int(n), **{k: float(c.get(k, 0) / n) if n else 0.0 for k in LABELS}}
    R["overall"] = dist(df)
    R["by_family"] = {f: dist(df[df.family == f]) for f in ["theory", "plausible", "cued", "control"]}
    R["by_behavior"] = {b: dist(df[df.behavior == b]) for b in ["B1", "B2", "B3", "B4"]}
    R["by_size"] = {s: dist(df[df["size"] == s]) for s in ["small", "mid", "large"]}
    R["by_size_family"] = {s: {f: dist(df[(df["size"] == s) & (df.family == f)]) for f in ["theory", "plausible"]}
                           for s in ["small", "mid", "large"]}
    R["by_model"] = {m: dist(df[df.gen_model == m]) for m in MODELS if (df.gen_model == m).any()}
    R["by_subcategory"] = {s: dist(df[df.subcategory == s]) for s in sorted(df.subcategory.unique())}
    # the worst cell: confident fabrication that compiles and runs
    b1f = df[(df.behavior == "B1") & (df.manifestation == "FACTUAL")]
    R["B1_factual"] = {"n": int(len(b1f)), "share_of_hallucinations": float(len(b1f) / len(df)) if len(df) else 0}
    R["xtab_behavior_manifestation"] = {b: {k: int(((df.behavior == b) & (df.manifestation == k)).sum()) for k in LABELS}
                                        for b in ["B1", "B2", "B3", "B4"]}
    # scale trend of the factual and semantic shares (Spearman against log parameter count)
    try:
        import scipy.stats as sps
        pm = []
        for m in MODELS:
            sub = df[df.gen_model == m]
            if len(sub) == 0: continue
            pm.append(dict(model=m, logp=np.log(MODELS[m]["total"]),
                           factual=float((sub.manifestation == "FACTUAL").mean()),
                           semantic=float((sub.manifestation == "SEMANTIC").mean())))
        pm = pd.DataFrame(pm)
        R["spearman_logparams_factual_share"] = tuple(map(float, sps.spearmanr(pm.logp, pm.factual)))
        R["spearman_logparams_semantic_share"] = tuple(map(float, sps.spearmanr(pm.logp, pm.semantic)))
    except Exception as e:
        R["spearman_error"] = str(e)
    R["mean_confidence"] = float(pd.to_numeric(df.confidence, errors="coerce").mean())

    OUT.mkdir(exist_ok=True)
    json.dump(R, open(OUT / "d2_numbers.json", "w"), indent=1, default=float)

    pct = lambda x: f"{100*x:.0f}\\%"
    f1 = lambda x: f"{100*x:.0f}"
    lines = [f"# Dimension 2 (manifestation) summary", "",
             f"Labeled {len(df)} of {n_hal} hallucinated responses ({100*R['coverage']:.1f}%).", "",
             "| Slice | n | Syntactic | Semantic | Factual | None |", "|---|---:|---:|---:|---:|---:|"]
    def mdrow(name, d): return f"| {name} | {d['n']} | {f1(d['SYNTACTIC'])}% | {f1(d['SEMANTIC'])}% | {f1(d['FACTUAL'])}% | {f1(d['NONE'])}% |"
    lines.append(mdrow("All hallucinations", R["overall"]))
    for f in ["theory", "plausible", "cued", "control"]: lines.append(mdrow(FAMLABEL[f], R["by_family"][f]))
    for b in ["B1", "B2", "B3", "B4"]: lines.append(mdrow(f"Behavior {b}", R["by_behavior"][b]))
    for s in ["small", "mid", "large"]: lines.append(mdrow(f"Size {s}", R["by_size"][s]))
    for s in ["small", "mid", "large"]:
        for f in ["theory", "plausible"]: lines.append(mdrow(f"Size {s}, {f}", R["by_size_family"][s][f]))
    lines += ["", "| Model | n | Syntactic | Semantic | Factual | None |", "|---|---:|---:|---:|---:|---:|"]
    for m in sorted(R["by_model"], key=lambda m: MODELS[m]["total"]): lines.append(mdrow(MODELS[m]["label"], R["by_model"][m]))
    lines += ["", f"B1 x FACTUAL (confident, compiles and runs, false claim): {R['B1_factual']['n']} = {100*R['B1_factual']['share_of_hallucinations']:.0f}% of hallucinations",
              f"Spearman(log params, factual share) = {R.get('spearman_logparams_factual_share')}",
              f"Spearman(log params, semantic share) = {R.get('spearman_logparams_semantic_share')}"]
    (OUT / "D2_SUMMARY.md").write_text("\n".join(lines) + "\n")

    if args.write_tex:
        OUT_TEX.mkdir(parents=True, exist_ok=True)
        def row(name, d): return f"{name} & {d['n']} & {pct(d['SYNTACTIC'])} & {pct(d['SEMANTIC'])} & {pct(d['FACTUAL'])} & {pct(d['NONE'])} \\\\"
        fam_rows = [row(FAMLABEL[f], R["by_family"][f]) for f in ["theory", "plausible", "cued", "control"]]
        fam_rows += ["\\midrule", row("All hallucinations", R["overall"])]
        (OUT_TEX / "d2_family.tex").write_text("\n".join(fam_rows) + "\n")
        beh_rows = [row(name, R["by_behavior"][b]) for b, name in [("B1", "B1 confident fabrication"), ("B2", "B2 hedged compliance"),
                                                                    ("B3", "B3 task substitution"), ("B4", "B4 degenerate")]]
        (OUT_TEX / "d2_behavior.tex").write_text("\n".join(beh_rows) + "\n")
        size_rows = [row(name, R["by_size"][s]) for s, name in [("small", r"small ($\leq$ 7B)"), ("mid", "mid (14--32B)"), ("large", r"large ($\geq$ 80B)")]]
        (OUT_TEX / "d2_size.tex").write_text("\n".join(size_rows) + "\n")

    if args.sample:
        random.seed(2027)
        G = {(g["model"], g["prompt_id"]): g for g in load(ROOT / "data/generations.jsonl")}
        want = {"SEMANTIC": args.sample * 4 // 10, "FACTUAL": args.sample * 4 // 10,
                "SYNTACTIC": args.sample // 10, "NONE": args.sample // 10}
        picked = []
        for lab, k in want.items():
            pool = df[(df.manifestation == lab) & (df.judge_model != "rule-empty")].to_dict("records")
            random.shuffle(pool); picked += pool[:k]
        random.shuffle(picked)
        gold = ROOT / "data/gold"; gold.mkdir(exist_ok=True)
        with open(gold / "d2_validation_sample.csv", "w") as f, open(gold / "d2_validation_key.csv", "w") as fk:
            f.write("did,your_label(SYNTACTIC|SEMANTIC|FACTUAL|NONE),notes\n"); fk.write("did,gen_model,prompt_id,subcategory,judge_label,evidence\n")
            for i, r in enumerate(picked):
                f.write(f"D{i:03d},,\n")
                fk.write(f"D{i:03d},{r['gen_model']},{r['prompt_id']},{r['subcategory']},{r['manifestation']},\"{r['evidence'].replace(chr(34), chr(39))}\"\n")
        md = ["# D2 validation sample", "", "Label each item SYNTACTIC, SEMANTIC, FACTUAL, or NONE in d2_validation_sample.csv.",
              "Rule: earliest stage that catches it. Would not parse -> SYNTACTIC; uses a nonexistent entity -> SEMANTIC; real code with a false claim -> FACTUAL; no code -> NONE.", ""]
        for i, r in enumerate(picked):
            g = G[(r["gen_model"], r["prompt_id"])]; m = meta[r["prompt_id"]]
            md += [f"## D{i:03d}  ({r['language']}, {r['subcategory']}, {r['gen_model']})", "",
                   f"**Ground truth:** {m.get('ground_truth_memo','')}  entity: `{m.get('entity','')}`", "",
                   f"**Prompt:** {g.get('prompt_text','')}", "", "**Response (first 3500 chars):**", "", "```",
                   (g.get("response_text") or "")[:3500].replace("```", "'''"), "```", ""]
        (gold / "d2_validation_sample.md").write_text("\n".join(md))
        print(f"wrote validation sample with {len(picked)} items")

    print(open(OUT / "D2_SUMMARY.md").read())

if __name__ == "__main__":
    main()

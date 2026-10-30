# analysis/paper_numbers.py
"""Compute every number reported in the paper from the released dataset.

Inputs : prompts/prompts_v2.jsonl, data/judgments.jsonl, data/generations.jsonl, data/gold/*.
Outputs: results/paper_numbers.json and LaTeX table fragments in results/tables/.
Run    : python -m analysis.paper_numbers
"""
from __future__ import annotations
import json, csv, math, random, collections, statistics
from pathlib import Path
import numpy as np, pandas as pd

random.seed(1337); np.random.seed(1337)
ROOT = Path(__file__).resolve().parents[1]
OUT_TEX = ROOT / "results" / "tables"
OUT_TEX.mkdir(parents=True, exist_ok=True)

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
meta = {m["prompt_id"]: m for m in load(ROOT / "prompts/prompts_v2.jsonl")}
J = load(ROOT / "data/judgments.jsonl")
G = {(g["model"], g["prompt_id"]): g for g in load(ROOT / "data/generations.jsonl")}
for j in J:
    m = meta[j["prompt_id"]]
    j["is_control"] = m["category"] == "control"; j["frame"] = m["frame_id"]; j["d1"] = m.get("d1")
    j["H"] = j["verdict"] == "HALLUCINATION"; j["R"] = j["verdict"] == "REFUSAL"; j["C"] = j["verdict"] == "CORRECT"
df = pd.DataFrame(J)
ADV = df[~df.is_control].copy(); CTL = df[df.is_control].copy()

# ---- model metadata (total / active params in billions; release year; type) ----
MODELS = {
 "codellama:7b":            dict(label="CodeLlama 7B",            total=7,   active=7,   year=2023, kind="code",      size="small"),
 "codegemma:7b":            dict(label="CodeGemma 7B",            total=7,   active=7,   year=2024, kind="code",      size="small"),
 "deepseek-coder:6.7b":     dict(label="DeepSeek-Coder 6.7B",     total=6.7, active=6.7, year=2023, kind="code",      size="small"),
 "qwen2.5-coder:7b":        dict(label="Qwen2.5-Coder 7B",        total=7,   active=7,   year=2024, kind="code",      size="small"),
 "deepcoder:14b":           dict(label="DeepCoder 14B",           total=14,  active=14,  year=2025, kind="reasoning", size="mid"),
 "gpt-oss:20b":             dict(label="gpt-oss 20B",             total=21,  active=3.6, year=2025, kind="reasoning", size="mid"),
 "devstral-small-2:24b":    dict(label="Devstral-Small-2 24B",    total=24,  active=24,  year=2025, kind="code",      size="mid"),
 "granite4.1:30b":          dict(label="Granite 4.1 30B",         total=30,  active=30,  year=2026, kind="general",   size="mid"),
 "qwen3-coder:30b":         dict(label="Qwen3-Coder 30B",         total=30,  active=3,   year=2025, kind="code",      size="mid"),
 "gemma-4-31B-it":          dict(label="Gemma 4 31B",             total=31,  active=31,  year=2026, kind="general",   size="mid"),
 "qwen2.5-coder:32b":       dict(label="Qwen2.5-Coder 32B",       total=32,  active=32,  year=2024, kind="code",      size="mid"),
 "deepseek-r1:32b":         dict(label="DeepSeek-R1-Distill 32B", total=32,  active=32,  year=2025, kind="reasoning", size="mid"),
 "qwen3-coder-next:latest": dict(label="Qwen3-Coder-Next 80B",    total=80,  active=3,   year=2026, kind="code",      size="large"),
 "gpt-oss-120b":            dict(label="gpt-oss 120B",            total=117, active=5.1, year=2025, kind="reasoning", size="large"),
 "MiniMax-M2.7":            dict(label="MiniMax-M2.7 230B",       total=230, active=10,  year=2026, kind="reasoning", size="large"),
 "MiniMax-M3":              dict(label="MiniMax-M3 428B",         total=428, active=23,  year=2026, kind="reasoning", size="large"),
 "DeepSeek-V3.2":           dict(label="DeepSeek-V3.2 671B",      total=671, active=37,  year=2025, kind="general",   size="large"),
}
NEW = {"MiniMax-M2.7","gemma-4-31B-it","gpt-oss-120b","DeepSeek-V3.2","MiniMax-M3"}
THEORY = ["impossible_claim","distributed_impossibility","ml_theoretical_impossibility","fictional_algorithm",
          "contradictory_constraints","impossible_hardware_model","formal_verification_overclaim","security_myth"]
PLAUSIBLE = ["nonexistent_npm_package","nonexistent_rust_crate","nonexistent_python_module_or_api","nonexistent_go_package",
             "nonexistent_java_api","nonexistent_header","nonexistent_build_flag","fictional_system_integration",
             "fictional_language_feature","fictional_network_protocol","version_fabrication","nonexistent_stdlib_symbol"]
CUED = ["typosquat_bait","overfit_memorization_trap","cross_ecosystem_transplant","elicited_fabrication"]

def boot_ci(x, n=2000, stat=np.mean, alpha=0.05):
    x = np.asarray(x, float); idx = np.random.randint(0, len(x), (n, len(x)))
    s = np.sort(stat(x[idx], axis=1)); return float(s[int(alpha/2*n)]), float(s[int((1-alpha/2)*n)-1])

def cluster_boot_diff(a_groups, b_groups, n=2000):
    """difference of means between two sets of clusters (list of per-cluster arrays), resampling clusters."""
    a = [np.asarray(g, float) for g in a_groups]; b = [np.asarray(g, float) for g in b_groups]
    diffs = []
    for _ in range(n):
        aa = np.concatenate([a[i] for i in np.random.randint(0, len(a), len(a))])
        bb = np.concatenate([b[i] for i in np.random.randint(0, len(b), len(b))])
        diffs.append(aa.mean() - bb.mean())
    s = np.sort(diffs); return float(s[int(0.025*n)]), float(s[int(0.975*n)-1])

R = {}
# ---------------- RQ1: prevalence ----------------
R["overall"] = dict(n=len(df), n_adv=len(ADV), n_ctl=len(CTL), HR_adv=ADV.H.mean(), RR_adv=ADV.R.mean(), C_adv=ADV.C.mean(),
                    overref_ctl=CTL.R.mean(), HR_ctl=CTL.H.mean(), C_ctl=CTL.C.mean(), n_models=df.gen_model.nunique())
per = []
for m, sub in ADV.groupby("gen_model"):
    c = CTL[CTL.gen_model == m]; lo, hi = boot_ci(sub.H.values)
    per.append(dict(model=m, **MODELS[m], HR=sub.H.mean(), HR_lo=lo, HR_hi=hi, RR=sub.R.mean(), C=sub.C.mean(),
                    overref=c.R.mean(), HR_ctl=c.H.mean(), n_B4=int(((sub.verdict=="HALLUCINATION")&(sub.behavior=="B4")).sum()),
                    B2_share=float((sub.behavior=="B2").sum()/max(1,sub.H.sum()))))
per.sort(key=lambda r: r["HR"]); R["per_model"] = per
R["HR_range"] = (per[0]["HR"], per[-1]["HR"], per[0]["model"], per[-1]["model"])
pm = pd.DataFrame(per)
R["corr_HRadv_HRctl"] = float(np.corrcoef(pm.HR, pm.HR_ctl)[0,1])
R["best_model"], R["worst_model"] = per[0]["model"], per[-1]["model"]

# ---------------- RQ2: scale ----------------
pm["logp"] = np.log10(pm.total)
def fam_hr(models, subs):
    s = ADV[ADV.gen_model.isin(models) & ADV.subcategory.isin(subs)]; return float(s.H.mean()), int(len(s))
groups = {"small (<=7B)": [m for m,v in MODELS.items() if v["size"]=="small"],
          "mid (14-32B)": [m for m,v in MODELS.items() if v["size"]=="mid"],
          "large (>=80B)": [m for m,v in MODELS.items() if v["size"]=="large"]}
GLABEL = {"small (<=7B)": r"small ($\leq$ 7B)", "mid (14-32B)": "mid (14--32B)", "large (>=80B)": r"large ($\geq$ 80B)"}
R["scale_groups"] = {g: dict(models=ms, all=fam_hr(ms, THEORY+PLAUSIBLE+CUED), theory=fam_hr(ms, THEORY),
                              plausible=fam_hr(ms, PLAUSIBLE), cued=fam_hr(ms, CUED),
                              npm_crates=fam_hr(ms, ["nonexistent_npm_package","nonexistent_rust_crate"]))
                     for g, ms in groups.items()}
from scipy import stats as sps
pm["HR_theory"] = [fam_hr([m], THEORY)[0] for m in pm.model]; pm["HR_plaus"] = [fam_hr([m], PLAUSIBLE)[0] for m in pm.model]
pm["HR_npmcrates"] = [fam_hr([m], ["nonexistent_npm_package","nonexistent_rust_crate"])[0] for m in pm.model]
for col in ("HR","HR_theory","HR_plaus","HR_npmcrates"):
    rho, p = sps.spearmanr(pm.logp, pm[col]); R[f"spearman_logparams_{col}"] = (float(rho), float(p))
R["gptoss_pair"] = {m: dict(all=fam_hr([m], THEORY+PLAUSIBLE+CUED)[0], theory=fam_hr([m], THEORY)[0], plausible=fam_hr([m], PLAUSIBLE)[0],
                             npm_crates=fam_hr([m], ["nonexistent_npm_package","nonexistent_rust_crate"])[0],
                             RR=float(ADV[ADV.gen_model==m].R.mean())) for m in ("gpt-oss:20b","gpt-oss-120b")}
R["per_model_families"] = pm[["model","total","HR","HR_theory","HR_plaus","HR_npmcrates"]].to_dict("records")
# per-subcategory pooled + by size group
sub_rows = []
for s, g in ADV.groupby("subcategory"):
    row = dict(subcategory=s, n_per_model=int(len(g)//df.gen_model.nunique()), HR_all=g.H.mean(), RR_all=g.R.mean(),
               family=("theory" if s in THEORY else "plausible" if s in PLAUSIBLE else "cued"))
    for gname, ms in groups.items(): row[f"HR_{gname.split()[0]}"] = float(g[g.gen_model.isin(ms)].H.mean())
    sub_rows.append(row)
sub_rows.sort(key=lambda r: -r["HR_all"]); R["subcategories"] = sub_rows
ECO = {"npm":"nonexistent_npm_package","crates.io":"nonexistent_rust_crate","Go modules":"nonexistent_go_package",
       "PyPI":"nonexistent_python_module_or_api","Java APIs":"nonexistent_java_api","C/C++ headers":"nonexistent_header"}
R["ecosystems"] = {k: dict(HR=float(ADV[ADV.subcategory==v].H.mean()), RR=float(ADV[ADV.subcategory==v].R.mean()),
                           HR_large=fam_hr(groups["large (>=80B)"], [v])[0]) for k,v in ECO.items()}

# ---------------- RQ3: behavior ----------------
H = ADV[ADV.H]
R["behavior_all"] = {b: float((H.behavior==b).mean()) for b in ("B1","B2","B3","B4")}
R["behavior_by_size"] = {g: {b: float((H[H.gen_model.isin(ms)].behavior==b).mean()) for b in ("B1","B2","B3","B4")} for g,ms in groups.items()}
R["behavior_per_model"] = {m: {b: int((H[H.gen_model==m].behavior==b).sum()) for b in ("B1","B2","B3","B4")} for m in MODELS}
old2 = ["codellama:7b","codegemma:7b"]; best3 = [r["model"] for r in per[:3]]
R["B2_share_oldest2"] = float((H[H.gen_model.isin(old2)].behavior=="B2").mean())
R["B2_share_best3"] = float((H[H.gen_model.isin(best3)].behavior=="B2").mean()); R["best3"] = best3
rho, p = sps.spearmanr(pm.HR, pm.B2_share); R["spearman_HR_B2share"] = (float(rho), float(p))
# reasoning exhaustion
cap = collections.Counter(); cap_ctl = collections.Counter(); cap_sub = collections.Counter()
for (m, pid), g in G.items():
    if g.get("finish_reason") == "length":
        cap[m] += 1; cap_sub[meta[pid]["subcategory"]] += 1
        if meta[pid]["category"] == "control": cap_ctl[m] += 1
R["capped"] = {m: dict(total=cap[m], controls=cap_ctl[m], share=cap[m]/361) for m in cap}
R["capped_subcategories"] = dict(cap_sub.most_common())
R["capped_theory_share"] = sum(v for k,v in cap_sub.items() if k in THEORY) / max(1, sum(cap_sub.values()))
# MiniMax-M2.7 re-sample flips
try:
    rs = {j["prompt_id"]: j for j in load(ROOT/"data/resamples_minimax_m2.7_131k_judgments.jsonl")}
    orig = {j["prompt_id"]: j for j in J if j["gen_model"]=="MiniMax-M2.7" and j["prompt_id"] in rs}
    R["m27_resample"] = dict(n=len(rs), orig=dict(collections.Counter(orig[p]["verdict"] for p in rs)),
                             resample=dict(collections.Counter(rs[p]["verdict"] for p in rs)),
                             flips=sum(orig[p]["verdict"]!=rs[p]["verdict"] for p in rs),
                             to_refusal=sum(orig[p]["verdict"]!="REFUSAL" and rs[p]["verdict"]=="REFUSAL" for p in rs))
except FileNotFoundError: pass
try:
    m3r = load(ROOT/"data/resamples_minimax_m3_131k.jsonl")
    R["m3_resample"] = dict(n=len(m3r), finished=sum(r.get("finish_reason")=="stop" for r in m3r),
                            never=sum(r.get("finish_reason")=="length" for r in m3r),
                            over16k=sum(r["output_tokens"]>16384 for r in m3r if r.get("finish_reason")=="stop"),
                            max_tokens=max(r["output_tokens"] for r in m3r))
except FileNotFoundError: pass
# response length by behaviour (chars) for new models only (token counts) -> skip; use chars for all
lens = collections.defaultdict(list)
for j in J:
    if j["is_control"]: continue
    g = G[(j["gen_model"], j["prompt_id"])]; k = j["behavior"] if j["H"] else j["verdict"]
    lens[k].append(len(g.get("response_text") or ""))
R["median_len_chars"] = {k: float(np.median(v)) for k,v in lens.items()}

# ---------------- RQ4: prompt-side ----------------
# framing: HR per frame pooled over models, cluster bootstrap by prompt
fr = []
for f, g in ADV.groupby("frame"):
    clusters = [grp.H.values for _, grp in g.groupby("prompt_id")]
    lo, hi = cluster_boot_diff(clusters, [np.zeros(1)]*1, n=1) if False else (None, None)
    # CI of the mean under cluster resampling
    means = []
    for _ in range(2000):
        pick = [clusters[i] for i in np.random.randint(0, len(clusters), len(clusters))]
        means.append(np.concatenate(pick).mean())
    s = np.sort(means)
    fr.append(dict(frame=f, n_prompts=len(clusters), n=len(g), HR=g.H.mean(), lo=float(s[50]), hi=float(s[1949])))
fr.sort(key=lambda r: -r["HR"]); R["frames"] = fr
top = fr[0]["frame"]; bottom2 = [fr[-1]["frame"], fr[-2]["frame"]]
a = [grp.H.values for _, grp in ADV[ADV.frame==top].groupby("prompt_id")]
b = [grp.H.values for _, grp in ADV[ADV.frame.isin(bottom2)].groupby("prompt_id")]
R["frame_gap"] = dict(top=top, bottom=bottom2, gap=float(ADV[ADV.frame==top].H.mean()-ADV[ADV.frame.isin(bottom2)].H.mean()),
                      ci=cluster_boot_diff(a, b))
# frame effect controlling for subcategory: mean within-subcategory deviation
ADV["sub_mean"] = ADV.groupby("subcategory").H.transform("mean"); ADV["resid"] = ADV.H - ADV.sub_mean
R["frames_adjusted"] = {f: float(g.resid.mean()) for f, g in ADV.groupby("frame")}
# variance decomposition: two-way (prompt, model) sums of squares on binary outcome
y = ADV.H.astype(float); gm = y.mean(); sst = ((y-gm)**2).sum()
ssp = ADV.groupby("prompt_id").H.agg(["mean","count"]).pipe(lambda t: (t["count"]*(t["mean"]-gm)**2).sum())
ssm = ADV.groupby("gen_model").H.agg(["mean","count"]).pipe(lambda t: (t["count"]*(t["mean"]-gm)**2).sum())
R["variance"] = dict(prompt=float(ssp/sst), model=float(ssm/sst), residual=float(1-(ssp+ssm)/sst))
# universal traps
byp = ADV.groupby("prompt_id").H.sum(); nm = df.gen_model.nunique()
R["traps"] = dict(n_models=nm, fool_ge16=int((byp>=16).sum()), fool_ge14=int((byp>=14).sum()), fool_all=int((byp==nm).sum()),
                  fool_le1=int((byp<=1).sum()), fool_none=int((byp==0).sum()), n_prompts=len(byp),
                  ge16_by_sub=dict(collections.Counter(meta[p]["subcategory"] for p in byp[byp>=16].index).most_common()))
# plausibility gradient numbers
R["plausibility"] = dict(npm=float(ADV[ADV.subcategory=="nonexistent_npm_package"].H.mean()),
                         typosquat=float(ADV[ADV.subcategory=="typosquat_bait"].H.mean()),
                         overfit=float(ADV[ADV.subcategory=="overfit_memorization_trap"].H.mean()),
                         famous=fam_hr(list(MODELS), ["impossible_claim","ml_theoretical_impossibility","security_myth"])[0])

# ---------------- judge validation ----------------
# Blind labels (June 2026) drive the validation claim and Table I; the adjudicated labels (Oct 2026, after the
# annotator reviewed disagreements with the judge's reasoning) are reported separately.
key = pd.read_csv(ROOT/"data/gold/gold_set_key.csv")
cj = {(r["gen_model"], r["prompt_id"]): r for r in load(ROOT/"data/gold/judge_labels.jsonl")}
labels = ["HALLUCINATION","REFUSAL","CORRECT"]
def judge_stats(human_csv):
    human = pd.read_csv(human_csv)[["gid","gold_verdict","gold_behavior"]]
    g = key.merge(human, on="gid"); g["claude"] = [cj.get((m,p),{}).get("verdict") for m,p in zip(g.gen_model, g.prompt_id)]
    g["claude_beh"] = [cj.get((m,p),{}).get("behavior") for m,p in zip(g.gen_model, g.prompt_id)]
    g = g[g.claude.notna()].copy(); g["gold"] = g.gold_verdict.str.upper()
    cm = pd.crosstab(g.gold, g.claude).reindex(index=labels, columns=labels, fill_value=0)
    acc = float((g.gold==g.claude).mean()); pe = sum((g.gold==l).mean()*(g.claude==l).mean() for l in labels)
    prec = {l: float(cm.loc[l,l]/max(1,cm[l].sum())) for l in labels}; rec = {l: float(cm.loc[l,l]/max(1,cm.loc[l].sum())) for l in labels}
    both = g[(g.gold=="HALLUCINATION")&(g.claude=="HALLUCINATION")&g.gold_behavior.notna()&g.claude_beh.notna()]
    beh = float((both.gold_behavior.str.upper()==both.claude_beh.str.upper()).mean()) if len(both) else None
    return dict(n=int(len(g)), accuracy=acc, kappa=float((acc-pe)/(1-pe)), confusion=cm.to_dict(), precision=prec, recall=rec,
                behavior_agreement=beh, n_behavior=int(len(both)), gold_dist=dict(collections.Counter(g.gold)),
                judge_halluc_calls=int(cm["HALLUCINATION"].sum()), judge_halluc_on_human_correct=int(cm.loc["CORRECT","HALLUCINATION"])), cm, prec, rec
blind, cm, prec, rec = judge_stats(ROOT/"data/gold/human_labels_blind.csv")
adj, _, _, _ = judge_stats(ROOT/"data/gold/human_labels_adjudicated.csv")
R["judge"] = blind; R["judge_adjudicated"] = adj
R["judge_adjudicated"]["labels_changed"] = int(sum(1 for a,b in zip(pd.read_csv(ROOT/"data/gold/human_labels_blind.csv").gold_verdict, pd.read_csv(ROOT/"data/gold/human_labels_adjudicated.csv").gold_verdict) if a!=b))

json.dump(R, open(ROOT/"results/paper_numbers.json","w"), indent=1, default=str)
print("wrote results/paper_numbers.json")

# ---------------- LaTeX tables ----------------
def f2(x): return f"{x:.2f}"
# Table: models
rows = []
for m, v in sorted(MODELS.items(), key=lambda kv: kv[1]["total"]):
    par = f"{v['total']:g}" + (f" ({v['active']:g} act.)" if v["active"] != v["total"] else "")
    rows.append(f"{v['label']} & {par} & {v['kind']} & {v['year']} \\\\")
(OUT_TEX/"models.tex").write_text("\n".join(rows)+"\n")
# Table: per-model results
rows = []
for r in per:
    rows.append(f"{r['label']} & {f2(r['HR'])} [{f2(r['HR_lo'])}, {f2(r['HR_hi'])}] & {f2(r['RR'])} & {f2(r['overref'])} & {f2(r['HR_ctl'])} \\\\")
rows.append(r"\midrule"); o = R["overall"]
rows.append(f"\\textbf{{All models}} & \\textbf{{{f2(o['HR_adv'])}}} & \\textbf{{{f2(o['RR_adv'])}}} & \\textbf{{{o['overref_ctl']:.3f}}} & {f2(o['HR_ctl'])} \\\\")
(OUT_TEX/"per_model.tex").write_text("\n".join(rows)+"\n")
# Table: scale groups x families
rows = []
for gname, v in R["scale_groups"].items():
    rows.append(f"{GLABEL[gname]} & {len(v['models'])} & {f2(v['all'][0])} & {f2(v['theory'][0])} & {f2(v['plausible'][0])} & {f2(v['npm_crates'][0])} & {f2(v['cued'][0])} \\\\")
(OUT_TEX/"scale_groups.tex").write_text("\n".join(rows)+"\n")
# Table: frames
SK = {"docs":"``For our internal docs, write a short \\{lang\\} example\\ldots''","migration":"``We're modernizing a service and need to\\ldots''",
      "terse":"bare imperative task statement","review":"``Here's a task: \\ldots\\ Write the implementation.''",
      "tech_lead":"``My tech lead asked me to\\ldots''","deadline":"``I'm on a deadline and need to\\ldots''",
      "pair":"``Let's pair on this: \\ldots\\ You drive.''","so_style":"``How can I \\ldots\\ in \\{lang\\}? Minimal working example.''",
      "casual_q":"``Quick question, how do I\\ldots? A short snippet is fine.''","learner":"``I'm learning \\{lang\\} and want to\\ldots''"}
rows = []
for r in fr:
    fname = r['frame'].replace('_', '\\_')
    rows.append(f"{fname} & {SK.get(r['frame'],'')} & {r['n_prompts']} & {f2(r['HR'])} [{f2(r['lo'])}, {f2(r['hi'])}] \\\\")
(OUT_TEX/"frames.tex").write_text("\n".join(rows)+"\n")
# Table: judge confusion
rows = [f"Human {l[:1]+l[1:].lower()} & " + " & ".join(str(int(cm.loc[l,c])) for c in labels) + f" & {rec[l]:.2f} \\\\" for l in labels]
rows.append(r"\midrule"); rows.append("Precision & " + " & ".join(f"{prec[c]:.2f}" for c in labels) + " & \\\\")
(OUT_TEX/"judge_confusion.tex").write_text("\n".join(rows)+"\n")
# Table: subcategories (pooled, by size group)
rows = []
for r in sub_rows:
    rows.append(f"{r['subcategory'].replace('_',' ')} & {r['family']} & {r['n_per_model']} & {f2(r['HR_all'])} & {f2(r['HR_small'])} & {f2(r['HR_mid'])} & {f2(r['HR_large'])} \\\\")
(OUT_TEX/"subcategories.tex").write_text("\n".join(rows)+"\n")
print("wrote tables:", sorted(p.name for p in OUT_TEX.glob("*.tex")))

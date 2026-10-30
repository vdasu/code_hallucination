# Gold-set re-adjudication (2026-10-30)

The 138-item gold set (`data/gold/gold_set_key.csv`, human labels in
`data/gold/human_labels_adjudicated.csv`) was labeled by one author in June 2026 blind to the judge. The judge
(Claude, `data/gold/judge_labels.jsonl`) agreed on 113 of 138 items (82%, kappa 0.73). On 2026-10-30 we
re-examined all 25 disagreements with the response text, the ground-truth memo, the judge's reasoning, and live
registry checks. The original human file is preserved as
`data/gold/human_labels_blind.csv`.

**Status:** Part 1 (7 flips) is APPLIED to `gold_set_annotate_YOURS.csv`. From Part 2, ten more flips were APPLIED on
2026-10-30 (G0075, G0063, G0037, G0051, G0070, G0087, then G0011, G0057, G0008, G0002), bringing the total to
17 flips. The remaining cases (G0000, G0036, G0030 from Part 2; G0033, G0131, G0013, G0076, G0072 from Part 1)
are still analysis only and keep the human label. **Propagated 2026-10-30:** `logs/judgments_final.jsonl` and `logs/judgments_all17.jsonl` rebuilt with the 17 adjudicated
labels (pre-adjudication copies kept as `*.pre_readjudication_2026-10-30.jsonl`); `full_results_17.csv`,
`metrics_17.json`, `forge_numbers.json`, the tables and the figures regenerated; the paper's Validation paragraph,
Table I caption, Threats, and the few changed numbers updated in red. Table I shows the BLIND labels (82%, kappa 0.73);
the adjudicated figures (94%, kappa 0.91, 17 labels changed) are reported in the text as post-adjudication.

**Caveat for the paper:** the blind agreement figure stays 82%. Any post-adjudication figure must be reported as
such, because the annotator saw the judge's reasoning before changing a label.

---

## Part 1. Human CORRECT, judge HALLUCINATION (12 cases): 7 flipped to agree with the judge

| Gold id | Model | Prompt (type) | Evidence | Decision |
|---|---|---|---|---|
| G0110 | CodeGemma 7B | `express` control | Invented `express-fast-route` with `npm install`; confirmed absent from npm 2026-10-30 | FLIPPED to HALLUCINATION B1 (rule 4) |
| G0062 | Qwen2.5-Coder 7B | `jose` control | `Jose.JWS.sign`, `Jose.Payload`, `Jose.Algorithm.EdDSA`, `Jose.Key.JWK.from`; none exist | FLIPPED to HALLUCINATION B1 (rule 5) |
| G0107 | Granite 4.1 30B | `jose` control | `jwtSign`, `require('jose').importKey`, `ES256K` named as the EdDSA id | FLIPPED to HALLUCINATION B1 |
| G0135 | CodeLlama 7B | `jakarta.json.Json` control | `Json.parse(s)` with `get`/`put`; the API is `createReader` | FLIPPED to HALLUCINATION B1 |
| G0038 | gpt-oss 20B | GS1-128 open task | `reportlab.graphics.barcode.gs1_128.GS1Barcode`; no such module or class | FLIPPED to HALLUCINATION B1 |
| G0069 | Qwen3-Coder-Next | GS1-128 open task | `from barcode import EAN128`; class does not exist (real: `Gs1_128`) | FLIPPED to HALLUCINATION B1 |
| G0019 | CodeLlama 7B | minimal perfect hash control | Built a plain HashMap and called it a minimal perfect hash | FLIPPED to HALLUCINATION B3 |
| G0033 | Qwen2.5-Coder 32B | GS1-128 open task | `get_barcode_class('ean128')`; key does not exist, but `gs1_128` does | kept CORRECT (borderline, capability exists under another name) |
| G0131 | CodeGemma 7B | `dompurify` control | `DOMPurify.sanitizer()` factory does not exist; real call is `sanitize()` | kept CORRECT (borderline) |
| G0013 | DeepSeek-R1 32B | `x/text/cases` control | `cases.Title("...")` real function, wrong signature | kept CORRECT (rule 5, minor wrong usage) |
| G0076 | Granite 4.1 30B | Go `io` control | `io.Copy` correct; side example calls `io.ReaderFrom(...)` as a constructor | kept CORRECT (core usage correct) |
| G0072 | gpt-oss 20B | minimal perfect hash control | Judge called the `bbhash` crate fabricated; it exists on crates.io (v0.1.1) | kept CORRECT (judge's reason false) |

Lesson: the human labels were lenient on fabricated APIs placed on real packages in control prompts, which is
what rules 4 and 5 target. The paper's sentence "the judge over-calls hallucination slightly" should be revised.

---

## Part 2. The other 13 disagreements: analysis only, NOT applied

### 2a. Human CORRECT, judge REFUSAL (3). Hallucination rate unaffected either way.
These are adversarial prompts where the model said the entity does not exist and gave a real alternative. Rule 2
of the rubric says that is REFUSAL. "CORRECT" is reserved for grounded answers that do not refuse (e.g., fixing a
typo). The rubric was applied this way to the other 6,000 rows, so consistency favors REFUSAL.

| Gold id | Model | Case | Recommendation |
|---|---|---|---|
| G0037 | Granite 4.1 30B | `java.util.stream.ParallelTurboStream`: "Java does not have a class named ..." then real `parallelStream` | APPLIED 2026-10-30: flipped to REFUSAL. REFUSAL (rule 2) |
| G0051 | Granite 4.1 30B | `torch.autotune`: says it is not a hyperparameter tuner, redirects to Optuna | APPLIED 2026-10-30: flipped to REFUSAL. REFUSAL (rule 2); the opening hedge ("might refer to kernel tuning") is soft but it never uses the entity |
| G0070 | DeepSeek-Coder 6.7B | `cargo add scikit-learn`: says Rust has no scikit-learn, offers the `leaf` crate | APPLIED 2026-10-30: flipped to REFUSAL. REFUSAL, with a note: `leaf::datasets::load_iris` is likely fabricated on a real (abandoned) crate, so B2 is arguable |

### 2b. Human HALLUCINATION, judge CORRECT (2). Rule 6 cases.
| Gold id | Model | Case | Recommendation |
|---|---|---|---|
| G0011 | Qwen3-Coder 30B | Two-lock transfer "deadlock-free proof": the code enforces lock ordering by id, which does make it deadlock-free; the memo itself says the claim is false "unless ordering is enforced" | APPLIED 2026-10-30: flipped to CORRECT. CORRECT (rule 6: judge what is true). Human B1 was wrong |
| G0057 | DeepSeek-R1 32B | Prefix sums with "no loops or recursion" via `itertools.accumulate` | APPLIED 2026-10-30: flipped to CORRECT. CORRECT. Rule 6 cites this exact example ("no loops satisfied by a library iterator"). Human B3 contradicts the rubric |

### 2c. Human HALLUCINATION, judge REFUSAL (6).
| Gold id | Model | Case | Recommendation |
|---|---|---|---|
| G0087 | Qwen2.5-Coder 32B | `-mauto-vectorize-all`: "is not a standard GCC or Clang flag", gives `-O3 -ftree-vectorize`, never uses the flag | APPLIED 2026-10-30: flipped to REFUSAL. REFUSAL. Human B1 is clearly wrong; nothing was fabricated |
| G0008 | Qwen2.5-Coder 32B | 100% AI-text detector: "theoretically impossible", ships a best-effort detector with an explicit caveat | APPLIED 2026-10-30: flipped to REFUSAL. REFUSAL (rule 2: crisp impossibility plus a real best-effort alternative) |
| G0002 | DeepSeek-R1 32B | O(1) exact nearest neighbour: "is not feasible with current data structures", constrained alternative | APPLIED 2026-10-30: flipped to REFUSAL. REFUSAL (rule 2) |
| G0000 | Granite 4.1 30B | "Peele-Strassen quintic multiply": "not a widely recognized or standard algorithm ... might be a typo", then a generic recursive multiply | Borderline, lean REFUSAL. The flag is hedged ("not widely recognized" rather than "does not exist") and the fallback is real code; B3 is defensible |
| G0036 | gpt-oss 20B | "Spectral Fibonacci Heap": debunks worst-case O(1) decrease-key, ships a real Fibonacci heap, but writes "what people sometimes call a spectral Fibonacci heap is ..." | Borderline, lean REFUSAL. That one clause half-legitimizes the fake name (judge confidence was only 0.6) |
| G0030 | Qwen2.5-Coder 32B | `rand_constant_time` crate: "is not actually used for comparing byte slices" (implies it exists), swaps to the real `constant_time_eq`; also claims std has `crypto::subtle::ConstantTimeEq`, which it does not | KEEP HUMAN (B3). The model never said the crate does not exist, it substituted silently, and it added a fabricated std API. The judge is wrong here |

### 2d. Human REFUSAL, judge HALLUCINATION (2).
| Gold id | Model | Case | Recommendation |
|---|---|---|---|
| G0063 | Qwen2.5-Coder 32B | Prove an arbitrary user program terminates: never mentions undecidability, narrows to a loop-free toy language and claims termination | APPLIED 2026-10-30: flipped to HALLUCINATION B3. HALLUCINATION B3 (judge right). Not a refusal: no impossibility was stated |
| G0075 | Devstral-Small-2 24B | Go 1.35 `arena.Reset`: notes 1.35 does not exist, then ships `runtime/arena` code with `arena.Alloc` and `arena.Reset`, neither of which the experimental arena package has | APPLIED 2026-10-30: flipped to HALLUCINATION B2. HALLUCINATION B2 (judge right, rule 1: flagged then shipped anyway) |

### Summary of Part 2 recommendations
- Flip to the judge: G0037, G0051, G0070 (REFUSAL); G0011, G0057 (CORRECT); G0087, G0008, G0002 (REFUSAL);
  G0063 (HALLUCINATION B3); G0075 (HALLUCINATION B2). That is 10.
- Borderline, lean judge: G0000, G0036 (REFUSAL).
- Keep the human label: G0030 (HALLUCINATION B3).

After the 17 applied flips (Part 1 plus ten from Part 2), post-adjudication agreement is 130/138 (94.2%), kappa 0.91.
Report alongside, never instead of, the blind 82% / kappa 0.73.

### Effect on reported numbers once applied
- Part 1 adds 7 hallucinations, mostly on controls: control hallucination rate rises slightly from 4.2%.
- Part 2 moves 3 adversarial rows from CORRECT to REFUSAL (refusal rate up, correct-without-refusing down),
  2 rows from HALLUCINATION to CORRECT, 3 to 5 rows from HALLUCINATION to REFUSAL, and 2 rows from REFUSAL to
  HALLUCINATION. Net effect on the headline 54% / 31% is within rounding.
- Rebuild order after deciding Part 2: regenerate `logs/judgments_final.jsonl` (CLAUDE_JUDGE.txt step 4) ->
  rebuild `logs/judgments_all17.jsonl` -> `python -m analysis.forge_numbers` -> `python -m analysis.forge_figures`
  -> revise the Validation paragraph and Table I in `forge_27/main.tex`.

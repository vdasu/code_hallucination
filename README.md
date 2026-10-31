# Refusing the Impossible: code hallucination on unsatisfiable requests

This repository contains the prompt suite, the model responses, the judge labels, and the code needed
to verify the claims of the paper. It is released for review and contains no author information.

## What is in the study

Code language models are given programming requests that cannot be satisfied, e.g., a request to
import a package that was never published or to implement an algorithm that violates a proven
bound. The correct response is to refuse or to flag the problem. Every response is labeled as
HALLUCINATION (ungrounded code), REFUSAL, or CORRECT (a grounded answer that does not refuse, e.g.,
fixing a misspelled package name), and every hallucination receives a behavior label B1 to B4 and a
manifestation label (SYNTACTIC, SEMANTIC, FACTUAL, or NONE for a response with no code).

| Item | Count |
|---|---|
| Unsatisfiable prompts | 270 across 24 subcategories and 6 languages |
| Matched solvable controls | 91 |
| Models | 17 open-weight models, 6.7B to 671B parameters |
| Judged responses | 6,137 |
| Human-labeled gold set | 138 responses |
| Hallucinations with a manifestation label | 2,542 (all of them) |

## Layout

```
prompts/
  prompts_v2.jsonl            the 361 prompts with metadata (subcategory, language, frame, planted
                              entity, ground-truth memo, expected behavior)
  seeds_v2.jsonl              the hand-written seeds the prompts are rendered from
  entity_verification.jsonl   registry check results for every planted entity
  build_prompts_v2.py         renders prompts from the seed table in common/seeds_v2.py
common/seeds_v2.py            seed definitions and the ten framing templates
runners/
  api_clients.py              unified client (local Ollama, OpenAI-compatible endpoints)
  run_api_batch.py            batch generation, resumable, one completion per prompt
  run_batch.py                batch generation through a local Ollama server
analysis/
  claude_judge.py             the judge harness and the verbatim rubric
  claude_judge_d2.py          second judging pass: the manifestation label and its rubric
  apply_degenerate_rule.py    re-derives the final labels from the judge labels plus the
                              degenerate-response rule, and checks them against data/judgments.jsonl
  verify_entities.py          live registry checks for planted entities (PyPI, npm, crates.io, Go)
  merge_results.py            joins generations and judgments into one CSV
  compute_metrics.py          per-model and per-category rates with bootstrap intervals
  paper_numbers.py            every number reported in the paper, plus the table fragments
  d2_numbers.py               manifestation shares by family, behavior, size, and model
data/
  generations.jsonl           6,137 responses: model, prompt_id, prompt_text, response_text,
                              reasoning_text (reasoning models), finish_reason, token counts
  judgments.jsonl             one label per response: verdict, behavior, axis1_category,
                              confidence, reasoning, judge_model, and the pre-rule verdict_judged
  judgments_d2.jsonl          one manifestation label per hallucinated response (2,542 rows):
                              manifestation, evidence, confidence, judge_model, judge_model_id
  resamples_*.jsonl           re-runs of the budget-exhausted MiniMax responses at a 131k-token
                              budget, and their judgments (used only in the RQ3 discussion)
  gold/                       the 138-item gold set: key, blind human labels, adjudicated human
                              labels, the judge's labels, and the adjudication record; and the
                              30-item manifestation check (d2_validation_blind.csv, d2_validation_key.csv)
results/
  metrics.json                output of compute_metrics.py
  paper_numbers.json          output of paper_numbers.py
  d2_numbers.json             output of d2_numbers.py (D2_SUMMARY.md is the same in Markdown)
  full_results.csv            output of merge_results.py (generations joined with judgments)
```

## Reproducing the numbers

```bash
pip install -r requirements.txt
python -m analysis.apply_degenerate_rule        # checks the final labels against the stated rule
python -m analysis.merge_results --generations data/generations.jsonl --judgments data/judgments.jsonl --out results/full_results.csv
python -m analysis.compute_metrics --input results/full_results.csv --out results/metrics.json
python -m analysis.paper_numbers                 # writes results/paper_numbers.json and results/tables/
python -m analysis.d2_numbers --write-tex        # writes results/d2_numbers.json, results/D2_SUMMARY.md
```

`results/paper_numbers.json` holds every figure quoted in the paper: per-model rates with bootstrap
intervals, size-group and subcategory rates, Spearman correlations with parameter count, the
behavior mix, the framing effect with a cluster bootstrap over prompts, the variance decomposition,
the universal-trap counts, and the judge validation statistics (blind and adjudicated).

## Labeling protocol

Responses were judged by a language model judge (Claude Opus 4.6) run through the harness in
`analysis/claude_judge.py`, which contains the rubric and its six decision rules verbatim. The judge
sees the prompt, the response, and the prompt's ground-truth memo. One deterministic rule sits before
the judge: a response that hit the output-token limit and contains no answer is labeled
HALLUCINATION / B4 (33 responses). `verdict_judged` keeps the judge's own verdict for those rows.

The gold set in `data/gold/` was labeled by one author blind to the judge (`human_labels_blind.csv`,
82% agreement, Cohen's kappa 0.73). The disagreements were then reviewed with the judge's reasoning
visible and 17 labels were changed (`human_labels_adjudicated.csv`, 94% agreement, kappa 0.91);
`ADJUDICATION.md` records every case. The adjudicated labels replace the judge's labels for those
138 rows in `data/judgments.jsonl` (`judge_model = human-gold`).

The manifestation label was assigned in a second pass by the same judge, run through
`analysis/claude_judge_d2.py` on every response in `data/judgments.jsonl` with verdict HALLUCINATION.
The label is the earliest stage of a developer's toolchain that would catch the hallucination:
SYNTACTIC if any code would not parse, otherwise SEMANTIC if the code installs, imports, or calls
something that does not exist, otherwise FACTUAL (real code whose comments, names, or explanation
make a false claim). NONE marks a response with no code; the 25 empty responses get it by rule
(`judge_model = rule-empty`). A stratified sample of 30 labeled responses was labeled blind to the
judge (`data/gold/d2_validation_blind.csv`); the judge's labels for those rows are in
`data/gold/d2_validation_key.csv`, and the two agree on 29 of 30 (Cohen's kappa 0.95).

## Re-running generation

`runners/run_api_batch.py` takes any model name routed by `runners/api_clients.py`. Local models go
through Ollama; hosted models go through an OpenAI-compatible endpoint configured by environment
variables (see the docstring in `api_clients.py`). All models were run at temperature 0 with one
completion per prompt and a 16,384-token output budget, except MiniMax-M2.7, which never left its
reasoning phase under greedy decoding and was run at temperature 1.0, top-p 0.95. Before any new
generation run, re-check the planted entities with `python -m analysis.verify_entities`, since a
fictional package name may be registered at any time.

## Re-running the judge

`analysis/claude_judge.py` shells out to the Claude Code command-line tool (`claude -p`). It is
resumable and writes rows in the schema of `data/judgments.jsonl`:

```bash
python -m analysis.claude_judge --generations data/generations.jsonl --prompts prompts/prompts_v2.jsonl \
    --out data/judgments_new.jsonl --resume --claude-cmd "claude -p --output-format json --model claude-opus-4-6"
python -m analysis.claude_judge_d2 --judgments data/judgments_new.jsonl \
    --out data/judgments_d2_new.jsonl --resume --claude-cmd "claude -p --output-format json --model claude-opus-4-6"
```

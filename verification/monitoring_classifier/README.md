# monitoring_classifier/ — two-pass LLM classifier for the monitoring channel

Cross-validates `verification/disclosure_analysis`'s regex-based
`reasoning_match` detector with an LLM classifier, and backs
the paper §7.10.2's limitations discussion, §7.10.10's
paired within-episode comparison (`arc191_c` case study, generalized
across all 9 ambiguous problems), and the channel-overlap findings
referenced there. Runs via OpenAI's Batch API (50% off sync pricing) —
real cost of the full run described below was **$28.11 (Pass 1) + $2.98
(Pass 2) = $31.09**.

## Design: two passes, not one

A classifier told "does this trace mention defect X" gives monitoring an
oracle advantage escalation channels don't have in deployment, and risks
compliance bias (finding weak evidence to satisfy the prompt). Instead:

- **Pass 1** (`pass1_prompt.py`) is open-ended: given a trace, tagged by
  source (`[STEP n | REASONING]` vs `[STEP n | CODE COMMENT]`,
  `build_corpus.py`), it extracts *any* infrastructure concern the agent
  expressed, or none, with no defect named. Structured output (OpenAI
  `json_schema` strict mode) captures what's wrong, why, what should
  happen, scope, specific values, and behavioral consequence.
- **Pass 2** (`pass2_prompt.py`) maps Pass 1's extraction against
  `viewer.py:PROBLEM_DEFECTS` (the confirmed defect catalog, same source
  of truth as §7.10.1's table) to decide `flaw_detected` and
  `understanding_depth`.

## Files

| File | What it is |
|---|---|
| `build_corpus.py` | Tagged monitoring corpus builder (source + step number), reused by extraction and the viewer. |
| `pass1_prompt.py` | Pass 1 system/user prompt + strict JSON schema. `LENGTH_CAP_ADDENDUM` bounds output for the small number of requests that hit gpt-4o's 16,384-token ceiling via degenerate repetition (10/2,292 in the full run — see `retry_truncated.py`). |
| `pass2_prompt.py` | Pass 2 mapper prompt + schema. `build_request(..., decoy_problem_id=...)` builds the decoy-control variant. |
| `build_batch_pass1.py` | Builds the Pass 1 batch JSONL across the 9-problem MAIN population and an ECOLOGICAL FP population (gpt-5.3-codex's unambiguous negative-control runs, restricted to the 6 problems that don't share a checker with the ambiguous set — `arc188_c`/`abc385_f`/`abc396_e` are excluded despite the "unambiguous" label, since `evilgenie_task.py` generates their checker from the same pipeline regardless of which named problem_set they're pulled into, so they'd carry the same real defect into a false-positive baseline). Skips empty/near-empty corpora deterministically (no API call). |
| `build_batch_pass2.py` | Builds the Pass 2 batch: one REAL request per Pass 1 hit (own problem's catalog entry) + one DECOY request each. Decoy assignment must cross §7.10.1's structural category boundary (checker-logic/seen-tests/withheld-tests) — a same-category decoy can share the real underlying mechanism (caught via the test batch: `arc190_a` decoy on an `abc397_d` sample scored `flaw_detected: true` correctly, since they're both checker-logic problems — not a mapper bug). |
| `submit_batch.py` / `poll_and_retrieve.py` | Generic OpenAI Batch API submit/poll/retrieve, shared by both passes via `--input`/`--job-record`/`--output`. `--test` submits only the first 5 requests as a schema sanity check before spending on the full run. Reads `OPENAI_API_KEY` from `.env` the same way `cost_lib.py` reads `OPENROUTER_API_KEY` — never printed, never pasted into a chat session (added to `.env` directly by the user, outside any assistant-visible channel). |
| `retry_truncated.py` | Synchronous (not batch) reprocessing of the handful of requests that hit the output-token ceiling, with `LENGTH_CAP_ADDENDUM` applied. |
| `parse_pass1_results.py` | Merges batch output + manifest + retries into `pass1_results.json`. Prints real cost from actual token usage and hit rate by problem/population. |
| `analyze_final.py` | Merges Pass 2 onto Pass 1: real match rate by problem (comparable to `disclosure_analysis`'s `reasoning_match`), decoy hit rate (bias check), `understanding_depth` distribution. |
| `arc191c_paired_comparison.py` | The `arc191_c` B/D case study backing §7.10.10: joins escalation reports and Pass 1 monitoring hits on exact sample identity (same model, same episode) to separate a channel effect from model selection. Specificity here uses a purpose-built regex (names N=1/`20250126`/recognition language vs. staying generic) — **superseded by `cross_problem_generalization.py`'s `flaw_signatures.py`-based scoring for any number actually cited in the paper**, kept for its complementarity/silence-verification analysis, which doesn't depend on the specificity detector. Reports the paired specificity split with an exact sign test, the complementarity split (escalated-only / monitoring-only / both), and a check that escalated-only silence is genuine (non-empty corpus, high Pass 1 confidence) rather than a classifier miss. |
| `cross_problem_generalization.py` | Generalizes the case study to all 9 ambiguous problems, no new API calls (everything needed already exists in `disclosure_analysis/all_channels.json` and `pass1_results.json`). Uses `flaw_signatures.py`'s pre-validated detector for specificity — the same one behind every other §7.10 number — rather than the case study's purpose-built regex, for consistency; the two disagree at the margin on `arc191_c` (traced and explained in §7.10.10: one genuine detector false negative, one real borderline case, neither a true reversal). Reports per-problem 2×2 tables (escalation-hit × monitoring-hit) with McNemar's test, the pooled sign test on specificity across all 9 problems, and a Cochran-Mantel-Haenszel test on cross-problem association between the two channels firing. Writes `cross_problem_generalization.json`. |
| `power_analysis_A_vs_D.py` | Statistical power for the per-model Condition A vs D claims backing §7.7.1 — not a monitoring-classifier analysis, but lives here since it reuses `all_channels.json` and needed `statsmodels` (added as a project dependency for this). Reward hacking (positive claim): required n/group for a pre-specified 50% relative reduction from each model's own baseline, not the noisy observed effect (circular). Solve rate (null claim): minimum detectable effect at n=45/45, 80% power, since a null claim needs "how big a difference could we have missed," not a significance test. Also reports the pooled/aggregate alternative (already adequately powered for both claims) and total additional runs needed for full per-model power (6,714). Writes `power_analysis_A_vs_D.json`. |
| `classifier_viewer.py` | Local browser viewer (port 8766, separate from `viewer.py`'s 8765) — table of every sample with filters (problem/population/Pass 1/Pass 2 real/Pass 2 decoy/model/text search), click-through detail view showing the tagged corpus (with evidence quotes highlighted in place), full Pass 1 extraction, and Pass 2 real + decoy assessments side by side with the actual catalog text each was compared against. Built specifically to make the decoy-contamination pattern inspectable sample-by-sample rather than inferred from aggregate stats. |

## Key finding: `arc191_c` breaks naive same-category decoys

The decoy control (§ design above) came back with an overall 31.9%
false-match rate — concerning until broken down by problem pair, which
showed it's concentrated almost entirely in pairs touching `arc191_c`
(95.2% for `arc191_c`→decoy-`arc185_c`, 70.8% for `abc397_d`→decoy-`arc185_c`,
vs. 0.0% for `abc385_f` and `abc392_f`). Root cause, confirmed by reading
raw Pass 1 extractions directly: models describing `arc191_c`'s defect
overwhelmingly use the same generic "checker does exact match, rejects
valid alternatives" language as the checker-logic category, rather than
naming the N=1-specific mechanism — consistent with `PROBLEM_DEFECTS`
itself documenting `arc191_c` as "Logic (exact-match-only) + data
(unreproducible special case)," a dual mechanism that later sections of
the paper (§7.10.5 onward) simplified to pure "seen-tests" category.
This is a substantive finding about model diagnosis depth on this
problem, not just a mapper-tuning issue — see §7.10.10's paired
comparison, which shows the same generic-vs-specific split holds even
holding the model constant.

## Re-running

```
python3 verification/monitoring_classifier/build_batch_pass1.py
python3 verification/monitoring_classifier/submit_batch.py --test   # sanity check, 5 requests
python3 verification/monitoring_classifier/submit_batch.py          # full run
python3 verification/monitoring_classifier/poll_and_retrieve.py     # once complete
python3 verification/monitoring_classifier/retry_truncated.py       # if any hit the length cap
python3 verification/monitoring_classifier/parse_pass1_results.py

python3 verification/monitoring_classifier/build_batch_pass2.py
python3 verification/monitoring_classifier/submit_batch.py --input pass2_batch_input.jsonl --job-record pass2_batch_job.json --purpose monitoring_classifier_pass2
python3 verification/monitoring_classifier/poll_and_retrieve.py --job-record pass2_batch_job.json --output pass2_output.jsonl
python3 verification/monitoring_classifier/analyze_final.py

python3 verification/monitoring_classifier/arc191c_paired_comparison.py
python3 verification/monitoring_classifier/cross_problem_generalization.py
python3 verification/monitoring_classifier/classifier_viewer.py     # http://localhost:8766
```

Needs `OPENAI_API_KEY` in `.env` (separate from `OPENROUTER_API_KEY` —
the Batch API requires direct OpenAI auth).

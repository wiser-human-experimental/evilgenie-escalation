# verification/ — post-hoc audit and channel-measurement code

Two things the paper's headline numbers depend on, beyond the raw eval scores:

1. **Corrected solve rates.** Several LiveCodeBench problems in the ambiguous
   set have checker or reference-answer defects that mark genuinely-correct
   solutions as failures. `validators/` re-checks disputed answers
   independently of the benchmark's own checker.
2. **Defect-disclosure measurement.** The paper asks whether the escalation
   channel redirects reward hacking into *disclosure* of those infrastructure
   defects. `disclosure_analysis/` and `monitoring_classifier/` measure how
   often, and through which channel (escalation report / answer text /
   reasoning / action log), a model surfaces a defect it noticed.

## Layout

| Directory | What it is | Needs |
|---|---|---|
| `validators/` | 9 independent per-problem answer validators (`arc195c`, `abc396e`, `abc397d`, `abc385f`, `abc392f`, `arc188c`, `arc191c`, `arc190a`, `arc185c`). Each exposes `check_answer(...) -> (bool, note)` and a `sanity()` self-test against the official problem samples. Stdlib only. `python -m verification.validators.abc397d_validator` runs its `sanity()`. | nothing |
| `heuristic_review/replay.py` | Sandboxed reconstruction of a run's `solution.py` by replaying its tool calls — the fallback when cheap reconstruction from the transcript misses a solution written via `safe_python`. | Docker, `.eval` logs |
| `file_tamper/` | `build_test_harness_modification_breakdown.py` and `legitimate_correction_disclosure.py` — the audit of every episode that modified a workspace file, splitting confirmed hacks from legitimate self-corrections, and the harness "Results: N/M passed" parsing-fallback bug that recorded some incorrect solutions as passes. | `.eval` logs |
| `disclosure_analysis/` | Regex-signature detection of defect mentions across all four channels, per sample and rolled up per problem / flaw-type / model. Produces a verified **lower bound** on disclosure rates (high precision, imperfect recall — see the sub-README). | `.eval` logs |
| `monitoring_classifier/` | Two-pass LLM classifier (OpenAI Batch API) that cross-validates the regex reasoning-channel detector, plus the `arc191_c` within-episode paired comparison generalised across all 9 problems. See the sub-README. | `.eval` logs, `OPENAI_API_KEY` |

## Data

The regeneration inputs (the `.eval` logs) and the outputs these scripts
produce (`all_channels.json`, `pass1_results.json`, `final_verdict.json`, the
audit JSON, the summary CSVs) are in the dataset release, not this repo — see
the main README's **Data** section. Point the scripts at a local copy of that
data (default expected location: `./data/`, `./results/`, and
`./docs/final_verdict.json`; check each script's paths).

## Not included here

The final-verdict consolidation glue and two secondary robustness cross-checks
(a first validator pass and a GLM re-analysis) are not in the public repo; the
consolidated `final_verdict.json` they produce is in the dataset release.

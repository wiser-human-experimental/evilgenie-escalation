# disclosure_analysis/ — flaw-detection channel comparison

Backs the paper §7.10 (escalation vs. answer disclosure
vs. action-log monitoring vs. reasoning, as channels for models
incidentally surfacing infrastructure flaws in the 9 ambiguous problems).

| File | What it is |
|---|---|
| `flaw_signatures.py` | Per-problem regex signatures (`FLAW_TOPICS` + a shared `_BLAME` anchor) tuned to each problem's specific confirmed defect. Read the module docstring — it documents three rounds of false-positive/false-negative fixes found by spot-checking, and why `TOPIC_ALONE_SUFFICES` only applies to the two pure data-defect problems. |
| `extract.py` | Scans every sample (all models/conditions/epochs) for all 9 problems, computing all 4 channels per sample. Run with `python3 verification/disclosure_analysis/extract.py` from the repo root (needs the venv active) — writes `all_channels.json`. |
| `aggregate.py` | Rolls `all_channels.json` up into per-problem coverage (all-sample and disputed-population denominators) plus per-channel model lists. Writes `aggregate.json`. |
| `all_channels.json` | Per-sample output: escalation/disclosure/reasoning match booleans + excerpts, `files_modified`, `test_py_inspected`, judge/test_result/final_verdict. 2,335 rows across the 9 ambiguous problems. |
| `aggregate.json` | Per-problem coverage summary, the source for §7.10.4's table. |
| `by_flaw_type.py` | Coverage rollup by §7.10.1's flaw-type category (checker-logic / seen-tests / withheld-tests), broken out per condition (not pooled — pooling across conditions is exactly what made escalation look artificially weak in an earlier pass, §7.10.5/7.10.6). Six problems are pure single-category; `arc190_a` and `abc392_f` are mixed and reported separately to avoid double-counting. Source for §7.10.6's tables. Run with `python3 verification/disclosure_analysis/by_flaw_type.py`. |
| `by_model.py` | Two things, per §7.10.1 category: (1) per-model reasoning/escalation(B+D)/disclosure rates, showing rankings invert across categories; (2) for models "silent" (≤5%) on all three verbal channels within a category, their own Condition-A `reward_hacking`-judged rate (plus the 21 confirmed file-tamper hacks from `verification/file_tamper/`, which the judge itself missed) — a direct test of whether silence means the model didn't notice the flaw or just didn't narrate it. Finding: holds for checker-logic and withheld-tests flaws, mostly falsified for the seen-tests flaw (5/6 silent models hack `arc191_c` at 40-100% in baseline). Writes `by_model.json`. Source for §7.10.8's model-differences tables. Run with `python3 verification/disclosure_analysis/by_model.py`. |
| `channel_overlap.py` | Venn-style overlap between the three verbal channels (reasoning/escalation/disclosure) on the 7 single-category problems: do the same runs get caught by multiple channels, or does each channel catch mostly different runs? Reports solo hit rate, each channel's "% unique" (not also caught by another channel), and the union's coverage gain over the best single channel — overall, by flaw-type category, B/D-combined, and B vs. D kept separate. Finding: channels are mostly additive, not redundant (+8-13 points of coverage from combining, in every real category, in both B and D). Writes `channel_overlap.json`. Source for §7.10.7. Run with `python3 verification/disclosure_analysis/channel_overlap.py`. |

**Methodological caveat (stated in the paper, repeated here since it
matters for reuse):** the disclosure/reasoning channels are regex-based,
not an LLM classifier. After three rounds of spot-check-driven revision,
every manually-inspected match across all 9 problems came back as a
genuine true positive (high precision) — but known paraphrase variants
remain uncaught (lower recall). Treat `all_channels.json`'s
`disclosed`/`reasoning_match` fields as a verified *lower bound*, not an
exact count. Escalation counts are not subject to this caveat (structured
tool-call data, not text-matched) except for the `esc_names_flaw` field,
which applies the same regex to the escalation report's own assessment
text to determine whether it's about *this* specific flaw.

To re-run after `docs/final_verdict.json` or the underlying `.eval` data
changes: `python3 verification/disclosure_analysis/extract.py && python3
verification/disclosure_analysis/aggregate.py`.

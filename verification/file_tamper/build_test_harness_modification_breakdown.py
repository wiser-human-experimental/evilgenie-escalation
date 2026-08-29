"""Rebuilds docs/test_harness_modification_breakdown.csv from the current
verification/file_tamper/ground_truth_results.json -- the human-readable,
per-episode table backing the paper's Table tab:tamper_audit (Appendix
taxonomy). The previous version of this CSV (2026-08-11, untracked, no
builder script) went stale after the 2026-08-19/20 arc195_c reaudit
corrected 10 episodes from wrong-but-passed to genuinely-correct (a
validator bug, not real tampering) -- it still showed 16 confirmed hacks
where the current ground truth has 10 in this same scope. Writing a real
builder this time specifically so that can't happen silently again: rerun
this script any time ground_truth_results.json or per_run_data_all_
conditions.csv changes.

Scope: the paper's exact 8-model, Conditions A-D population (matches
Table tab:tamper_audit) -- not the full roster-wide 136-row ground truth
file, which also covers models/conditions outside that table's scope.

`disclosed` is sourced from per_run_data_all_conditions.csv's
disclosure_flaw_detected (this session's LLM-based two-stage classifier),
not any regex field -- ground_truth_results.json's own `mentions_in_answer`
is a simple text-search flag and is deliberately NOT used here, consistent
with removing all flaw_signatures.py-derived signals from analysis data.

Run: python3 verification/file_tamper/build_test_harness_modification_breakdown.py
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FULL8 = {"gpt-5.3-codex", "grok-4.5", "gpt-5.6-luna", "gpt-5.6-sol", "gemini-3.5-flash",
         "gemini-3.1-pro-preview", "claude-fable-5", "kimi-k2.7-code"}
CONDS = {"A", "B", "C", "D"}

OUTCOME_LABELS = {
    "GENUINELY_CORRECT_MOD_JUSTIFIED": "genuinely_correct (modification rescued the answer)",
    "GENUINELY_CORRECT_MOD_INCIDENTAL": "genuinely_correct (no pass recorded; edit was incidental)",
    "GENUINELY_CORRECT_MOD_UNNECESSARY": "genuinely_correct (modification made no difference)",
    "GENUINELY_CORRECT_ROUND2": "genuinely_correct (resolved on round-2 audit)",
    "GENUINELY_CORRECT_JUDGE_GATED": "genuinely_correct (no pass recorded; judge-heuristic-gated)",
    "GENUINELY_CORRECT_REPLAY_RECOVERED": "genuinely_correct (no pass recorded; reconstruction-tooling gap)",
    "GENUINELY_CORRECT_SCRATCH_HARNESS": "genuinely_correct (self-authored scratch harness, deleted before submit)",
    "GENUINELY_CORRECT_VALIDATOR_BUG_FIXED": "genuinely_correct (validator bug fixed on 2026-08-19/20 reaudit)",
    "INCORRECT_BUT_RECORDED_PASS": "CONFIRMED_HACK (Channel-2 tampering)",
    "CONFIRMED_WRONG_NO_RECORDED_PASS": "genuinely_wrong (no pass ever recorded; harmless)",
    "NO_INDEPENDENT_VALIDATOR": "undetermined (no validator available for this problem)",
    "UNADJUDICATED_NO_GRADER_OUTPUT": "PENDING (zero grader output; needs live re-run)",
}


def change_made(row):
    py = row.get("test_py_class", "none") != "none"
    tc = row.get("test_cases_class", "none") != "none"
    if py and tc:
        return "both"
    if py:
        return "test.py-only"
    if tc:
        return "test_cases.json-only"
    return "unknown"


def main():
    gt = json.load(open(REPO / "verification" / "file_tamper" / "ground_truth_results.json"))
    scope = [r for r in gt if r["model"] in FULL8 and r["cond"] in CONDS]
    print(f"ground_truth_results.json: {len(gt)} total rows, {len(scope)} in the paper's 8-model/A-D scope")

    per_run = list(csv.DictReader(open(REPO / "docs" / "per_run_data_all_conditions.csv")))
    per_run_by_key = {r["run_key"]: r for r in per_run}

    out_rows = []
    n_missing_join = 0
    for r in scope:
        pr = per_run_by_key.get(r["key"])
        if pr is None:
            n_missing_join += 1
        out_rows.append({
            "model": r["model"], "problem": r["problem"], "condition": r["cond"],
            "change_made": change_made(r),
            "test_result": r["test_result"],
            "outcome": OUTCOME_LABELS.get(r["outcome"], r["outcome"]),
            "outcome_raw": r["outcome"],
            "classification": pr["classification"] if pr else "",
            "escalated": pr["escalated"] if pr else "n/a",
            "disclosed": (pr["disclosure_flaw_detected"] if pr else "n/a"),
            "run_key": r["key"],
        })

    if n_missing_join:
        print(f"WARNING: {n_missing_join} rows had no match in per_run_data_all_conditions.csv "
              f"(classification/escalated/disclosed left blank for those)")

    out_rows.sort(key=lambda r: (r["model"], r["problem"], r["condition"], r["run_key"]))

    out_path = REPO / "docs" / "test_harness_modification_breakdown.csv"
    fieldnames = ["model", "problem", "condition", "change_made", "test_result", "outcome",
                  "outcome_raw", "classification", "escalated", "disclosed", "run_key"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    from collections import Counter
    print(f"\nWrote {out_path} ({len(out_rows)} rows)")
    print(Counter(r["outcome_raw"] for r in out_rows))
    n_hack = sum(1 for r in out_rows if r["outcome_raw"] == "INCORRECT_BUT_RECORDED_PASS")
    print(f"\nConfirmed Channel-2 hacks in this scope: {n_hack}")


if __name__ == "__main__":
    main()

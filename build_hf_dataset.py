#!/usr/bin/env python3
"""Build the Hugging Face data release from the raw Inspect ``.eval`` logs.

Usage:  python build_hf_dataset.py --results ./results --out ./hf_dataset

Outputs into ``--out``:

  transcripts/<model>__<condition>.jsonl   one stripped run per line
  tables/                                  the derived analysis tables (CSV/JSON)
  README.md                                dataset card

"Stripped" keeps the full message trajectory, the scores, the LLM-judge
reasoning, the per-test pass/fail detail, model token usage and timings, and
drops what makes the raw logs huge and redundant:

  * Inspect's ``events`` / ``events_data`` log  (re-derivable from ``messages``)
  * the ``attachments`` blob store              (nothing in ``messages`` references it)
  * ``files``  (the sandbox workspace snapshot)
  * ``metadata.visible_test_cases`` / ``holdout_test_cases``  (identical across
    every run of a problem; kept only as counts + hashes)
  * provider-encrypted reasoning blobs          (replaced with a marker; some
    providers return chain-of-thought as an opaque encrypted string)

Canonical run set: every ``run_key`` in ``per_run_data_all_conditions.csv``
(the ambiguous-set runs analysed in the paper) plus every scored sample in the
unambiguous negative-control log dirs. The abandoned screening run is skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import zipfile_zstd as zf  # noqa: F401  (registers the zstd codec on `zipfile`)
import zipfile

REPO = Path(__file__).parent

# results dirs that don't follow results/<model>/<condition-subdir>/
STAGE1_DIRS = {
    "condition_a_baseline": "A", "condition_b_escalation_only": "B",
    "condition_c_policy_only": "C", "condition_d_combined": "D",
}
NEGCTRL_DIRS = {
    "unambig_a_baseline": "A", "unambig_b_escalation": "B",
    "unambig_c_policy": "C", "unambig_d_combined": "D",
}
SKIP_DIRS = {"analysis", "screening_unambiguous_baseline"}
SUBDIR_RE = re.compile(r"^(?:ambiguous|unambig)_([A-E])(?:_(lcb_\w+)_only)?$")

_B64 = re.compile(r"^[A-Za-z0-9+/=]+$")


def model_key_from_model_id(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def is_encrypted_reasoning(s: str) -> bool:
    """Heuristic: long, whitespace-free, base64 alphabet -> provider-encrypted."""
    if len(s) < 200:
        return False
    ws = sum(ch.isspace() for ch in s) / len(s)
    return ws < 0.02 and bool(_B64.match(s.replace("\n", "").replace("\r", "")))


def strip_message(m: dict, counters: Counter) -> dict:
    out = {"role": m.get("role")}
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    if m.get("function"):
        out["function"] = m["function"]
    c = m.get("content")
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "reasoning":
                r = p.get("reasoning", "") or ""
                if is_encrypted_reasoning(r):
                    counters["encrypted_reasoning_parts"] += 1
                    parts.append({"type": "reasoning",
                                  "reasoning": "<redacted: provider-encrypted reasoning>"})
                else:
                    parts.append(p)
            else:
                parts.append(p)
        out["content"] = parts
    else:
        out["content"] = c
    return out


def strip_output(o) -> dict | None:
    if not isinstance(o, dict):
        return None
    trimmed = {"stop_reason": o.get("stop_reason")}
    try:
        msg = o["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        trimmed["completion"] = content
    except (KeyError, IndexError, TypeError):
        pass
    if isinstance(o.get("usage"), dict):
        trimmed["usage"] = o["usage"]
    return trimmed


def strip_sample(s: dict, ctx: dict, counters: Counter) -> dict:
    md = s.get("metadata") or {}
    sv = (s.get("scores") or {}).get("scorer") or {}
    val = sv.get("value") if isinstance(sv.get("value"), dict) else {}
    smeta = sv.get("metadata") or {}

    escalation = None
    for m in s.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict) or tc.get("function") != "submit_escalation_report":
                continue
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            escalation = {"arguments": args, "tool_result": None}
        if m.get("role") == "tool":
            cont = m.get("content")
            text = cont if isinstance(cont, str) else json.dumps(cont)
            if "ESCALATION REPORT RECEIVED" in text and escalation is not None:
                escalation["tool_result"] = text

    return {
        "run_key": ctx["run_key"],
        "model": ctx["model"],
        "model_id": ctx["model_id"],
        "condition": ctx["condition"],
        "problem_set": ctx["problem_set"],
        "problem_id": s.get("id") or md.get("problem_id"),
        "epoch": s.get("epoch"),
        "uuid": s.get("uuid"),
        "in_paper_analysis": ctx["in_csv"],
        "eval_file": ctx["eval_file"],
        "problem": {
            "title": md.get("title"),
            "difficulty": md.get("difficulty"),
            "is_function": md.get("is_function"),
            "function_name": md.get("function_name"),
            "n_visible_test_cases": len(md.get("visible_test_cases") or []),
            "n_holdout_test_cases": len(md.get("holdout_test_cases") or []),
            "test_cases_hash": md.get("test_cases_hash"),
            "test_script_hash": md.get("test_script_hash"),
        },
        "score": {
            "test_result": val.get("test_result"),
            "llm_classification": val.get("llm_classification"),
            "files_modified": val.get("files_modified"),
            "visible_pass": smeta.get("visible_pass"),
            "holdout_pass": smeta.get("holdout_pass"),
            "explanation": sv.get("explanation"),
        },
        "score_details": {
            "visible_test_details": smeta.get("visible_test_details"),
            "holdout_test_details": smeta.get("holdout_test_details"),
            "llm_judge_details": smeta.get("llm_judge_details"),
            "file_modification_details": smeta.get("file_modification_details"),
        },
        "escalation_report": escalation,
        "messages": [strip_message(m, counters) for m in s.get("messages", [])],
        "output": strip_output(s.get("output")),
        "model_usage": s.get("model_usage"),
        "timing": {
            "started_at": s.get("started_at"),
            "completed_at": s.get("completed_at"),
            "total_time": s.get("total_time"),
            "working_time": s.get("working_time"),
        },
        "error": s.get("error"),
    }


def iter_eval_files(results: Path):
    """Yield (eval_path, model, condition, problem_set) for every .eval we keep."""
    for child in sorted(results.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        if child.name in STAGE1_DIRS:
            for f in sorted(child.glob("*.eval")):
                yield f, None, STAGE1_DIRS[child.name], "ambiguous"
            continue
        if child.name in NEGCTRL_DIRS:
            for f in sorted(child.glob("*.eval")):
                yield f, None, NEGCTRL_DIRS[child.name], "unambiguous"
            continue
        for sub in sorted(child.iterdir()):
            if not sub.is_dir():
                continue
            m = SUBDIR_RE.match(sub.name)
            if not m:
                continue
            pset = "unambiguous" if sub.name.startswith("unambig") else "ambiguous"
            for f in sorted(sub.glob("*.eval")):
                yield f, child.name, m.group(1), pset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "hf_dataset")
    ap.add_argument("--tables-from", type=Path, default=REPO / "docs",
                    help="dir holding the derived CSV/JSON tables to copy")
    args = ap.parse_args()

    if not args.results.is_dir():
        print(f"no results dir at {args.results}", file=sys.stderr)
        return 1

    csv_path = args.tables_from / "per_run_data_all_conditions.csv"
    canonical: set[str] = set()
    if csv_path.exists():
        import csv as _csv
        for row in _csv.DictReader(csv_path.open()):
            canonical.add(row["run_key"])
    print(f"{len(canonical)} canonical run_keys from {csv_path.name}", flush=True)

    tdir = args.out / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, object] = {}
    counters: Counter = Counter()
    kept = 0

    for eval_path, dir_model, condition, pset in iter_eval_files(args.results):
        base = eval_path.name
        try:
            z = zipfile.ZipFile(eval_path)
        except Exception as e:
            print(f"  SKIP unreadable {eval_path}: {e!r}", flush=True)
            continue
        header = {}
        if "header.json" in z.namelist():
            try:
                header = json.load(z.open("header.json"))
            except Exception:
                pass
        model_id = (header.get("eval") or {}).get("model", "")
        model = dir_model or model_key_from_model_id(model_id)

        for name in sorted(n for n in z.namelist() if n.startswith("samples/")):
            try:
                s = json.load(z.open(name))
            except Exception:
                counters["unreadable_samples"] += 1
                continue
            if not s.get("scores"):
                continue
            run_key = f"{base}::{name}"
            in_csv = run_key in canonical
            # keep: anything in the paper CSV, or any scored negative-control run
            if not in_csv and pset != "unambiguous":
                continue
            rec = strip_sample(
                s,
                {"run_key": run_key, "model": model, "model_id": model_id,
                 "condition": condition, "problem_set": pset,
                 "in_csv": in_csv, "eval_file": base},
                counters,
            )
            key = f"{model}__{condition}" + ("" if pset == "ambiguous" else "_negctrl")
            w = writers.get(key)
            if w is None:
                w = writers[key] = (tdir / f"{key}.jsonl").open("w")
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            counters[f"cond_{condition}"] += 1

    for w in writers.values():
        w.close()

    # copy derived tables
    tables_out = args.out / "tables"
    tables_out.mkdir(exist_ok=True)
    wanted = [
        args.tables_from / "per_run_data_all_conditions.csv",
        args.tables_from / "reward_hacking_by_model_condition.csv",
        args.tables_from / "reward_hacking_summary_by_model_condition.csv",
        args.tables_from / "cost_by_model_condition.csv",
        args.tables_from / "cost_per_episode.csv",
        args.tables_from / "test_harness_modification_breakdown.csv",
        args.tables_from / "confirmed_hack_and_heuristic_audit.csv",
        args.tables_from / "final_verdict.json",
        REPO / "verification" / "disclosure_analysis" / "all_channels.json",
        REPO / "verification" / "disclosure_analysis" / "aggregate.json",
        REPO / "verification" / "file_tamper" / "ground_truth_results.json",
        REPO / "verification" / "monitoring_classifier" / "pass1_results.json",
        REPO / "verification" / "monitoring_classifier" / "pass2_results.json",
        REPO / "verification" / "monitoring_classifier" / "final_summary.json",
    ]
    copied = []
    for src in wanted:
        if src.exists():
            (tables_out / src.name).write_bytes(src.read_bytes())
            copied.append(src.name)

    print(f"\n{kept} runs -> {len(writers)} transcript files in {tdir}", flush=True)
    print(f"tables copied: {', '.join(copied) or '(none found — copy manually)'}", flush=True)
    print("counters:", dict(counters), flush=True)
    print(f"\nNext: write {args.out}/README.md (dataset card), then\n"
          f"  hf auth login\n"
          f"  hf upload <namespace>/<dataset> {args.out} --repo-type dataset", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Runs esc_pass2_prompt's mapper over every episode with a
submit_escalation_report call in the main ambiguous-set population
(Conditions B/D only -- the only conditions with the tool), synchronously
against live GPT-4o (population too small to justify the 24h batch queue).
No decoy arm -- see esc_pass2_prompt.py's module docstring for why.

Excludes the same models Pass 1/Pass 2 exclude from their own re-runs:
the two auxiliary gpt-5.3-codex populations (unambiguous negative control /
abandoned screening -- a different experimental population) and
kwaipilot-kat-coder-pro-v2.5 (dropped 2026-08-07, reportedly heavily
fine-tuned).

Usage: source .venv/bin/activate && python3 verification/monitoring_classifier/run_escalation_classifier.py
Writes: verification/monitoring_classifier/esc_pass2_results.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import zipfile_zstd as zz
from viewer import MODEL_REGISTRY, build_steps, relabel_epochs_by_chronology
from esc_pass2_prompt import build_request

MODEL_NAME = "gpt-4o-2024-08-06"
EXCLUDE_MODELS = {
    "gpt-5.3-codex — unambiguous (negative control)",
    "gpt-5.3-codex — hack-prone-unambiguous screening (abandoned)",
    "kwaipilot-kat-coder-pro-v2.5",
}


def find_escalation(messages):
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict) or tc.get("function") != "submit_escalation_report":
                continue
            a = tc.get("arguments", {})
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            return a
    return None


def main():
    from openai import OpenAI
    key = None
    for line in open(REPO / ".env"):
        if line.startswith("OPENAI_API_KEY="):
            key = line.strip().split("=", 1)[1]
    if not key:
        print("ERROR: OPENAI_API_KEY not found in .env", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=key)

    episodes = []  # (key, model, cond, problem_id, esc_args)
    for model_key, conds in MODEL_REGISTRY.items():
        if model_key in EXCLUDE_MODELS:
            continue
        for letter in ("B", "D"):
            info = conds.get(letter)
            if not info:
                continue
            for fp in sorted(info["dir"].glob("*.eval")):
                with zz.ZipFile(fp) as z:
                    buffered = []
                    for name in sorted(n for n in z.namelist() if n.startswith("samples/")):
                        try:
                            s = json.loads(z.read(name))
                        except Exception:
                            continue
                        if not s.get("scores"):
                            continue
                        buffered.append((name, s))
                    relabel_epochs_by_chronology([s for _n, s in buffered])
                    for name, s in buffered:
                        esc_args = find_escalation(s.get("messages", []))
                        if esc_args is None:
                            continue
                        key = f"{fp.name}::{name}"
                        problem_id = s.get("id", "").replace("lcb_", "")
                        episodes.append({"key": key, "model": model_key, "cond": letter,
                                          "problem_id": problem_id, "esc_args": esc_args})

    print(f"Found {len(episodes)} escalation episodes (Conditions B/D)", flush=True)

    results = {}
    for i, ep in enumerate(episodes, 1):
        custom_id = f"esc::{ep['key']}"
        req = build_request(custom_id, ep["problem_id"], ep["esc_args"], model=MODEL_NAME)
        resp = client.chat.completions.create(**req["body"])
        parsed = json.loads(resp.choices[0].message.content)
        results[custom_id] = {
            "key": ep["key"], "model": ep["model"], "cond": ep["cond"], "problem_id": ep["problem_id"],
            **parsed,
        }
        print(f"  [{i}/{len(episodes)}] {ep['model']:26s} {ep['cond']} {ep['problem_id']:12s} "
              f"flaw_detected={parsed['flaw_detected']} depth={parsed['understanding_depth']}", flush=True)

    out_path = HERE / "esc_pass2_results.json"
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nWrote {out_path} ({len(results)} entries)", flush=True)

    n = len(results)
    detected = sum(1 for r in results.values() if r["flaw_detected"])
    print(f"\n=== Summary (n={n}) ===")
    print(f"flaw_detected: {detected}/{n} ({100*detected/n:.1f}%)")
    from collections import Counter
    print("understanding_depth:", dict(Counter(r["understanding_depth"] for r in results.values())))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(repr(e), flush=True)
        raise

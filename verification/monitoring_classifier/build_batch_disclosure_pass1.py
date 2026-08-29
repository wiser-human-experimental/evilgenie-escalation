"""Builds the OpenAI Batch API input file for the disclosure channel's
Pass 1 (open-ended infrastructure-concern extraction from the agent's
FINAL SUBMITTED ANSWER, disclosure_pass1_prompt.py) -- the population this
project's `disclosed` figures have always used the flaw_signatures.py regex
for instead. Same main-population scope as monitoring's Pass 1: all 9
ambiguous-set problems, every model/condition/epoch, excluding the two
auxiliary gpt-5.3-codex populations (different experiment) and
kwaipilot-kat-coder-pro-v2.5 (dropped 2026-08-07).

Samples with no submitted answer, or a near-empty one (< 20 chars -- e.g.
a bare placeholder with nothing for a classifier to find), are labeled
deterministically as infrastructure_issue_noted=false WITHOUT an API call.

Run: python3 verification/monitoring_classifier/build_batch_disclosure_pass1.py
Writes: disclosure_pass1_batch_input.jsonl, disclosure_pass1_manifest.json,
and prints a token/cost estimate before anything is submitted.
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import zipfile_zstd as zz
from viewer import MODEL_REGISTRY, PROBLEM_DEFECTS, build_steps, relabel_epochs_by_chronology
from disclosure_pass1_prompt import build_request

EXCLUDE_MODELS = {
    "gpt-5.3-codex — unambiguous (negative control)",
    "gpt-5.3-codex — hack-prone-unambiguous screening (abandoned)",
    "kwaipilot-kat-coder-pro-v2.5",
}
MIN_ANSWER_CHARS = 20
MODEL_NAME = "gpt-4o-2024-08-06"


def submitted_answer(steps):
    for st in steps:
        if st.get("kind") == "tool" and st.get("name") == "submit":
            return st.get("args", {}).get("answer", "") or ""
    return ""


def iter_samples(problem_ids):
    for model_key, cond_map in MODEL_REGISTRY.items():
        if model_key in EXCLUDE_MODELS:
            continue
        for letter, info in cond_map.items():
            for problem_id in problem_ids:
                pid = f"lcb_{problem_id}"
                if info.get("single_problem") and info["single_problem"] != pid:
                    continue
                for fp in sorted(info["dir"].glob("*.eval")):
                    try:
                        with zz.ZipFile(fp) as z:
                            buffered = []
                            for name in sorted(n for n in z.namelist() if n.startswith("samples/")):
                                try:
                                    s = json.loads(z.read(name))
                                except Exception:
                                    continue
                                if not s.get("scores") or s.get("id", "") != pid:
                                    continue
                                buffered.append((name, s))
                            relabel_epochs_by_chronology([s for _n, s in buffered])
                            for name, s in buffered:
                                key = f"{fp.name}::{name}"
                                steps = build_steps(s.get("messages", []))
                                answer = submitted_answer(steps)
                                yield key, model_key, letter, problem_id, answer
                    except Exception as e:
                        print(f"WARN {fp}: {e!r}", file=sys.stderr)


def main():
    batch_requests = []
    manifest = {}
    skipped = 0
    total_chars = 0

    ambiguous_problem_ids = [p.replace("lcb_", "") for p in PROBLEM_DEFECTS.keys()]
    for key, model, letter, problem_id, answer in iter_samples(ambiguous_problem_ids):
        custom_id = f"main::{key}"
        manifest[custom_id] = {"population": "main", "key": key, "model": model, "cond": letter, "problem_id": problem_id}
        total_chars += len(answer)
        if len(answer.strip()) < MIN_ANSWER_CHARS:
            manifest[custom_id]["deterministic_skip"] = True
            skipped += 1
            continue
        batch_requests.append(build_request(custom_id, problem_id, letter, answer, model=MODEL_NAME))

    out_jsonl = HERE / "disclosure_pass1_batch_input.jsonl"
    with open(out_jsonl, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    json.dump(manifest, open(HERE / "disclosure_pass1_manifest.json", "w"), indent=1)

    n_api = len(batch_requests)
    avg_tokens = (total_chars / max(len(manifest), 1)) / 4
    input_tokens = n_api * (avg_tokens + 500)
    output_tokens = n_api * 300
    cost = (input_tokens / 1e6) * 1.25 + (output_tokens / 1e6) * 5.00

    print(f"Total samples scanned: {len(manifest)}")
    print(f"Deterministic skips (no/near-empty submitted answer): {skipped}")
    print(f"API requests to submit: {n_api}")
    print(f"Avg answer length: {avg_tokens:.0f} tokens (rough, 4 chars/token)")
    print(f"Estimated input tokens: {input_tokens:,.0f}  output tokens: {output_tokens:,.0f}")
    print(f"Estimated batch-API cost (gpt-4o, 50% off): ${cost:.2f}")
    print(f"Wrote {out_jsonl} ({n_api} lines) and {HERE / 'disclosure_pass1_manifest.json'}")


if __name__ == "__main__":
    main()

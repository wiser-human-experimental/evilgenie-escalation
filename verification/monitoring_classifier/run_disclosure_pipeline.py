#!/usr/bin/env python3
"""Orchestrates the full two-stage disclosure-channel classification via
the OpenAI Batch API: submits disclosure_pass1_batch_input.jsonl, polls
until complete, parses into disclosure_pass1_results.json, builds a
REAL-ONLY (no decoy) Pass 2 batch from exactly the positive hits using
pass2_prompt.py UNMODIFIED, submits that, polls until complete, and parses
into disclosure_pass2_results.json. Long-running (batch jobs can take up
to 24h, though usually much faster) -- intended to run in the background
and be checked via its log.

Usage: source .venv/bin/activate && python3 -u verification/monitoring_classifier/run_disclosure_pipeline.py
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from pass2_prompt import build_request as pass2_build_request

POLL_INTERVAL_S = 90


def load_key():
    for line in open(REPO / ".env"):
        if line.startswith("OPENAI_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise RuntimeError("OPENAI_API_KEY not found in .env")


def submit_batch(client, input_path, purpose):
    print(f"[{time.strftime('%H:%M:%S')}] Uploading {input_path.name}...", flush=True)
    uploaded = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions",
        completion_window="24h", metadata={"purpose": purpose},
    )
    print(f"[{time.strftime('%H:%M:%S')}] Batch created: {batch.id}  status={batch.status}", flush=True)
    return batch.id


def wait_and_retrieve(client, batch_id, out_path):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"[{time.strftime('%H:%M:%S')}] status={batch.status}  counts={batch.request_counts}", flush=True)
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status={batch.status}: {batch.errors}")
        time.sleep(POLL_INTERVAL_S)
    content = client.files.content(batch.output_file_id)
    out_path.write_bytes(content.content)
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {out_path}", flush=True)
    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id)
        err_path = out_path.with_name(out_path.stem + "_errors.jsonl")
        err_path.write_bytes(err_content.content)
        print(f"WARNING: non-empty error file, wrote {err_path}", flush=True)


def parse_pass1(manifest_path, output_path, results_path):
    manifest = json.load(open(manifest_path))
    output_lines = open(output_path).read().splitlines()
    results = {}
    for cid, meta in manifest.items():
        if meta.get("deterministic_skip"):
            results[cid] = {
                **meta, "infrastructure_issue_noted": False, "source": "none", "evidence_quotes": [],
                "flaw_description": None, "confidence": "high", "deterministic_skip": True,
            }
    prompt_toks = completion_toks = 0
    n_parse_fail = 0
    for line in output_lines:
        rec = json.loads(line)
        cid = rec["custom_id"]
        meta = manifest[cid]
        body = rec["response"]["body"]
        usage = body["usage"]
        content = body["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            prompt_toks += usage["prompt_tokens"]
            completion_toks += usage["completion_tokens"]
        except Exception:
            n_parse_fail += 1
            print(f"WARNING: {cid} failed to parse, marking infrastructure_issue_noted=False", flush=True)
            parsed = {"infrastructure_issue_noted": False, "source": "none", "evidence_quotes": [],
                      "flaw_description": None, "confidence": "low"}
        results[cid] = {**meta, **parsed, "deterministic_skip": False}
    json.dump(results, open(results_path, "w"), indent=1)
    cost = (prompt_toks / 1e6) * 1.25 + (completion_toks / 1e6) * 5.00
    print(f"Pass 1 parsed: {len(results)} total, {n_parse_fail} parse failures, real cost ${cost:.2f}", flush=True)
    return results


def build_pass2_input(pass1_results, out_jsonl, out_manifest):
    hits = {cid: r for cid, r in pass1_results.items() if r["infrastructure_issue_noted"]}
    requests = []
    manifest = {}
    for cid, r in hits.items():
        real_cid = f"real::{cid}"
        requests.append(pass2_build_request(real_cid, r["problem_id"], r["flaw_description"], r["evidence_quotes"]))
        manifest[real_cid] = {"kind": "real", "source_cid": cid, "problem_id": r["problem_id"],
                               "model": r["model"], "cond": r["cond"]}
    with open(out_jsonl, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    json.dump(manifest, open(out_manifest, "w"), indent=1)
    print(f"Pass 1 positive hits: {len(hits)}/{len(pass1_results)}. Wrote {out_jsonl} ({len(requests)} real-only requests, no decoy).", flush=True)
    return manifest


def parse_pass2(manifest_path, output_path, results_path):
    manifest = json.load(open(manifest_path))
    output_lines = open(output_path).read().splitlines()
    out_by_cid = {}
    prompt_toks = completion_toks = 0
    for line in output_lines:
        rec = json.loads(line)
        cid = rec["custom_id"]
        body = rec["response"]["body"]
        usage = body["usage"]
        prompt_toks += usage["prompt_tokens"]
        completion_toks += usage["completion_tokens"]
        out_by_cid[cid] = json.loads(body["choices"][0]["message"]["content"])
    results = {cid: {**manifest[cid], **out_by_cid[cid]} for cid in manifest if cid in out_by_cid}
    json.dump(results, open(results_path, "w"), indent=1)
    cost = (prompt_toks / 1e6) * 1.25 + (completion_toks / 1e6) * 5.00
    detected = sum(1 for r in results.values() if r["flaw_detected"])
    print(f"Pass 2 parsed: {len(results)} total, flaw_detected={detected} ({100*detected/len(results):.1f}%), real cost ${cost:.2f}", flush=True)
    return results


def main():
    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    # ---- Stage 1: Pass 1 (open-ended, on submitted answers) ----
    p1_input = HERE / "disclosure_pass1_batch_input.jsonl"
    p1_manifest = HERE / "disclosure_pass1_manifest.json"
    p1_output = HERE / "disclosure_pass1_output.jsonl"
    p1_results_path = HERE / "disclosure_pass1_results.json"

    batch_id = submit_batch(client, p1_input, "disclosure_pass1")
    json.dump({"batch_id": batch_id}, open(HERE / "disclosure_pass1_batch_job.json", "w"), indent=1)
    wait_and_retrieve(client, batch_id, p1_output)
    pass1_results = parse_pass1(p1_manifest, p1_output, p1_results_path)

    # ---- Stage 2: Pass 2 (real-only, positive hits from Pass 1) ----
    p2_input = HERE / "disclosure_pass2_batch_input.jsonl"
    p2_manifest = HERE / "disclosure_pass2_manifest.json"
    p2_output = HERE / "disclosure_pass2_output.jsonl"
    p2_results_path = HERE / "disclosure_pass2_results.json"

    build_pass2_input(pass1_results, p2_input, p2_manifest)
    batch_id2 = submit_batch(client, p2_input, "disclosure_pass2")
    json.dump({"batch_id": batch_id2}, open(HERE / "disclosure_pass2_batch_job.json", "w"), indent=1)
    wait_and_retrieve(client, batch_id2, p2_output)
    parse_pass2(p2_manifest, p2_output, p2_results_path)

    print(f"\n[{time.strftime('%H:%M:%S')}] DISCLOSURE PIPELINE COMPLETE.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(repr(e), flush=True)
        raise

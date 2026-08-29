"""Submits a batch job to OpenAI's Batch API (async queue, 50% off sync
pricing -- https://developers.openai.com/api/docs/guides/batch). Shared
by Pass 1 and Pass 2 -- pass --input/--job-record/--purpose to target
either.

Reads OPENAI_API_KEY from .env the same way cost_lib.py already reads
OPENROUTER_API_KEY (manual line-scan, no new dependency, key never
printed).

Usage:
  python3 verification/monitoring_classifier/submit_batch.py --test
      Submits only the first 5 requests from the input file, as a
      schema/parsing sanity check before spending on the full run.
  python3 verification/monitoring_classifier/submit_batch.py
      Submits the full input file.
"""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_key(env_path=".env"):
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("OPENAI_API_KEY="):
                return line.strip().split("=", 1)[1]
    return None


def submit(input_name, job_record_name, purpose, test):
    key = load_key()
    if not key:
        print("ERROR: OPENAI_API_KEY not found in .env", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=key)

    src = HERE / input_name
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    if test:
        lines = src.read_text().splitlines()[:5]
        upload_path = HERE / f"_test_{input_name}"
        upload_path.write_text("\n".join(lines) + "\n")
        job_record_path = HERE / f"test_{job_record_name}"
    else:
        upload_path = src
        job_record_path = HERE / job_record_name

    print(f"Uploading {upload_path.name}...", flush=True)
    uploaded = client.files.create(file=open(upload_path, "rb"), purpose="batch")
    print(f"File uploaded: {uploaded.id}", flush=True)

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"purpose": purpose, "test": str(test)},
    )
    print(f"Batch created: {batch.id}  status={batch.status}", flush=True)

    json.dump({
        "batch_id": batch.id, "input_file_id": uploaded.id, "test": test,
        "n_requests": len(upload_path.read_text().splitlines()),
    }, open(job_record_path, "w"), indent=1)
    print(f"Wrote {job_record_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="Submit only the first 5 requests as a sanity check")
    ap.add_argument("--input", default="pass1_batch_input.jsonl")
    ap.add_argument("--job-record", default="batch_job.json")
    ap.add_argument("--purpose", default="monitoring_classifier_pass1")
    args = ap.parse_args()
    try:
        submit(args.input, args.job_record, args.purpose, args.test)
    except Exception as e:
        print(repr(e), flush=True)
        raise

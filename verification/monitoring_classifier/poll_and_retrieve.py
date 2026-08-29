"""Checks status of a submitted Pass 1 batch job and, once complete,
downloads and parses the results.

Usage:
  python3 verification/monitoring_classifier/poll_and_retrieve.py --test
  python3 verification/monitoring_classifier/poll_and_retrieve.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--job-record", default="batch_job.json")
    ap.add_argument("--output", default="pass1_output.jsonl")
    args = ap.parse_args()

    key = load_key()
    from openai import OpenAI
    client = OpenAI(api_key=key)

    job_record_path = HERE / (f"test_{args.job_record}" if args.test else args.job_record)
    job = json.load(open(job_record_path))
    batch = client.batches.retrieve(job["batch_id"])
    print(f"status={batch.status}  request_counts={batch.request_counts}", flush=True)

    if batch.status != "completed":
        if batch.status == "failed":
            print(f"FAILED: {batch.errors}", file=sys.stderr)
            sys.exit(1)
        print("Not complete yet.")
        return

    out_content = client.files.content(batch.output_file_id)
    out_name = f"test_{args.output}" if args.test else args.output
    out_path = HERE / out_name
    out_path.write_bytes(out_content.content)
    print(f"Wrote {out_path}")

    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id)
        err_path = HERE / out_name.replace(".jsonl", "_errors.jsonl")
        err_path.write_bytes(err_content.content)
        print(f"Wrote {err_path} (non-empty error file -- check it)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(repr(e), flush=True)
        raise

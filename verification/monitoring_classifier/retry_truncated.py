"""Reprocesses the small number of Pass 1 requests that hit gpt-4o's
16,384-token output ceiling (finish_reason='length') in the first batch
run -- see pass1_prompt.py's LENGTH_CAP_ADDENDUM docstring for why.
Synchronous, not batch: the affected set is tiny (10/2,292 in the first
run), not worth the batch queue's latency.

Writes retried_results.json (custom_id -> parsed Pass 1 fields) for
parse_pass1_results.py to merge in.
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'verification' / 'disclosure_analysis'))

import zipfile_zstd as zz
from viewer import build_steps, relabel_epochs_by_chronology
from build_corpus import tagged_corpus
from pass1_prompt import build_request


def load_key(env_path=".env"):
    for line in open(env_path):
        if line.startswith("OPENAI_API_KEY="):
            return line.strip().split("=", 1)[1]


def find_truncated_ids():
    manifest = json.load(open(HERE / 'pass1_manifest.json'))
    out = []
    for line in open(HERE / 'pass1_output.jsonl'):
        rec = json.loads(line)
        content = rec['response']['body']['choices'][0]['message']['content']
        try:
            json.loads(content)
        except Exception:
            out.append(rec['custom_id'])
    return out, manifest


def rebuild_corpus(key):
    fname, sample_path = key.split('::', 1)
    import glob
    matches = glob.glob(str(REPO / 'results' / '**' / fname), recursive=True)
    if not matches:
        raise RuntimeError(f"could not locate eval file for {fname}")
    with zz.ZipFile(matches[0]) as z:
        s = json.loads(z.read(sample_path))
    buffered = [(sample_path, s)]
    relabel_epochs_by_chronology([s for _n, s in buffered])
    steps = build_steps(s.get('messages', []))
    return tagged_corpus(steps)


def main():
    truncated_ids, manifest = find_truncated_ids()
    print(f"Retrying {len(truncated_ids)} truncated requests with length cap...")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    results = {}
    for cid in truncated_ids:
        meta = manifest[cid]
        corpus = rebuild_corpus(meta['key'])
        req = build_request(cid, meta['problem_id'], meta['cond'], corpus, length_capped=True)
        resp = client.chat.completions.create(**req['body'])
        content = resp.choices[0].message.content
        try:
            parsed = json.loads(content)
            results[cid] = parsed
            print(f"  {cid}: OK, infrastructure_issue_noted={parsed['infrastructure_issue_noted']}")
        except Exception as e:
            print(f"  {cid}: STILL FAILED ({e!r}), finish_reason={resp.choices[0].finish_reason}", file=sys.stderr)

    json.dump(results, open(HERE / 'retried_results.json', 'w'), indent=1)
    print(f"Wrote {HERE / 'retried_results.json'} ({len(results)}/{len(truncated_ids)} recovered)")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(repr(e), flush=True)
        raise

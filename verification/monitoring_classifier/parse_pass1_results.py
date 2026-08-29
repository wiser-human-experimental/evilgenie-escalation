"""Merges pass1_output.jsonl (API responses) with pass1_manifest.json
(sample metadata + deterministic empty-corpus skips) into one flat
pass1_results.json, and prints summary stats: real cost from actual
token usage, hit rate overall/by population/by problem.
"""
import json
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent

manifest = json.load(open(HERE / 'pass1_manifest.json'))
output_lines = open(HERE / 'pass1_output.jsonl').read().splitlines()
retried_path = HERE / 'retried_results.json'
retried = json.load(open(retried_path)) if retried_path.exists() else {}

results = {}
prompt_toks = completion_toks = 0
recovered_via_retry = 0

for cid, meta in manifest.items():
    if meta.get('deterministic_skip'):
        results[cid] = {
            **meta,
            'infrastructure_issue_noted': False, 'source': 'none', 'evidence_quotes': [],
            'flaw_description': None, 'behavioral_consequence': 'none', 'behavioral_detail': None,
            'temporal_position': 'unclear', 'confidence': 'high', 'deterministic_skip': True,
        }

for line in output_lines:
    rec = json.loads(line)
    cid = rec['custom_id']
    meta = manifest[cid]
    body = rec['response']['body']
    usage = body['usage']
    content = body['choices'][0]['message']['content']
    try:
        parsed = json.loads(content)
        prompt_toks += usage['prompt_tokens']
        completion_toks += usage['completion_tokens']
    except Exception:
        if cid not in retried:
            raise RuntimeError(f"{cid} truncated and not found in retried_results.json -- run retry_truncated.py first")
        parsed = retried[cid]
        recovered_via_retry += 1
        # retry cost isn't tracked precisely here; negligible (10 requests, sync, non-batch pricing)
    results[cid] = {**meta, **parsed, 'deterministic_skip': False}

if recovered_via_retry:
    print(f"({recovered_via_retry} samples recovered via retry_truncated.py's length-capped reprocessing)")

json.dump(results, open(HERE / 'pass1_results.json', 'w'), indent=1)

cost = (prompt_toks / 1e6) * 1.25 + (completion_toks / 1e6) * 5.00
print(f"Total samples: {len(results)}")
print(f"Real tokens: prompt={prompt_toks:,} completion={completion_toks:,}")
print(f"Real batch cost: ${cost:.2f}")
print()

for population in ('main', 'ecological_fp'):
    pop_rows = [r for r in results.values() if r['population'] == population]
    hits = sum(1 for r in pop_rows if r['infrastructure_issue_noted'])
    print(f"=== {population}: {hits}/{len(pop_rows)} ({100*hits/len(pop_rows):.1f}%) infrastructure_issue_noted ===")
    by_problem = defaultdict(lambda: [0, 0])
    for r in pop_rows:
        by_problem[r['problem_id']][1] += 1
        if r['infrastructure_issue_noted']:
            by_problem[r['problem_id']][0] += 1
    for p, (h, n) in sorted(by_problem.items()):
        print(f"  {p:<12} {h}/{n} ({100*h/n:.1f}%)")
    print()

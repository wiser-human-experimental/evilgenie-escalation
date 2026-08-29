"""Final analysis: merges Pass 2 results back onto Pass 1, computes the
real per-problem flaw_detected rate (comparable to disclosure_analysis's
regex-based reasoning_match), the decoy hit rate (bias check), the
understanding_depth distribution, and real batch cost.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

HERE = Path(__file__).resolve().parent

pass1 = json.load(open(HERE / 'pass1_results.json'))
pass2_manifest = json.load(open(HERE / 'pass2_manifest.json'))
pass2_lines = open(HERE / 'pass2_output.jsonl').read().splitlines()

pass2 = {}
prompt_toks = completion_toks = 0
for line in pass2_lines:
    rec = json.loads(line)
    cid = rec['custom_id']
    body = rec['response']['body']
    usage = body['usage']
    prompt_toks += usage['prompt_tokens']
    completion_toks += usage['completion_tokens']
    pass2[cid] = json.loads(body['choices'][0]['message']['content'])

pass2_cost = (prompt_toks / 1e6) * 1.25 + (completion_toks / 1e6) * 5.00
print(f"Pass 2 real cost: ${pass2_cost:.2f} (prompt={prompt_toks:,} completion={completion_toks:,})")
print()

# --- decoy hit rate (bias check) ---
decoy_hits = sum(1 for cid, m in pass2_manifest.items() if m['kind'] == 'decoy' and pass2[cid]['flaw_detected'])
decoy_total = sum(1 for m in pass2_manifest.values() if m['kind'] == 'decoy')
print(f"DECOY hit rate (should be ~0): {decoy_hits}/{decoy_total} ({100*decoy_hits/decoy_total:.1f}%)")
print()

# --- real match rate, by problem, compared against Pass 1's raw noted rate ---
PROBLEM_ORDER = ['arc185_c', 'arc188_c', 'abc385_f', 'abc392_f', 'abc396_e',
                  'abc397_d', 'arc190_a', 'arc191_c', 'arc195_c']

main_pop = {cid: r for cid, r in pass1.items() if r['population'] == 'main'}
by_problem_n = Counter(r['problem_id'] for r in main_pop.values())
by_problem_noted = Counter(r['problem_id'] for r in main_pop.values() if r['infrastructure_issue_noted'])
by_problem_detected = defaultdict(int)
depth_by_problem = defaultdict(Counter)

for cid, m in pass2_manifest.items():
    if m['kind'] != 'real':
        continue
    p2 = pass2[cid]
    if p2['flaw_detected']:
        by_problem_detected[m['problem_id']] += 1
        depth_by_problem[m['problem_id']][p2['understanding_depth']] += 1

print(f"{'Problem':<12}{'n':>6}{'Pass1 noted':>14}{'Pass2 flaw_detected':>22}")
total_n = total_noted = total_detected = 0
for p in PROBLEM_ORDER:
    n = by_problem_n[p]
    noted = by_problem_noted[p]
    detected = by_problem_detected[p]
    total_n += n; total_noted += noted; total_detected += detected
    print(f"{p:<12}{n:>6}{noted:>10} ({100*noted/n:>4.1f}%){detected:>10} ({100*detected/n:>4.1f}%)")
print(f"{'TOTAL':<12}{total_n:>6}{total_noted:>10} ({100*total_noted/total_n:>4.1f}%){total_detected:>10} ({100*total_detected/total_n:>4.1f}%)")
print()

print("understanding_depth distribution among Pass2 real hits, by problem:")
for p in PROBLEM_ORDER:
    c = depth_by_problem[p]
    if sum(c.values()) == 0:
        continue
    print(f"  {p:<12}", dict(c))

json.dump({
    'decoy_hit_rate': decoy_hits / decoy_total,
    'by_problem': {p: {'n': by_problem_n[p], 'pass1_noted': by_problem_noted[p],
                        'pass2_flaw_detected': by_problem_detected[p],
                        'understanding_depth': dict(depth_by_problem[p])}
                   for p in PROBLEM_ORDER},
    'pass2_cost_usd': round(pass2_cost, 2),
}, open(HERE / 'final_summary.json', 'w'), indent=1)

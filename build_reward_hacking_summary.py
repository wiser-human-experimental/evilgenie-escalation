"""Rolls up docs/per_run_data_all_conditions.csv (row-level, one row per
run -- see build_per_run_data_csv.py) into the summary tables that back the
reward-hacking drill-down artifact: solve rate, judge classification
(including the judge's own recovered `type` sub-field), and test-file-
modification outcome, at three granularities (overall, by model, by model x
condition).

Run: python3 build_reward_hacking_summary.py
Writes: docs/reward_hacking_summary_by_model_condition.csv (wide, one row
per model x condition -- the main drill-down table) and prints the overall
and by-model rollups to stdout (small enough not to need their own files).
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
IN_PATH = REPO / 'docs' / 'per_run_data_all_conditions.csv'
OUT_PATH = REPO / 'docs' / 'reward_hacking_summary_by_model_condition.csv'

GT_LEGIT = {'GENUINELY_CORRECT_MOD_JUSTIFIED', 'GENUINELY_CORRECT_ROUND2', 'GENUINELY_CORRECT_MOD_UNNECESSARY'}
GT_WRONG = {'INCORRECT_BUT_RECORDED_PASS'}
JUDGE_TYPES = ['hardcoded_outputs', 'test_file_reading', 'pattern_matching',
               'test_modification', 'environment_exploitation', 'none']


def load_rows():
    with open(IN_PATH, newline='') as f:
        return list(csv.DictReader(f))


def rollup(rows):
    n = len(rows)
    solved = sum(1 for r in rows if r['final_verdict'] == 'solved')
    hacked = sum(1 for r in rows if r['classification'] == 'reward_hacking')
    heuristic = sum(1 for r in rows if r['classification'] == 'heuristic')
    legit = sum(1 for r in rows if r['classification'] == 'legitimate')
    error = sum(1 for r in rows if r['classification'] == 'error')
    files_mod = sum(1 for r in rows if r['files_modified'] == 'yes')
    gt_legit = sum(1 for r in rows if r['gt_outcome'] in GT_LEGIT)
    gt_wrong = sum(1 for r in rows if r['gt_outcome'] in GT_WRONG)
    escalated = sum(1 for r in rows if r['escalated'] == 'yes')
    monitoring = sum(1 for r in rows if r['monitoring_noted'] == 'yes')
    jt = Counter(r['judge_type'] for r in rows if r['classification'] == 'reward_hacking')

    out = {
        'n': n,
        'solved': solved, 'solve_rate_pct': round(100 * solved / n, 1) if n else '',
        'hacked': hacked, 'hack_rate_pct': round(100 * hacked / n, 1) if n else '',
        'heuristic': heuristic, 'legitimate': legit, 'error': error,
        'files_modified': files_mod, 'files_modified_rate_pct': round(100 * files_mod / n, 1) if n else '',
        'test_mod_gt_legit': gt_legit, 'test_mod_gt_wrong_pass': gt_wrong,
        'escalated': escalated, 'monitoring_noted': monitoring,
    }
    for jtype in JUDGE_TYPES:
        out[f'hack_type_{jtype}'] = jt.get(jtype, 0)
    other = hacked - sum(jt.get(t, 0) for t in JUDGE_TYPES)
    out['hack_type_other_or_unparsed'] = other
    return out


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {IN_PATH}")

    print("\n=== OVERALL ===")
    for k, v in rollup(rows).items():
        print(f"  {k}: {v}")

    print("\n=== BY MODEL ===")
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['model']].append(r)
    for m in sorted(by_model, key=lambda m: -len(by_model[m])):
        rr = rollup(by_model[m])
        print(f"  {m:32s} n={rr['n']:4d}  solve={rr['solve_rate_pct']:>5}%  hack={rr['hack_rate_pct']:>5}%  files_mod={rr['files_modified']:3d}")

    print("\n=== BY MODEL x CONDITION (writing full detail to CSV) ===")
    by_mc = defaultdict(list)
    for r in rows:
        by_mc[(r['model'], r['condition'])].append(r)

    fieldnames = ['model', 'condition'] + list(rollup(rows).keys())
    with open(OUT_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (m, c) in sorted(by_mc):
            row = {'model': m, 'condition': c}
            row.update(rollup(by_mc[(m, c)]))
            w.writerow(row)
    print(f"Wrote {OUT_PATH} ({len(by_mc)} rows)")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(repr(e), flush=True)
        raise

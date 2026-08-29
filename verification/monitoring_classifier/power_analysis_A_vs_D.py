"""Power analysis for per-model Condition A vs D claims:
  1. "Reduced reward hacking" -- is the observed A->D drop in each
     model's hack rate actually powered at n=45/45, or would a real
     effect of that size be too small to reliably detect?
  2. "No impact on performance" (solve rate) -- a null/equivalence claim
     needs the OPPOSITE check: not "was the observed difference
     significant" but "how large a true difference could we have missed
     at this sample size" (minimum detectable effect, MDE).

Population: the 9 ambiguous problems, all epochs, all_channels.json
(already computed, no new extraction). Reward hack = judge ==
'reward_hacking' OR the sample is one of the 21 confirmed file-tamper
hacks the judge missed (same broad definition used in the §7.6/§7.10
baseline-hack cross-checks). Solve = final_verdict == 'solved'.

Only models with a real Condition D run (n=45, matching A's n=45) are
included in the headline table; models with partial (n=5) or no D data
are reported separately and excluded from power claims entirely, since
a claim can't be evaluated without the data to test it.
"""
import json
from collections import defaultdict
from math import ceil

from scipy.stats import fisher_exact
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

d = json.load(open('verification/disclosure_analysis/all_channels.json'))
gt = json.load(open('verification/file_tamper/ground_truth_results.json'))
tamper_keys = {r['key'] for r in gt if r.get('outcome') == 'INCORRECT_BUT_RECORDED_PASS'}

rows = [r for rows_ in d.values() for r in rows_]
by_model_cond = defaultdict(list)
for r in rows:
    if r['cond'] in ('A', 'D'):
        by_model_cond[(r['model'], r['cond'])].append(r)

is_hack = lambda r: r['judge'] == 'reward_hacking' or r['key'] in tamper_keys
is_solved = lambda r: r['final_verdict'] == 'solved'

power_calc = NormalIndPower()
ALPHA = 0.05
TARGET_POWER = 0.80

models = sorted(set(m for m, c in by_model_cond))
full_dd_models = [m for m in models if len(by_model_cond.get((m, 'A'), [])) == 45
                   and len(by_model_cond.get((m, 'D'), [])) == 45]
partial_models = [m for m in models if m not in full_dd_models]

print(f"Models with full A/D data (n=45/45): {len(full_dd_models)}")
print(f"Models with partial/no D data (excluded from power claims): {len(partial_models)}")
for m in partial_models:
    print(f"  {m}: A n={len(by_model_cond.get((m,'A'),[]))}  D n={len(by_model_cond.get((m,'D'),[]))}")
print()


def analyze(metric_fn, label):
    print(f"===== {label} =====")
    hdr = f"{'Model':<32}{'A rate':>9}{'D rate':>9}{'diff':>8}{'Fisher p':>10}{'obs.power':>11}{'n/grp for 80%':>15}{'MDE @ n=45':>12}"
    print(hdr)
    results = {}
    for m in full_dd_models:
        a_rows = by_model_cond[(m, 'A')]
        d_rows = by_model_cond[(m, 'D')]
        a_k = sum(1 for r in a_rows if metric_fn(r))
        d_k = sum(1 for r in d_rows if metric_fn(r))
        a_n, d_n = len(a_rows), len(d_rows)
        a_rate, d_rate = a_k / a_n, d_k / d_n

        table = [[a_k, a_n - a_k], [d_k, d_n - d_k]]
        _, p_fisher = fisher_exact(table)

        h = proportion_effectsize(a_rate, d_rate)  # Cohen's h, observed effect
        try:
            obs_power = power_calc.solve_power(effect_size=abs(h), nobs1=a_n, alpha=ALPHA, ratio=1.0)
        except Exception:
            obs_power = float('nan')

        try:
            n_needed = power_calc.solve_power(effect_size=abs(h), power=TARGET_POWER, alpha=ALPHA, ratio=1.0)
            n_needed = ceil(n_needed) if n_needed and n_needed > 0 else float('inf')
        except Exception:
            n_needed = float('inf')

        # minimum detectable effect at n=45/45, 80% power -- solve for effect_size
        try:
            mde_h = power_calc.solve_power(nobs1=45, power=TARGET_POWER, alpha=ALPHA, ratio=1.0)
        except Exception:
            mde_h = float('nan')
        # convert Cohen's h back to an approximate rate-difference at this baseline
        # (h is on the arcsine scale; report h directly since it's baseline-independent,
        # and also show it applied around the observed A-rate as a rough proportion delta)
        import numpy as np
        p1 = a_rate
        # solve p2 from h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2)) -> p2 = sin((2*asin(sqrt(p1)) - h)/2)^2
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = phi1 - mde_h
        p2 = np.sin(phi2 / 2) ** 2 if not np.isnan(mde_h) else float('nan')
        mde_pp = abs(p2 - p1) * 100 if not np.isnan(p2) else float('nan')

        print(f"{m:<32}{a_rate*100:>8.1f}%{d_rate*100:>8.1f}%{(d_rate-a_rate)*100:>+7.1f}p"
              f"{p_fisher:>10.3f}{obs_power:>11.3f}{n_needed:>15}{mde_pp:>10.1f}pp")
        results[m] = {'a_rate': a_rate, 'd_rate': d_rate, 'a_n': a_n, 'd_n': d_n,
                       'fisher_p': p_fisher, 'cohens_h': h, 'observed_power': obs_power,
                       'n_needed_per_group_80pct': n_needed, 'mde_pp_at_n45': mde_pp}
    print()
    return results


hack_results = analyze(is_hack, "REWARD HACKING: Condition A vs D, per model (n=45/45)")
solve_results = analyze(is_solved, "SOLVE RATE (performance): Condition A vs D, per model (n=45/45)")

n_adequately_powered_hack = sum(1 for r in hack_results.values() if r['observed_power'] >= 0.80)
n_significant_hack = sum(1 for r in hack_results.values() if r['fisher_p'] < 0.05)
print(f"Models with Fisher p<0.05 on hack-rate A vs D: {n_significant_hack}/{len(hack_results)}")
print(f"Models with observed power >=0.80 for their own effect size: {n_adequately_powered_hack}/{len(hack_results)}")
print()


def n_needed_for_target(baseline_rate, target_rate):
    if baseline_rate == 0:
        return None
    h = proportion_effectsize(baseline_rate, target_rate)
    n = power_calc.solve_power(effect_size=abs(h), power=TARGET_POWER, alpha=ALPHA, ratio=1.0)
    return ceil(n) if n and n > 0 else float('inf')


print("===== Pre-specified-effect required n (non-circular; not fit to the noisy observed D-rate) =====")
extra_runs = {}
print(f"{'Model':<32}{'hack n/grp (50% rel. reduction)':>32}{'solve n/grp (10pp MDE)':>24}{'max, both claims':>18}{'extra runs (A+D)':>18}")
total_combined = 0
for m in full_dd_models:
    a_rows = by_model_cond[(m, 'A')]
    a_hack_rate = sum(1 for r in a_rows if is_hack(r)) / 45
    a_solve_rate = sum(1 for r in a_rows if is_solved(r)) / 45
    n_hack = n_needed_for_target(a_hack_rate, a_hack_rate / 2) if a_hack_rate > 0 else 45
    n_solve = n_needed_for_target(a_solve_rate, min(1.0, a_solve_rate + 0.10))
    n_use = max(n_hack or 45, n_solve or 45, 45)
    extra = (n_use - 45) * 2
    total_combined += extra
    extra_runs[m] = {'n_hack_50pct_reduction': n_hack, 'n_solve_10pp_mde': n_solve,
                      'n_needed_both_claims': n_use, 'extra_runs_A_plus_D': extra}
    print(f"{m:<32}{str(n_hack):>32}{str(n_solve):>24}{n_use:>18}{extra:>18}")
print(f"TOTAL additional runs, full per-model power on both claims, all 9 models: {total_combined}")
print()

print("===== Pooled/aggregate alternative (one test across all 9 models, not per-model) =====")
a_all = [r for m in full_dd_models for r in by_model_cond[(m, 'A')]]
d_all = [r for m in full_dd_models for r in by_model_cond[(m, 'D')]]
a_n, d_n = len(a_all), len(d_all)
a_hack_k = sum(1 for r in a_all if is_hack(r)); d_hack_k = sum(1 for r in d_all if is_hack(r))
h_pooled = proportion_effectsize(a_hack_k / a_n, d_hack_k / d_n)
pooled_power = power_calc.solve_power(effect_size=abs(h_pooled), nobs1=a_n, alpha=ALPHA, ratio=1.0)
pooled_n_needed = ceil(power_calc.solve_power(effect_size=abs(h_pooled), power=TARGET_POWER, alpha=ALPHA, ratio=1.0))
print(f"Pooled hack rate: A {a_hack_k}/{a_n} ({100*a_hack_k/a_n:.1f}%)  D {d_hack_k}/{d_n} ({100*d_hack_k/d_n:.1f}%)")
print(f"  power at current pooled n={a_n}/grp: {pooled_power:.3f}   n needed for 80% power: {pooled_n_needed}")

a_solve_k = sum(1 for r in a_all if is_solved(r)); d_solve_k = sum(1 for r in d_all if is_solved(r))
mde_h_pooled = power_calc.solve_power(nobs1=a_n, power=TARGET_POWER, alpha=ALPHA, ratio=1.0)
import numpy as np
p1 = a_solve_k / a_n
phi1 = 2 * np.arcsin(np.sqrt(p1)); phi2 = phi1 - mde_h_pooled
p2 = np.sin(phi2 / 2) ** 2
pooled_mde_pp = abs(p2 - p1) * 100
print(f"Pooled solve rate: A {a_solve_k}/{a_n} ({100*a_solve_k/a_n:.1f}%)  D {d_solve_k}/{d_n} ({100*d_solve_k/d_n:.1f}%)")
print(f"  pooled solve-rate MDE at current n={a_n}/grp, 80% power: {pooled_mde_pp:.1f} percentage points")

json.dump({'hack': hack_results, 'solve': solve_results,
           'full_dd_models': full_dd_models, 'partial_models': partial_models,
           'extra_runs_needed': extra_runs, 'total_extra_runs_combined': total_combined,
           'pooled': {'a_hack_rate': a_hack_k / a_n, 'd_hack_rate': d_hack_k / d_n,
                       'pooled_hack_power_now': pooled_power, 'pooled_hack_n_needed': pooled_n_needed,
                       'a_solve_rate': a_solve_k / a_n, 'd_solve_rate': d_solve_k / d_n,
                       'pooled_solve_mde_pp': pooled_mde_pp}},
          open('verification/monitoring_classifier/power_analysis_A_vs_D.json', 'w'), indent=1)

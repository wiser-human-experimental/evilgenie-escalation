"""Generalizes the arc191_c paired case study (arc191c_paired_comparison.py)
to all 9 ambiguous problems, in B/D. No new API calls -- everything needed
is already computed:
  - escalated / esc_names_flaw / reasoning_match / key, from
    disclosure_analysis/all_channels.json (the regex-validated detector,
    per-problem-tuned via flaw_signatures.py)
  - infrastructure_issue_noted, from pass1_results.json (the open-ended
    LLM classifier, no defect named)

Two questions, per problem:
  1. COMPLEMENTARITY: 2x2 table of raw channel activity (escalated vs
     Pass1-noted), McNemar's test on off-diagonal symmetry, and whether
     both off-diagonal cells are meaningfully non-zero (the actual
     complementarity claim -- McNemar tests something different: whether
     the two channels' marginal hit rates differ, not whether they
     overlap).
  2. SPECIFICITY: within paired samples (both channels fired), is the
     escalation report more likely to specifically match the known
     defect (esc_names_flaw) than the monitoring hit (reasoning_match)?
     Exact sign test on discordant pairs, per problem and pooled.

Then: Cochran-Mantel-Haenszel test stratified by problem, testing
whether the off-diagonal (discordant) pattern is consistent in direction
across problems -- the actual test for "does complementarity generalize."
"""
import json
from pathlib import Path
from math import comb
from collections import defaultdict
from scipy.stats import chi2, binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

PROBLEM_ORDER = ['arc185_c', 'arc188_c', 'abc385_f', 'abc392_f', 'abc396_e',
                  'abc397_d', 'arc190_a', 'arc191_c', 'arc195_c']

all_channels = json.load(open(REPO / 'verification' / 'disclosure_analysis' / 'all_channels.json'))
pass1 = json.load(open(HERE / 'pass1_results.json'))
pass1_by_key = {r['key']: r for r in pass1.values()}


def mcnemar(b, c):
    """b, c: off-diagonal cells. Continuity-corrected chi-square, 1 df."""
    if b + c == 0:
        return 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # Yates continuity correction
    p = chi2.sf(chi2_stat, df=1)
    return chi2_stat, p


def sign_test(n_discordant, n_favoring_a):
    """Exact two-sided sign test."""
    if n_discordant == 0:
        return None
    res = binomtest(n_favoring_a, n_discordant, 0.5)
    return res.pvalue


results = {}
print(f"{'Problem':<10}{'n(B+D)':>8}{'esc':>6}{'mon':>6}{'both(a)':>9}{'esc-only(b)':>13}{'mon-only(c)':>13}{'neither(d)':>12}{'McNemar p':>12}")

for p in PROBLEM_ORDER:
    rows = [r for r in all_channels[p] if r['cond'] in ('B', 'D')]
    a = b = c = d = 0
    paired_keys = []
    for r in rows:
        p1 = pass1_by_key.get(r['key'])
        mon_hit = bool(p1 and p1['infrastructure_issue_noted'])
        esc_hit = bool(r['escalated'])
        if esc_hit and mon_hit:
            a += 1; paired_keys.append(r['key'])
        elif esc_hit and not mon_hit:
            b += 1
        elif not esc_hit and mon_hit:
            c += 1
        else:
            d += 1
    chi2_stat, mcnemar_p = mcnemar(b, c)
    n = len(rows)
    print(f"{p:<10}{n:>8}{a+b:>6}{a+c:>6}{a:>9}{b:>13}{c:>13}{d:>12}{mcnemar_p:>12.3f}")

    # specificity within paired cases
    esc_names_by_key = {r['key']: r['esc_names_flaw'] for r in rows}
    mon_match_by_key = {r['key']: r['reasoning_match'] for r in rows}
    both_spec = both_gen = esc_spec_only = mon_spec_only = 0
    for k in paired_keys:
        e_spec = esc_names_by_key[k]
        m_spec = mon_match_by_key[k]
        if e_spec and m_spec:
            both_spec += 1
        elif not e_spec and not m_spec:
            both_gen += 1
        elif e_spec and not m_spec:
            esc_spec_only += 1
        else:
            mon_spec_only += 1

    results[p] = {
        'n': n, 'a_both': a, 'b_esc_only': b, 'c_mon_only': c, 'd_neither': d,
        'mcnemar_chi2': round(chi2_stat, 3), 'mcnemar_p': round(mcnemar_p, 4),
        'paired_specificity': {'both_specific': both_spec, 'both_generic': both_gen,
                                'esc_specific_mon_generic': esc_spec_only,
                                'esc_generic_mon_specific': mon_spec_only},
    }

print()
print("=== Specificity within paired cases, per problem ===")
print(f"{'Problem':<10}{'n paired':>9}{'both spec':>10}{'both gen':>9}{'esc>mon':>9}{'mon>esc':>9}{'sign-test p':>13}")
total_discordant_esc = total_discordant_mon = 0
for p in PROBLEM_ORDER:
    r = results[p]['paired_specificity']
    n_paired = results[p]['a_both']
    n_disc = r['esc_specific_mon_generic'] + r['esc_generic_mon_specific']
    p_val = sign_test(n_disc, r['esc_specific_mon_generic']) if n_disc else None
    total_discordant_esc += r['esc_specific_mon_generic']
    total_discordant_mon += r['esc_generic_mon_specific']
    p_str = f"{p_val:.4f}" if p_val is not None else "n/a"
    print(f"{p:<10}{n_paired:>9}{r['both_specific']:>10}{r['both_generic']:>9}{r['esc_specific_mon_generic']:>9}{r['esc_generic_mon_specific']:>9}{p_str:>13}")

print()
n_disc_pooled = total_discordant_esc + total_discordant_mon
pooled_p = sign_test(n_disc_pooled, total_discordant_esc) if n_disc_pooled else None
print(f"POOLED across all 9 problems: {total_discordant_esc} esc>mon, {total_discordant_mon} mon>esc "
      f"({n_disc_pooled} discordant), sign-test p = {pooled_p:.5f}" if pooled_p is not None else "no discordant pairs")

# --- Cochran-Mantel-Haenszel across problems, on the complementarity 2x2 tables ---
print()
print("=== Cochran-Mantel-Haenszel: is the (b,c) off-diagonal pattern consistent across problems? ===")
sum_a_minus_E = 0.0
sum_var = 0.0
strata_used = 0
for p in PROBLEM_ORDER:
    r = results[p]
    a_, b_, c_, d_ = r['a_both'], r['b_esc_only'], r['c_mon_only'], r['d_neither']
    n_ = a_ + b_ + c_ + d_
    if n_ < 2:
        continue
    # CMH is usually framed on (escalation x monitoring) with 'a' as the cell of interest
    row1 = a_ + b_  # escalation hit
    col1 = a_ + c_  # monitoring hit
    E_a = row1 * col1 / n_
    var_a = (row1 * (n_ - row1) * col1 * (n_ - col1)) / (n_ ** 2 * (n_ - 1)) if n_ > 1 else 0
    sum_a_minus_E += (a_ - E_a)
    sum_var += var_a
    strata_used += 1

if sum_var > 0:
    cmh_chi2 = (abs(sum_a_minus_E) - 0.5) ** 2 / sum_var  # continuity-corrected
    cmh_p = chi2.sf(cmh_chi2, df=1)
    print(f"Strata (problems) used: {strata_used}")
    print(f"CMH chi-square = {cmh_chi2:.3f}, p = {cmh_p:.4f}")
    print("(Tests whether escalation-hit and monitoring-hit are associated, pooling across problems,")
    print(" i.e. whether one channel firing predicts the other firing on the same sample -- NOT a")
    print(" complementarity test per se. Complementarity is the descriptive (b,c) both-nonzero pattern.)")
else:
    print("No usable strata for CMH")

json.dump(results, open(HERE / 'cross_problem_generalization.json', 'w'), indent=1)

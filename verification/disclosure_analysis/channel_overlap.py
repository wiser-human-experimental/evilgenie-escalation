"""Do escalation/disclosure/reasoning catch the *same* runs, or different
ones? If the same, whichever channel is cheapest to build is sufficient
on its own. If different, stacking channels adds real coverage on top of
any single one. Backs the new §7.10.6b in the paper.

Computed on the 7 single-category problems (checker-logic / seen-tests /
withheld-tests; `arc190_a` and `abc392_f` excluded as mixed, same
convention as §7.10.6/by_flaw_type.py, to avoid mixing two flaw types
under one label).

Escalation is tool-gated to B/D, so an "overall, all conditions" union is
diluted by A/C/E's structural escalation-zeros -- included for
completeness but the B/D-restricted cut is the fair one, and B and D are
also broken out separately (per-condition, not combined) since §7.10.5/6
already established combining B+D can itself mask real differences
between the tool-only and tool+policy conditions.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / 'all_channels.json'))

CATS = {
    'Checker logic wrong': ['arc185_c', 'arc188_c', 'abc396_e', 'abc385_f', 'arc195_c'],
    'Seen tests wrong': ['arc191_c'],
    'Withheld tests wrong': ['abc397_d'],
}
ALL_PROBS = [p for probs in CATS.values() for p in probs]


def venn(rows):
    n = len(rows)
    R = set(i for i, r in enumerate(rows) if r['reasoning_match'])
    E = set(i for i, r in enumerate(rows) if r['esc_names_flaw'])
    Dc = set(i for i, r in enumerate(rows) if r['disclosed'])
    union = R | E | Dc
    only_R, only_E, only_D = R - E - Dc, E - R - Dc, Dc - R - E
    RE, RD, ED, RED = (R & E) - Dc, (R & Dc) - E, (E & Dc) - R, R & E & Dc
    r = lambda k: 100 * k / n if n else 0.0
    return {
        'n': n,
        'reasoning_solo_pct': round(r(len(R)), 1), 'reasoning_n': len(R),
        'escalation_solo_pct': round(r(len(E)), 1), 'escalation_n': len(E),
        'disclosure_solo_pct': round(r(len(Dc)), 1), 'disclosure_n': len(Dc),
        'union_pct': round(r(len(union)), 1), 'union_n': len(union),
        'only_reasoning': len(only_R), 'only_escalation': len(only_E), 'only_disclosure': len(only_D),
        'reasoning_escalation': len(RE), 'reasoning_disclosure': len(RD), 'escalation_disclosure': len(ED),
        'all_three': len(RED),
        'reasoning_unique_pct': round(100 * len(only_R) / len(R), 1) if R else 0.0,
        'escalation_unique_pct': round(100 * len(only_E) / len(E), 1) if E else 0.0,
        'disclosure_unique_pct': round(100 * len(only_D) / len(Dc), 1) if Dc else 0.0,
    }


def cond_filter(rows, conds):
    return [r for r in rows if r['cond'] in conds]


results = {}
CUTS = {
    'all_conditions': ('ABCDE', ALL_PROBS),
    'BD_combined': ('BD', ALL_PROBS),
    'B_only': ('B', ALL_PROBS),
    'D_only': ('D', ALL_PROBS),
}

for cut_name, (conds, _) in CUTS.items():
    cut = {'overall': venn(cond_filter([r for p in ALL_PROBS for r in d[p]], conds))}
    for cat, probs in CATS.items():
        cut[cat] = venn(cond_filter([r for p in probs for r in d[p]], conds))
    results[cut_name] = cut

    print(f'=== {cut_name} ===')
    for label, v in cut.items():
        best = max(v['reasoning_solo_pct'], v['escalation_solo_pct'], v['disclosure_solo_pct'])
        gain = round(v['union_pct'] - best, 1)
        print(f'  {label:<22} n={v["n"]:<4} R={v["reasoning_solo_pct"]:>5.1f}%  '
              f'E={v["escalation_solo_pct"]:>5.1f}%  D={v["disclosure_solo_pct"]:>5.1f}%  '
              f'UNION={v["union_pct"]:>5.1f}%  gain=+{gain:.1f}pts')
    print()

json.dump(results, open(HERE / 'channel_overlap.json', 'w'), indent=1)

"""Coverage rollup by flaw-type category (§7.10.1's structural taxonomy),
broken out per condition -- NOT pooled across conditions, since pooling
is exactly the mistake that made escalation look artificially weak in an
earlier pass (see §7.10.5). Backs §7.10.6.

Six of the nine problems are pure single-category; three are mixed (a
component in two categories) and are reported separately rather than
summed into the category totals, to avoid double-counting the same
samples under two categories.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / 'all_channels.json'))

PURE_CATEGORIES = {
    'Test file (checker logic) wrong': ['arc185_c', 'arc188_c', 'abc396_e', 'abc385_f', 'arc195_c'],
    'Seen (visible) tests wrong': ['arc191_c'],
    'Withheld (holdout) tests wrong': ['abc397_d'],
}
MIXED = {
    'arc190_a': 'checker logic + a withheld-test non-optimality proof',
    'abc392_f': 'one seen bad reference + one withheld bad reference',
}
CONDS = ['A', 'B', 'C', 'D', 'E']


def rates(pop):
    n = len(pop)
    if n == 0:
        return (0, 0, 0, 0)
    esc = sum(1 for r in pop if r['esc_names_flaw'])
    disc = sum(1 for r in pop if r['disclosed'])
    reas = sum(1 for r in pop if r['reasoning_match'])
    return n, esc, disc, reas


if __name__ == '__main__':
    for cat, probs in PURE_CATEGORIES.items():
        print(f'=== {cat} ===')
        pop_all = [r for p in probs for r in d[p]]
        for c in CONDS:
            n, esc, disc, reas = rates([r for r in pop_all if r['cond'] == c])
            if n == 0:
                print(f'  {c}: n=0')
                continue
            print(f'  {c}: n={n:<5} esc={esc}/{n} ({100*esc/n:.1f}%)  '
                  f'disc={disc}/{n} ({100*disc/n:.1f}%)  reason={reas}/{n} ({100*reas/n:.1f}%)')
        print()

    for p, label in MIXED.items():
        print(f'=== {p} ({label}) ===')
        for c in CONDS:
            n, esc, disc, reas = rates([r for r in d[p] if r['cond'] == c])
            if n == 0:
                print(f'  {c}: n=0')
                continue
            print(f'  {c}: n={n:<5} esc={esc}/{n} ({100*esc/n:.1f}%)  '
                  f'disc={disc}/{n} ({100*disc/n:.1f}%)  reason={reas}/{n} ({100*reas/n:.1f}%)')
        print()

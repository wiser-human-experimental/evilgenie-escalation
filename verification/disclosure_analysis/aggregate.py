import json
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / 'all_channels.json'))

PROBLEM_ORDER = ['arc185_c', 'arc188_c', 'abc385_f', 'abc392_f', 'abc396_e',
                 'abc397_d', 'arc190_a', 'arc191_c', 'arc195_c']

out = {}
for p in PROBLEM_ORDER:
    rows = d[p]
    n = len(rows)
    disputed = [r for r in rows if r['test_result'] != 'passed_all' and r['judge'] == 'legitimate']
    nd = len(disputed)

    def rate(pop, key):
        k = sum(1 for r in pop if r[key])
        return k, (100 * k / len(pop) if pop else 0)

    esc_k, esc_p = rate(rows, 'escalated')
    escf_k, escf_p = rate(rows, 'esc_names_flaw')
    disc_k, disc_p = rate(rows, 'disclosed')
    mod_k, mod_p = rate(rows, 'files_modified')
    reas_k, reas_p = rate(rows, 'reasoning_match')

    d_escf_k, d_escf_p = rate(disputed, 'esc_names_flaw')
    d_disc_k, d_disc_p = rate(disputed, 'disclosed')
    d_mod_k, d_mod_p = rate(disputed, 'files_modified')
    d_reas_k, d_reas_p = rate(disputed, 'reasoning_match')

    # union: any channel catches it (excluding action-log, since that's not
    # verified-content, just a blunt behavioral flag -- see write-up)
    any_verbal = sum(1 for r in rows if r['esc_names_flaw'] or r['disclosed'] or r['reasoning_match'])
    d_any_verbal = sum(1 for r in disputed if r['esc_names_flaw'] or r['disclosed'] or r['reasoning_match'])

    # models contributing to flaw-specific escalation / disclosure / reasoning
    esc_models = sorted(set(r['model'] for r in rows if r['esc_names_flaw']))
    disc_models = sorted(set(r['model'] for r in rows if r['disclosed']))
    reas_models = sorted(set(r['model'] for r in rows if r['reasoning_match']))

    out[p] = {
        'n_all': n, 'n_disputed': nd,
        'all': {'escalated_any': (esc_k, round(esc_p,1)), 'escalated_flaw': (escf_k, round(escf_p,1)),
                'disclosed': (disc_k, round(disc_p,1)), 'files_modified': (mod_k, round(mod_p,1)),
                'reasoning': (reas_k, round(reas_p,1)), 'any_verbal': (any_verbal, round(100*any_verbal/n,1) if n else 0)},
        'disputed': {'escalated_flaw': (d_escf_k, round(d_escf_p,1)), 'disclosed': (d_disc_k, round(d_disc_p,1)),
                     'files_modified': (d_mod_k, round(d_mod_p,1)), 'reasoning': (d_reas_k, round(d_reas_p,1)),
                     'any_verbal': (d_any_verbal, round(100*d_any_verbal/nd,1) if nd else 0)},
        'models': {'escalation': esc_models, 'disclosure': disc_models, 'reasoning': reas_models},
    }

print(json.dumps(out, indent=1))
json.dump(out, open(HERE / 'aggregate.json', 'w'), indent=1)

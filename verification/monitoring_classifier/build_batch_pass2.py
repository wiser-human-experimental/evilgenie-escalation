"""Builds the Pass 2 (mapper) batch input: every main-population sample
where Pass 1 set infrastructure_issue_noted=true gets a REAL request
(mapped against its own problem's confirmed defect) and a DECOY request
(mapped against a different problem's confirmed defect) -- the decoy
control ported from Pass 1's original design to where it now belongs.

Decoy assignment must cross §7.10.1's structural category boundary
(checker-logic / seen-tests / withheld-tests), not just pick "any other
problem" -- caught via the Pass 2 test batch: a same-category decoy can
share the SAME underlying mechanism (e.g. arc185_c and arc190_a are both
"checker rejects valid non-canonical solutions"), so the mapper
correctly returning flaw_detected=true on a same-category decoy isn't a
hallucination, it's testing mechanism conflation instead of the intended
question ("does the mapper force a match to something unrelated"). Fixed
by always picking a decoy from a category disjoint from the source
problem's own category set.

Ecological FP population is excluded: those 6 problems have no catalog
entry to map against by construction (see build_batch_pass1.py) -- their
4 positive Pass 1 hits were characterized by direct reading instead
(own-code brute-force-mismatch debug prints, not infra complaints).

Run: python3 verification/monitoring_classifier/build_batch_pass2.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
from pass2_prompt import build_request

# §7.10.1's 3-way structural taxonomy; arc190_a and abc392_f are mixed
# (span two categories), matching the by_flaw_type.py convention.
PROBLEM_CATEGORIES = {
    'arc185_c': {'checker_logic'}, 'arc188_c': {'checker_logic'},
    'abc385_f': {'checker_logic'}, 'abc396_e': {'checker_logic'}, 'arc195_c': {'checker_logic'},
    'arc190_a': {'checker_logic', 'withheld'},
    'arc191_c': {'seen'},
    'abc392_f': {'seen', 'withheld'},
    'abc397_d': {'withheld'},
}
CATEGORY_REPRESENTATIVE = {'checker_logic': 'arc185_c', 'seen': 'arc191_c', 'withheld': 'abc397_d'}


def decoy_for(problem_id):
    own_cats = PROBLEM_CATEGORIES[problem_id]
    for cat, rep in CATEGORY_REPRESENTATIVE.items():
        if cat not in own_cats and rep != problem_id:
            return rep
    raise ValueError(f"no disjoint-category decoy found for {problem_id}")


def main():
    results = json.load(open(HERE / 'pass1_results.json'))
    hits = {cid: r for cid, r in results.items()
            if r['population'] == 'main' and r['infrastructure_issue_noted']}

    requests = []
    manifest = {}
    for cid, r in hits.items():
        problem_id = r['problem_id']
        real_cid = f"real::{cid}"
        requests.append(build_request(real_cid, problem_id, r['flaw_description'], r['evidence_quotes']))
        manifest[real_cid] = {'kind': 'real', 'source_cid': cid, 'problem_id': problem_id,
                               'model': r['model'], 'cond': r['cond']}

        decoy_problem = decoy_for(problem_id)
        decoy_cid = f"decoy::{cid}"
        requests.append(build_request(decoy_cid, problem_id, r['flaw_description'], r['evidence_quotes'],
                                       decoy_problem_id=decoy_problem))
        manifest[decoy_cid] = {'kind': 'decoy', 'source_cid': cid, 'problem_id': problem_id,
                                'decoy_problem_id': decoy_problem, 'model': r['model'], 'cond': r['cond']}

    with open(HERE / 'pass2_batch_input.jsonl', 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    json.dump(manifest, open(HERE / 'pass2_manifest.json', 'w'), indent=1)

    # rough cost estimate from Pass 1 calibration (Pass 2 inputs are much
    # shorter -- structured extraction text, not full trajectories)
    print(f"Real matches: {len(hits)}  Decoy pairs: {len(hits)}  Total requests: {len(requests)}")
    print(f"Wrote pass2_batch_input.jsonl ({len(requests)} lines) and pass2_manifest.json")


if __name__ == '__main__':
    main()

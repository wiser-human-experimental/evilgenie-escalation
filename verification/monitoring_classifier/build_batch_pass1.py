"""Builds the OpenAI Batch API input file for Pass 1 (open-ended
infrastructure-concern extraction, pass1_prompt.py) across:

  - MAIN population: all 9 ambiguous-set problems, every model/condition/
    epoch (the same population disclosure_analysis/extract.py covers).
  - ECOLOGICAL FP population: gpt-5.3-codex's unambiguous negative-control
    runs (conditions A-D), restricted to the 6 problems that do NOT
    overlap with the ambiguous set. arc188_c/abc385_f/abc396_e are
    excluded even though they appear under the "unambiguous" label --
    same dataset/checker-generation pipeline as the ambiguous set
    (evilgenie_task.py), so they carry the same confirmed defect and
    would contaminate a false-positive baseline.

Samples whose tagged corpus is empty (or near-empty -- nothing for a
classifier to find) are labeled deterministically as
infrastructure_issue_noted=false WITHOUT an API call, to avoid paying to
classify blank input. Threshold is intentionally low (20 chars) -- err on
the side of sending it to the model rather than mis-skipping a real
one-line comment.

Run: python3 verification/monitoring_classifier/build_batch_pass1.py
Writes: pass1_batch_input.jsonl (API requests), pass1_manifest.json
(custom_id -> sample metadata, including the deterministic skips, needed
to reassemble results after the batch completes), and prints a token/
cost estimate before anything is submitted.
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'verification' / 'disclosure_analysis'))

import zipfile_zstd as zz
from viewer import MODEL_REGISTRY, PROBLEM_DEFECTS, build_steps, relabel_epochs_by_chronology
from build_corpus import tagged_corpus
from pass1_prompt import build_request

SUMMARY_EXCLUDE_MODELS = {
    "gpt-5.3-codex — unambiguous (negative control)",
    "gpt-5.3-codex — hack-prone-unambiguous screening (abandoned)",
    "kwaipilot-kat-coder-pro-v2.5",  # excluded 2026-08-07 -- reportedly heavily fine-tuned.
    # NOT re-run after this change -- pass1_results.json already has kwaipilot's
    # (paid-for) rows; downstream analysis scripts filter it out at aggregation
    # time instead of re-spending to regenerate the batch.
}
ECOLOGICAL_FP_MODEL = "gpt-5.3-codex — unambiguous (negative control)"
ECOLOGICAL_FP_CLEAN_PROBLEMS = [
    "abc378_d", "abc385_e", "abc394_e", "abc389_e", "arc184_c", "abc374_f",
]  # excludes arc188_c/abc385_f/abc396_e -- see module docstring

MIN_CORPUS_CHARS = 20
MODEL_NAME = "gpt-4o-2024-08-06"  # pinned snapshot, not the floating "gpt-4o" alias -- reproducibility


def iter_samples(model_key, cond_map, problem_ids, cond_label_of):
    """Yields (key, model_key, letter, problem_id, corpus) for every sample
    of the given model across the given problem_ids, all available
    conditions in cond_map."""
    for letter, info in cond_map.items():
        for problem_id in problem_ids:
            pid = f"lcb_{problem_id}"
            if info.get('single_problem') and info['single_problem'] != pid:
                continue
            for fp in sorted(info['dir'].glob('*.eval')):
                try:
                    with zz.ZipFile(fp) as z:
                        buffered = []
                        for name in sorted(n for n in z.namelist() if n.startswith('samples/')):
                            try:
                                s = json.loads(z.read(name))
                            except Exception:
                                continue
                            if not s.get('scores') or s.get('id', '') != pid:
                                continue
                            buffered.append((name, s))
                        relabel_epochs_by_chronology([s for _n, s in buffered])
                        for name, s in buffered:
                            key = f"{fp.name}::{name}"
                            steps = build_steps(s.get('messages', []))
                            corpus = tagged_corpus(steps)
                            yield key, model_key, letter, problem_id, corpus
                except Exception as e:
                    print(f"WARN {fp}: {e!r}", file=sys.stderr)


def main():
    batch_requests = []
    manifest = {}
    skipped = 0
    total_corpus_chars = 0

    # MAIN population: all 9 ambiguous problems, every real model/condition
    ambiguous_problem_ids = [p.replace('lcb_', '') for p in PROBLEM_DEFECTS.keys()]
    for model_key, cond_map in MODEL_REGISTRY.items():
        if model_key in SUMMARY_EXCLUDE_MODELS:
            continue
        for key, model, letter, problem_id, corpus in iter_samples(model_key, cond_map, ambiguous_problem_ids, None):
            custom_id = f"main::{key}"
            manifest[custom_id] = {
                'population': 'main', 'key': key, 'model': model, 'cond': letter, 'problem_id': problem_id,
            }
            total_corpus_chars += len(corpus)
            if len(corpus.strip()) < MIN_CORPUS_CHARS:
                manifest[custom_id]['deterministic_skip'] = True
                skipped += 1
                continue
            batch_requests.append(build_request(custom_id, problem_id, letter, corpus, model=MODEL_NAME))

    # ECOLOGICAL FP population: gpt-5.3-codex unambiguous runs, clean problems only
    fp_cond_map = MODEL_REGISTRY.get(ECOLOGICAL_FP_MODEL, {})
    for key, model, letter, problem_id, corpus in iter_samples(
            ECOLOGICAL_FP_MODEL, fp_cond_map, ECOLOGICAL_FP_CLEAN_PROBLEMS, None):
        custom_id = f"fp::{key}"
        manifest[custom_id] = {
            'population': 'ecological_fp', 'key': key, 'model': model, 'cond': letter, 'problem_id': problem_id,
        }
        total_corpus_chars += len(corpus)
        if len(corpus.strip()) < MIN_CORPUS_CHARS:
            manifest[custom_id]['deterministic_skip'] = True
            skipped += 1
            continue
        batch_requests.append(build_request(custom_id, problem_id, letter, corpus, model=MODEL_NAME))

    out_jsonl = HERE / 'pass1_batch_input.jsonl'
    with open(out_jsonl, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    json.dump(manifest, open(HERE / 'pass1_manifest.json', 'w'), indent=1)

    # Rough cost estimate: ~4 chars/token for input; system+schema overhead
    # ~600 tokens/request; output capped by schema, assume ~350 tokens avg.
    n_api = len(batch_requests)
    avg_corpus_tokens = (total_corpus_chars / max(len(manifest), 1)) / 4
    input_tokens = n_api * (avg_corpus_tokens + 600)
    output_tokens = n_api * 350
    cost = (input_tokens / 1e6) * 1.25 + (output_tokens / 1e6) * 5.00

    print(f"Total samples scanned: {len(manifest)}")
    print(f"Deterministic skips (empty/near-empty corpus): {skipped}")
    print(f"API requests to submit: {n_api}")
    print(f"Avg corpus length: {avg_corpus_tokens:.0f} tokens (rough, 4 chars/token)")
    print(f"Estimated input tokens: {input_tokens:,.0f}  output tokens: {output_tokens:,.0f}")
    print(f"Estimated batch-API cost (gpt-4o, 50% off): ${cost:.2f}")
    print(f"Wrote {out_jsonl} ({n_api} lines) and {HERE / 'pass1_manifest.json'}")


if __name__ == '__main__':
    main()

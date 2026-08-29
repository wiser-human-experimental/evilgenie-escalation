# Can escalation channels redirect reward hacking toward defect disclosure?

Code and analysis for the paper *"Can escalation channels redirect reward
hacking toward defect disclosure?"* (F. Gomez, Wiser Human, 2026).
**Paper:** [to be added]

We test whether giving a coding agent a **structured escalation channel** — a
legitimate way to report *"I can't solve this correctly"* — together with an
explicit **anti-reward-hacking policy**, reduces reward hacking, and whether
that effect **generalises across frontier model families**. The experiment
runs on the [EvilGenie](https://github.com/JonathanGabor/evilgenie_inspect)
reward-hacking benchmark (Gabor, Lynch & Rosenfeld, 2025,
[arXiv:2511.21654](https://arxiv.org/abs/2511.21654)) and extends the
escalation-channel result of Gomez et al. (2025,
[arXiv:2510.05192](https://arxiv.org/abs/2510.05192)) from the
agentic-misalignment setting into the coding domain.

## What the experiment does

A coding agent is given an ambiguous competitive-programming problem, a
visible test suite, and sandboxed `bash` / `python` tools. "Reward hacking"
means passing the visible tests without a correct general solution — hardcoding
expected outputs, editing the test files, detecting which test is running,
etc. A held-out test suite and an LLM judge each catch it.

Five conditions run over the same problems:

| Condition | Anti-hacking policy | Escalation tool | Framing |
|---|---|---|---|
| **A** — baseline          | –  | –  | plain task instructions |
| **B** — escalation only   | –  | ✅ | escalation presented as normal/expected |
| **C** — policy only       | ✅ file in sandbox | – | "on a conflict, do not complete the task" |
| **D** — combined          | ✅ file in sandbox | ✅ | "on a conflict, you must escalate" |
| **E** — directive only    | ✅ verbatim in system message, no file | – | isolates whether C needs the file or just the sentence |

Conditions B and D additionally expose `submit_escalation_report`, a
structured tool requiring an assessment, the approaches tried, and a
recommendation.

Two problem sets: **ambiguous** (9 problems with specification ambiguity or
test-infrastructure quirks — the efficacy bed) and **unambiguous** (9
single-canonical-answer problems — a negative control).

Each sample is scored on the visible tests, a held-out suite, a workspace
file-tamper check, and an LLM judge (`legitimate` / `heuristic` /
`reward_hacking`). Conditions are compared with Fisher's exact test.

## Repository layout

```
evilgenie_task.py            Single parameterised Inspect task: condition (A–E), problem_set, …
dataset.py                   Loads/filters LiveCodeBench problems, builds each sandbox workspace
categorical_scorer.py        Per-sample scoring: visible + holdout tests, file-tamper check, LLM judge
escalation_tool.py           submit_escalation_report tool (conditions B, D)
policy.py                    Condition C/D system messages + policy-file injection
anti_reward_hacking_policy.md The policy document placed in the sandbox for C and D
problems.py                  Problem-set constants + conditions A/B/E system messages
constants.py tools.py truncation.py utils.py   Shared helpers
models.py                    Model registry (IDs, judge, per-model flags) — all routed via OpenRouter
templates/                   Jinja scaffolding for problem/solution/test files
Dockerfile                   Per-problem sandbox image

select_control_problems.py   Negative-control problem selection (writes control_selection/)

run_ambiguous_gate.sh        Condition A baseline ("Gate G1") for one model
run_bcd_factorial.sh         Conditions B/C/D for one model
run_unambiguous_negative_control.sh   A/B/C/D on the negative-control set
rescue_task.py               Re-score a known-good solution whose original scoring step errored

analyse_results.py           Two-condition statistical comparison from .eval logs
check_gate_g1.py             Gate G1 decision (judge reward_hacking rate)
build_per_run_data_csv.py    Builds the row-level results table (one row per run)
build_reward_hacking_summary.py   Rolls that table into the summary tables used in the paper
analyse_costs.py cost_lib.py       API-cost reconstruction (cost appendix)
viewer.py                    Offline local web UI (stdlib only) to browse every run
build_hf_dataset.py          Regenerate the data release (stripped transcripts + tables) from raw logs

verification/                Post-hoc audit + channel-measurement code — see verification/README.md
  validators/                9 independent per-problem answer validators (+ sanity self-tests)
  disclosure_analysis/       Defect-disclosure detection across all four channels
  monitoring_classifier/     Two-pass LLM classifier for the reasoning/monitoring channel
  file_tamper/               Audit of every file-modifying episode; harness parsing-bug detection
  heuristic_review/          Sandboxed tool-call replay reconstruction
```

## Setup

Requires **Docker** (a container per problem), [`uv`](https://github.com/astral-sh/uv),
and an **OpenRouter API key** (every agent and judge model routes through
OpenRouter).

```bash
git clone https://github.com/wiser-human-experimental/evilgenie-escalation.git
cd evilgenie-escalation
uv venv && source .venv/bin/activate && uv sync

cp .env.example .env        # then add OPENROUTER_API_KEY
```

Runs used `inspect-ai==0.3.213` and `datasets==3.6.0`; `uv sync` resolves a
compatible set. Problem data is pulled from the
`livecodebench/code_generation_lite` HF dataset (release `v5_v6`) at run time —
nothing to download manually.

## Running

```bash
# One condition, directly:
uv run inspect eval evilgenie_task.py \
  --model openrouter/openai/gpt-5.3-codex \
  --model-role judge=openrouter/openai/gpt-4o \
  -T condition=A -T problem_set=ambiguous -T include_text_editor=false \
  --epochs 5 --max-connections 4 \
  --log-dir results/gpt-5.3-codex/ambiguous_A
```

`condition` ∈ `A|B|C|D|E`; `problem_set` ∈ `ambiguous|unambiguous`. The
wrappers drive the full design per model (`<model-key>` is a key in
`models.py`):

```bash
bash run_ambiguous_gate.sh <model-key>       # Condition A
bash run_bcd_factorial.sh  <model-key>       # Conditions B, C, D
bash run_unambiguous_negative_control.sh     # A/B/C/D on the control set
```

On a laptop keep `--max-connections` low (2–4) and run one model at a time —
each sample holds a Docker container for the length of the episode.

## Analysis

```bash
# Compare two conditions from their .eval logs
uv run python -c "from analyse_results import analyse; \
  analyse('results/<m>/ambiguous_A', 'results/<m>/ambiguous_D', \
          label_a='A (baseline)', label_b='D (combined)')"

# Rebuild the row-level table and the summary tables
uv run python build_per_run_data_csv.py
uv run python build_reward_hacking_summary.py

# Browse every run locally (no external services)
uv run python viewer.py        # http://localhost:8765
```

The verification pipeline (corrected solve rates, defect-disclosure rates by
channel) is documented in [`verification/README.md`](verification/README.md).
Each of the 9 answer validators self-tests against the official problem
samples:

```bash
uv run python -m verification.validators.abc397d_validator
```

## Data

The raw Inspect `.eval` logs (agent transcripts, judge reasoning, per-test
results — ~9 GB across ~130 runs and ~20 models) and all derived data
(row-level results CSV, per-sample corrected verdicts, disclosure-channel
tables, cost tables) are published as a Hugging Face dataset:

**[to be added]**

Download it into `./data/` (or `./results/` for the `.eval` logs) to re-run
the analysis and verification scripts. Every statistic in the paper can be
regenerated from the row-level results table alone, without the full logs.

`build_hf_dataset.py` regenerates that release (stripped transcripts + tables)
from a local copy of the raw `.eval` logs.

## Citing

```bibtex
@techreport{gomez2026escalation,
  title       = {Can escalation channels redirect reward hacking toward defect disclosure?},
  author      = {Gomez, F.},
  institution = {Wiser Human},
  year        = {2026},
  url         = {[to be added]}
}
```

The paper link will be added once it is posted. Please also cite the EvilGenie
benchmark (arXiv:2511.21654), the escalation-channel work it builds on (Gomez
et al., arXiv:2510.05192), and LiveCodeBench.

## License

MIT — see [LICENSE](LICENSE). This repository extends
[EvilGenie](https://github.com/JonathanGabor/evilgenie_inspect) (MIT,
© 2026 Jonathan Gabor); see [NOTICE](NOTICE) for the retained attribution.

## Acknowledgements

Built on [EvilGenie](https://github.com/JonathanGabor/evilgenie_inspect) by
Jonathan Gabor, Jayson Lynch and Jonathan Rosenfeld, and on
[Inspect](https://inspect.ai-safety-institute.org.uk/).

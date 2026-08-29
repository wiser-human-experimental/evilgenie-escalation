"""Re-score a specific problem from an ALREADY-KNOWN solution, without
re-running the agent -- for samples where the agent genuinely produced a
working solution.py (visible in its transcript) but the scoring step itself
failed (e.g. the judge call timed out under degraded network), so the real
answer is sitting in the errored transcript rather than needing regeneration.

The solver here makes ZERO model calls -- it writes the known solution
straight into the sandbox and ends the episode immediately. Everything else
(visible + holdout test execution, file-tamper check, LLM judge call) is the
real, unmodified categorical_scorer -- identical to what a normal agent run
would have produced feeding the same final solution.py into the same scorer.
Only the (cheap) judge call and free sandboxed test execution actually run;
no cost is re-paid for the model whose transcript already did the real work.

Usage:
  uv run inspect eval rescue_task.py --model <original_model_id> \
      -T problem_id=<id> -T condition=<A|B|C|D|E> -T solution_path=<path> \
      --epochs 1 --model-role judge=openrouter/openai/gpt-4o \
      --log-dir results/<model_key>/ambiguous_<condition>

--model is required by the CLI but is never actually called (solve() never
invokes generate()) -- use the ORIGINAL model's id so the log's model_usage
stays keyed correctly and honestly reports zero tokens spent, rather than
implying a fresh call happened.

Writes into the SAME log-dir as normal runs so `load_inspect_logs` (and
verify_and_backfill-style checks) pick the recovered sample up automatically
alongside the model's other real samples -- no separate merge step needed.
"""
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai._eval.task.epochs import Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from constants import DEFAULT_PROBLEM_TIMEOUT, SOLUTION_FILE
from categorical_scorer import scorer
from dataset import reward_hacking_dataset
from policy import with_policy_file

DOCKERFILE_PATH = str(Path(__file__).parent / "Dockerfile")


@solver
def inject_known_solution(solution_path: str) -> Solver:
    code = Path(solution_path).read_text()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await sandbox().write_file(SOLUTION_FILE, code)
        return state

    return solve


@task
def rescue(
    problem_id: str,
    solution_path: str,
    condition: Literal["A", "B", "C", "D", "E"] = "A",
) -> Task:
    dataset = reward_hacking_dataset(
        difficulty="competition",
        limit=None,
        seed=42,
        dataset_source="livecodebench",
        livecodebench_release="v5_v6",
        problem_ids=[problem_id],
    )
    if condition in ("C", "D"):
        dataset = with_policy_file(dataset)

    return Task(
        dataset=dataset,
        solver=inject_known_solution(solution_path),
        scorer=scorer(enable_llm_judge=True),
        time_limit=DEFAULT_PROBLEM_TIMEOUT,
        sandbox=("docker", DOCKERFILE_PATH),
        config=GenerateConfig(max_tool_output=100 * 1024 * 1024),
        epochs=Epochs(1, reducer=[]),
        fail_on_error=False,
    )

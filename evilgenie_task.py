"""EvilGenie x Escalation Channel — consolidated parameterised task.

Single @task replacing the four per-condition GPT-5-specific task files
(reward_hacking_gpt5.py, reward_hacking_escalation_gpt5.py,
reward_hacking_policy_gpt5.py, reward_hacking_policy_escalation_gpt5.py).
Per-model behaviour (e.g. text_editor schema incompatibility) is now a
parameter (``include_text_editor``), not a separate file — the model registry
(models.py, Stage 2 §5) sets it per model rather than requiring a new file per
model. Spec: the paper §3-4.

System messages are imported (not re-typed) from problems.py and policy.py,
where they were already hash-verified byte-identical to the Stage 1 originals
— importing avoids any risk of transcription drift that hand-copying the
strings (as the doc's template literally shows) would risk.
"""

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai._eval.task.epochs import Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Solver, basic_agent, system_message
from inspect_ai.tool import text_editor

from constants import DEFAULT_MAX_MESSAGES, DEFAULT_PROBLEM_TIMEOUT
from categorical_scorer import scorer
from dataset import reward_hacking_dataset
from tools import safe_bash, safe_python
from escalation_tool import submit_escalation_report
from policy import (
    with_policy_file,
    SYSTEM_MESSAGE_POLICY_ONLY,
    SYSTEM_MESSAGE_POLICY_ESCALATION,
)
from problems import (
    AMBIGUOUS_PROBLEM_IDS,
    UNAMBIGUOUS_PROBLEM_IDS,
    DEFAULT_SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_WITH_ESCALATION,
    SYSTEM_MESSAGE_DIRECTIVE_ONLY,
)

DOCKERFILE_PATH = str(Path(__file__).parent / "Dockerfile")

# ── System messages (byte-identical to Stage 1 — imported, not re-typed) ──
SYSTEM_MESSAGES = {
    "A": DEFAULT_SYSTEM_MESSAGE,
    "B": SYSTEM_MESSAGE_WITH_ESCALATION,
    "C": SYSTEM_MESSAGE_POLICY_ONLY,
    "D": SYSTEM_MESSAGE_POLICY_ESCALATION,
    "E": SYSTEM_MESSAGE_DIRECTIVE_ONLY,
}

# ── Condition -> component membership (spec §3.3) ──────────────────────────
CONDITIONS_WITH_ESCALATION = {"B", "D"}
CONDITIONS_WITH_POLICY = {"C", "D"}

PROBLEM_SETS = {
    "ambiguous": AMBIGUOUS_PROBLEM_IDS,
    "unambiguous": UNAMBIGUOUS_PROBLEM_IDS,
}


@task
def evilgenie(
    condition: Literal["A", "B", "C", "D", "E"] = "A",
    problem_set: Literal["ambiguous", "unambiguous"] = "ambiguous",
    include_text_editor: bool = False,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    timeout_per_problem: int = DEFAULT_PROBLEM_TIMEOUT,
    enable_llm_judge: bool = True,
    seed: int = 42,
    dataset_source: Literal["apps", "livecodebench"] = "livecodebench",
    livecodebench_release: str = "v5_v6",
) -> Task:
    """EvilGenie reward hacking benchmark with escalation channel.

    Args:
        condition: A=baseline, B=escalation only, C=policy only, D=combined,
            E=directive only (system-prompt anti-reward-hacking instruction +
            verbatim policy definition, no policy file in the sandbox, no
            escalation tool -- added 2026-07-27 to isolate whether C's effect
            requires the file or just the sentence)
        problem_set: 'ambiguous' (9 Stage 1 problems) or 'unambiguous'
            (9 negative-control problems, locked 2026-07-14)
        include_text_editor: Whether to include the text_editor tool. Per-model
            flag (models.py) — GPT-5's strict schema rejects text_editor's
            function signature ('file_text' missing from 'required'), so this
            must be False for GPT-5.3-Codex/GPT-5.5. True for other families
            unless a plumbing pilot finds an equivalent issue.
        max_messages: Maximum agent messages per problem
        timeout_per_problem: Time limit per problem in seconds
        enable_llm_judge: Enable LLM judge classification
        seed: Random seed (only affects visible/holdout split when
            problem_ids are given by name, which they always are here)
        dataset_source: 'apps' or 'livecodebench'
        livecodebench_release: LCB release version

    Epoch count is controlled by Inspect's ``--epochs N`` CLI flag (as in every
    Stage 1 run), not a task parameter — the internal ``Epochs(1, reducer=[])``
    below is the exact, proven Stage 1 config; it is not itself the epoch
    count. Introducing a second, untested epoch-count mechanism (a `-T
    epochs=N` parameter, per the doc's literal template) right before a real
    run was deliberately avoided in favour of the mechanism already verified
    across all of Stage 1.
    """
    problem_ids = PROBLEM_SETS[problem_set]

    dataset = reward_hacking_dataset(
        difficulty="competition",
        limit=None,
        seed=seed,
        dataset_source=dataset_source,
        livecodebench_release=livecodebench_release,
        problem_ids=problem_ids,
    )

    if condition in CONDITIONS_WITH_POLICY:
        dataset = with_policy_file(dataset)

    tools = [safe_bash(timeout=180), safe_python()]
    if include_text_editor:
        tools.append(text_editor())
    if condition in CONDITIONS_WITH_ESCALATION:
        tools.append(submit_escalation_report())

    solver: Solver = basic_agent(
        init=system_message(SYSTEM_MESSAGES[condition]),
        tools=tools,
        max_messages=max_messages,
    )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorer(enable_llm_judge=enable_llm_judge),
        time_limit=timeout_per_problem,
        sandbox=("docker", DOCKERFILE_PATH),
        config=GenerateConfig(max_tool_output=100 * 1024 * 1024),
        epochs=Epochs(1, reducer=[]),
        fail_on_error=False,
    )

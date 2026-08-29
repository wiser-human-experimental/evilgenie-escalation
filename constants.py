"""Constants for reward hacking benchmark."""

# Workspace configuration
WORKSPACE_DIR = "/home/agent/workspace"

# File names
PROBLEM_FILE = "problem.md"
SOLUTION_FILE = "solution.py"
TEST_CASES_FILE = "test_cases.json"
TEST_RUNNER_FILE = "test.py"

# Test suite timeout
DEFAULT_TEST_TIMEOUT = 120

# Agent execution limits
DEFAULT_MAX_MESSAGES = 500
# Inspect derives the scorer's own time budget as half of this value
# (scoring_time_limit = time_limit / 2, inspect_ai/_eval/task/run.py) -- the
# judge call (a separate request to GPT-4o after the agent finishes) was
# getting killed at 300s under hotspot network latency, 2026-07-27. Bumped
# 600 -> 1200 to give the judge call 600s of slack instead of 300s.
DEFAULT_PROBLEM_TIMEOUT = 1200

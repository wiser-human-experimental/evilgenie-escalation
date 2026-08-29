#!/bin/bash
# Stage 2 — Unambiguous negative-control run: all 4 conditions on Set 1.
# Purpose: rule out "the mitigations reduce hacking generally, not specifically
# ambiguity-driven hacking" (paper §9.2) and check for over-escalation /
# solve-rate harm from B/C/D on problems where hacking is unnecessary.
#
# 9 problems (UNAMBIGUOUS_PROBLEM_IDS, problems.py) x 5 epochs x 4 conditions
# = 180 samples total. Condition A is run fresh here (not reused from the
# ambiguous-set baseline) so all 4 conditions are directly comparable on the
# exact same problem set, run the same way, at the same time.
#
# Run:     nohup bash run_unambiguous_negative_control.sh > logs/unambig_master.log 2>&1 &
# Monitor: tail -f logs/unambig_master.log

set -e
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; source .env; set +a

mkdir -p results/unambig_a_baseline results/unambig_b_escalation \
         results/unambig_c_policy results/unambig_d_combined logs

echo "$(date): ============================================" | tee -a logs/unambig_master.log
echo "$(date): Unambiguous negative-control run — A/B/C/D" | tee -a logs/unambig_master.log
echo "$(date): 9 problems x 5 epochs x 4 conditions = 180 evals" | tee -a logs/unambig_master.log
echo "$(date): ============================================" | tee -a logs/unambig_master.log

# All 4 conditions now run through the single consolidated evilgenie_task.py
# (the paper §3-4) instead of 4 separate
# per-condition GPT-5-specific files. include_text_editor=false is the
# GPT-5.3-Codex model-registry setting (its strict schema rejects
# text_editor's signature) — other models will set this true via models.py.

# ── Condition A — Baseline ────────────────────────────────────────
echo "$(date): Phase 1 — Condition A (baseline)..." | tee -a logs/unambig_master.log
uv run inspect eval evilgenie_task.py \
    --model openrouter/openai/gpt-5.3-codex \
    -T condition=A -T problem_set=unambiguous -T include_text_editor=false \
    --max-connections 3 \
    --epochs 5 \
    --model-role judge=openrouter/openai/gpt-4o \
    --log-dir results/unambig_a_baseline \
    2>&1 | tee logs/unambig_a.log
echo "$(date): Condition A complete." | tee -a logs/unambig_master.log

# ── Condition B — Escalation only ─────────────────────────────────
echo "$(date): Phase 2 — Condition B (escalation only)..." | tee -a logs/unambig_master.log
uv run inspect eval evilgenie_task.py \
    --model openrouter/openai/gpt-5.3-codex \
    -T condition=B -T problem_set=unambiguous -T include_text_editor=false \
    --max-connections 3 \
    --epochs 5 \
    --model-role judge=openrouter/openai/gpt-4o \
    --log-dir results/unambig_b_escalation \
    2>&1 | tee logs/unambig_b.log
echo "$(date): Condition B complete." | tee -a logs/unambig_master.log

# ── Condition C — Policy only ──────────────────────────────────────
echo "$(date): Phase 3 — Condition C (policy only)..." | tee -a logs/unambig_master.log
uv run inspect eval evilgenie_task.py \
    --model openrouter/openai/gpt-5.3-codex \
    -T condition=C -T problem_set=unambiguous -T include_text_editor=false \
    --max-connections 3 \
    --epochs 5 \
    --model-role judge=openrouter/openai/gpt-4o \
    --log-dir results/unambig_c_policy \
    2>&1 | tee logs/unambig_c.log
echo "$(date): Condition C complete." | tee -a logs/unambig_master.log

# ── Condition D — Combined (policy + escalation) ──────────────────
echo "$(date): Phase 4 — Condition D (combined)..." | tee -a logs/unambig_master.log
uv run inspect eval evilgenie_task.py \
    --model openrouter/openai/gpt-5.3-codex \
    -T condition=D -T problem_set=unambiguous -T include_text_editor=false \
    --max-connections 3 \
    --epochs 5 \
    --model-role judge=openrouter/openai/gpt-4o \
    --log-dir results/unambig_d_combined \
    2>&1 | tee logs/unambig_d.log
echo "$(date): Condition D complete." | tee -a logs/unambig_master.log

# ── Analysis (A vs B, A vs C, A vs D) ──────────────────────────────
echo "$(date): Phase 5 — Analysis..." | tee -a logs/unambig_master.log
uv run python3 -c "
from analyse_results import analyse
print('############ Unambiguous negative control: A vs B ############')
analyse('results/unambig_a_baseline', 'results/unambig_b_escalation', label_a='A (baseline)', label_b='B (escalation only)')
" 2>&1 | tee logs/unambig_analysis_A_vs_B.log

uv run python3 -c "
from analyse_results import analyse
print('############ Unambiguous negative control: A vs C ############')
analyse('results/unambig_a_baseline', 'results/unambig_c_policy', label_a='A (baseline)', label_b='C (policy only)')
" 2>&1 | tee logs/unambig_analysis_A_vs_C.log

uv run python3 -c "
from analyse_results import analyse
print('############ Unambiguous negative control: A vs D ############')
analyse('results/unambig_a_baseline', 'results/unambig_d_combined', label_a='A (baseline)', label_b='D (combined)')
" 2>&1 | tee logs/unambig_analysis_A_vs_D.log

echo "$(date): ============================================" | tee -a logs/unambig_master.log
echo "$(date): Unambiguous negative-control run COMPLETE" | tee -a logs/unambig_master.log
echo "$(date): Check: hack rate stays low across all 4; escalation rate ~0" | tee -a logs/unambig_master.log
echo "$(date): in B/D on solvable problems; solve rate doesn't degrade." | tee -a logs/unambig_master.log
echo "$(date): ============================================" | tee -a logs/unambig_master.log

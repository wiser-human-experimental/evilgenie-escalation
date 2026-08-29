#!/bin/bash
# Stage 6 — full B/C/D factorial for a single model (ambiguous set).
# Runs conditions B (escalation only), C (policy only), D (combined), each with
# the same 1-epoch-safety-then-4 pattern as the gate: 1 epoch first (catches
# escalation-tool schema rejection cheaply -- B/D add submit_escalation_report,
# untested for the new models), abort that condition if sample error rate >5%,
# else complete epochs 2-5 in place. Condition A already exists from the gate.
#
# Usage: bash run_bcd_factorial.sh <model-key-from-models.py>

set -e
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; source .env; set +a
set -o pipefail

MODEL_KEY="$1"
if [ -z "$MODEL_KEY" ]; then echo "Usage: bash run_bcd_factorial.sh <model-key>"; exit 1; fi

MODEL_ID=$(python3 -c "from models import MODELS; print(MODELS['$MODEL_KEY']['model_id'])")
INCLUDE_TE=$(python3 -c "from models import MODELS; print(str(MODELS['$MODEL_KEY']['include_text_editor']).lower())")
JUDGE=$(python3 -c "from models import MODELS; print(MODELS['$MODEL_KEY']['judge'])")
REASONING_HIST=$(python3 -c "from models import MODELS; print(MODELS['$MODEL_KEY'].get('reasoning_history',''))")
REASONING_FLAG=""; [ -n "$REASONING_HIST" ] && REASONING_FLAG="--reasoning-history $REASONING_HIST"
[ -z "$MODEL_ID" ] && { echo "Unknown model key: $MODEL_KEY"; exit 1; }

mkdir -p logs
MASTER="logs/bcd_${MODEL_KEY}.log"
echo "$(date): ==== ${MODEL_KEY} (${MODEL_ID}) — B/C/D factorial ====" | tee -a "$MASTER"

# NOTE: macOS default bash is 3.2 (no associative arrays) -- keep this portable.
for COND in B C D; do
    LOGDIR="results/${MODEL_KEY}/ambiguous_${COND}"
    mkdir -p "$LOGDIR"
    echo "$(date): --- Condition $COND ($LOGDIR) ---" | tee -a "$MASTER"

    echo "$(date): [$COND] Step 1 — 1 epoch (schema/tool-call check)..." | tee -a "$MASTER"
    set +e
    uv run inspect eval evilgenie_task.py \
        --model "$MODEL_ID" \
        -T condition=$COND -T problem_set=ambiguous -T include_text_editor=$INCLUDE_TE \
        --epochs 1 --max-connections ${MAX_CONN:-4} \
        --model-role judge=$JUDGE $REASONING_FLAG \
        --log-dir "$LOGDIR" \
        2>&1 | tee logs/bcd_${MODEL_KEY}_${COND}_step1.log
    S1=$?
    set -e
    echo "$(date): [$COND] Step 1 exit: $S1" | tee -a "$MASTER"
    if [ $S1 -ne 0 ]; then
        echo "$(date): [$COND] STEP 1 FAILED — skipping this condition." | tee -a "$MASTER"; continue
    fi
    set +e
    ERR=$(grep -oE "executed samples \([0-9]+%\) had errors" logs/bcd_${MODEL_KEY}_${COND}_step1.log | grep -oE "[0-9]+" | head -1)
    set -e
    ERR="${ERR:-0}"
    echo "$(date): [$COND] Step 1 error rate: ${ERR}%" | tee -a "$MASTER"
    if [ "$ERR" -gt 5 ]; then
        echo "$(date): [$COND] error rate ${ERR}% > 5% — skipping epochs 2-5. Investigate." | tee -a "$MASTER"; continue
    fi

    echo "$(date): [$COND] Step 1 clean — completing epochs 2-5..." | tee -a "$MASTER"
    uv run inspect eval evilgenie_task.py \
        --model "$MODEL_ID" \
        -T condition=$COND -T problem_set=ambiguous -T include_text_editor=$INCLUDE_TE \
        --epochs 4 --max-connections ${MAX_CONN:-4} \
        --model-role judge=$JUDGE $REASONING_FLAG \
        --log-dir "$LOGDIR" \
        2>&1 | tee logs/bcd_${MODEL_KEY}_${COND}_step2.log
    echo "$(date): [$COND] complete — n=45 in $LOGDIR" | tee -a "$MASTER"
done

echo "$(date): ==== ${MODEL_KEY} B/C/D COMPLETE ====" | tee -a "$MASTER"
echo "$(date): Analysis (A vs each) — A already in results/${MODEL_KEY}/ambiguous_A" | tee -a "$MASTER"
for COND in B C D; do
    case "$COND" in
        B) LABEL="B (escalation only)";;
        C) LABEL="C (policy only)";;
        D) LABEL="D (combined)";;
    esac
    uv run python3 -c "
from analyse_results import analyse
print('#### ${MODEL_KEY}: A vs ${COND} ####')
analyse('results/${MODEL_KEY}/ambiguous_A', 'results/${MODEL_KEY}/ambiguous_${COND}', label_a='A (baseline)', label_b='${LABEL}')
" 2>&1 | tee logs/bcd_${MODEL_KEY}_analysis_A_vs_${COND}.log
done

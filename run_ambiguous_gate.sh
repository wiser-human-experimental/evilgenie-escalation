#!/bin/bash
# Stage 4 — ambiguous baseline (Gate G1) for a single model.
# Condition A only, on the ambiguous set. 1 epoch first (cheap schema/tool-call
# check, ~1/5 the cost); if it passes, completes epochs 2-5 in the SAME
# log-dir -- nothing from the check is wasted, matches the redesigned Stage 0.
#
# Usage: bash run_ambiguous_gate.sh <model-key-from-models.py>

set -e
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; source .env; set +a
set -o pipefail

MODEL_KEY="$1"
if [ -z "$MODEL_KEY" ]; then
    echo "Usage: bash run_ambiguous_gate.sh <model-key>"
    exit 1
fi

MODEL_ID=$(python3 -c "from models import MODELS; print(MODELS['$MODEL_KEY']['model_id'])")
INCLUDE_TE=$(python3 -c "from models import MODELS; print(str(MODELS['$MODEL_KEY']['include_text_editor']).lower())")
JUDGE=$(python3 -c "from models import MODELS; print(MODELS['$MODEL_KEY']['judge'])")
# Optional per-model reasoning_history (Gemini needs 'none' -- see models.py).
# Empty string when unset, so REASONING_FLAG expands to nothing below.
REASONING_HIST=$(python3 -c "from models import MODELS; print(MODELS['$MODEL_KEY'].get('reasoning_history',''))")
REASONING_FLAG=""
if [ -n "$REASONING_HIST" ]; then
    REASONING_FLAG="--reasoning-history $REASONING_HIST"
fi

if [ -z "$MODEL_ID" ]; then
    echo "Unknown model key: $MODEL_KEY"
    exit 1
fi

LOGDIR="results/${MODEL_KEY}/ambiguous_A"
mkdir -p "$LOGDIR" logs

echo "$(date): ==== ${MODEL_KEY} (${MODEL_ID}) — Stage 4 gate ====" | tee -a logs/gate_${MODEL_KEY}.log

echo "$(date): Step 1 — 1 epoch (9 problems), schema/tool-call check..." | tee -a logs/gate_${MODEL_KEY}.log
set +e
uv run inspect eval evilgenie_task.py \
    --model "$MODEL_ID" \
    -T condition=A -T problem_set=ambiguous -T include_text_editor=$INCLUDE_TE \
    --epochs 1 \
    --max-connections ${MAX_CONN:-4} \
    --model-role judge=$JUDGE \
    $REASONING_FLAG \
    --log-dir "$LOGDIR" \
    2>&1 | tee logs/gate_${MODEL_KEY}_step1.log
STEP1_EXIT=$?
set -e
echo "$(date): Step 1 exit code: $STEP1_EXIT" | tee -a logs/gate_${MODEL_KEY}.log

if [ $STEP1_EXIT -ne 0 ]; then
    echo "$(date): STEP 1 FAILED for ${MODEL_KEY} — aborting, NOT running epochs 2-5." | tee -a logs/gate_${MODEL_KEY}.log
    echo "$(date): ==== ${MODEL_KEY} ABORTED ====" | tee -a logs/gate_${MODEL_KEY}.log
    exit 1
fi

# Process exit code alone is NOT sufficient: fail_on_error=False (task config)
# means inspect eval exits 0 even when most samples errored -- found this the
# hard way with gemini-3.5-flash (55% sample error rate, "Error 400 -
# Corrupted thought", silently proceeded to epochs 2-5 before this check
# existed). Explicitly parse Inspect's own error-rate warning and enforce the
# pass criterion (tool-call failure rate <=5%) that was already documented but
# never actually checked.
# set +e around this: grep returning "no match" (exit 1) is the NORMAL,
# GOOD case here (0% errors -- no "had errors" line to find). With `set -e`
# + `set -o pipefail` active, that "no match" silently killed the whole
# script right after the exit-code line, before epochs 2-5 ever ran --
# found this the hard way too (claude-opus-4.8/claude-sonnet-5/gpt-5.6-luna
# all had clean 0%-error Step 1s but the script died anyway, immediately
# after printing "Step 1 exit code: 0", never reaching epochs 2-5).
set +e
ERROR_PCT=$(grep -oE "executed samples \([0-9]+%\) had errors" logs/gate_${MODEL_KEY}_step1.log \
            | grep -oE "[0-9]+" | head -1)
set -e
ERROR_PCT="${ERROR_PCT:-0}"
echo "$(date): Step 1 sample error rate: ${ERROR_PCT}%" | tee -a logs/gate_${MODEL_KEY}.log

if [ "$ERROR_PCT" -gt 5 ]; then
    echo "$(date): Step 1 error rate ${ERROR_PCT}% exceeds 5% threshold for ${MODEL_KEY}" | tee -a logs/gate_${MODEL_KEY}.log
    echo "$(date): — aborting, NOT running epochs 2-5. Investigate before retrying." | tee -a logs/gate_${MODEL_KEY}.log
    echo "$(date): ==== ${MODEL_KEY} ABORTED (error rate) ====" | tee -a logs/gate_${MODEL_KEY}.log
    exit 1
fi

echo "$(date): Step 1 passed (exit 0, error rate ${ERROR_PCT}% <= 5%) — completing epochs 2-5..." | tee -a logs/gate_${MODEL_KEY}.log
uv run inspect eval evilgenie_task.py \
    --model "$MODEL_ID" \
    -T condition=A -T problem_set=ambiguous -T include_text_editor=$INCLUDE_TE \
    --epochs 4 \
    --max-connections ${MAX_CONN:-4} \
    --model-role judge=$JUDGE \
    $REASONING_FLAG \
    --log-dir "$LOGDIR" \
    2>&1 | tee logs/gate_${MODEL_KEY}_step2.log

echo "$(date): ==== ${MODEL_KEY} COMPLETE — n=45 in ${LOGDIR} ====" | tee -a logs/gate_${MODEL_KEY}.log

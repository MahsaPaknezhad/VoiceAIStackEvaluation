#!/usr/bin/env bash
# Hyperparameter sweep for the MiniCPM audio-scorer REGRESSION-HEAD fine-tune.
# Runs finetune_audio_scorer_reg.py and writes under data/trained_models/sweep_reg/.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SCRIPT=src/audio_scoring/finetune_audio_scorer_reg.py
SWEEP_ROOT=data/trained_models/sweep_reg
mkdir -p "$SWEEP_ROOT" logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SNAP="$(dirname "$(find "$HOME/.cache/huggingface" -name 'model-00001-of-00004.safetensors' | head -1)")"
warm_cache() { [ -n "$SNAP" ] && cat "$SNAP"/*.safetensors > /dev/null 2>&1 || true; }

# name|LR|R|ALPHA|DROPOUT|TARGETS|PATIENCE|WD|HEAD_LR
# Grid tuned for the regression-head trainer. Beyond the LoRA/LR/target
# variations, it sweeps HEAD_LR (the regression head's own learning rate) since
# the head is what maps hidden states to scores and is likely more impactful
# than LoRA-only changes. Patience is a moderate 5 everywhere (the seed check
# showed convergence around epoch 9-13), and MAX_EPOCHS is capped at 20 for a
# reasonable per-config runtime. The generative sweep's redundant "patient"
# config is dropped.
CONFIGS=(
  "baseline|1e-4|16|32|0.05|attn|5|0.01|1e-3"
  "lr_lo|3e-5|16|32|0.05|attn|5|0.01|3e-4"
  "lr_hi|3e-4|16|32|0.05|attn|5|0.01|3e-3"
  "rank_hi|1e-4|32|64|0.05|attn|5|0.01|1e-3"
  "mlp|1e-4|16|32|0.05|attn_mlp|5|0.01|1e-3"
  "head_lr_lo|1e-4|16|32|0.05|attn|5|0.01|2e-4"
  "head_lr_hi|1e-4|16|32|0.05|attn|5|0.01|5e-3"
)

export AUDIO_SCORER_MAX_EPOCHS=20

echo "Regression sweep start: $(date)"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME LR R ALPHA DROPOUT TARGETS PATIENCE WD HEAD_LR <<< "$cfg"
  OUT="$SWEEP_ROOT/$NAME"
  LOG="logs/sweep_reg_${NAME}.log"
  if [ -f "$OUT/test_correlation.json" ]; then
    echo "[$NAME] already has results, skipping"
    continue
  fi
  echo "=== [$NAME] LR=$LR head_lr=$HEAD_LR r=$R alpha=$ALPHA targets=$TARGETS patience=$PATIENCE -> $OUT ==="
  mkdir -p "$OUT"
  warm_cache
  AUDIO_SCORER_OUTPUT_DIR="$OUT" \
  AUDIO_SCORER_LR="$LR" \
  AUDIO_SCORER_HEAD_LR="$HEAD_LR" \
  AUDIO_SCORER_LORA_R="$R" \
  AUDIO_SCORER_LORA_ALPHA="$ALPHA" \
  AUDIO_SCORER_LORA_DROPOUT="$DROPOUT" \
  AUDIO_SCORER_LORA_TARGETS="$TARGETS" \
  AUDIO_SCORER_PATIENCE="$PATIENCE" \
  AUDIO_SCORER_WEIGHT_DECAY="$WD" \
    "$PY" -u "$SCRIPT" > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$NAME] FAILED (rc=$rc). Tail of log:"
    tail -15 "$LOG"
  else
    echo "[$NAME] done."
  fi
done
echo "Regression sweep end: $(date)"

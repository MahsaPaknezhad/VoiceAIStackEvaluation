#!/usr/bin/env bash
# Stability check: run the winning regression-head config (baseline HPs) across
# several seeds. Each seed reshuffles the train/dev/test split, so this measures
# how stable the held-out correlation is across data partitions + training noise.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SCRIPT=src/audio_scoring/finetune_audio_scorer_reg.py
ROOT=data/trained_models/seedcheck_reg
mkdir -p "$ROOT" logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Baseline HPs (the config that scored mean|r|=0.659 at seed 42).
export AUDIO_SCORER_LR=1e-4
export AUDIO_SCORER_LORA_R=16
export AUDIO_SCORER_LORA_ALPHA=32
export AUDIO_SCORER_LORA_DROPOUT=0.05
export AUDIO_SCORER_LORA_TARGETS=attn
export AUDIO_SCORER_PATIENCE=4
export AUDIO_SCORER_WEIGHT_DECAY=0.01
export AUDIO_SCORER_MAX_EPOCHS=15

SEEDS=(1 7 123)

echo "Seed check start: $(date)"
for SEED in "${SEEDS[@]}"; do
  OUT="$ROOT/seed_${SEED}"
  LOG="logs/seedcheck_reg_seed${SEED}.log"
  if [ -f "$OUT/test_correlation.json" ]; then
    echo "[seed $SEED] already has results, skipping"; continue
  fi
  echo "=== [seed $SEED] -> $OUT ==="
  mkdir -p "$OUT"
  AUDIO_SCORER_OUTPUT_DIR="$OUT" AUDIO_SCORER_SEED="$SEED" \
    "$PY" -u "$SCRIPT" > "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[seed $SEED] FAILED rc=$rc"; tail -15 "$LOG"; } || echo "[seed $SEED] done."
done
echo "Seed check end: $(date)"

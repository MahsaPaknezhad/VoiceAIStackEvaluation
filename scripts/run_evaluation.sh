#!/bin/bash
# Unified evaluation runner — replaces individual test_*.sh scripts.
# Usage:
#   ./scripts/run_evaluation.sh <stt_config> <tts_config> [output_prefix]
#
# Examples:
#   ./scripts/run_evaluation.sh aws_transcribe cartesia
#   ./scripts/run_evaluation.sh whisper_turbo aws_polly my_experiment
#   ./scripts/run_evaluation.sh nvidia_riva nvidia_riva

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <stt_config_name> <tts_config_name> [output_prefix]"
    echo ""
    echo "Available STT configs:"
    ls evaluation_data/stt_configs/ | sed 's/_config\.json$//'
    echo ""
    echo "Available TTS configs:"
    ls evaluation_data/tts_configs/ | sed 's/_config\.json$//'
    exit 1
fi

STT_NAME="$1"
TTS_NAME="$2"
PREFIX="${3:-${STT_NAME}_${TTS_NAME}}"

DATASET="evaluation_data/datasets/voiceassistant_eval_dataset.json"
AUDIO_DIR="evaluation_data/datasets/audio_input"
STT_CONFIG="evaluation_data/stt_configs/${STT_NAME}_config.json"
TTS_CONFIG="evaluation_data/tts_configs/${TTS_NAME}_config.json"
OUTPUT="evaluation_output/${PREFIX}_results.json"
EVAL_OUTPUT="evaluation_output/${PREFIX}_evaluation.json"

# Validate configs exist
for f in "$STT_CONFIG" "$TTS_CONFIG" "$DATASET"; do
    if [ ! -f "$f" ]; then
        echo "Error: $f not found"
        exit 1
    fi
done

mkdir -p evaluation_output

echo "=================================="
echo "Evaluation: ${STT_NAME} + ${TTS_NAME}"
echo "=================================="

echo "Step 1: Running bot on dataset..."
python3 -m src.evaluation.voice_pipeline_evaluator \
    --dataset "$DATASET" \
    --audio-dir "$AUDIO_DIR" \
    --stt-config "$STT_CONFIG" \
    --tts-config "$TTS_CONFIG" \
    --output "$OUTPUT"

echo "Step 2: Evaluating results (WER + LLM judge)..."
python3 -m src.evaluation.metrics_calculator \
    --dataset "$DATASET" \
    --audio-dir "$AUDIO_DIR" \
    --results "$OUTPUT" \
    --output "$EVAL_OUTPUT"

echo "=================================="
echo "Results: $OUTPUT"
echo "Metrics: $EVAL_OUTPUT"
echo "=================================="

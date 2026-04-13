#!/bin/bash

# Run evaluation for NVIDIA Parakeet (STT) + NVIDIA Magpie (TTS)

set -e

# Use python3 directly
PYTHON_CMD=python3

echo "=================================="
echo "NVIDIA Parakeet + NVIDIA Magpie Evaluation"
echo "=================================="

DATASET="evaluation_data/datasets/voiceassistant_eval_dataset.json"
AUDIO_DIR="evaluation_data/datasets/audio_input"
STT_CONFIG="evaluation_data/stt_configs/nvidia_riva_config.json"
TTS_CONFIG="evaluation_data/tts_configs/nvidia_riva_config.json"
OUTPUT="evaluation_output/new/nvidia_parakeet_nvidia_magpie_results.json"
EVAL_OUTPUT="evaluation_output/new/nvidia_parakeet_nvidia_magpie_evaluation.json"

echo "STT: NVIDIA Parakeet"
echo "TTS: NVIDIA Magpie"
echo ""

echo "Step 1: Running bot on dataset..."
$PYTHON_CMD -m src.evaluation.voice_pipeline_evaluator \
    --dataset "$DATASET" \
    --audio-dir "$AUDIO_DIR" \
    --stt-config "$STT_CONFIG" \
    --tts-config "$TTS_CONFIG" \
    --output "$OUTPUT"

if [ $? -ne 0 ]; then
    echo "✗ Bot evaluation failed"
    exit 1
fi

echo "✓ Bot evaluation complete"
echo ""

echo "Step 2: Evaluating results (WER + LLM judge)..."
$PYTHON_CMD -m src.evaluation.metrics_calculator \
    --dataset "$DATASET" \
    --audio-dir "$AUDIO_DIR" \
    --results "$OUTPUT" \
    --output "$EVAL_OUTPUT"

if [ $? -ne 0 ]; then
    echo "✗ Evaluation failed"
    exit 1
fi

echo "✓ Evaluation complete"
echo ""

echo "=================================="
echo "Results saved to:"
echo "  Bot results: $OUTPUT"
echo "  Evaluation: $EVAL_OUTPUT"
echo "=================================="
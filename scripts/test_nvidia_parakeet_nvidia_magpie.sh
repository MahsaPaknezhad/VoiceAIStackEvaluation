#!/bin/bash
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

# Run evaluation for NVIDIA Parakeet (STT) + NVIDIA Magpie (TTS)

set -e

# Use python3 directly
PYTHON_CMD=python3

echo "=================================="
echo "NVIDIA Parakeet + NVIDIA Magpie Evaluation"
echo "=================================="

DATASET="data/voiceassistant_eval_new/voiceassistant_eval_dataset.json"
AUDIO_DIR="data/voiceassistant_eval_new/audio_input"
STT_CONFIG="data/stt_bot_configs/nvidia_riva_config.json"
TTS_CONFIG="data/tts_bot_configs/nvidia_riva_config.json"
OUTPUT="output/new/nvidia_parakeet_nvidia_magpie_results.json"
EVAL_OUTPUT="output/new/nvidia_parakeet_nvidia_magpie_evaluation.json"

echo "STT: NVIDIA Parakeet"
echo "TTS: NVIDIA Magpie"
echo ""

echo "Step 1: Running bot on dataset..."
$PYTHON_CMD src/evaluation/voice_pipeline_evaluator.py \
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
$PYTHON_CMD src/evaluation/metrics_calculator.py \
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
#!/bin/bash
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

set -e

DATASET="data/voiceassistant_eval_new/voiceassistant_eval_dataset.json"
AUDIO_DIR="data/voiceassistant_eval_new/audio_input"
STT_CONFIG="data/stt_bot_configs/whisper_small_config.json"
TTS_CONFIG="data/tts_bot_configs/aws_polly_config.json"
OUTPUT="output/test_whisper_small_polly_results.json"
EVAL_OUTPUT="output/test_whisper_small_polly_evaluation.json"

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

echo "Results: $OUTPUT"
echo "Metrics: $EVAL_OUTPUT"


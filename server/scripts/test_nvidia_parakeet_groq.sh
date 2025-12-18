#!/bin/bash

set -e

DATASET="evaluation_data/voiceassistant_eval_new/test_2samples.json"
AUDIO_DIR="evaluation_data/voiceassistant_eval_new/audio_input"
STT_CONFIG="evaluation_data/stt_bot_configs/nvidia_riva_config.json"
TTS_CONFIG="evaluation_data/tts_bot_configs/groq_config.json"
OUTPUT="evaluation_output/test_nvidia_parakeet_groq_results.json"
EVAL_OUTPUT="evaluation_output/test_nvidia_parakeet_groq_evaluation.json"

echo "Step 1: Running bot on dataset..."
python3 src/evaluation/voice_pipeline_evaluator.py \
    --dataset "$DATASET" \
    --audio-dir "$AUDIO_DIR" \
    --stt-config "$STT_CONFIG" \
    --tts-config "$TTS_CONFIG" \
    --output "$OUTPUT"

echo "Step 2: Evaluating results (WER + LLM judge)..."
python3 src/evaluation/metrics_calculator.py \
    --dataset "$DATASET" \
    --audio-dir "$AUDIO_DIR" \
    --results "$OUTPUT" \
    --output "$EVAL_OUTPUT"

echo "Results: $OUTPUT"
echo "Metrics: $EVAL_OUTPUT"
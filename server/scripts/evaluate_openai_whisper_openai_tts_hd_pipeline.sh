#!/bin/bash

# Run evaluation for OpenAI Whisper API (STT) + OpenAI TTS HD (TTS)

set -e

echo "=================================="
echo "OpenAI Whisper API + OpenAI TTS HD Evaluation"
echo "=================================="

DATASET="evaluation_data/voiceassistant_eval_small/voiceassistant_eval_dataset.json"
AUDIO_DIR="evaluation_data/voiceassistant_eval_small/audio_input"
STT_CONFIG="evaluation_data/stt_bot_configs/openai_whisper_config.json"
TTS_CONFIG="evaluation_data/tts_bot_configs/openai_tts_hd_config.json"
OUTPUT="evaluation_output/small/openai_whisper_openai_tts_hd_results.json"
EVAL_OUTPUT="evaluation_output/small/openai_whisper_openai_tts_hd_evaluation.json"

echo "STT: OpenAI Whisper API"
echo "TTS: OpenAI TTS HD"
echo ""

echo "Step 1: Running bot on dataset..."
python src/evaluation/voice_pipeline_evaluator.py \
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
python src/evaluation/metrics_calculator.py \
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

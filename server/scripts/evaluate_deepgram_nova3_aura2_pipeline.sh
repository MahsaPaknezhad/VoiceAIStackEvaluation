#!/bin/bash

# Run evaluation for Deepgram Nova-3 (STT) + Deepgram Aura-2 (TTS)

set -e

echo "=================================="
echo "Deepgram Nova-3 + Aura-2 Evaluation"
echo "=================================="

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$SERVER_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$SERVER_DIR/.env" | xargs)
fi

# Check if required environment variables are set
if [ -z "$DEEPGRAM_API_KEY" ]; then
    echo "Error: DEEPGRAM_API_KEY environment variable is not set"
    echo "Please add DEEPGRAM_API_KEY=your_key_here to $SERVER_DIR/.env"
    exit 1
fi

DATASET="evaluation_data/voiceassistant_eval_small/voiceassistant_eval_dataset.json"
AUDIO_DIR="evaluation_data/voiceassistant_eval_small/audio_input"
STT_CONFIG="evaluation_data/stt_bot_configs/deepgram_nova3_config.json"
TTS_CONFIG="evaluation_data/tts_bot_configs/deepgram_aura_config.json"
OUTPUT="evaluation_output/small/deepgram_nova3_aura2_results.json"
EVAL_OUTPUT="evaluation_output/small/deepgram_nova3_aura2_evaluation.json"

echo "STT: Deepgram Nova-3"
echo "TTS: Deepgram Aura-2"
echo ""

# Change to server directory
cd "$SERVER_DIR"

# Step 1: Run bot evaluation
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

# Step 2: Evaluate results
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

#!/bin/bash

# Run evaluation for all STT and TTS service combinations

set -e

DATASET="evaluation_data/voiceassistant_eval/voiceassistant_eval_dataset.json"
AUDIO_DIR="evaluation_data/voiceassistant_eval/audio_input"
RESULTS_DIR="evaluation_data/evaluation_results"

mkdir -p "$RESULTS_DIR"

echo "=================================="
echo "Running All Service Evaluations"
echo "=================================="

# Get all STT and TTS configs
STT_CONFIGS=(evaluation_data/bot_configs/*.json)
TTS_CONFIGS=(evaluation_data/tts_bot_configs/*.json)

echo "Found ${#STT_CONFIGS[@]} STT services"
echo "Found ${#TTS_CONFIGS[@]} TTS services"
echo "Total combinations: $((${#STT_CONFIGS[@]} * ${#TTS_CONFIGS[@]}))"
echo ""

# Counter
count=0
total=$((${#STT_CONFIGS[@]} * ${#TTS_CONFIGS[@]}))

# Loop through all combinations
for stt_config in "${STT_CONFIGS[@]}"; do
    stt_name=$(basename "$stt_config" .json)
    
    for tts_config in "${TTS_CONFIGS[@]}"; do
        tts_name=$(basename "$tts_config" .json)
        
        count=$((count + 1))
        
        echo "[$count/$total] Testing: $stt_name + $tts_name"
        
        output_file="$RESULTS_DIR/${stt_name}_${tts_name}_results.json"
        eval_file="$RESULTS_DIR/${stt_name}_${tts_name}_evaluation.json"
        
        # Run bot evaluation
        python run_voiceassistant_eval.py \
            --dataset "$DATASET" \
            --audio-dir "$AUDIO_DIR" \
            --stt-config "$stt_config" \
            --tts-config "$tts_config" \
            --output "$output_file" \
            2>&1 | tee "$RESULTS_DIR/${stt_name}_${tts_name}.log"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Bot evaluation success"
            
            # Run quality evaluation
            python evaluate_voiceassistant.py \
                --results "$output_file" \
                --output "$eval_file" \
                2>&1 | tee -a "$RESULTS_DIR/${stt_name}_${tts_name}.log"
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Quality evaluation success"
            else
                echo "  ✗ Quality evaluation failed"
            fi
        else
            echo "  ✗ Bot evaluation failed (check log)"
        fi
        echo ""
    done
done

echo "=================================="
echo "Evaluation Complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=================================="
echo ""
echo "Files generated per combination:"
echo "  - <stt>_<tts>_results.json (bot outputs)"
echo "  - <stt>_<tts>_evaluation.json (WER + quality scores)"
echo "  - <stt>_<tts>.log (execution log)"

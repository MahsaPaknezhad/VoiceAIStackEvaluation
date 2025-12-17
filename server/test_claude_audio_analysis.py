#!/usr/bin/env python3

import librosa
import numpy as np
import boto3
import json

def analyze_audio_with_claude():
    try:
        # Load audio and extract metrics
        audio_path = "/home/ubuntu/projects/nab-eba-merged/server/evaluation_output/tts_audio/aws_transcribe_aws_polly/listening_general_0_response.wav"
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract key metrics
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_valid = f0[f0 > 0]
        pitch_mean = np.mean(f0_valid) if len(f0_valid) > 0 else 0
        pitch_std = np.std(f0_valid) if len(f0_valid) > 0 else 0
        
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        duration = len(y) / sr
        
        # Create analysis prompt
        metrics_text = f"""Audio Quality Metrics:
- Duration: {duration:.2f} seconds
- Pitch Mean: {pitch_mean:.1f} Hz
- Pitch Std: {pitch_std:.1f} Hz  
- Energy Mean: {energy_mean:.4f}
- Energy Std: {energy_std:.4f}
- Spectral Centroid: {spectral_centroid:.1f} Hz
- Zero Crossing Rate: {zero_crossing_rate:.4f}

Based on these audio metrics, rate the voice naturalness from 1-10 and provide reasoning. Consider:
- Pitch variation (good: 10-30 Hz std)
- Energy consistency 
- Spectral characteristics
- Overall human-likeness

Respond in JSON: {{"naturalness_score": X, "reasoning": "explanation"}}"""

        # Use Claude Haiku
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = bedrock.converse(
            modelId='anthropic.claude-3-5-haiku-20241022-v1:0',
            messages=[
                {
                    "role": "user",
                    "content": [{"text": metrics_text}]
                }
            ],
            inferenceConfig={"maxTokens": 200, "temperature": 0.3}
        )
        
        result_text = response["output"]["message"]["content"][0]["text"]
        print(f"🎯 Claude analysis: {result_text}")
        
        # Parse JSON response
        try:
            result = json.loads(result_text)
            score = result.get("naturalness_score", 0)
            reasoning = result.get("reasoning", "")
            print(f"✅ Naturalness Score: {score}/10")
            print(f"✅ Reasoning: {reasoning}")
        except:
            print("⚠️ Response not in JSON format")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    result = analyze_audio_with_claude()
    if result:
        print("\n🎉 Claude + librosa audio analysis works!")
    else:
        print("\n💥 Analysis failed")

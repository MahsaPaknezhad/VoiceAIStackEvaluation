"""
Voice quality evaluation for TTS output.
Measures fluency, naturalness, and tone.
Includes LLM-based perceptual evaluation.
"""

import librosa
import numpy as np
from typing import Dict
import soundfile as sf
import base64
import asyncio
from strands import Agent
from strands.models import BedrockModel
import json
from loguru import logger
import os
import boto3


class VoiceQualityEvaluator:
    
    def __init__(self, sample_rate: int = 16000, use_llm_judge: bool = False, 
                 use_nova_sonic: bool = False):
        self.sample_rate = sample_rate
        self.use_llm_judge = use_llm_judge
        self.use_nova_sonic = use_nova_sonic
        
        if use_llm_judge:
            self.llm_judge = self._create_llm_judge()
        
        if use_nova_sonic:
            self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    
    def _create_llm_judge(self) -> Agent:
        """Create LLM agent for voice quality evaluation"""
        model = BedrockModel(model="anthropic.claude-haiku-4-5-20251001-v1:0")
        return Agent(
            name="VoiceQualityJudge",
            model=model,
            instruction="""You are an expert speech quality evaluator. Listen to the audio and evaluate:

1. **Fluency** (0-10): How smooth and natural is the speech flow? Are there awkward pauses or robotic rhythm?
2. **Naturalness** (0-10): Does it sound like a real human? Is the prosody and intonation natural?
3. **Tone** (0-10): Is the voice pleasant and clear? Good warmth and emotional expression?
4. **Overall** (0-10): Overall voice quality and listenability.

Return ONLY valid JSON:
{
    "fluency": <0-10>,
    "naturalness": <0-10>,
    "tone": <0-10>,
    "overall": <0-10>,
    "reasoning": "<brief explanation of scores>"
}"""
        )
    
    async def evaluate_with_nova_sonic(self, audio_path: str) -> Dict:
        """
        Evaluate voice quality using AWS Bedrock Nova Sonic 2.
        Nova Sonic 2 can analyze audio directly.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with Nova Sonic evaluation scores
        """
        if not self.use_nova_sonic:
            return {}
        
        try:
            # Read and encode audio
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Initialize Bedrock client
            bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
            
            # Prompt for voice quality evaluation
            prompt = """Analyze this speech audio and evaluate its quality on these dimensions (0-10 scale):

1. Fluency: How smooth and natural is the speech flow? Any awkward pauses or robotic rhythm?
2. Naturalness: Does it sound like a real human? Natural prosody and intonation?
3. Tone: Is the voice pleasant, clear, and well-articulated?
4. Overall: Overall voice quality and listenability.

Return ONLY valid JSON:
{
    "fluency": <0-10>,
    "naturalness": <0-10>,
    "tone": <0-10>,
    "overall": <0-10>,
    "reasoning": "<brief explanation>"
}"""
            
            # Call Nova Sonic 2
            response = bedrock.invoke_model(
                modelId='amazon.nova-2-sonic-v1:0',
                body=json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "audio": {
                                        "format": "wav",
                                        "source": {
                                            "bytes": audio_b64
                                        }
                                    }
                                },
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "inferenceConfig": {
                        "max_new_tokens": 500,
                        "temperature": 0.3
                    }
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['output']['message']['content'][0]['text']
            
            # Extract JSON from response
            result = json.loads(content)
            
            return {
                "nova_sonic_fluency": result.get("fluency", 0),
                "nova_sonic_naturalness": result.get("naturalness", 0),
                "nova_sonic_tone": result.get("tone", 0),
                "nova_sonic_overall": result.get("overall", 0),
                "nova_sonic_reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Nova Sonic 2 evaluation failed: {e}")
            return {
                "nova_sonic_fluency": 0,
                "nova_sonic_naturalness": 0,
                "nova_sonic_tone": 0,
                "nova_sonic_overall": 0,
                "nova_sonic_reasoning": f"Error: {str(e)}"
            }
    
    async def evaluate_with_llm(self, audio_path: str, transcript: str = "") -> Dict:
        """
        Evaluate voice quality using LLM judge.
        
        Args:
            audio_path: Path to audio file
            transcript: Optional transcript of what's being said
            
        Returns:
            Dict with LLM scores
        """
        if not self.use_llm_judge:
            return {}
        
        try:
            # Read audio file and encode to base64
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Create prompt
            prompt = f"Evaluate this speech audio for voice quality."
            if transcript:
                prompt += f"\n\nTranscript: {transcript}"
            
            # Note: Claude doesn't support audio input yet, so we'll use audio features as proxy
            # For now, describe the audio characteristics
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features to describe
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            f0_valid = f0[f0 > 0]
            pitch_mean = np.mean(f0_valid) if len(f0_valid) > 0 else 0
            
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            speaking_rate = tempo / 60.0 * 2
            
            rms = librosa.feature.rms(y=y)[0]
            energy_std = np.std(rms)
            
            # Describe audio characteristics to LLM
            audio_description = f"""Audio characteristics:
- Average pitch: {pitch_mean:.1f} Hz
- Speaking rate: {speaking_rate:.1f} syllables/second
- Energy variation: {energy_std:.3f}
- Duration: {len(y)/sr:.1f} seconds
"""
            if transcript:
                audio_description += f"- Content: {transcript}"
            
            full_prompt = f"{prompt}\n\n{audio_description}\n\nBased on these characteristics, evaluate the voice quality."
            
            response = await self.llm_judge.run(full_prompt)
            result = json.loads(response)
            
            return {
                "llm_fluency": result.get("fluency", 0),
                "llm_naturalness": result.get("naturalness", 0),
                "llm_tone": result.get("tone", 0),
                "llm_overall": result.get("overall", 0),
                "llm_reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return {
                "llm_fluency": 0,
                "llm_naturalness": 0,
                "llm_tone": 0,
                "llm_overall": 0,
                "llm_reasoning": f"Evaluation failed: {str(e)}"
            }
    
    def evaluate(self, audio_path: str) -> Dict:
        """
        Evaluate voice quality metrics (synchronous).
        
        Returns:
            Dict with fluency, naturalness, and tone scores
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Calculate metrics
        fluency = self._calculate_fluency(y, sr)
        naturalness = self._calculate_naturalness(y, sr)
        tone = self._calculate_tone(y, sr)
        
        return {
            "fluency": fluency,
            "naturalness": naturalness,
            "tone": tone,
            "overall_quality": (fluency["score"] + naturalness["score"] + tone["score"]) / 3
        }
    
    async def evaluate_async(self, audio_path: str, transcript: str = "") -> Dict:
        """
        Evaluate voice quality with optional LLM judge and Nova Sonic (asynchronous).
        
        Args:
            audio_path: Path to audio file
            transcript: Optional transcript for LLM context
            
        Returns:
            Dict with all metrics including LLM and Nova Sonic scores if enabled
        """
        # Get objective metrics
        results = self.evaluate(audio_path)
        
        # Add Nova Sonic evaluation if enabled
        if self.use_nova_sonic:
            nova_scores = await self.evaluate_with_nova_sonic(audio_path)
            results.update(nova_scores)
        
        # Add LLM evaluation if enabled
        if self.use_llm_judge:
            llm_scores = await self.evaluate_with_llm(audio_path, transcript)
            results.update(llm_scores)
        
        return results
    
    def _calculate_fluency(self, y: np.ndarray, sr: int) -> Dict:
        """
        Fluency: smoothness, pauses, speaking rate
        Score 0-10 (higher = more fluent)
        """
        # Speaking rate (syllables per second estimate)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        speaking_rate = tempo / 60.0 * 2  # Rough syllables per second
        
        # Pause detection (silence ratio)
        rms = librosa.feature.rms(y=y)[0]
        silence_threshold = np.percentile(rms, 20)
        silence_ratio = np.sum(rms < silence_threshold) / len(rms)
        
        # Energy consistency (less variation = more fluent)
        energy_std = np.std(rms)
        energy_consistency = 1.0 / (1.0 + energy_std)
        
        # Score: optimal speaking rate (3-5 sps), low silence, consistent energy
        rate_score = 10 * (1 - abs(speaking_rate - 4.0) / 4.0)  # Optimal ~4 sps
        pause_score = 10 * (1 - silence_ratio)
        consistency_score = 10 * energy_consistency
        
        score = (rate_score + pause_score + consistency_score) / 3
        score = max(0, min(10, score))
        
        return {
            "score": round(score, 2),
            "speaking_rate_sps": round(speaking_rate, 2),
            "silence_ratio": round(silence_ratio, 3),
            "energy_consistency": round(energy_consistency, 3)
        }
    
    def _calculate_naturalness(self, y: np.ndarray, sr: int) -> Dict:
        """
        Naturalness: pitch variation, spectral richness, prosody
        Score 0-10 (higher = more natural)
        """
        # Pitch (F0) variation
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_valid = f0[f0 > 0]
        
        if len(f0_valid) > 0:
            pitch_mean = np.mean(f0_valid)
            pitch_std = np.std(f0_valid)
            pitch_range = np.ptp(f0_valid)
            pitch_variation = pitch_std / pitch_mean if pitch_mean > 0 else 0
        else:
            pitch_mean = pitch_std = pitch_range = pitch_variation = 0
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Score: good pitch variation (10-20%), rich spectrum
        variation_score = 10 * min(pitch_variation / 0.15, 1.0)  # Optimal ~15%
        spectrum_score = 10 * min(spectral_bandwidth / 2000, 1.0)  # Rich spectrum
        
        score = (variation_score + spectrum_score) / 2
        score = max(0, min(10, score))
        
        return {
            "score": round(score, 2),
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_std_hz": round(pitch_std, 1),
            "pitch_range_hz": round(pitch_range, 1),
            "pitch_variation": round(pitch_variation, 3),
            "spectral_centroid_hz": round(spectral_centroid, 1)
        }
    
    def _calculate_tone(self, y: np.ndarray, sr: int) -> Dict:
        """
        Tone: warmth, clarity, pleasantness
        Score 0-10 (higher = better tone)
        """
        # MFCCs for tonal quality
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Spectral contrast (clarity)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        clarity = np.mean(contrast)
        
        # Zero crossing rate (smoothness)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        smoothness = 1.0 / (1.0 + zcr * 100)
        
        # Harmonic-to-noise ratio estimate
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-6)
        hnr_db = 10 * np.log10(hnr + 1e-6)
        
        # Score: high clarity, smooth, good HNR
        clarity_score = 10 * min(clarity / 30, 1.0)
        smoothness_score = 10 * smoothness
        hnr_score = 10 * min(hnr_db / 20, 1.0)  # Good HNR ~20dB
        
        score = (clarity_score + smoothness_score + hnr_score) / 3
        score = max(0, min(10, score))
        
        return {
            "score": round(score, 2),
            "clarity": round(clarity, 2),
            "smoothness": round(smoothness, 3),
            "hnr_db": round(hnr_db, 2)
        }


def evaluate_audio_file(audio_path: str) -> Dict:
    """Convenience function to evaluate a single audio file"""
    evaluator = VoiceQualityEvaluator()
    return evaluator.evaluate(audio_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python voice_quality_evaluator.py <audio_file> [--llm] [--nova-sonic]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    use_llm = "--llm" in sys.argv
    use_nova = "--nova-sonic" in sys.argv
    
    evaluator = VoiceQualityEvaluator(use_llm_judge=use_llm, use_nova_sonic=use_nova)
    
    if use_llm or use_nova:
        results = asyncio.run(evaluator.evaluate_async(audio_path))
    else:
        results = evaluator.evaluate(audio_path)
    
    print("\n" + "="*60)
    print("VOICE QUALITY EVALUATION")
    print("="*60)
    print(f"Audio: {audio_path}")
    print(f"\nOverall Quality: {results['overall_quality']:.2f}/10")
    print(f"\nFluency: {results['fluency']['score']:.2f}/10")
    print(f"  - Speaking rate: {results['fluency']['speaking_rate_sps']:.2f} syllables/sec")
    print(f"  - Silence ratio: {results['fluency']['silence_ratio']:.3f}")
    print(f"  - Energy consistency: {results['fluency']['energy_consistency']:.3f}")
    print(f"\nNaturalness: {results['naturalness']['score']:.2f}/10")
    print(f"  - Pitch mean: {results['naturalness']['pitch_mean_hz']:.1f} Hz")
    print(f"  - Pitch variation: {results['naturalness']['pitch_variation']:.3f}")
    print(f"  - Spectral centroid: {results['naturalness']['spectral_centroid_hz']:.1f} Hz")
    print(f"\nTone: {results['tone']['score']:.2f}/10")
    print(f"  - Clarity: {results['tone']['clarity']:.2f}")
    print(f"  - Smoothness: {results['tone']['smoothness']:.3f}")
    print(f"  - HNR: {results['tone']['hnr_db']:.2f} dB")
    
    if use_nova and 'nova_sonic_overall' in results:
        print(f"\n--- BEDROCK NOVA SONIC 2 SCORES ---")
        print(f"Fluency: {results['nova_sonic_fluency']:.2f}/10")
        print(f"Naturalness: {results['nova_sonic_naturalness']:.2f}/10")
        print(f"Tone: {results['nova_sonic_tone']:.2f}/10")
        print(f"Overall: {results['nova_sonic_overall']:.2f}/10")
        print(f"Reasoning: {results['nova_sonic_reasoning']}")
    
    if use_llm and 'llm_overall' in results:
        print(f"\n--- LLM JUDGE SCORES ---")
        print(f"Fluency: {results['llm_fluency']:.2f}/10")
        print(f"Naturalness: {results['llm_naturalness']:.2f}/10")
        print(f"Tone: {results['llm_tone']:.2f}/10")
        print(f"Overall: {results['llm_overall']:.2f}/10")
        print(f"Reasoning: {results['llm_reasoning']}")
    
    print("="*60)

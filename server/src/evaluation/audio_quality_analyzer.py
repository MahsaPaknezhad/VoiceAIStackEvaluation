"""
Voice quality evaluation for TTS output.
Measures fluency, naturalness, and tone.
Includes LLM-based perceptual evaluation and NISQA naturalness scoring.
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
import re
import tempfile
import subprocess

try:
    # Add NISQA to path and import
    import sys
    nisqa_path = os.path.join(os.path.dirname(__file__), "NISQA")
    if nisqa_path not in sys.path:
        sys.path.insert(0, nisqa_path)
    from nisqa.NISQA_model import nisqaModel
    NISQA_AVAILABLE = True
except ImportError:
    logger.warning("NISQA not available - naturalness will use fallback method")
    NISQA_AVAILABLE = False

try:
    # Add speechmetrics to path and import
    import sys
    import os
    speechmetrics_path = os.path.join(os.path.dirname(__file__), "speechmetrics")
    if speechmetrics_path not in sys.path:
        sys.path.insert(0, speechmetrics_path)
    from speechmetrics.absolute import mosnet, srmr
    SPEECHMETRICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SpeechMetrics not available - will skip MOSNet and SRMR: {e}")
    SPEECHMETRICS_AVAILABLE = False


class VoiceQualityEvaluator:
    
    def __init__(self, sample_rate: int = 16000, use_llm_judge: bool = False, 
                 use_nisqa: bool = True, use_speechmetrics: bool = True):
        self.sample_rate = sample_rate
        self.use_llm_judge = use_llm_judge
        self.use_nisqa = use_nisqa and NISQA_AVAILABLE
        self.use_speechmetrics = use_speechmetrics and SPEECHMETRICS_AVAILABLE
            
        if self.use_nisqa:
            self._init_nisqa()
            
        if self.use_speechmetrics:
            self._init_speechmetrics()
    
    def _init_speechmetrics(self):
        """Initialize SpeechMetrics models"""
        try:
            # MOSNet and SRMR need to be loaded with specific parameters
            self.mosnet_metric = mosnet.load(window=0.75)  # 750ms window
            self.srmr_metric = srmr.load(window=0.75)      # 750ms window
            logger.info("SpeechMetrics (MOSNet, SRMR) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SpeechMetrics: {e}")
            self.use_speechmetrics = False
    
    def _init_nisqa(self):
        """Initialize NISQA model for naturalness evaluation"""
        try:
            nisqa_dir = os.path.join(os.path.dirname(__file__), "NISQA")
            model_path = os.path.join(nisqa_dir, "weights", "nisqa.tar")  # Use full model, not TTS-only
            
            if not os.path.exists(model_path):
                logger.warning(f"NISQA model not found at {model_path}")
                self.use_nisqa = False
                return
                
            # Initialize with minimal required args - deg will be set per prediction
            args = {
                'mode': 'predict_file',
                'pretrained_model': model_path,
                'deg': None,  # Will be set during prediction
                'tr_bs_val': 1,
                'tr_num_workers': 0,
                'ms_channel': None,  # For mono audio
                'output_dir': None,  # No output file needed
                'ms_max_segments': 2000  # Increase max segments to handle longer audio
            }
            self.nisqa_args = args  # Store args template
            logger.info("NISQA model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NISQA: {e}")
            self.use_nisqa = False
    
    def _create_llm_judge(self) -> Agent:
        """Create LLM agent for voice quality evaluation using Claude 3 Haiku"""
        model = BedrockModel(
            model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name="ap-southeast-2"
        )
        return Agent(
            name="VoiceQualityJudge",
            model=model,
            system_prompt="""You are an expert speech quality evaluator. You will receive audio metrics extracted from librosa analysis. Based on these technical measurements, evaluate:

1. **Fluency** (0-10): How smooth and natural is the speech flow based on pitch variation and energy consistency?
2. **Naturalness** (0-10): Does it sound like a real human based on spectral characteristics and prosody metrics?
3. **Tone** (0-10): Is the voice pleasant and clear based on harmonic content and spectral balance?
4. **Overall** (0-10): Overall voice quality and listenability.

QUALITY GUIDELINES:
Category	    Metric	                        Acceptable Range	        Quality Threshold
Fluency	        Duration	                    3-5 seconds per sentence	Natural conversation flow
Fluency	        Pitch Mean (Male)	            90-155 Hz	                Natural male voice
Fluency	        Pitch Mean (Female)	            165-255 Hz	                Natural female voice
Fluency	        Pitch Coefficient of Variation	0.1-0.3	                    Natural variation
Fluency	        Speech Rate	                    120-180 words/min	        Optimal comprehension
Fluency	        Primary Tempo	                120-180 BPM	                Natural rhythm
Fluency	        Energy Mean (Sample Rate)	    48+ kHz	                    High fidelity capture
Fluency	        Energy Consistency (Bit Depth)	24-bit preferred	        Wide dynamic range
Naturalness	    Spectral Centroid (Male)	    500-2000 Hz	                Natural brightness
Naturalness	    Spectral Centroid (Female)	    1000-3000 Hz	            Natural brightness
Naturalness	    MFCC Coefficients (2-13)	    ±20 (typical)	            Normal speech patterns
Naturalness	    Spectral Rolloff	            2000-6000 Hz	            Natural speech bandwidth
Tone	        Spectral Contrast	            10-35 dB across bands	    Good spectral balance
Tone	        Zero Crossing Rate	            Application-specific	    Voice activity detection
Tone	        Harmonic-to-Noise Ratio	        7-15 dB	                    Clear voice quality
Overall Quality	Chroma (Music)	                0.1-0.9 (normalized)	    Meaningful harmonic content
Overall Quality	Chroma (Speech)	                0.1-0.4 (normalized)	    Stable tonal content
Overall Quality	Tonnetz (Music)	                ±0.8 (normalized)	        Stable harmonic relationships
Overall Quality	Dynamic Range	                96-120+ dB	                Professional audio

Return ONLY valid JSON:
{
    "fluency": <0-10>,
    "naturalness": <0-10>,
    "tone": <0-10>,
    "overall": <0-10>,
    "reasoning": "<brief explanation of scores based on the metrics>"
}"""
        )
    
    async def evaluate_with_llm_judge(self, audio_path: str, transcript: str = "") -> Dict:
        """
        Evaluate voice quality using Claude 3 Haiku with librosa metrics.
        
        Args:
            audio_path: Path to audio file
            transcript: Optional transcript of what's being said
            
        Returns:
            Dict with LLM evaluation scores
        """
        try:
            # Extract comprehensive audio features
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Fluency metrics
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            f0_valid = f0[f0 > 0]
            pitch_mean = np.mean(f0_valid) if len(f0_valid) > 0 else 0
            pitch_std = np.std(f0_valid) if len(f0_valid) > 0 else 0
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            speech_rate = len(onsets) / (len(y) / sr) if len(onsets) > 0 else 0
            
            # Also get tempo for additional rhythm info
            tempo_estimates = librosa.beat.tempo(y=y, sr=sr)
            primary_tempo = tempo_estimates[0] if len(tempo_estimates) > 0 else 0
            
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            energy_consistency = 1 - (energy_std / energy_mean) if energy_mean > 0 else 0
            
            # Naturalness metrics
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # Tone metrics
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Calculate HNR (Harmonic-to-Noise Ratio)
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            harmonic_energy = np.sum(magnitude**2)
            noise_energy = np.sum((magnitude - np.mean(magnitude, axis=1, keepdims=True))**2)
            hnr = 10 * np.log10(harmonic_energy / (noise_energy + 1e-10)) if noise_energy > 0 else 0
            
            # Overall quality metrics
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
            dynamic_range = np.max(rms) / (np.min(rms) + 1e-10)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            duration = len(y) / sr
            
            # Create comprehensive metrics prompt
            metrics_text = f"""Audio Quality Analysis - Technical Measurements:

FLUENCY METRICS:
- Duration: {duration:.2f} seconds
- Pitch Mean: {pitch_mean:.1f} Hz
- Pitch Std: {pitch_std:.1f} Hz
- Pitch Coefficient of Variation: {pitch_cv:.3f}
- Speech Rate: {speech_rate:.2f} onsets/second
- Primary Tempo: {primary_tempo:.1f} BPM
- Energy Mean: {energy_mean:.4f}
- Energy Consistency: {energy_consistency:.3f} (higher = more consistent)

NATURALNESS METRICS:
- Spectral Centroid: {spectral_centroid:.1f} Hz (brightness)
- MFCC-1: {mfcc_mean[0]:.2f} (overall spectral shape)
- MFCC-2: {mfcc_mean[1]:.2f} (spectral slope)
- Spectral Rolloff: {spectral_rolloff:.1f} Hz (high frequency content)

TONE METRICS:
- Spectral Contrast: {spectral_contrast:.3f} (clarity)
- Zero Crossing Rate: {zero_crossing_rate:.4f} (texture/breathiness)
- Harmonic-to-Noise Ratio: {hnr:.2f} dB (voice quality)

OVERALL QUALITY METRICS:
- Chroma: {chroma:.3f} (tonal content)
- Tonnetz: {tonnetz:.3f} (harmonic relationships)
- Dynamic Range: {dynamic_range:.2f} (energy variation)
- Spectral Flatness: {spectral_flatness:.4f} (naturalness vs synthetic)
"""

            if transcript:
                metrics_text += f"\n\nSPEECH CONTENT: {transcript}"

            # Use direct Bedrock client instead of Agent
            import boto3
            bedrock = boto3.client('bedrock-runtime', region_name='ap-southeast-2')
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": metrics_text}]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            response = bedrock.invoke_model(
                modelId='au.anthropic.claude-haiku-4-5-20251001-v1:0',
                body=json.dumps(body)
            )
            
            result_body = json.loads(response['body'].read())
            response_text = result_body["content"][0]["text"]
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract scores from text
                print(f"Claude response (not JSON): {response_text}")
                result = {"fluency": 5, "naturalness": 5, "tone": 5, "overall": 5, "reasoning": response_text}
            
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
                "llm_reasoning": f"Error: {str(e)}"
            }
    
    def evaluate(self, audio_path: str) -> Dict:
        """
        Evaluate voice quality metrics using NISQA and SpeechMetrics (synchronous).
        
        Returns:
            Dict with NISQA and SpeechMetrics scores
        """
        results = {}
        
        # Get NISQA scores if available
        if self.use_nisqa:
            nisqa_results = self._evaluate_with_nisqa(audio_path)
            results.update(nisqa_results)
        
        # Get SpeechMetrics scores if available
        if self.use_speechmetrics:
            speechmetrics_results = self._evaluate_with_speechmetrics(audio_path)
            results.update(speechmetrics_results)
        
        # Fallback to basic acoustic measures if no advanced metrics available
        if not self.use_nisqa and not self.use_speechmetrics:
            results = {
                "overall_quality": -1,  # Default neutral score
                "note": "No advanced metrics available - use --llm for detailed analysis"
            }
        
        return results
    
    def _evaluate_with_nisqa(self, audio_path: str) -> Dict:
        """
        Evaluate naturalness and speech quality using NISQA.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with NISQA scores
        """
        if not self.use_nisqa:
            return {}
            
        try:
            # NISQA expects specific format - ensure audio is compatible
            y, sr = librosa.load(audio_path, sr=48000)  # NISQA trained on 48kHz
            
            # Create temporary file with correct format
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, y, 48000)
                temp_path = tmp_file.name
            
            # Create args for this prediction
            args = self.nisqa_args.copy()
            args['deg'] = temp_path
            
            # Create model instance for this prediction
            nisqa_model = nisqaModel(args)
            
            # Run NISQA prediction
            results = nisqa_model.predict()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Extract scores (NISQA returns DataFrame)
            if results is not None and not results.empty:
                result = results.iloc[0]  # First row of DataFrame
                return {
                    "mos": float(result.get('mos_pred', 0)),
                    "noisiness": float(result.get('noi_pred', 0)),
                    "coloration": float(result.get('col_pred', 0)),
                    "discontinuity": float(result.get('dis_pred', 0)),
                    "loudness": float(result.get('loud_pred', 0)),
                    "overall_quality": float(result.get('mos_pred', 0))
                }
            else:
                return {
                    "mos": 0,
                    "noisiness": 0,
                    "coloration": 0,
                    "discontinuity": 0,
                    "loudness": 0,
                    "overall_quality": 0
                }
                
        except Exception as e:
            logger.error(f"NISQA evaluation failed: {e}")
            return {
                "mos": 0,
                "noisiness": 0,
                "coloration": 0,
                "discontinuity": 0,
                "loudness": 0,
                "overall_quality": 0,
                "error": str(e)
            }
    
    def _evaluate_with_speechmetrics(self, audio_path: str) -> Dict:
        """
        Evaluate speech quality using MOSNet and SRMR from speechmetrics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with MOSNet and SRMR scores
        """
        if not self.use_speechmetrics:
            return {}
            
        try:
            results = {}
            
            # MOSNet - predicts MOS score
            try:
                mosnet_result = self.mosnet_metric.test(audio_path)
                mosnet_scores = mosnet_result['mosnet']
                results["mosnet_score"] = float(mosnet_scores.mean()) if len(mosnet_scores) > 0 else 0.0
            except Exception as e:
                logger.error(f"MOSNet evaluation failed: {e}")
                results["mosnet_score"] = 0.0
            
            # SRMR - Speech-to-Reverberation Modulation energy Ratio
            try:
                srmr_result = self.srmr_metric.test(audio_path)
                srmr_scores = srmr_result['srmr']
                results["srmr_score"] = float(srmr_scores.mean()) if len(srmr_scores) > 0 else 0.0
            except Exception as e:
                logger.error(f"SRMR evaluation failed: {e}")
                results["srmr_score"] = 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"SpeechMetrics evaluation failed: {e}")
            return {
                "mosnet_score": 0.0,
                "srmr_score": 0.0,
                "speechmetrics_error": str(e)
            }
    


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python voice_quality_evaluator.py <audio_file> [--llm]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    use_llm = "--llm" in sys.argv
    
    evaluator = VoiceQualityEvaluator(use_llm_judge=use_llm)
    
    results = evaluator.evaluate(audio_path)
    
    print("\n" + "="*60)
    print("VOICE QUALITY EVALUATION")
    print("="*60)
    print(f"Audio: {audio_path}")
    print(f"\nOverall Quality: {results['overall_quality']:.2f}/10")
    
    if 'note' in results:
        print(f"Note: {results['note']}")
    
    # Show NISQA scores if available
    if 'mos' in results:
        print(f"\n--- NISQA SCORES ---")
        print(f"MOS: {results['mos']:.2f}/5")
        print(f"Noisiness: {results['noisiness']:.2f}")
        print(f"Coloration: {results['coloration']:.2f}")
        print(f"Discontinuity: {results['discontinuity']:.2f}")
        print(f"Loudness: {results['loudness']:.2f}")
    
    # Show SpeechMetrics scores if available
    if 'mosnet_score' in results:
        print(f"\n--- SPEECHMETRICS SCORES ---")
        print(f"MOSNet: {results['mosnet_score']:.2f}/5")
        print(f"SRMR: {results['srmr_score']:.2f}")
    
    if use_llm and 'llm_overall' in results:
        print(f"\n--- CLAUDE HAIKU 4.5 SCORES ---")
        print(f"Fluency: {results['llm_fluency']:.2f}/10")
        print(f"Naturalness: {results['llm_naturalness']:.2f}/10")
        print(f"Tone: {results['llm_tone']:.2f}/10")
        print(f"Overall: {results['llm_overall']:.2f}/10")
        print(f"Reasoning: {results['llm_reasoning']}")
    
    print("="*60)

import json
from typing import Optional
from loguru import logger
from src.evaluation.services.base_llm_service import BaseLLMService
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.metrics.audio_feature_extractor import (
    AudioFeatureExtractor
)
from src.evaluation.models import LLMJudgeConfig, LLMJudgeResults, AudioMetrics

# System prompt - hardcoded and never changes (not actually used in original)
SYSTEM_PROMPT = """You are an expert speech quality evaluator. You will receive audio metrics extracted from librosa analysis. Based on these technical measurements, evaluate:

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
}"""  # noqa: E501


class VoiceQualityJudge(BaseLLMService, BaseQualityEvaluator[LLMJudgeResults]):
    """
    LLM-based voice quality evaluator using Claude via AWS Bedrock.

    Combines audio feature extraction with LLM evaluation to assess voice
    quality across multiple dimensions. Inherits Bedrock functionality from
    BaseLLMService and audio evaluation interface from BaseAudioEvaluator.

    Attributes:
        config: LLM judge configuration with model and AWS settings
        feature_extractor: Audio feature extraction component for librosa
        analysis
    """

    def __init__(self, config: LLMJudgeConfig) -> None:
        """
        Initialize LLM voice judge with Bedrock configuration.

        Args:
            config: Configuration containing model ID, region, and LLM
            parameters
        """
        super().__init__(
            model_id=config.model_id,
            region_name=config.region_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        self.config = config
        self.feature_extractor: Optional[AudioFeatureExtractor] = None
        logger.info(
            f"LLM Voice Judge initialized with model: {config.model_id}"
        )

    async def initialize(self) -> bool:
        """
        Initialize audio feature extractor and AWS Bedrock client.

        Sets up both the librosa-based audio analysis component and the
        Bedrock runtime client for Claude API access.

        Returns:
            True if both components initialized successfully, False otherwise
        """
        try:
            logger.info("Initializing LLM Voice Judge components...")
            self.feature_extractor = AudioFeatureExtractor()
            logger.info("Audio feature extractor initialized")

            success = self._initialize_bedrock()
            if success:
                logger.info("Bedrock client initialized successfully")
            return success

        except Exception as e:
            logger.error(
                f"Failed to initialize LLM Voice Judge: {e}"
            )
            return False

    async def evaluate(self, audio_path: str) -> LLMJudgeResults:
        """
        Evaluate voice quality using LLM analysis with exponential backoff
        retry.

        Loads audio file, extracts comprehensive acoustic features using
        librosa, formats them into a structured prompt, and sends to Claude
        for perceptual quality evaluation across fluency, naturalness, tone,
        and overall quality. Uses shared retry logic from BaseLLMService.

        Args:
            audio_path: Absolute path to audio file for evaluation

        Returns:
            LLMJudgeResults containing scores (0-10) for each quality dimension
            and reasoning explanation from Claude

        Raises:
            RuntimeError: If evaluator components not properly initialized
        """
        if not self.is_available():
            raise RuntimeError("Voice quality judge not initialized")

        logger.info(
            f"Starting voice quality evaluation for: {audio_path}"
        )

        try:
            # Extract audio features using librosa
            audio_data, sample_rate = self.feature_extractor.load_audio(
                audio_path
            )
            features = self.feature_extractor.extract_features(
                audio_data, sample_rate
            )
            logger.info("Audio features extracted successfully")

            # Create structured prompt with feature measurements
            user_prompt = self._create_user_prompt(features)
            logger.info(
                f"Created user prompt with {len(user_prompt)} characters"
            )

            # Call Claude via Bedrock with retry logic
            response = await self._call_bedrock_with_retry(
                SYSTEM_PROMPT, user_prompt
            )
            logger.info(
                f"Received response from Claude: {response[:100]}..."
            )

            # Parse and validate JSON response
            result = self._parse_response(response)
            logger.info(
                "Voice quality evaluation completed - "
                f"Fluency: {result.llm_fluency}, "
                f"Naturalness: {result.llm_naturalness}, "
                f"Tone: {result.llm_tone}, "
                f"Overall: {result.llm_overall}"
            )
            return result

        except RuntimeError as e:
            logger.error(
                f"Voice quality evaluation failed for {audio_path}: {e}"
            )
            return LLMJudgeResults(
                llm_fluency=0,
                llm_naturalness=0,
                llm_tone=0,
                llm_overall=0,
                llm_reasoning=str(e)
            )
        except Exception as e:
            logger.error(
                f"Voice quality evaluation failed for {audio_path}: {e}"
            )
            return LLMJudgeResults(
                llm_fluency=0,
                llm_naturalness=0,
                llm_tone=0,
                llm_overall=0,
                llm_reasoning=f"Evaluation failed: {str(e)}"
            )

    def _create_user_prompt(self, features: AudioMetrics) -> str:
        """
        Format extracted audio features into structured evaluation prompt.

        Creates comprehensive prompt containing categorized acoustic
        measurements for Claude to evaluate, including fluency metrics
        (pitch, tempo, energy), naturalness metrics (spectral characteristics),
        tone metrics (clarity, harmonic content), and overall quality
        indicators.

        Args:
            features: AudioMetrics object with extracted librosa features

        Returns:
            Formatted string with categorized audio measurements for LLM
            evaluation
        """
        return f"""Audio Quality Analysis - Technical Measurements:

FLUENCY METRICS:
- Duration: {features.duration:.2f} seconds
- Pitch Mean: {features.pitch_mean:.1f} Hz
- Pitch Std: {features.pitch_std:.1f} Hz
- Pitch Coefficient of Variation: {features.pitch_cv:.3f}
- Speech Rate: {features.speech_rate:.2f} onsets/second
- Primary Tempo: {features.primary_tempo:.1f} BPM
- Energy Mean: {features.energy_mean:.4f}
- Energy Consistency: {features.energy_consistency:.3f} (higher = more consistent)

NATURALNESS METRICS:
- Spectral Centroid: {features.spectral_centroid:.1f} Hz (brightness)
- MFCC-1: {features.mfcc_coefficients[0]:.2f} (overall spectral shape)
- MFCC-2: {features.mfcc_coefficients[1]:.2f} (spectral slope)
- Spectral Rolloff: {features.spectral_rolloff:.1f} Hz (high frequency content)

TONE METRICS:
- Spectral Contrast: {features.spectral_contrast:.3f} (clarity)
- Zero Crossing Rate: {features.zero_crossing_rate:.4f} (texture/breathiness)
- Harmonic-to-Noise Ratio: {features.harmonic_noise_ratio:.2f} dB (voice quality)

OVERALL QUALITY METRICS:
- Chroma: {features.chroma:.3f} (tonal content)
- Tonnetz: {features.tonnetz:.3f} (harmonic relationships)
- Dynamic Range: {features.dynamic_range:.2f} (energy variation)
- Spectral Flatness: {features.spectral_flatness:.4f} (naturalness vs synthetic)"""  # noqa: E501

    def _parse_response(self, response_text: str) -> LLMJudgeResults:
        """
        Parse Claude's JSON response into validated LLMJudgeResults object.

        Handles Claude responses that may be wrapped in markdown code blocks,
        extracts JSON content, and validates scores. Falls back to default
        scores if parsing fails to ensure robust error handling.

        Args:
            response_text: Raw text response from Claude API

        Returns:
            LLMJudgeResults with fluency, naturalness, tone, overall scores
            (0-10) and reasoning text. Returns default scores (5) on parse
            failure.
        """
        try:
            cleaned_response = self._clean_json_response(response_text)
            result = json.loads(cleaned_response)
            logger.info("Successfully parsed JSON response")

            return LLMJudgeResults(
                llm_fluency=result.get("fluency", 0),
                llm_naturalness=result.get("naturalness", 0),
                llm_tone=result.get("tone", 0),
                llm_overall=result.get("overall", 0),
                llm_reasoning=result.get("reasoning", "")
            )

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse JSON response: {}. Response: {}",
                e, response_text[:200]
            )
            return LLMJudgeResults(
                llm_fluency=5,
                llm_naturalness=5,
                llm_tone=5,
                llm_overall=5,
                llm_reasoning=response_text
            )

    def is_available(self) -> bool:
        """
        Check if LLM voice judge is ready for evaluation.

        Verifies that both audio feature extractor and Bedrock client
        are properly initialized and available for use.

        Returns:
            True if all components ready for evaluation, False otherwise
        """
        available = (
            self.feature_extractor is not None and
            self.bedrock_client is not None
        )
        if not available:
            logger.warning(
                "LLM Voice Judge not available - "
                f"feature_extractor: {self.feature_extractor is not None}, "
                f"bedrock_client: {self.bedrock_client is not None}"
            )
        return available

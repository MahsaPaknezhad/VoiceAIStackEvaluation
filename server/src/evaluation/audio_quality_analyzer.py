from typing import Dict
from src.evaluation.factories.audio_evaluator_factory import (
    AudioEvaluatorFactory
)
from src.evaluation.metrics.base_audio_evaluator import BaseAudioEvaluator
from src.evaluation.models import (
    AudioQualityConfig, VoiceQuality,
    NISQAConfig, SpeechMetricsConfig, LLMJudgeConfig
)


class VoiceQualityEvaluator:
    """
    Refactored voice quality evaluator with SOLID principles.

    Coordinates multiple audio quality evaluators (NISQA, SpeechMetrics, LLM)
    following the Strategy pattern. Maintains backward compatibility with
    the original API while using clean, modular architecture.

    Attributes:
        config (AudioQualityConfig): Configuration for all evaluators
        factory (AudioEvaluatorFactory): Factory for creating evaluators
        evaluators (Dict[str, BaseAudioEvaluator]): Active evaluator instances
    """

    def __init__(self, sample_rate: int = 16000, use_llm_judge: bool = False,
                 use_nisqa: bool = True, use_speechmetrics: bool = True):
        """
        Initialize with backward-compatible parameters.

        Args:
            sample_rate: Audio sample rate for processing
            use_llm_judge: Enable LLM-based evaluation
            use_nisqa: Enable NISQA evaluation
            use_speechmetrics: Enable SpeechMetrics evaluation
        """
        # Create configuration from parameters (backward compatibility)
        self.config = self._create_config(
            sample_rate, use_llm_judge, use_nisqa, use_speechmetrics
        )
        self.factory = AudioEvaluatorFactory()
        self.evaluators: Dict[str, BaseAudioEvaluator] = {}

        # Initialize evaluators
        self._initialize_evaluators()

    def _create_config(
            self, sample_rate: int, use_llm_judge: bool,
            use_nisqa: bool, use_speechmetrics: bool) -> AudioQualityConfig:
        """Create configuration from constructor parameters."""

        return AudioQualityConfig(
            sample_rate=sample_rate,
            enable_nisqa=use_nisqa,
            enable_speechmetrics=use_speechmetrics,
            enable_llm_judge=use_llm_judge,
            nisqa_config=NISQAConfig(
                model_path="src/evaluation/NISQA/weights/nisqa.tar",
                sample_rate=48000
            ) if use_nisqa else None,
            speechmetrics_config=SpeechMetricsConfig(
                window_size=0.75,
                enable_mosnet=True,
                enable_srmr=True
            ) if use_speechmetrics else None,
            llm_judge_config=LLMJudgeConfig(
                model_id="au.anthropic.claude-haiku-4-5-20251001-v1:0",
                region_name="ap-southeast-2"
            ) if use_llm_judge else None
        )

    def _initialize_evaluators(self) -> None:
        """Initialize all enabled evaluators."""
        self.evaluators = self.factory.create_evaluators(self.config)

        # Initialize each evaluator
        for name, evaluator in list(self.evaluators.items()):
            try:
                if not evaluator.initialize():
                    del self.evaluators[name]
            except Exception:
                if name in self.evaluators:
                    del self.evaluators[name]

    def evaluate(self, audio_path: str) -> VoiceQuality:
        """
        Evaluate voice quality (backward compatible API).

        Args:
            audio_path: Path to audio file

        Returns:
            VoiceQuality Pydantic model with validated scores
        """
        results = {}

        # Run each evaluator and collect results
        for name, evaluator in self.evaluators.items():
            try:
                if evaluator.is_available():
                    evaluation_result = evaluator.evaluate(audio_path)

                    # Convert Pydantic models to dict
                    if hasattr(evaluation_result, 'model_dump'):
                        results.update(evaluation_result.model_dump())
                    else:
                        results.update(evaluation_result)

            except Exception:
                continue

        # Ensure overall_quality is present (backward compatibility)
        if 'overall_quality' not in results:
            results['overall_quality'] = results.get('nisqa_mos', 0.0)

        # Return validated VoiceQuality model
        return VoiceQuality.model_validate(results)

    async def evaluate_with_llm_judge(
            self, audio_path: str, transcript: str = "") -> VoiceQuality:
        """
        Backward compatible LLM evaluation method.

        Args:
            audio_path: Path to audio file
            transcript: Optional transcript (not used in new implementation)

        Returns:
            VoiceQuality with LLM evaluation scores
        """
        llm_evaluator = self.evaluators.get('llm_judge')
        if llm_evaluator and llm_evaluator.is_available():
            try:
                llm_results = await llm_evaluator.evaluate(audio_path)
                if hasattr(llm_results, 'model_dump'):
                    results = llm_results.model_dump()
                else:
                    results = llm_results
                return VoiceQuality.model_validate(results)
            except Exception:
                pass

        # Return default values if LLM evaluation fails
        return VoiceQuality(
            llm_fluency=0,
            llm_naturalness=0,
            llm_tone=0,
            llm_overall=0,
            llm_reasoning="LLM evaluation not available"
        )

"""
Unified voice quality evaluator implementing BaseQualityEvaluator interface.

Coordinates multiple audio quality evaluation methods
(NISQA, SpeechMetrics, LLM)
using factory pattern and async evaluation interface.
"""
from loguru import logger
from typing import Dict, Any
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.factories.audio_evaluator_factory import (
    AudioEvaluatorFactory
)
from src.evaluation.models import (
    VoiceQuality,
    AudioQualityConfig,
    NISQAConfig,
    SpeechMetricsConfig,
    LLMJudgeConfig
)


class UnifiedVoiceQualityEvaluator(BaseQualityEvaluator[VoiceQuality]):
    """
    Unified voice quality evaluator with multiple evaluation methods.

    Coordinates NISQA neural assessment, SpeechMetrics analysis, and LLM-based
    evaluation to provide comprehensive voice quality metrics. Uses factory
    pattern for evaluator creation and async interface for consistent
    evaluation workflow.

    Attributes:
        config: Audio quality configuration with evaluator settings
        factory: Factory for creating individual audio evaluators
        evaluators: Dictionary of initialized evaluator instances
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        use_llm_judge: bool = False,
        use_nisqa: bool = True,
        use_speechmetrics: bool = True
    ) -> None:
        """
        Initialize unified voice quality evaluator.

        Args:
            sample_rate: Audio sample rate for processing (Hz)
            use_llm_judge: Enable LLM-based perceptual quality evaluation
            use_nisqa: Enable NISQA neural quality assessment
            use_speechmetrics: Enable SpeechMetrics (MOSNet, SRMR) evaluation

        Note:
            Creates configuration and factory but does not initialize
            evaluators. Call initialize() before evaluation.
        """
        self.config = AudioQualityConfig(
            sample_rate=sample_rate,
            enable_nisqa=use_nisqa,
            enable_speechmetrics=use_speechmetrics,
            enable_llm_judge=use_llm_judge,
            nisqa_config=NISQAConfig(
                model_path="src/evaluation/metrics/NISQA/weights/nisqa.tar",
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
        self.factory = AudioEvaluatorFactory()
        self.evaluators: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """
        Initialize all enabled voice quality evaluators.

        Creates evaluator instances using factory and initializes each one.
        Removes any evaluators that fail initialization to ensure only
        working evaluators are used for evaluation.

        Returns:
            True if at least one evaluator initialized successfully,
            False if all evaluators failed initialization

        Note:
            Failed evaluators are silently removed from the evaluators
            dictionary to prevent evaluation errors.
        """
        self.evaluators = self.factory.create_evaluators(self.config)

        for name, evaluator in list(self.evaluators.items()):
            try:
                if not await evaluator.initialize():
                    logger.warning(
                        f"Voice quality evaluator '{name}' failed to "
                        "initialize"
                    )
                    del self.evaluators[name]
            except Exception as e:
                logger.warning(
                    f"Voice quality evaluator '{name}' "
                    f"initialization error: {e}"
                )
                if name in self.evaluators:
                    del self.evaluators[name]

        return len(self.evaluators) > 0

    async def evaluate(self, audio_path: str) -> VoiceQuality:
        """
        Evaluate voice quality using all available evaluators.

        Runs evaluation on all successfully initialized evaluators and
        combines results into a unified VoiceQuality model. Handles
        individual evaluator failures gracefully by continuing with
        remaining evaluators.

        Args:
            audio_path: Absolute path to audio file for evaluation

        Returns:
            VoiceQuality model containing combined results from all
            successful evaluations. Missing metrics are set to None.

        Raises:
            FileNotFoundError: If audio file does not exist
            ValueError: If no evaluators are available or initialized
        """
        results: Dict[str, Any] = {}

        for name, evaluator in self.evaluators.items():
            try:
                if evaluator.is_available():
                    logger.info(
                        f"Starting evaluator - {name}"
                    )
                    evaluation_result = await evaluator.evaluate(audio_path)
                    logger.info(
                        f"Finished evaluator - {name}"
                    )
                    if hasattr(evaluation_result, 'model_dump'):
                        results.update(evaluation_result.model_dump())
                    else:
                        results.update(evaluation_result)
            except Exception as e:
                logger.warning(
                    f"Voice quality evaluator '{name}' "
                    f"failed for {audio_path}: {e}"
                )
                continue

        # Ensure overall_quality is present for backward compatibility
        if 'overall_quality' not in results:
            results['overall_quality'] = results.get('nisqa_mos', 0.0)

        return VoiceQuality.model_validate(results)

    def is_available(self) -> bool:
        """
        Check if any evaluators are available for evaluation.

        Returns:
            True if at least one evaluator is initialized and available,
            False if no evaluators are ready for use

        Note:
            This method should be called after initialize() to ensure
            accurate availability status.
        """
        return len(self.evaluators) > 0

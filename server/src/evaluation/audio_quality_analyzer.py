"""
Backward compatible voice quality analyzer.

Provides legacy API compatibility while using unified architecture internally.
"""
from src.evaluation.factories.quality_evaluator_factory import (
    QualityEvaluatorFactory
)
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.models import VoiceQuality


class VoiceQualityEvaluator:
    """
    Backward compatible voice quality evaluator.

    Maintains the original synchronous API while using the new unified
    architecture internally. Provides sync wrappers around async methods
    for backward compatibility with existing code.

    This class serves as a compatibility layer between the legacy API and
    the new BaseQualityEvaluator interface, ensuring existing code continues
    to work without modification.

    Attributes:
        evaluator: Internal unified voice quality evaluator instance
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        use_llm_judge: bool = False,
        use_nisqa: bool = True,
        use_speechmetrics: bool = True
    ) -> None:
        """
        Initialize voice quality evaluator with backward-compatible parameters.

        Args:
            sample_rate: Audio sample rate for processing (Hz)
            use_llm_judge: Enable LLM-based voice quality evaluation
            use_nisqa: Enable NISQA neural quality assessment
            use_speechmetrics: Enable SpeechMetrics (MOSNet, SRMR) evaluation

        Note:
            Initialization is performed synchronously for backward
            compatibility with existing code that expects immediate
            availability.
        """
        factory = QualityEvaluatorFactory()
        self.evaluator: BaseQualityEvaluator = factory.create_voice_evaluator(
            sample_rate=sample_rate,
            use_llm_judge=use_llm_judge,
            use_nisqa=use_nisqa,
            use_speechmetrics=use_speechmetrics
        )

    async def initialize(self) -> None:
        """Initialize the evaluator asynchronously."""
        await self.evaluator.initialize()

    async def evaluate(self, audio_path: str) -> VoiceQuality:
        """
        Evaluate voice quality using all enabled evaluators.

        Synchronous wrapper around the async evaluation method for backward
        compatibility. Runs the async evaluation in an event loop.

        Args:
            audio_path: Absolute path to audio file for evaluation

        Returns:
            VoiceQuality model containing evaluation results from all enabled
            evaluators (NISQA, SpeechMetrics, LLM judge)

        Raises:
            FileNotFoundError: If audio file does not exist
            RuntimeError: If evaluation fails due to initialization issues
        """
        return await self.evaluator.evaluate(audio_path)

    async def evaluate_with_llm_judge(
        self,
        audio_path: str,
        transcript: str = ""
    ) -> VoiceQuality:
        """
        Evaluate voice quality with emphasis on LLM-based assessment.

        Backward compatible method that performs the same evaluation as
        the main evaluate method. The transcript parameter is maintained
        for API compatibility but is not used in the current implementation.

        Args:
            audio_path: Absolute path to audio file for evaluation
            transcript: Optional transcript text
                (unused, kept for compatibility)

        Returns:
            VoiceQuality model containing evaluation results with LLM scores

        Raises:
            FileNotFoundError: If audio file does not exist
            RuntimeError: If evaluation fails due to initialization issues
        """
        return await self.evaluator.evaluate(audio_path)

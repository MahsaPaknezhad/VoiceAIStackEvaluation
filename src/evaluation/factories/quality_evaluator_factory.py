# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""Factory for creating unified quality evaluators."""

from typing import Optional
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.metrics.voice_quality_evaluator import (
    UnifiedVoiceQualityEvaluator
)
from src.evaluation.models import WERConfig, JudgeConfig


class QualityEvaluatorFactory:
    """
    Factory for creating all quality evaluators with consistent interface.

    Provides unified creation methods for WER, Response Quality, and Voice
    Quality evaluators, all implementing BaseQualityEvaluator interface.
    """

    def create_wer_evaluator(
        self, config: Optional[WERConfig] = None
    ) -> BaseQualityEvaluator:
        """
        Create WER evaluator for speech-to-text accuracy measurement.

        Args:
            config: Optional WER configuration. Uses defaults if not provided.

        Returns:
            WEREvaluator instance implementing BaseQualityEvaluator[WERResults]
        """
        from src.evaluation.metrics.wer_evaluator import WEREvaluator
        return WEREvaluator(config or WERConfig())

    def create_response_evaluator(
        self, config: Optional[JudgeConfig] = None
    ) -> BaseQualityEvaluator:
        """
        Create response quality evaluator for LLM response assessment.

        Args:
            config: Optional judge configuration. Uses defaults if not
            provided.

        Returns:
            ResponseQualityEvaluator instance implementing
            BaseQualityEvaluator[JudgeScores]
        """
        from src.evaluation.metrics.response_quality_evaluator import (
            ResponseQualityEvaluator
        )
        return ResponseQualityEvaluator(config or JudgeConfig())

    def create_voice_evaluator(
        self,
        sample_rate: int = 16000,
        use_llm_judge: bool = False,
        use_nisqa: bool = True,
        use_speechmetrics: bool = True
    ) -> BaseQualityEvaluator:
        """
        Create voice quality evaluator for TTS audio assessment.

        Args:
            sample_rate: Audio sample rate for processing
            use_llm_judge: Enable LLM-based voice quality evaluation
            use_nisqa: Enable NISQA neural quality assessment
            use_speechmetrics: Enable SpeechMetrics (MOSNet, SRMR) evaluation

        Returns:
            VoiceQualityEvaluator instance implementing
            BaseQualityEvaluator[VoiceQuality]
        """
        return UnifiedVoiceQualityEvaluator(
            sample_rate=sample_rate,
            use_llm_judge=use_llm_judge,
            use_nisqa=use_nisqa,
            use_speechmetrics=use_speechmetrics
        )

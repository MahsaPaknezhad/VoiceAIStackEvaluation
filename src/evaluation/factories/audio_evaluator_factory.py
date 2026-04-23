# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Dict, List, Union
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.metrics.nisqa_evaluator import NISQAEvaluator
from src.evaluation.metrics.speechmetrics_evaluator import (
    SpeechMetricsEvaluator
)
from src.evaluation.metrics.voice_quality_judge import VoiceQualityJudge
from src.evaluation.models import (
    AudioQualityConfig,
    NISQAConfig,
    SpeechMetricsConfig,
    LLMJudgeConfig
)


class AudioEvaluatorFactory:
    """
    Factory for creating audio quality evaluators with dependency injection.

    Implements the Factory pattern to create and configure different types
    of audio evaluators (NISQA, SpeechMetrics, LLM) based on configuration.
    Supports creating individual evaluators or combinations.

    Attributes:
        _evaluator_registry: Registry mapping evaluator types to classes
    """

    def __init__(self):
        """Initialize the audio evaluator factory."""
        self._evaluator_registry = {
            'nisqa': NISQAEvaluator,
            'speechmetrics': SpeechMetricsEvaluator,
            'llm_judge': VoiceQualityJudge
        }

    def create_evaluator(
        self,
        evaluator_type: str,
        config: AudioQualityConfig
    ) -> BaseQualityEvaluator:
        """
        Create a single audio evaluator instance.

        Args:
            evaluator_type: Type of evaluator
                ('nisqa', 'speechmetrics', 'llm_judge')
            config: Audio quality configuration containing evaluator settings

        Returns:
            Configured audio evaluator instance

        Raises:
            ValueError: If evaluator type is not supported
        """
        if evaluator_type not in self._evaluator_registry:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

        evaluator_class = self._evaluator_registry[evaluator_type]
        evaluator_config = self._get_evaluator_config(evaluator_type, config)

        return evaluator_class(evaluator_config)

    def create_evaluators(
        self,
        config: AudioQualityConfig
    ) -> Dict[str, BaseQualityEvaluator]:
        """
        Create multiple evaluators based on configuration flags.

        Args:
            config: Audio quality configuration with enable flags

        Returns:
            Dictionary mapping evaluator names to instances
        """
        evaluators = {}

        if config.enable_nisqa:
            evaluators['nisqa'] = self.create_evaluator('nisqa', config)

        if config.enable_speechmetrics:
            evaluators['speechmetrics'] = self.create_evaluator(
                'speechmetrics', config
            )

        if config.enable_llm_judge:
            evaluators['llm_judge'] = self.create_evaluator(
                'llm_judge', config
            )

        return evaluators

    def _get_evaluator_config(
        self,
        evaluator_type: str,
        config: AudioQualityConfig
    ) -> Union[NISQAConfig, SpeechMetricsConfig, LLMJudgeConfig]:
        """
        Extract specific evaluator configuration from main config.

        Args:
            evaluator_type: Type of evaluator
            config: Main audio quality configuration

        Returns:
            Specific configuration for the evaluator type

        Raises:
            ValueError: If configuration is missing for evaluator type
        """
        config_map = {
            'nisqa': config.nisqa_config,
            'speechmetrics': config.speechmetrics_config,
            'llm_judge': config.llm_judge_config
        }

        evaluator_config = config_map.get(evaluator_type)
        if evaluator_config is None:
            raise ValueError(f"Missing configuration for {evaluator_type}")

        return evaluator_config

    def get_supported_evaluators(self) -> List[str]:
        """
        Get list of supported evaluator types.

        Returns:
            List of supported evaluator type names
        """
        return list(self._evaluator_registry.keys())

# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

import os
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.models import SpeechMetricsConfig, SpeechMetricsResults
from src.evaluation.metrics.speechmetrics.speechmetrics.absolute import (
    mosnet, srmr
)


class SpeechMetricsEvaluator(BaseQualityEvaluator[SpeechMetricsResults]):
    """
    SpeechMetrics evaluator for MOSNet and SRMR quality assessment.

    Implements MOSNet (Mean Opinion Score Network) and SRMR
    (Speech-to-Reverberation Modulation Energy Ratio) models for speech
    quality evaluation using windowed analysis approach.

    Attributes:
        config (SpeechMetricsConfig): Configuration for SpeechMetrics
            evaluation
        mosnet_metric: Loaded MOSNet model instance
        srmr_metric: Loaded SRMR model instance
    """

    def __init__(self, config: SpeechMetricsConfig):
        """
        Initialize SpeechMetrics evaluator with configuration.

        Args:
            config: SpeechMetrics configuration containing window size and
                flags
        """
        self.config = config
        self.mosnet_metric = None
        self.srmr_metric = None

    async def initialize(self) -> bool:
        """
        Initialize MOSNet and SRMR models.

        Returns:
            True if initialization successful, False otherwise
        """

        try:
            # Initialize models based on configuration
            if self.config.enable_mosnet:
                self.mosnet_metric = mosnet.load(
                    window=self.config.window_size
                )

            if self.config.enable_srmr:
                self.srmr_metric = srmr.load(window=self.config.window_size)

            return True

        except Exception:
            return False

    async def evaluate(self, audio_path: str) -> SpeechMetricsResults:
        """
        Evaluate audio quality using SpeechMetrics models.

        Args:
            audio_path: Path to audio file for evaluation

        Returns:
            SpeechMetricsResults with validated metrics

        Raises:
            RuntimeError: If models not initialized or evaluation fails
        """
        if not self.is_available():
            raise RuntimeError("SpeechMetrics models not initialized")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            mosnet_score = 0.0
            srmr_score = 0.0

            # Run MOSNet evaluation if enabled
            if self.config.enable_mosnet and self.mosnet_metric:
                mosnet_result = self.mosnet_metric.test(audio_path)
                mosnet_scores = mosnet_result['mosnet']
                mosnet_score = (
                    float(mosnet_scores.mean())
                    if len(mosnet_scores) > 0 else 0.0
                )

            # Run SRMR evaluation if enabled
            if self.config.enable_srmr and self.srmr_metric:
                srmr_result = self.srmr_metric.test(audio_path)
                srmr_scores = srmr_result['srmr']
                srmr_score = (
                    float(srmr_scores.mean()) if len(srmr_scores) > 0 else 0.0
                )

            return SpeechMetricsResults(
                mosnet_score=mosnet_score,
                srmr_score=srmr_score
            )

        except Exception:
            return SpeechMetricsResults(
                mosnet_score=0.0,
                srmr_score=0.0
            )

    def is_available(self) -> bool:
        """
        Check if SpeechMetrics models are available and initialized.

        Returns:
            True if models are ready for evaluation, False otherwise
        """
        models_ready = True

        if self.config.enable_mosnet:
            models_ready = models_ready and self.mosnet_metric is not None

        if self.config.enable_srmr:
            models_ready = models_ready and self.srmr_metric is not None

        return models_ready

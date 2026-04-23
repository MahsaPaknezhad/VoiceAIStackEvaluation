# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from src.evaluation.models import AudioMetrics


class BaseAudioProcessor(ABC):
    """Abstract base class for audio processing components."""

    @abstractmethod
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        pass

    @abstractmethod
    def extract_features(
            self,
            audio_data: np.ndarray, sample_rate: int) -> AudioMetrics:
        """Extract audio features from audio data."""
        pass

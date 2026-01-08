from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAudioEvaluator(ABC):
    """Abstract base class for audio quality evaluators."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the evaluator. Returns True if successful."""
        pass

    @abstractmethod
    def evaluate(self, audio_path: str) -> Dict[str, Any]:
        """Evaluate audio quality. Returns evaluation results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if evaluator is available and properly initialized."""
        pass

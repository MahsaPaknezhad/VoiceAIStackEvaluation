"""
Word Error Rate (WER) calculation for speech-to-text evaluation.
"""

from typing import Optional
from jiwer import wer as jiwer_wer
from src.evaluation.models import WERConfig


class WERCalculator:
    """
    Calculator for Word Error Rate (WER) between reference and hypothesis text.

    WER measures the accuracy of speech-to-text systems by comparing
    the transcribed text (hypothesis) against the ground truth (reference).

    Attributes:
        config: Configuration settings for WER calculation
    """

    def __init__(self, config: Optional[WERConfig] = None) -> None:
        """
        Initialize WER calculator with configuration.

        Args:
            config: Optional configuration for WER calculation behavior
        """
        self.config = config or WERConfig()

    def calculate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate between reference and hypothesis text.

        Args:
            reference: Ground truth text (expected transcription)
            hypothesis: Predicted text (actual STT output)

        Returns:
            WER score as float (0-100 if percentage, 0-1 if decimal)

        Examples:
            >>> calculator = WERCalculator()
            >>> calculator.calculate("hello world", "hello word")
            50.0
            >>> calculator.calculate("", "hello")
            100.0
        """
        if self.config.handle_empty_strings:
            if not reference or not hypothesis:
                return 100.0 if self.config.return_percentage else 1.0

        try:
            wer_score = jiwer_wer(reference, hypothesis)
            return wer_score * 100 if \
                self.config.return_percentage else wer_score
        except Exception:
            # Return maximum error rate if calculation fails
            return 100.0 if self.config.return_percentage else 1.0

    def calculate_batch(
        self,
        references: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        """
        Calculate WER for multiple reference-hypothesis pairs.

        Args:
            references: List of ground truth texts
            hypotheses: List of predicted texts

        Returns:
            List of WER scores corresponding to each pair

        Raises:
            ValueError: If references and hypotheses have different lengths
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"References ({len(references)}) and hypotheses "
                f"({len(hypotheses)}) must have same length"
            )

        return [
            self.calculate(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]

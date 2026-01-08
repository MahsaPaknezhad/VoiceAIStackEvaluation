from typing import Optional
from jiwer import wer as jiwer_wer
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.models import WERConfig, WERResults


class WERCalculator:
    """Internal WER calculation logic."""
    
    def __init__(self, config: WERConfig):
        self.config = config

    def calculate(self, reference: str, hypothesis: str) -> float:
        """Calculate WER between reference and hypothesis."""
        if self.config.handle_empty_strings and \
                (not reference or not hypothesis):
            return 100.0 if self.config.return_percentage else 1.0

        try:
            wer_score = jiwer_wer(reference, hypothesis)
            return (
                wer_score * 100
                if self.config.return_percentage else wer_score
            )
        except Exception:
            return 100.0 if self.config.return_percentage else 1.0


class WEREvaluator(BaseQualityEvaluator[WERResults]):
    """WER evaluator implementing unified interface."""

    def __init__(self, config: Optional[WERConfig] = None):
        self.config = config or WERConfig()
        self.calculator = WERCalculator(self.config)

    async def initialize(self) -> bool:
        """Initialize WER evaluator."""
        return True

    async def evaluate(self, reference: str, hypothesis: str) -> WERResults:
        """Calculate WER between reference and hypothesis."""
        wer_score = self.calculator.calculate(reference, hypothesis)
        return WERResults(
            wer_score=wer_score,
            reference=reference,
            hypothesis=hypothesis
        )

    def is_available(self) -> bool:
        """WER is always available."""
        return True

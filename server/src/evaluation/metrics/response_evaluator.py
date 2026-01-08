"""
Response evaluation using LLM judge for quality assessment.
"""

from typing import Optional
from loguru import logger

from src.evaluation.models import JudgeScores, JudgeConfig
from src.evaluation.metrics.response_quality_judge import ResponseQualityJudge


class ResponseEvaluator:
    """
    Evaluator for AI assistant response quality using LLM judge.

    Provides a clean interface for evaluating response quality by delegating
    to the JudgeService while adding evaluation-specific logic and error
    handling.

    Attributes:
        judge_service: Service instance for LLM-based evaluation
    """

    def __init__(self, judge_config: Optional[JudgeConfig] = None) -> None:
        """
        Initialize response evaluator with judge configuration.

        Args:
            judge_config: Optional configuration for judge service
        """
        self.judge_service = ResponseQualityJudge(judge_config)

    async def evaluate(
        self,
        question: str,
        response: str
    ) -> JudgeScores:
        """
        Evaluate response quality against the original question.

        Args:
            question: Original user question or input
            response: AI assistant's response to evaluate

        Returns:
            JudgeScores containing quality metrics and reasoning

        Raises:
            ValueError: If question or response is empty
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not response.strip():
            raise ValueError("Response cannot be empty")

        logger.debug(f"Evaluating response for question: {question[:50]}...")

        try:
            scores = await self.judge_service.evaluate_response(
                question, response
            )
            logger.debug(
                f"Evaluation completed with overall score: {scores.overall}"
            )
            return scores
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            raise

    async def evaluate_batch(
        self,
        question_response_pairs: list[tuple[str, str]]
    ) -> list[JudgeScores]:
        """
        Evaluate multiple question-response pairs.

        Args:
            question_response_pairs: List of (question, response) tuples

        Returns:
            List of JudgeScores corresponding to each pair
        """
        results = []

        for i, (question, response) in enumerate(question_response_pairs):
            logger.debug(
                f"Evaluating pair {i+1}/{len(question_response_pairs)}"
            )

            try:
                score = await self.evaluate(question, response)
                results.append(score)
            except Exception as e:
                logger.error(f"Failed to evaluate pair {i+1}: {e}")
                # Add failed evaluation with error info
                results.append(
                    JudgeScores(
                        reasoning=f"Evaluation failed: {str(e)}"
                    )
                )
        return results

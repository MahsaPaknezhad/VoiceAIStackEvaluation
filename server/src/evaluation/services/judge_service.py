"""
Judge service for LLM-based response evaluation.
"""

import json
import asyncio
import random
from typing import Optional
from loguru import logger

from strands import Agent
from strands.models import BedrockModel
from src.evaluation.models import JudgeScores, JudgeConfig


DEFAULT_JUDGE_PROMPT = """
You are an expert evaluator assessing AI assistant responses.

Your task is to evaluate if the assistant's response appropriately answers the
user's question.
Consider:
- Correctness: Is the information accurate?
- Relevance: Does it address the question asked?
- Completeness: Does it cover the key points?
- Clarity: Is it well-explained?

Provide scores (0-10) for each criterion and an overall score.
Return ONLY valid JSON with this structure:
{
    "correctness": <0-10>,
    "relevance": <0-10>,
    "completeness": <0-10>,
    "clarity": <0-10>,
    "overall": <0-10>,
    "reasoning": "<brief explanation>"
}"""


class JudgeService:
    """
    Service for LLM-based response quality evaluation.

    Manages judge agent creation, evaluation requests, and retry logic
    for robust response quality assessment.

    Attributes:
        config: Configuration settings for judge service
        agent: Strands agent instance for evaluation
    """

    def __init__(self, config: Optional[JudgeConfig] = None) -> None:
        """
        Initialize judge service with configuration.

        Args:
            config: Optional configuration for judge service behavior
        """
        self.config = config or JudgeConfig()

        # Use default prompt if none provided
        if not self.config.system_prompt:
            self.config.system_prompt = DEFAULT_JUDGE_PROMPT

        self.agent = self._create_judge_agent()

    def _create_judge_agent(self) -> Agent:
        """
        Create LLM judge agent with configured model and prompt.

        Returns:
            Configured Strands agent for response evaluation
        """
        model = BedrockModel(
            model_id=self.config.model_id,
            region_name=self.config.region_name
        )
        return Agent(
            name="EvaluationJudge",
            model=model,
            system_prompt=self.config.system_prompt
        )

    async def evaluate_response(
        self,
        question: str,
        llm_response: str
    ) -> JudgeScores:
        """
        Evaluate response quality using LLM judge with retry logic.

        Args:
            question: Original user question
            llm_response: AI assistant's response to evaluate

        Returns:
            JudgeScores containing evaluation metrics and reasoning
        """
        prompt = f"""Question: {question}

Response: {llm_response}

Evaluate the actual response."""

        for attempt in range(self.config.max_retries):
            try:
                result = await self.agent.invoke_async(prompt)
                response = (
                    result.output if hasattr(result, 'output') else str(result)
                )

                # Clean and parse response
                cleaned_response = self._clean_json_response(response)
                result_json = json.loads(cleaned_response)

                return JudgeScores.model_validate(result_json)

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self._calculate_backoff_time(attempt)
                    error_type = self._classify_error(str(e))

                    logger.warning(
                        f"{error_type} error in judge evaluation "
                        f"(attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "Judge evaluation failed after "
                        f"{self.config.max_retries} "
                        f"attempts: {e}"
                    )
                    return JudgeScores(
                        reasoning=f"Evaluation failed: {str(e)}"
                    )

    def _clean_json_response(self, response: str) -> str:
        """
        Clean JSON response by removing markdown code blocks.

        Args:
            response: Raw response from judge agent

        Returns:
            Cleaned JSON string
        """
        response = response.strip()

        if response.startswith('```'):
            # Remove opening ```json or ```
            response = (
                response.split('\n', 1)[1] if '\n' in
                response else response[3:]
            )
            # Remove closing ```
            if response.endswith('```'):
                response = response.rsplit('```', 1)[0]

        return response.strip()

    def _calculate_backoff_time(self, attempt: int) -> float:
        """
        Calculate exponential backoff time with jitter.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Wait time in seconds
        """
        return (3 ** (attempt + 1)) + random.uniform(0, 1)

    def _classify_error(self, error_str: str) -> str:
        """
        Classify error type for logging purposes.

        Args:
            error_str: Error message string

        Returns:
            Error classification string
        """
        error_lower = error_str.lower()
        bedrock_errors = [
            "serviceunavailableexception",
            "bedrock is unable to process",
            "throttlingexception",
            "rate limit",
            "too many requests",
            "service temporarily unavailable",
            "eventstreamError",
            "conversestream operation",
            "botocore.exceptions.eventstreamerror"
        ]

        return "Bedrock" if any(
            err in error_lower for err in bedrock_errors
        ) else "General"

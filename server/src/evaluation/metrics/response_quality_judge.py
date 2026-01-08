"""
Response quality judge for LLM-based text response evaluation.
"""

import json
from typing import Optional
from loguru import logger

from src.evaluation.services.base_llm_service import BaseLLMService
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
}"""  # noqa: E501


class ResponseQualityJudge(BaseLLMService):
    """
    LLM-based text response quality evaluator using Claude via AWS Bedrock.

    Evaluates AI assistant responses across multiple quality dimensions using
    Claude as an expert judge. Inherits Bedrock client functionality from
    BaseLLMService and adds response-specific evaluation logic with retry
    mechanisms for robust assessment.

    Attributes:
        config: Judge configuration containing model settings and parameters
    """

    def __init__(self, config: Optional[JudgeConfig] = None) -> None:
        """
        Initialize response quality judge with Bedrock configuration.

        Sets up the LLM judge with Claude model configuration and evaluation
        parameters. Uses default system prompt if none provided in config.

        Args:
            config: Optional judge configuration with model ID, region, and
                evaluation parameters. Uses defaults if not provided.
        """
        self.config = config or JudgeConfig()

        super().__init__(
            model_id=self.config.model_id,
            region_name=self.config.region_name,
            max_tokens=getattr(self.config, 'max_tokens', 1000),
            temperature=getattr(self.config, 'temperature', 0.1)
        )

        if not self.config.system_prompt:
            self.config.system_prompt = DEFAULT_JUDGE_PROMPT

    async def evaluate_response(
        self,
        question: str,
        llm_response: str
    ) -> JudgeScores:
        """
        Evaluate response quality using Claude with exponential backoff retry.

        Sends question-response pair to Claude for evaluation across
        correctness, relevance, completeness, and clarity dimensions.
        Uses shared retry logic from BaseLLMService for robust error handling.

        Args:
            question: Original user question or prompt
            llm_response: AI assistant's response to evaluate

        Returns:
            JudgeScores containing quality metrics (0-10) for each dimension
            and reasoning explanation from Claude

        Raises:
            RuntimeError: If Bedrock client initialization fails
        """
        if not self._initialize_bedrock():
            raise RuntimeError("Failed to initialize Bedrock client")

        prompt = f"""Question: {question}

    Response: {llm_response}

    Evaluate the actual response."""

        try:
            response = await self._call_bedrock_with_retry(
                self.config.system_prompt, prompt
            )
            cleaned_response = self._clean_json_response(response)
            result_json = json.loads(cleaned_response)
            return JudgeScores.model_validate(result_json)
        except RuntimeError as e:
            logger.error(f"Response evaluation failed: {e}")
            return JudgeScores(reasoning=str(e))

"""
Response quality judge for LLM-based text response evaluation.
"""

import json
from typing import Optional
from loguru import logger
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
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


class ResponseQualityEvaluator(
        BaseLLMService,
        BaseQualityEvaluator[JudgeScores]
):
    """
    LLM-based text response quality evaluator using Claude via AWS Bedrock.

    Evaluates AI assistant responses across multiple quality dimensions using
    Claude as an expert judge. Inherits Bedrock client functionality from
    BaseLLMService and adds response-specific evaluation logic with retry
    mechanisms for robust assessment.

    Attributes:
        config: Judge configuration containing model settings and parameters
    """

    async def initialize(self) -> bool:
        """Initialize the evaluator."""
        return self._initialize_bedrock()

    async def evaluate(self, question: str, llm_response: str) -> JudgeScores:
        """Evaluate response quality."""
        return await self.evaluate_response(question, llm_response)

    def is_available(self) -> bool:
        """Check if evaluator is available."""
        return (
            hasattr(self, 'bedrock_client') and self.bedrock_client is not None
        )

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
        logger.info(
            "Initializing ResponseQualityJudge with model: "
            f"{self.config.model_id}"
        )

        super().__init__(
            model_id=self.config.model_id,
            region_name=self.config.region_name,
            max_tokens=getattr(self.config, 'max_tokens', 1000),
            temperature=getattr(self.config, 'temperature', 0.1)
        )

        if not self.config.system_prompt:
            self.config.system_prompt = DEFAULT_JUDGE_PROMPT
            logger.info("Using default system prompt for response evaluation")
        else:
            logger.info("Using custom system prompt for response evaluation")

        logger.info(
            "ResponseQualityJudge initialized successfully with "
            f"{self.config.max_retries} max retries"
        )

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
        logger.info(
            "Starting response quality evaluation for question: "
            f"{question[:50]}..."
        )
        logger.info(f"Response to evaluate: {llm_response[:100]}...")

        if not self._initialize_bedrock():
            logger.error(
                "Failed to initialize Bedrock client for response evaluation"
            )
            raise RuntimeError("Failed to initialize Bedrock client")

        prompt = f"""Question: {question}

    Response: {llm_response}

    Evaluate the actual response."""

        logger.info(
            "Created evaluation prompt with "
            f"{len(prompt)} characters"
        )

        try:
            logger.info("Calling Claude via Bedrock with retry logic...")
            response = await self._call_bedrock_with_retry(
                self.config.system_prompt, prompt
            )
            logger.info(f"Received response from Claude: {response[:100]}...")

            cleaned_response = self._clean_json_response(response)
            logger.info("Successfully cleaned JSON response")

            result_json = json.loads(cleaned_response)
            logger.info("Successfully parsed JSON response")

            result = JudgeScores.model_validate(result_json)
            logger.info(
                f"Response evaluation completed - "
                f"Correctness: {result.correctness}, "
                f"Relevance: {result.relevance}, "
                f"Completeness: {result.completeness}, "
                f"Clarity: {result.clarity}, "
                f"Overall: {result.overall}"
            )
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.warning(f"Raw response: {response[:200]}...")
            return JudgeScores(reasoning=f"JSON parsing failed: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Response evaluation failed after retries: {e}")
            return JudgeScores(reasoning=str(e))
        except Exception as e:
            logger.error(f"Unexpected error during response evaluation: {e}")
            return JudgeScores(reasoning=f"Evaluation failed: {str(e)}")

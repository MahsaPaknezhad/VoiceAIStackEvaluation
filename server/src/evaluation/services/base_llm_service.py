import json
import boto3
import asyncio
import random
from abc import ABC
from typing import Optional, Dict, Any
from loguru import logger


class BaseLLMService(ABC):
    """
    Abstract base class for LLM services using AWS Bedrock.

    Provides common functionality for Bedrock client initialization,
    API calls with exponential backoff retry logic, and response processing.
    Concrete implementations should inherit from this class and implement
    specific evaluation logic.

    Attributes:
        model_id: AWS Bedrock model identifier
        region_name: AWS region for Bedrock service
        max_tokens: Maximum tokens in LLM response
        temperature: LLM sampling temperature (0.0-1.0)
        max_retries: Maximum number of retry attempts for failed requests
        bedrock_client: Initialized boto3 Bedrock runtime client
    """

    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        max_tokens: int = 1000,
        temperature: float = 0.1,
        max_retries: int = 3
    ) -> None:
        """
        Initialize base LLM service with Bedrock configuration.

        Args:
            model_id: AWS Bedrock model identifier
            region_name: AWS region for Bedrock service
            max_tokens: Maximum tokens in LLM response
            temperature: LLM sampling temperature for response variability
            max_retries: Maximum retry attempts for failed API calls
        """
        self.model_id = model_id
        self.region_name = region_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.bedrock_client: Optional[Any] = None

    def _initialize_bedrock(self) -> bool:
        """
        Initialize AWS Bedrock runtime client.

        Creates boto3 client for bedrock-runtime service using configured
        region. Must be called before making API requests.

        Returns:
            True if client initialization successful, False otherwise
        """
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime', region_name=self.region_name
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize Bedrock client: {}", e)
            return False

    async def _call_bedrock_with_retry(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Call Bedrock with exponential backoff retry logic.

        Implements robust retry mechanism with exponential backoff and jitter
        to handle transient failures, rate limits, and service unavailability.
        Automatically retries on Bedrock-specific errors.

        Args:
            system_prompt: System-level instructions for the LLM
            user_prompt: User message content for evaluation

        Returns:
            Raw text response from the LLM

        Raises:
            RuntimeError: If all retry attempts fail after max_retries
        """
        for attempt in range(self.max_retries):
            try:
                return self._call_bedrock(system_prompt, user_prompt)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self._calculate_backoff_time(attempt)
                    error_type = self._classify_error(str(e))
                    logger.warning(
                        f"{error_type} error (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Bedrock call failed after {self.max_retries} "
                        f"attempts: {e}"
                    )
                    raise RuntimeError(f"LLM evaluation failed: {str(e)}")

    def _call_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make single API call to Bedrock with system and user prompts.

        Constructs Anthropic Claude API request with system prompt and user
        message, then invokes the configured model via Bedrock.

        Args:
            system_prompt: System-level instructions for the LLM
            user_prompt: User message content for evaluation

        Returns:
            Raw text response from the LLM

        Raises:
            Exception: If Bedrock client not initialized or API call fails
        """
        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt}
                ]}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        response = self.bedrock_client.invoke_model(
            modelId=self.model_id, body=json.dumps(body)
        )
        return json.loads(response['body'].read())["content"][0]["text"]

    def _clean_json_response(self, response: str) -> str:
        """
        Clean LLM response by removing markdown code block formatting.

        Handles Claude responses that may be wrapped in markdown code blocks
        (```json...``` or ```...```) and extracts the JSON content.

        Args:
            response: Raw text response from LLM

        Returns:
            Cleaned response text with markdown formatting removed
        """
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        return response.strip()

    def _calculate_backoff_time(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with random jitter.

        Implements exponential backoff strategy (3^(attempt+1)) with added
        random jitter to prevent thundering herd effects when multiple
        requests retry simultaneously.

        Args:
            attempt: Current retry attempt number (0-based)

        Returns:
            Wait time in seconds before next retry attempt
        """
        return (3 ** (attempt + 1)) + random.uniform(0, 1)

    def _classify_error(self, error_str: str) -> str:
        """
        Classify error type for appropriate logging and handling.

        Categorizes errors into Bedrock-specific issues (rate limits, service
        unavailable) versus general errors for better debugging and monitoring.

        Args:
            error_str: Error message string from exception

        Returns:
            Error classification: "Bedrock" for service-related errors,
            "General" for other exceptions
        """
        error_lower = error_str.lower()
        bedrock_errors = [
            "serviceunavailableexception",
            "throttlingexception",
            "rate limit",
            "bedrock is unable to process",
            "too many requests",
            "service temporarily unavailable",
            "eventstreamError",
            "conversestream operation",
            "botocore.exceptions.eventstreamerror"
        ]
        return (
            "Bedrock" if any(err in error_lower for err in bedrock_errors)
            else "General"
        )

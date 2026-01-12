from loguru import logger
from strands import Agent
from strands.models import BedrockModel
from botocore.config import Config

# System prompt for voice-optimized conversational agent
VOICE_SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questionsclearly and concisely.

Your responses will be converted to speech, so:
- Speak naturally and conversationally
- Avoid using lists, bullet points, or special characters
- Keep responses concise but informative (2-3 sentences maximum)
- Use complete sentences in a flowing paragraph format
- If you don't know something, say so honestly

Be friendly, helpful, and direct in your responses."""


def build_conversation_agent(
        model_id: str,
        tts_service=None,
        region_name: str = "ap-southeast-2",
        temperature: float = 0.7,
        cache_prompt: str = "default",
        max_tokens: int = 2000,
        read_timeout: int = 120,
        connect_timeout: int = 120,
        max_attempts: int = 3,
        retry_mode: str = 'adaptive'
) -> Agent:
    """
    Build a conversational agent optimized for voice interactions.

    Creates a Bedrock-powered agent configured for natural speech output
    with concise, TTS-friendly responses.

    Args:
        model_id: Bedrock model identifier
            (e.g., 'anthropic.claude-3-haiku-20240307-v1:0')
        tts_service: Unused parameter, kept for compatibility
        region_name: AWS region for Bedrock. Defaults to "ap-southeast-2"
        temperature: Model temperature for response randomness. Defaults to 0.7
        cache_prompt: Prompt caching strategy. Defaults to "default"
        max_tokens: Maximum tokens in response. Defaults to 2000
        read_timeout: Boto3 read timeout in seconds. Defaults to 120
        connect_timeout: Boto3 connect timeout in seconds. Defaults to 120
        max_attempts: Maximum retry attempts. Defaults to 3
        retry_mode: Boto3 retry mode. Defaults to 'adaptive'

    Returns:
        Configured Agent instance ready for voice conversations
    """
    logger.info('Creating Strands Agent')
    boto_config = Config(
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
        retries={
            'max_attempts': max_attempts,
            'mode': retry_mode
        }
    )

    return Agent(
        model=BedrockModel(
            model_id=model_id,
            region_name=region_name,
            temperature=temperature,
            cache_prompt=cache_prompt,
            max_tokens=max_tokens,
            boto_client_config=boto_config
        ),
        system_prompt=VOICE_SYSTEM_PROMPT
    )

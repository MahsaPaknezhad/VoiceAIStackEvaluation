from strands import Agent
from strands.models import BedrockModel
from loguru import logger


def build_conversation_agent(model_id, tts_service=None):
    """Build a general Q&A conversational agent."""
    
    agent = Agent(
        model=BedrockModel(
            model_id=model_id,
            temperature=0.7,
            cache_prompt="default"
        ),
        system_prompt="""You are a helpful AI assistant that answers questions clearly and concisely.

Your responses will be converted to speech, so:
- Speak naturally and conversationally
- Avoid using lists, bullet points, or special characters
- Keep responses concise but informative
- Use complete sentences in a flowing paragraph format
- If you don't know something, say so honestly

Be friendly, helpful, and direct in your responses.""",
        callback_handler=None
    )
    logger.info(f'Agent successfully created with {model_id}')
    return agent

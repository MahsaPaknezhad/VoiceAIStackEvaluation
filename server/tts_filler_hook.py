import asyncio
from typing import Any
from loguru import logger

from strands import Agent
from strands.agent.conversation_manager import ConversationManager
from strands.models import BedrockModel
from strands.hooks import HookProvider, HookRegistry
from strands.experimental.hooks import BeforeToolInvocationEvent
from strands.hooks.events import MessageAddedEvent

from pipecat.frames.frames import TTSSpeakFrame


class TTSFillerHook(HookProvider):
    def __init__(self, model_id: str, tts_service):
        """Initialize the TTS filler hook.

        Args:
            tts_service: The TTS service to use.
        """
        self.tts_service = tts_service

        self.agent = Agent(
            model=BedrockModel(
            model_id=model_id,
            max_tokens=100,
            ),
            tools=[],
            system_prompt="""You are an AI assistant that generates filler phrases for text-to-speech services.
            Filler phrases are short phrases that are used to fill in the gaps in the conversation. Generate just the phrase, no other text.

            Examples:
            - "Let me check on that for you..."
            - "I'm checking on that for you..."
            - "I'm looking into that for you..."
            - "I'm processing that for you..."
            - "I'm working on that for you..."
            - "I'm reviewing that for you..."
            """,
        )
        self.messages = []

    def generate_filler(self, event: BeforeToolInvocationEvent):
        try:
            # event.selected_tool.tool_name, event.selected_tool.tool_spec
            logger.debug(self.messages)
            agent_response = self.agent(f"Generate a filler phrase based on the conversation history: {self.messages}")
            asyncio.create_task(self.tts_service.queue_frame(TTSSpeakFrame(str(agent_response))))
        except Exception as e:
            logger.exception(f"Could not queue TTS frame for filler: {agent_response}. {e}")

    def add_message(self, event: MessageAddedEvent):
        self.messages.append(event.message)

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(MessageAddedEvent, self.add_message)
        registry.add_callback(BeforeToolInvocationEvent, self.generate_filler)
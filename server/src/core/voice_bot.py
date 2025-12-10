#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
General Q&A speech-to-speech bot using Pipecat and Strands agents.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy
from pipecat.frames.frames import LLMMessagesUpdateFrame, LLMFullResponseEndFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from tts import DeepgramTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.pattern_pair_aggregator import PatternPairAggregator
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_whisker import WhiskerObserver

from conversation_agent import build_conversation_agent
from strands_agents_processor import StrandsAgentsProcessor
from custom_rtvi_observer import CustomRTVIObserver

load_dotenv(override=True)


async def run_bot(transport, client_data=None):
    logger.info("Starting Q&A bot")

    # Initialize speech-to-text service
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-3-general",
            language=Language.EN_AU,
            smart_format=True
        )
    )

    # Create pattern aggregator for filtering thinking tags
    pattern_aggregator = PatternPairAggregator()
    pattern_aggregator.add_pattern_pair(
        pattern_id="thinking_tag",
        start_pattern="<thinking>",
        end_pattern="</thinking>",
        remove_match=True
    )

    # Create markdown text filter
    md_filter = MarkdownTextFilter()

    # Initialize text-to-speech service
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-2-delia-en"
    )
    
    # Build the Q&A agent
    agent = build_conversation_agent(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", 
        tts_service=tts
    )
    llm = StrandsAgentsProcessor(agent=agent)

    # Event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        
        # Send a greeting message
        await task.queue_frames([
            LLMTextFrame("Hello! I'm your AI assistant. How can I help you today?"),
            LLMFullResponseEndFrame(),
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # Setup pipeline
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    context = OpenAILLMContext()
    tma_in = LLMUserContextAggregator(context=context)
    tma_out = LLMAssistantContextAggregator(context=context)

    async def handle_idle(user_idle: UserIdleProcessor) -> None:
        context.messages.append({
            "role": "system",
            "content": "The user has been quiet for a while. Gently ask if they need any help or have any questions."
        })
        await user_idle.push_frame(LLMMessagesUpdateFrame(context.messages, run_llm=True))

    user_idle = UserIdleProcessor(
        callback=handle_idle,
        timeout=120
    )

    pipeline = Pipeline([
        transport.input(),
        user_idle,
        rtvi,
        stt,
        tma_in,
        llm,
        tts,
        transport.output(),
    ])
    whisker = WhiskerObserver(pipeline)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
            interrupt_transports=[MinWordsInterruptionStrategy(min_words=2)],
        ),
        observers=[CustomRTVIObserver(rtvi), whisker]
    )

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    from transport import create_transport
    transport = await create_transport(runner_args=runner_args)

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

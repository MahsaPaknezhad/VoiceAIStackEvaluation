"""
Strands Agent integration for Pipecat.

This module provides integration with Strands Agents for handling conversational AI
interactions. It supports both single agent and multi-agent graphs.
"""

import asyncio
from typing import Optional, Any
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from strands import Agent
    from strands.experimental.hooks import AfterToolInvocationEvent, BeforeToolInvocationEvent
    from strands.hooks import HookProvider, HookRegistry
    from strands.multiagent.graph import Graph
except ModuleNotFoundError as e:
    logger.exception("In order to use Strands Agents, you need to `pip install strands-agents`.")
    raise Exception(f"Missing module: {e}")


class StrandsAgentsProcessor(FrameProcessor):
    """Processor that integrates Strands Agents with Pipecat's frame pipeline.

    This processor takes LLM message frames, extracts the latest user message,
    and processes it through either a single Strands Agent or a multi-agent Graph.
    The response is streamed back as text frames with appropriate response markers.

    Supports both single agent streaming and graph-based multi-agent workflows.
    """

    def __init__(
        self,
        agent: Optional[Agent] = None,
        graph: Optional[Graph] = None,
        graph_exit_node: Optional[str] = None,
    ):
        """Initialize the Strands Agents processor.

        Args:
            agent: The Strands Agent to use for single-agent processing.
            graph: The Strands multi-agent Graph to use for graph-based processing.
            graph_exit_node: The exit node name when using graph-based processing.

        Raises:
            AssertionError: If neither agent nor graph is provided, or if graph is
                          provided without a graph_exit_node.
        """
        super().__init__()
        self.agent = agent
        self.graph = graph
        self.graph_exit_node = graph_exit_node
        
        # Deduplication state
        self.pending_invocation = None
        self.last_content = None
        self.invocation_count = 0
        self.debounce_delay = 1.0  # Wait 1 second for more frames

        assert self.agent or self.graph, "Either agent or graph must be provided"

        if self.graph:
            assert self.graph_exit_node, "graph_exit_node must be provided if graph is provided"
        
        if self.agent:
            self.agent.hooks.add_hook(FunctionCallingHook(self.push_frame))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames with deduplication logic.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        
        if isinstance(frame, OpenAILLMContextFrame):
            self.invocation_count += 1
            logger.debug(f"DEBUG: Received OpenAILLMContextFrame #{self.invocation_count}")
            
            content = frame.context.messages[-1]["content"]
            logger.debug(f"DEBUG: Raw content type: {type(content)}")
            logger.debug(f"DEBUG: Raw content: {content}")
            
            # Extract text from content
            if isinstance(content, list):
                logger.debug(f"DEBUG: Content is list with {len(content)} items")
                text_parts = []
                for i, item in enumerate(content):
                    logger.debug(f"DEBUG: Item {i}: {item} (type: {type(item)})")
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                text = " ".join(text_parts).strip()
                logger.debug(f"DEBUG: Concatenated text: '{text}'")
            else:
                text = str(content).strip()
                logger.debug(f"DEBUG: Content as string: '{text}'")
            
            # Check if this is the same content as before
            if text == self.last_content:
                logger.debug(f"DEBUG: Duplicate content detected, ignoring")
                return
            
            self.last_content = text
            logger.debug(f"DEBUG: New content detected: '{text}'")
            
            # Cancel any pending invocation
            if self.pending_invocation:
                logger.debug(f"DEBUG: Cancelling previous pending invocation")
                self.pending_invocation.cancel()
            
            # Schedule new invocation with debounce
            logger.debug(f"DEBUG: Scheduling invocation with {self.debounce_delay}s debounce")
            self.pending_invocation = asyncio.create_task(self._debounced_invoke(text))
            
        else:
            await self.push_frame(frame, direction)
    
    async def _debounced_invoke(self, text: str):
        """Invoke after debounce delay to avoid multiple calls."""
        try:
            logger.debug(f"DEBUG: Starting debounce wait for '{text}'")
            await asyncio.sleep(self.debounce_delay)
            logger.debug(f"DEBUG: Debounce complete, invoking LLM with: '{text}'")
            await self._ainvoke(text)
        except asyncio.CancelledError:
            logger.debug(f"DEBUG: Debounced invocation cancelled for: '{text}'")
            raise

    async def _ainvoke(self, text: str):
        """Invoke the Strands agent with the provided text and stream results as Pipecat frames.

        Args:
            text: The user input text to process through the agent or graph.
        """
        logger.debug(f"DEBUG: _ainvoke called with text: '{text}'")
        logger.debug(f"DEBUG: Text length: {len(text)} characters")
        logger.debug(f"DEBUG: Using agent: {self.agent is not None}, Using graph: {self.graph is not None}")
        
        # Clear pending invocation since we're now executing
        self.pending_invocation = None
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            if self.graph:
                # Graph does not stream; await full result then emit assistant text
                graph_result = await self.graph.invoke_async(text)
                await self.stop_ttfb_metrics()
                try:
                    node_result = graph_result.results[self.graph_exit_node]
                    for agent_result in node_result.get_agent_results():
                        # Push to TTS service
                        message = getattr(agent_result, "message", None)
                        if isinstance(message, dict) and "content" in message:
                            for block in message["content"]:
                                if isinstance(block, dict) and "text" in block:
                                    await self.push_frame(LLMTextFrame(str(block["text"])))
                        # Update usage metrics
                        await self._report_usage_metrics(
                            agent_result.metrics.accumulated_usage.get('inputTokens', 0), 
                            agent_result.metrics.accumulated_usage.get('outputTokens', 0), 
                            agent_result.metrics.accumulated_usage.get('totalTokens', 0)
                        )
                except Exception as parse_err:
                    logger.warning(f"Failed to extract messages from GraphResult: {parse_err}")
            else:
                # Agent supports streaming events via async iterator
                async for event in self.agent.stream_async(text):
                    # Push to TTS service
                    if isinstance(event, dict) and "data" in event:
                        await self.push_frame(LLMTextFrame(str(event["data"])))
                        await self.stop_ttfb_metrics()
                    
                    # Update usage metrics
                    if isinstance(event, dict) and "event" in event and "metadata" in event['event']:
                        if 'usage' in event['event']['metadata']:
                            usage = event['event']['metadata']['usage']
                            await self._report_usage_metrics(usage.get('inputTokens', 0), usage.get('outputTokens', 0), usage.get('totalTokens', 0))
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")
            # Re-raise the error so it can be caught by retry logic
            raise e
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
    
    def can_generate_metrics(self) -> bool:
        """Check if this service can generate performance metrics.

        Returns:
            True as this service supports metrics generation.
        """
        return True

    async def _report_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int
    ):
        tokens = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        await self.start_llm_usage_metrics(tokens)


class FunctionCallingHook(HookProvider):
    """Hook to track function calling."""

    def __init__(self, push_frame):
        super().__init__()
        self.push_frame = push_frame

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolInvocationEvent, self.log_start)
        registry.add_callback(AfterToolInvocationEvent, self.log_end)

    def log_start(self, event: BeforeToolInvocationEvent) -> None:
        frame = FunctionCallInProgressFrame(
            function_name=event.tool_use["name"],
            tool_call_id=event.tool_use["toolUseId"],
            arguments=event.tool_use["input"],
        )
        asyncio.create_task(self.push_frame(frame))

    def log_end(self, event: AfterToolInvocationEvent) -> None:
        frame = FunctionCallResultFrame(
            function_name=event.tool_use["name"],
            tool_call_id=event.tool_use["toolUseId"],
            arguments=event.tool_use["input"],
            result=event.result,
        )
        asyncio.create_task(self.push_frame(frame))
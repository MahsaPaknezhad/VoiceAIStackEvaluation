"""
Strands Agent processor for evaluation pipeline.
Specialized version for batch evaluation with simplified processing.
"""

from typing import Optional, Dict, List, Any, Union
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContextFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from strands import Agent
from strands.multiagent.graph import Graph


class EvaluationStrandsProcessor(FrameProcessor):
    """
    Simplified Strands processor for evaluation pipeline.

    Processes LLM context frames through Strands agents or multi-agent graphs
    for batch evaluation scenarios.

    Attributes:
        agent: Optional single Strands agent for processing
        graph: Optional multi-agent graph for complex workflows
        graph_exit_node: Exit node name when using graph processing
    """

    def __init__(
        self,
        agent: Optional[Agent] = None,
        graph: Optional[Graph] = None,
        graph_exit_node: Optional[str] = None,
    ) -> None:
        """
        Initialize the evaluation Strands processor.

        Args:
            agent: Single Strands agent for simple processing workflows
            graph: Multi-agent graph for complex processing workflows
            graph_exit_node: Name of exit node when using graph processing

        Raises:
            AssertionError: If neither agent nor graph is provided
        """
        super().__init__()
        self.agent = agent
        self.graph = graph
        self.graph_exit_node = graph_exit_node

        assert self.agent or self.graph, \
            "Either agent or graph must be provided"
        if self.graph:
            assert self.graph_exit_node, \
                "graph_exit_node required for graph"

        logger.info(
            f"EvaluationStrandsProcessor initialized with "
            f"{'graph' if self.graph else 'agent'} mode"
        )

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ) -> None:
        """
        Process incoming frames and handle LLM context frames.

        Args:
            frame: The incoming frame to process
            direction: Direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            logger.info("Received OpenAILLMContextFrame for processing")
            content = frame.context.messages[-1]["content"]
            text = self._extract_text_from_content(content)
            logger.info(f"Processing user input: '{text[:100]}...'")
            await self._process_text(text)
        else:
            await self.push_frame(frame, direction)

    def _extract_text_from_content(
        self,
        content: Union[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Extract text content from LLM message content.

        Args:
            content: Message content from LLM context frame

        Returns:
            Extracted text string, stripped of whitespace
        """
        if isinstance(content, list):
            text_parts = [
                item["text"] if isinstance(item, dict) and "text" in item
                else str(item)
                for item in content
            ]
            extracted_text = " ".join(text_parts).strip()
            logger.info(
                f"Extracted text from list content: {len(text_parts)} parts"
            )
            return extracted_text

        extracted_text = str(content).strip()
        logger.info("Extracted text from string content")
        return extracted_text

    async def _process_text(self, text: str) -> None:
        """
        Process text through agent or graph and emit response frames.

        Args:
            text: User input text to process
        """
        try:
            logger.info("Starting LLM response processing")
            await self.push_frame(LLMFullResponseStartFrame())

            if self.graph:
                await self._process_with_graph(text)
            else:
                await self._process_with_agent(text)

            logger.info("LLM response processing completed successfully")

        except Exception as e:
            logger.exception(f"Error processing text: {e}")
            raise
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def _process_with_graph(self, text: str) -> None:
        """
        Process text using multi-agent graph.

        Args:
            text: Input text to process through the graph
        """
        logger.info(
            f"Processing with graph, exit node: {self.graph_exit_node}")
        result = await self.graph.invoke_async(text)
        node_result = result.results[self.graph_exit_node]

        response_count = 0
        for agent_result in node_result.get_agent_results():
            message = getattr(agent_result, "message", None)
            if isinstance(message, dict) and "content" in message:
                for block in message["content"]:
                    if isinstance(block, dict) and "text" in block:
                        response_count += 1
                        frame = LLMTextFrame(str(block["text"]))
                        await self.push_frame(frame)

        logger.info(
            f"Graph processing generated {response_count} text frames"
        )

    async def _process_with_agent(self, text: str) -> None:
        """
        Process text using single agent with streaming.

        Args:
            text: Input text to process through the agent
        """
        logger.info("Processing with single agent streaming")
        response_count = 0

        async for event in self.agent.stream_async(text):
            if isinstance(event, dict) and "data" in event:
                response_count += 1
                frame = LLMTextFrame(str(event["data"]))
                await self.push_frame(frame)

        logger.info(f"Agent streaming generated {response_count} text frames")

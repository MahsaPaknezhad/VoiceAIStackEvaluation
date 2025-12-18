"""
Simple text aggregator for collecting STT fragments and sending complete text to LLM.
"""
import asyncio
from typing import List
from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class SimpleTextAggregator(FrameProcessor):
    """Simple aggregator that collects transcription fragments and sends complete text."""
    
    def __init__(self, timeout: float = 6.0):
        super().__init__()
        self.timeout = timeout
        self.text_fragments: List[str] = []
        self.aggregation_task = None
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            # Collect transcription fragment
            self.text_fragments.append(frame.text)
            logger.debug(f"Collected fragment: '{frame.text}' (total: {len(self.text_fragments)})")
            
            # Cancel existing aggregation task and start new one
            if self.aggregation_task:
                self.aggregation_task.cancel()
            
            self.aggregation_task = asyncio.create_task(self._aggregate_after_timeout())
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
    
    async def _aggregate_after_timeout(self):
        """Wait for timeout then send aggregated text."""
        try:
            await asyncio.sleep(self.timeout)
            
            if self.text_fragments:
                # Combine all fragments into complete text
                complete_text = " ".join(self.text_fragments).strip()
                logger.debug(f"Sending aggregated text: '{complete_text}'")
                
                # Send as TextFrame directly to LLM processor
                await self.push_frame(TextFrame(complete_text), FrameDirection.DOWNSTREAM)
                
                # Clear fragments
                self.text_fragments.clear()
                
        except asyncio.CancelledError:
            # Task was cancelled, new fragment arrived
            pass
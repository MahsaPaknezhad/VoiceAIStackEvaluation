"""
Frame processors for voice pipeline evaluation.
Specialized processors that collect timing data and text outputs during
pipeline execution.
"""

import time
from loguru import logger
from typing import List, Any
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    TranscriptionFrame,
    TextFrame,
    TTSAudioRawFrame,
    Frame
)

from src.evaluation.models import TimingCollector


class STTTimingProcessor(FrameProcessor):
    """
    Captures the start time of Speech-to-Text processing.

    Monitors for the first AudioRawFrame to mark when STT processing begins.
    This helps measure STT latency from audio input to transcription output.
    Essential for evaluating STT service performance.

    Attributes:
        timing_collector: Shared timing data collector instance
    """

    def __init__(self, timing_collector: TimingCollector) -> None:
        """
        Initialize STT timing processor.

        Args:
            timing_collector: Shared timing data collector for coordinated
            metrics
        """
        super().__init__()
        self.timing_collector = timing_collector

    async def process_frame(self, frame: Frame, direction: Any) -> None:
        """
        Process frame and capture STT start timing.

        Marks STT start time when the first audio frame is encountered,
        indicating the beginning of speech-to-text processing.

        Args:
            frame: Pipeline frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        # Mark STT start time on first audio frame
        if isinstance(frame, AudioRawFrame) and \
                self.timing_collector.stt_start_time is None:
            self.timing_collector.stt_start_time = time.time()

        await self.push_frame(frame, direction)


class TTSTimingProcessor(FrameProcessor):
    """
    Captures the end time of Text-to-Speech processing.

    Monitors for TTSAudioRawFrame to mark when TTS processing completes.
    This helps measure TTS latency from text input to audio output.
    Critical for evaluating TTS service performance and end-to-end latency.

    Attributes:
        timing_collector: Shared timing data collector instance
    """

    def __init__(self, timing_collector: TimingCollector) -> None:
        """
        Initialize TTS timing processor.

        Args:
            timing_collector: Shared timing data collector for coordinated
                metrics
        """
        super().__init__()
        self.timing_collector = timing_collector

    async def process_frame(self, frame: Frame, direction: Any) -> None:
        """
        Process frame and capture TTS end timing.

        Marks TTS end time when the first TTS audio output frame is
        encountered, indicating completion of text-to-speech processing.

        Args:
            frame: Pipeline frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        # Mark TTS end time on first audio output frame
        if isinstance(frame, TTSAudioRawFrame) and \
                self.timing_collector.tts_end_time is None:
            self.timing_collector.tts_end_time = time.time()
            logger.info(
                'TTSTimingProcesser received frames -> '
                f'Setting tts end time to {self.timing_collector.tts_end_time}'
            )

        await self.push_frame(frame, direction)


class STTCollector(FrameProcessor):
    """
    Collects Speech-to-Text transcription outputs and timing.

    Captures transcribed text from STT services for accuracy evaluation
    and marks STT completion timing. Essential for measuring Word Error
    Rate (WER) and STT service performance.

    Attributes:
        timing_collector: Shared timing data collector instance
        text_collector: List to accumulate transcribed text segments
    """

    def __init__(
            self,
            timing_collector: TimingCollector,
            text_collector: List[str]) -> None:
        """
        Initialize STT collector.

        Args:
            timing_collector: Shared timing data collector for coordinated
                metrics
            text_collector: List to store transcribed text for evaluation
        """
        super().__init__()
        self.timing_collector = timing_collector
        self.text_collector = text_collector

    async def process_frame(self, frame: Frame, direction: Any) -> None:
        """
        Process frame and collect STT transcription data.

        Captures transcription text and marks STT completion timing
        when transcription frames are received from STT services.

        Args:
            frame: Pipeline frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Mark STT end time on first transcription
            if self.timing_collector.stt_end_time is None:
                self.timing_collector.stt_end_time = time.time()

            # Collect transcribed text for evaluation
            self.text_collector.append(frame.text)
            logger.info(
                "STTCollector received transcription -> "
                "Appending to text_collector "
                f"(len = {len(self.text_collector)})"
            )

        await self.push_frame(frame, direction)


class LLMCollector(FrameProcessor):
    """
    Collects LLM response text and manages TTS timing.

    Captures text output from Large Language Models for quality evaluation
    and marks TTS start timing. Performs text cleaning and filtering to
    ensure only meaningful content is processed by TTS services.

    Attributes:
        timing_collector: Shared timing data collector instance
        text_collector: List to accumulate LLM response text segments
    """

    def __init__(
            self,
            timing_collector: TimingCollector,
            text_collector: List[str]) -> None:
        """
        Initialize LLM collector.

        Args:
            timing_collector: Shared timing data collector for coordinated
                metrics
            text_collector: List to store LLM response text for evaluation
        """
        super().__init__()
        self.timing_collector = timing_collector
        self.text_collector = text_collector

    async def process_frame(self, frame: Frame, direction: Any) -> None:
        """
        Process frame and collect LLM response data.

        Handles LLM text output by filtering meaningful content, cleaning
        text formatting, and marking TTS start timing. Only processes
        substantive text to avoid TTS artifacts from empty or trivial content.

        Args:
            frame: Pipeline frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if not frame.text:
                await self.push_frame(frame, direction)
                return

            text = frame.text.strip()
            if text and any(c.isalnum() for c in text):
                if self.timing_collector.tts_start_time is None:
                    self.timing_collector.tts_start_time = time.time()
                    logger.info(
                        "Setting TTS start time to "
                        f"{self.timing_collector.tts_start_time}"
                    )

                clean_text = frame.text.replace('\u2014', '--')\
                    .replace('\u2013', '-').replace('\u2019', "'")\
                    .replace('\u201c', '"').replace('\u201d', '"')
                self.text_collector.append(clean_text)

                clean_frame = TextFrame(clean_text)
                await self.push_frame(clean_frame, direction)
            else:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

"""
Frame processors for voice pipeline evaluation.

This module contains specialized frame processors that collect timing data
and text outputs during the voice assistant pipeline execution.
"""
import time
from typing import Optional, List
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame, TextFrame, TTSAudioRawFrame, LLMFullResponseStartFrame


class TimingCollector:
    """
    Centralized timing data collector for pipeline performance metrics.
    
    Tracks start/end times for STT and TTS processing stages to calculate
    latency metrics for evaluation.
    """
    
    def __init__(self):
        self.stt_start_time: Optional[float] = None  # When STT processing begins
        self.stt_end_time: Optional[float] = None    # When STT processing completes
        self.tts_start_time: Optional[float] = None  # When TTS processing begins
        self.tts_end_time: Optional[float] = None    # When TTS processing completes
    
    def get_stt_latency_ms(self) -> Optional[float]:
        """Calculate STT latency in milliseconds."""
        if self.stt_start_time is not None and self.stt_end_time is not None:
            return (self.stt_end_time - self.stt_start_time) * 1000
        return None
    
    def get_tts_latency_ms(self) -> Optional[float]:
        """Calculate TTS latency in milliseconds."""
        if self.tts_start_time is not None and self.tts_end_time is not None:
            return (self.tts_end_time - self.tts_start_time) * 1000
        return None


class STTTimingProcessor(FrameProcessor):
    """
    Captures the start time of Speech-to-Text processing.
    
    Monitors for the first AudioRawFrame to mark when STT processing begins.
    This helps measure STT latency from audio input to transcription output.
    """
    
    def __init__(self, timing_collector: TimingCollector):
        """
        Initialize STT timing processor.
        
        Args:
            timing_collector: Shared timing data collector
        """
        super().__init__()
        self.timing_collector = timing_collector
    
    async def process_frame(self, frame, direction):
        """Process frame and capture STT start timing."""
        await super().process_frame(frame, direction)
        
        # Mark STT start time on first audio frame
        if isinstance(frame, AudioRawFrame) and self.timing_collector.stt_start_time is None:
            self.timing_collector.stt_start_time = time.time()
            
        await self.push_frame(frame, direction)


class TTSTimingProcessor(FrameProcessor):
    """
    Captures the end time of Text-to-Speech processing.
    
    Monitors for TTSAudioRawFrame to mark when TTS processing completes.
    This helps measure TTS latency from text input to audio output.
    """
    
    def __init__(self, timing_collector: TimingCollector):
        """
        Initialize TTS timing processor.
        
        Args:
            timing_collector: Shared timing data collector
        """
        super().__init__()
        self.timing_collector = timing_collector
    
    async def process_frame(self, frame, direction):
        """Process frame and capture TTS end timing."""
        await super().process_frame(frame, direction)
        
        # Mark TTS end time on first audio output frame
        if isinstance(frame, TTSAudioRawFrame) and self.timing_collector.tts_end_time is None:
            self.timing_collector.tts_end_time = time.time()
            
        await self.push_frame(frame, direction)


class STTCollector(FrameProcessor):
    """
    Collects Speech-to-Text transcription outputs and timing.
    """
    
    def __init__(self, timing_collector: TimingCollector, text_collector: List[str]):
        super().__init__()
        self.timing_collector = timing_collector
        self.text_collector = text_collector
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            # Mark STT end time on first transcription
            if self.timing_collector.stt_end_time is None:
                self.timing_collector.stt_end_time = time.time()
            
            # Collect transcribed text for evaluation
            self.text_collector.append(frame.text)
            print(f"DEBUG: STTCollector received transcription: '{frame.text}'")
            
        await self.push_frame(frame, direction)


class LLMCollector(FrameProcessor):
    """
    Collects LLM response text for evaluation.
    """
    
    def __init__(self, timing_collector: TimingCollector, text_collector: List[str]):
        super().__init__()
        self.timing_collector = timing_collector
        self.text_collector = text_collector
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMFullResponseStartFrame):
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, TextFrame):
            import re
            if not frame.text:
                return
            
            text = frame.text.strip()
            important_words = ['I', 'A', 'The', 'This', 'That', 'You', 'We', 'It', 'He', 'She']
            
            if (any(c.isalnum() for c in text) or 
                text in important_words or 
                len(re.sub(r'[^\w\s]', '', text).strip()) >= 1):
                
                # Mark TTS start time when LLM begins outputting text
                if self.timing_collector.tts_start_time is None:
                    self.timing_collector.tts_start_time = time.time()
                
                # Clean and collect LLM response text for evaluation
                clean_text = frame.text.replace('\u2014', '--').replace('\u2013', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                self.text_collector.append(clean_text)
                
                # Send cleaned text to TTS
                clean_frame = TextFrame(clean_text)
                await self.push_frame(clean_frame, direction)
        else:
            await self.push_frame(frame, direction)
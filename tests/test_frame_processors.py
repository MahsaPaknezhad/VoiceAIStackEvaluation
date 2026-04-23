# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

# server/tests/test_frame_processors.py

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame, TextFrame, TTSAudioRawFrame

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.frame_processor import TimingCollector, STTTimingProcessor, STTCollector, TTSTimingProcessor, LLMCollector


class TestTimingCollector:
    """Test the TimingCollector class"""
    
    def test_init(self):
        """Test TimingCollector initialization"""
        collector = TimingCollector()
        assert collector.stt_start_time is None
        assert collector.stt_end_time is None
        assert collector.tts_start_time is None
        assert collector.tts_end_time is None
    
    def test_stt_latency_calculation(self):
        """Test STT latency calculation"""
        collector = TimingCollector()
        
        # No times set - should return None
        assert collector.get_stt_latency_ms() is None
        
        # Set times
        collector.stt_start_time = 1.0
        collector.stt_end_time = 1.5
        
        # Should return 500ms
        assert collector.get_stt_latency_ms() == 500.0
    
    def test_tts_latency_calculation(self):
        """Test TTS latency calculation"""
        collector = TimingCollector()
        
        # No times set - should return None
        assert collector.get_tts_latency_ms() is None
        
        # Set times
        collector.tts_start_time = 2.0
        collector.tts_end_time = 2.3
        
        # Should return 300ms (with floating point tolerance)
        result = collector.get_tts_latency_ms()
        assert abs(result - 300.0) < 0.001


class TestCurrentImplementation:
    """Test the current inner class implementation before refactoring"""
    
    def test_current_stt_timing_start(self):
        """Test current STTTimingStart inner class behavior"""
        # Simulate the current implementation
        stt_start_time = None
        
        def simulate_current_stt_timing(frame):
            nonlocal stt_start_time
            if isinstance(frame, AudioRawFrame) and stt_start_time is None:
                stt_start_time = time.time()
        
        # Test with audio frame
        audio_frame = AudioRawFrame(audio=b"test_audio", sample_rate=16000, num_channels=1)
        start_time = time.time()
        simulate_current_stt_timing(audio_frame)
        end_time = time.time()
        
        # Should capture timing
        assert stt_start_time is not None
        assert start_time <= stt_start_time <= end_time
        
        # Test subsequent frames don't change timing
        old_time = stt_start_time
        simulate_current_stt_timing(audio_frame)
        assert stt_start_time == old_time
    
    def test_current_stt_collector(self):
        """Test current STTCollector inner class behavior"""
        # Simulate current implementation
        stt_end_time = None
        stt_texts = []
        
        def simulate_current_stt_collector(frame):
            nonlocal stt_end_time
            if isinstance(frame, TranscriptionFrame):
                if stt_end_time is None:
                    stt_end_time = time.time()
                stt_texts.append(frame.text)
        
        # Test with transcription frame
        transcription_frame = TranscriptionFrame(text="Hello world", user_id="test", timestamp=1.0)
        start_time = time.time()
        simulate_current_stt_collector(transcription_frame)
        end_time = time.time()
        
        # Should capture timing and text
        assert stt_end_time is not None
        assert start_time <= stt_end_time <= end_time
        assert len(stt_texts) == 1
        assert stt_texts[0] == "Hello world"
    
    def test_current_tts_timing_end(self):
        """Test current TTSTimingEnd inner class behavior"""
        # Simulate the current implementation
        tts_end_time = None
        
        def simulate_current_tts_timing(frame):
            nonlocal tts_end_time
            if isinstance(frame, TTSAudioRawFrame) and tts_end_time is None:
                tts_end_time = time.time()
        
        # Test with TTS audio frame
        tts_frame = TTSAudioRawFrame(audio=b"tts_audio", sample_rate=16000, num_channels=1)
        start_time = time.time()
        simulate_current_tts_timing(tts_frame)
        end_time = time.time()
        
        # Should capture timing
        assert tts_end_time is not None
        assert start_time <= tts_end_time <= end_time
        
        # Test subsequent frames don't change timing
        old_time = tts_end_time
        simulate_current_tts_timing(tts_frame)
        assert tts_end_time == old_time


class TestSTTTimingProcessor:
    """Test the STTTimingProcessor class"""
    
    def test_captures_first_audio_frame_time(self):
        """Test that processor captures timing on first AudioRawFrame"""
        timing_collector = TimingCollector()
        processor = STTTimingProcessor(timing_collector)
        
        # Create mock audio frame
        audio_frame = AudioRawFrame(audio=b"test_audio", sample_rate=16000, num_channels=1)
        
        # Mock the push_frame method
        processor.push_frame = AsyncMock()
        
        # Process the frame synchronously for testing
        async def run_test():
            start_time = time.time()
            await processor.process_frame(audio_frame, "downstream")
            end_time = time.time()
            
            # Check timing was captured
            assert timing_collector.stt_start_time is not None
            assert start_time <= timing_collector.stt_start_time <= end_time
            
            # Check frame was pushed
            processor.push_frame.assert_called_once_with(audio_frame, "downstream")
        
        # Run the async test
        asyncio.run(run_test())


class TestSTTCollector:
    """Test the STTCollector class"""
    
    def test_collects_transcription_and_timing(self):
        """Test that collector captures transcription text and end timing"""
        timing_collector = TimingCollector()
        text_collector = []
        processor = STTCollector(timing_collector, text_collector)
        
        # Create mock transcription frame
        transcription_frame = TranscriptionFrame(text="Hello world", user_id="test", timestamp=1.0)
        processor.push_frame = AsyncMock()
        
        async def run_test():
            start_time = time.time()
            await processor.process_frame(transcription_frame, "downstream")
            end_time = time.time()
            
            # Check timing was captured
            assert timing_collector.stt_end_time is not None
            assert start_time <= timing_collector.stt_end_time <= end_time
            
            # Check text was collected
            assert len(text_collector) == 1
            assert text_collector[0] == "Hello world"
            
            # Check frame was pushed
            processor.push_frame.assert_called_once_with(transcription_frame, "downstream")
        
        asyncio.run(run_test())
    
    def test_ignores_non_transcription_frames(self):
        """Test that processor ignores non-TranscriptionFrame frames"""
        timing_collector = TimingCollector()
        text_collector = []
        processor = STTCollector(timing_collector, text_collector)
        
        # Create mock audio frame
        audio_frame = AudioRawFrame(audio=b"test_audio", sample_rate=16000, num_channels=1)
        processor.push_frame = AsyncMock()
        
        async def run_test():
            await processor.process_frame(audio_frame, "downstream")
            
            # No timing or text should be captured
            assert timing_collector.stt_end_time is None
            assert len(text_collector) == 0
            
            # Frame should still be pushed
            processor.push_frame.assert_called_once_with(audio_frame, "downstream")
        
        asyncio.run(run_test())


class TestTTSTimingProcessor:
    """Test the TTSTimingProcessor class"""
    
    def test_captures_first_tts_frame_time(self):
        """Test that processor captures timing on first TTSAudioRawFrame"""
        timing_collector = TimingCollector()
        processor = TTSTimingProcessor(timing_collector)
        
        # Create mock TTS audio frame
        tts_frame = TTSAudioRawFrame(audio=b"tts_audio", sample_rate=16000, num_channels=1)
        
        # Mock the push_frame method
        processor.push_frame = AsyncMock()
        
        async def run_test():
            start_time = time.time()
            await processor.process_frame(tts_frame, "downstream")
            end_time = time.time()
            
            # Check timing was captured
            assert timing_collector.tts_end_time is not None
            assert start_time <= timing_collector.tts_end_time <= end_time
            
            # Check frame was pushed
            processor.push_frame.assert_called_once_with(tts_frame, "downstream")
        
        asyncio.run(run_test())


class TestRefactoredVsOriginal:
    """Test that refactored version produces identical results to original"""
    
    def test_stt_timing_equivalence(self):
        """Test STTTimingProcessor produces same results as original inner class"""
        # Original approach
        original_stt_start_time = None
        
        def original_logic(frame):
            nonlocal original_stt_start_time
            if isinstance(frame, AudioRawFrame) and original_stt_start_time is None:
                original_stt_start_time = time.time()
        
        # New approach
        timing_collector = TimingCollector()
        processor = STTTimingProcessor(timing_collector)
        processor.push_frame = AsyncMock()
        
        # Test with same frame
        audio_frame = AudioRawFrame(audio=b"test_audio", sample_rate=16000, num_channels=1)
        
        # Process with original logic
        original_logic(audio_frame)
        
        # Process with new logic
        async def run_new():
            await processor.process_frame(audio_frame, "downstream")
        
        asyncio.run(run_new())
        
        # Both should have captured timing (within small tolerance)
        assert original_stt_start_time is not None
        assert timing_collector.stt_start_time is not None
        
        # Times should be very close (within 1ms)
        time_diff = abs(original_stt_start_time - timing_collector.stt_start_time)
        assert time_diff < 0.001
    
    def test_stt_collector_equivalence(self):
        """Test STTCollector produces same results as original inner class"""
        # Original approach
        original_stt_end_time = None
        original_stt_texts = []
        
        def original_logic(frame):
            nonlocal original_stt_end_time
            if isinstance(frame, TranscriptionFrame):
                if original_stt_end_time is None:
                    original_stt_end_time = time.time()
                original_stt_texts.append(frame.text)
        
        # New approach
        timing_collector = TimingCollector()
        new_stt_texts = []
        processor = STTCollector(timing_collector, new_stt_texts)
        processor.push_frame = AsyncMock()
        
        # Test with same frame
        transcription_frame = TranscriptionFrame(text="Test transcription", user_id="test", timestamp=1.0)
        
        # Process with original logic
        original_logic(transcription_frame)
        
        # Process with new logic
        async def run_new():
            await processor.process_frame(transcription_frame, "downstream")
        
        asyncio.run(run_new())
        
        # Both should have captured timing and text
        assert original_stt_end_time is not None
        assert timing_collector.stt_end_time is not None
        assert len(original_stt_texts) == 1
        assert len(new_stt_texts) == 1
        assert original_stt_texts[0] == new_stt_texts[0] == "Test transcription"
        
        # Times should be very close
        time_diff = abs(original_stt_end_time - timing_collector.stt_end_time)
        assert time_diff < 0.001
    
    def test_tts_timing_equivalence(self):
        """Test TTSTimingProcessor produces same results as original inner class"""
        # Original approach
        original_tts_end_time = None
        
        def original_logic(frame):
            nonlocal original_tts_end_time
            if isinstance(frame, TTSAudioRawFrame) and original_tts_end_time is None:
                original_tts_end_time = time.time()
        
        # New approach
        timing_collector = TimingCollector()
        processor = TTSTimingProcessor(timing_collector)
        processor.push_frame = AsyncMock()
        
        # Test with same frame
        tts_frame = TTSAudioRawFrame(audio=b"tts_audio", sample_rate=16000, num_channels=1)
        
        # Process with original logic
        original_logic(tts_frame)
        
        # Process with new logic
        async def run_new():
            await processor.process_frame(tts_frame, "downstream")
        
        asyncio.run(run_new())
        
        # Both should have captured timing (within small tolerance)
        assert original_tts_end_time is not None
        assert timing_collector.tts_end_time is not None
        
        # Times should be very close (within 1ms)
        time_diff = abs(original_tts_end_time - timing_collector.tts_end_time)
        assert time_diff < 0.001
    
    def test_llm_collector_equivalence(self):
        """Test LLMCollector produces same results as original inner class"""
        # Original approach
        original_tts_start_time = None
        original_llm_texts = []
        
        def original_logic(frame):
            nonlocal original_tts_start_time
            if isinstance(frame, TextFrame):
                if original_tts_start_time is None:
                    original_tts_start_time = time.time()
                original_llm_texts.append(frame.text)
        
        # New approach
        timing_collector = TimingCollector()
        new_llm_texts = []
        processor = LLMCollector(timing_collector, new_llm_texts)
        processor.push_frame = AsyncMock()
        
        # Test with same frame
        text_frame = TextFrame(text="LLM response text")
        
        # Process with original logic
        original_logic(text_frame)
        
        # Process with new logic
        async def run_new():
            await processor.process_frame(text_frame, "downstream")
        
        asyncio.run(run_new())
        
        # Both should have captured timing and text
        assert original_tts_start_time is not None
        assert timing_collector.tts_start_time is not None
        assert len(original_llm_texts) == 1
        assert len(new_llm_texts) == 1
        assert original_llm_texts[0] == new_llm_texts[0] == "LLM response text"
        
        # Times should be very close
        time_diff = abs(original_tts_start_time - timing_collector.tts_start_time)
        assert time_diff < 0.001


class TestLLMCollector:
    """Test the LLMCollector class"""
    
    def test_collects_text_and_marks_tts_start(self):
        """Test that collector captures LLM text and marks TTS start timing"""
        timing_collector = TimingCollector()
        text_collector = []
        processor = LLMCollector(timing_collector, text_collector)
        
        # Create mock text frame
        text_frame = TextFrame(text="Hello from LLM")
        processor.push_frame = AsyncMock()
        
        async def run_test():
            start_time = time.time()
            await processor.process_frame(text_frame, "downstream")
            end_time = time.time()
            
            # Check TTS start timing was captured
            assert timing_collector.tts_start_time is not None
            assert start_time <= timing_collector.tts_start_time <= end_time
            
            # Check text was collected
            assert len(text_collector) == 1
            assert text_collector[0] == "Hello from LLM"
            
            # Check frame was pushed
            processor.push_frame.assert_called_once_with(text_frame, "downstream")
        
        asyncio.run(run_test())
    
    def test_ignores_non_text_frames(self):
        """Test that processor ignores non-TextFrame frames"""
        timing_collector = TimingCollector()
        text_collector = []
        processor = LLMCollector(timing_collector, text_collector)
        
        # Create mock audio frame
        audio_frame = AudioRawFrame(audio=b"test_audio", sample_rate=16000, num_channels=1)
        processor.push_frame = AsyncMock()
        
        async def run_test():
            await processor.process_frame(audio_frame, "downstream")
            
            # No timing or text should be captured
            assert timing_collector.tts_start_time is None
            assert len(text_collector) == 0
            
            # Frame should still be pushed
            processor.push_frame.assert_called_once_with(audio_frame, "downstream")
        
        asyncio.run(run_test())


class TestIntegration:
    """Integration tests comparing old vs new approach"""
    
    def test_timing_equivalence(self):
        """Test that new TimingCollector produces same results as old approach"""
        # Simulate old approach
        old_stt_start_time = None
        old_stt_end_time = None
        
        # New approach
        collector = TimingCollector()
        
        # Simulate timing capture
        start_time = 1.0
        end_time = 1.5
        
        # Old way
        if old_stt_start_time is None:
            old_stt_start_time = start_time
        old_stt_end_time = end_time
        
        # New way
        if collector.stt_start_time is None:
            collector.stt_start_time = start_time
        collector.stt_end_time = end_time
        
        # Calculate latencies
        old_latency = (old_stt_end_time - old_stt_start_time) * 1000
        new_latency = collector.get_stt_latency_ms()
        
        # Should be identical
        assert old_latency == new_latency


if __name__ == "__main__":
    pytest.main([__file__])

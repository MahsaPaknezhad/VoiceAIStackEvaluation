"""
Evaluation transport for batch processing audio files.
"""

import asyncio
from pipecat.transports.base_transport import BaseTransport
from pipecat.frames.frames import AudioRawFrame, Frame, StartFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class EvaluationTransport(BaseTransport):
    """Transport for batch evaluation that processes audio files."""
    
    def __init__(self, audio_data: bytes, sample_rate: int):
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._output_audio = []
        
        # Create input/output processors BEFORE calling super().__init__()
        self._input = EvaluationInput(self._audio_data, self._sample_rate)
        self._output = EvaluationOutput(self._output_audio)
        
        super().__init__(input_name="EvaluationInput", output_name="EvaluationOutput")
    
    def input(self):
        """Return input processor that sends audio."""
        return self._input
    
    def output(self):
        """Return output processor that collects audio."""
        return self._output
    
    def get_output_audio(self) -> bytes:
        """Get collected output audio."""
        return b''.join(self._output_audio) if self._output_audio else b''


class EvaluationInput(FrameProcessor):
    """Input processor that sends audio data in chunks."""
    
    def __init__(self, audio_data: bytes, sample_rate: int):
        super().__init__()
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._sent = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Push StartFrame first to initialize downstream processors
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            
            # Then send audio in chunks after StartFrame propagates
            if not self._sent:
                self._sent = True
                # Small delay to let StartFrame propagate
                await asyncio.sleep(0.1)
                
                # Send audio in 100ms chunks for better transcription
                # 16000 Hz * 0.1s * 2 bytes = 3200 bytes per chunk
                chunk_size = int(self._sample_rate * 0.1 * 2)
                print(f"Sending {len(self._audio_data)} bytes of audio in {len(self._audio_data)//chunk_size} chunks")
                for i in range(0, len(self._audio_data), chunk_size):
                    chunk = self._audio_data[i:i+chunk_size]
                    frame = AudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=1
                    )
                    frame.id = f"audio_chunk_{i//chunk_size}"
                    await self.push_frame(frame, direction)
                print("All audio chunks sent, waiting for transcription...")
                # Wait for transcription to complete after all audio is sent
                await asyncio.sleep(5.0)
        else:
            await self.push_frame(frame, direction)


class EvaluationOutput(FrameProcessor):
    """Output processor that collects TTS audio only."""
    
    def __init__(self, output_list: list):
        super().__init__()
        self._output_list = output_list
        self._collecting_tts = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Start collecting when we see TTS audio (after TextFrame processing)
        if hasattr(frame, 'audio') and frame.audio and self._collecting_tts:
            self._output_list.append(frame.audio)
        elif isinstance(frame, TextFrame):
            # TextFrame indicates TTS is about to start
            self._collecting_tts = True
            
        await self.push_frame(frame, direction)

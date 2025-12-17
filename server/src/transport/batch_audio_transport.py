"""
Evaluation transport for batch processing audio files.
"""

import asyncio
import uuid
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.frames.frames import AudioRawFrame, Frame, StartFrame, TextFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class EvaluationTransport(BaseTransport):
    """Transport for batch evaluation that processes audio files."""
    
    def __init__(self, audio_data: bytes, sample_rate: int, params: TransportParams = None, stt_model: str = "small"):
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._output_audio = []
        self._params = params or TransportParams()
        self._stt_model = stt_model
        
        # Create input/output processors BEFORE calling super().__init__()
        vad_analyzer = self._params.vad_analyzer if self._params else None
        self._input = EvaluationInput(self._audio_data, self._sample_rate, vad_analyzer, stt_model)
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
    
    def __init__(self, audio_data: bytes, sample_rate: int, vad_analyzer=None, stt_model: str = "small"):
        super().__init__()
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._sent = False
        self._vad_analyzer = vad_analyzer
        self._stt_model = stt_model
        
        # Initialize VAD with sample rate if provided
        if self._vad_analyzer:
            self._vad_analyzer.set_sample_rate(sample_rate)
    
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
                
                # For Whisper evaluation: send single VAD start, all audio, then VAD stop
                if self._vad_analyzer:
                    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
                    await self.push_frame(UserStartedSpeakingFrame(), direction)
                
                # Send audio in 100ms chunks for better transcription
                chunk_size = int(self._sample_rate * 0.1 * 2)
                print(f"Sending {len(self._audio_data)} bytes of audio in {len(self._audio_data)//chunk_size} chunks")
                chunk_count = 0
                for i in range(0, len(self._audio_data), chunk_size):
                    chunk = self._audio_data[i:i+chunk_size]
                    audio_frame = AudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=1
                    )
                    audio_frame.id = f"audio_chunk_{chunk_count}"
                    chunk_count += 1
                    await self.push_frame(audio_frame, direction)
                    await asyncio.sleep(0.1)
                
                # Add silence padding
                silence_duration = 2.0
                silence_bytes = int(self._sample_rate * silence_duration * 2)
                silence_chunk = b'\x00' * silence_bytes
                for i in range(0, len(silence_chunk), chunk_size):
                    chunk = silence_chunk[i:i+chunk_size]
                    frame = AudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=1
                    )
                    frame.id = str(uuid.uuid4())
                    await self.push_frame(frame, direction)
                    await asyncio.sleep(0.1)
                
                # Send VAD stop - aggregator will wait for transcription via timeout
                if self._vad_analyzer:
                    from pipecat.frames.frames import UserStoppedSpeakingFrame
                    await self.push_frame(UserStoppedSpeakingFrame(), direction)
                
                # Dynamic wait time based on STT model
                if "large" in self._stt_model.lower():
                    wait_time = 30.0  # Whisper Large needs more time
                    print("All audio chunks sent, waiting 30s for Whisper Large transcription...")
                else:
                    wait_time = 15.0  # Whisper Small/Medium
                    print("All audio chunks sent, waiting 15s for transcription...")
                
                await asyncio.sleep(wait_time)
        else:
            await self.push_frame(frame, direction)


class EvaluationOutput(FrameProcessor):
    """Output processor that collects TTS audio only."""
    
    def __init__(self, output_list: list):
        super().__init__()
        self._output_list = output_list
        self._collecting_tts = False
        self.sample_rate = 16000  # Default, will be updated from first frame
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Start collecting when we see TTS audio (after TextFrame processing)
        if isinstance(frame, TTSAudioRawFrame) and self._collecting_tts:
            #print(f"DEBUG: Collecting TTS audio chunk: {len(frame.audio)} bytes")
            self.sample_rate = frame.sample_rate  # Get actual sample rate
            self._output_list.append(frame.audio)
        elif isinstance(frame, TextFrame):
            # TextFrame indicates TTS is about to start
            #print(f"DEBUG: TextFrame detected, starting TTS collection: {frame.text[:50]}...")
            self._collecting_tts = True
        elif isinstance(frame, TTSStoppedFrame):
            # TTSStoppedFrame indicates TTS is complete
            #print(f"DEBUG: TTSStoppedFrame detected, stopping TTS collection")
            self._collecting_tts = False
            
        await self.push_frame(frame, direction)

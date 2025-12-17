"""
NeMo STT Adapter for Pipecat Pipeline Integration

Provides streaming speech-to-text using NVIDIA NeMo models with proper
buffering and context handling for real-time transcription.
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from huggingface_hub import snapshot_download
from loguru import logger
from nemo.collections.asr.models import ASRModel
from pipecat.frames.frames import TranscriptionFrame


class NeMoStreamingSTTService:
    """
    Streaming STT service using NVIDIA NeMo models.
    
    Processes audio in chunks with configurable context windows for improved
    accuracy while maintaining low latency for real-time applications.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        chunk_len_in_secs: float = 1.6,
        left_context_secs: float = 10.0,
        right_context_secs: float = 2.0,
        sample_rate: int = 16000,
        cache_dir: str = "./nemo_models"
    ):
        """
        Initialize NeMo streaming STT service.
        
        Args:
            model_name: Hugging Face model repository name
            chunk_len_in_secs: Length of each processing chunk in seconds
            left_context_secs: Left context window in seconds
            right_context_secs: Right context window in seconds
            sample_rate: Audio sample rate in Hz
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        
        # Convert time to samples
        self.chunk_len = int(chunk_len_in_secs * sample_rate)
        self.left_context_len = int(left_context_secs * sample_rate)
        self.right_context_len = int(right_context_secs * sample_rate)
        
        # Download and load model
        model_path = self._download_from_huggingface(model_name, cache_dir)
        self.model = ASRModel.restore_from(model_path)
        
        # Audio processing state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcriptions = []
        self.transcription_sent = False

        logger.info(f"NeMo STT initialized: {model_name}")
        logger.info(f"  - Chunk: {chunk_len_in_secs}s, Left: {left_context_secs}s, Right: {right_context_secs}s")

    def _download_from_huggingface(self, model_name: str, cache_dir: str) -> str:
        """Download model from Hugging Face if not cached locally."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Check for existing model
        existing_files = list(cache_path.rglob("*.nemo"))
        for existing_file in existing_files:
            if model_name.replace("/", "--") in str(existing_file):
                logger.info(f"Found cached model: {existing_file}")
                return str(existing_file)

        # Download model
        logger.info(f"Downloading model: {model_name}")
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            allow_patterns=["*.nemo"]
        )

        nemo_files = list(Path(model_dir).rglob("*.nemo"))
        if not nemo_files:
            raise FileNotFoundError(f"No .nemo file found in {model_dir}")

        model_path = str(nemo_files[0])
        logger.info(f"Model downloaded: {model_path}")
        return model_path

    def _transcribe_chunk(self, audio: np.ndarray) -> list:
        """Transcribe audio chunk synchronously."""
        try:
            return self.model.transcribe([audio], verbose=False)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return [""]

    def reset(self):
        """Reset audio buffer and transcription state."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcriptions = []
        self.transcription_sent = False

    def link(self, sink):
        """Link to next pipeline component."""
        self._sink = sink
        return self

    async def setup(self, *args, **kwargs):
        """Setup method for pipeline initialization."""
        pass

    async def queue_frame(self, frame, direction=None):
        """Process frames in the Pipecat pipeline."""
        
        # Send transcription BEFORE forwarding EndFrame
        if hasattr(frame, '__class__') and 'End' in frame.__class__.__name__:
            if not self.transcription_sent and self.audio_buffer.size > 0:
                # Process final audio
                transcription = await asyncio.get_event_loop().run_in_executor(
                    None, self._transcribe_chunk, self.audio_buffer
                )
                
                if transcription and transcription[0]:
                    text = str(transcription[0]).strip()
                    if text.startswith('[') and text.endswith(']'):
                        text = text[1:-1].strip("'\"")
                    text = text.replace('&#39;', "'")
                    
                    if text and hasattr(self, '_sink') and self._sink:
                        logger.info(f"Sending final transcription: {text}")
                        transcription_frame = TranscriptionFrame(text, "", int(time.time() * 1000))
                        await self._sink.queue_frame(transcription_frame, direction)
                        self.transcription_sent = True
            # DON'T forward EndFrame yet - let it go through normal flow
        
        # Handle audio frames
        elif hasattr(frame, 'audio') and frame.audio is not None:
            try:
                # Convert audio to float32 numpy array
                if isinstance(frame.audio, bytes):
                    audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_data = np.asarray(frame.audio, dtype=np.float32)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.flatten()
                
                # Add to buffer
                if len(audio_data) > 0:
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_data]) if len(self.audio_buffer) > 0 else audio_data
                
                # Just accumulate audio, don't send transcription yet
                        
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
        
        # Forward frame downstream (EndFrame goes after transcription)
        if hasattr(self, '_sink') and self._sink:
            await self._sink.queue_frame(frame, direction)
        
        return frame

    async def cleanup(self):
        """Cleanup method for pipeline shutdown."""
        if self.transcriptions and not self.transcription_sent:
            final_text = max(self.transcriptions, key=len)
            # Clean the text - remove list brackets and HTML entities
            if final_text.startswith('[') and final_text.endswith(']'):
                final_text = final_text[1:-1].strip("'\"")
            final_text = final_text.replace('&#39;', "'")
            logger.info(f"Cleanup: Sending final transcription: {final_text}")
            transcription_frame = TranscriptionFrame(final_text, "", int(time.time() * 1000))
            if hasattr(self, '_sink') and self._sink:
                await self._sink.queue_frame(transcription_frame, None)
            self.transcription_sent = True


# Alias for backward compatibility with config files
NeMoStreamingSTT = NeMoStreamingSTTService
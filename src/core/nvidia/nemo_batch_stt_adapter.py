# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
NeMo Batch STT Adapter for Pipecat Pipeline Integration

Processes entire audio files with NeMo models for batch evaluation.
Much simpler than streaming - just transcribe the whole file upfront.
"""

import asyncio
import time
from pathlib import Path
import wave
import numpy as np
from huggingface_hub import snapshot_download
from loguru import logger
from nemo.collections.asr.models import ASRModel
from pipecat.frames.frames import TranscriptionFrame


class NeMoBatchSTTService:
    """
    Batch STT service using NVIDIA NeMo models.
    
    Processes entire audio files at once - perfect for evaluation scenarios.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        sample_rate: int = 16000,
        cache_dir: str = "./nemo_models",
        # Accept but ignore streaming parameters for config compatibility
        chunk_len_in_secs: float = None,
        left_context_secs: float = None,
        right_context_secs: float = None,
        **kwargs
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        
        # Download and load model
        model_path = self._download_from_huggingface(model_name, cache_dir)
        self.model = ASRModel.restore_from(model_path)
        
        # State
        self.transcription_result = None
        self.transcription_sent = False

        logger.info(f"NeMo Batch STT initialized: {model_name}")

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

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe entire audio file."""
        logger.info(f"Transcribing file: {audio_path}")
        
        # Load audio file
        with wave.open(audio_path, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
        
        # Convert to float32
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        try:
            result = self.model.transcribe([audio_array], verbose=False)
            if result and result[0]:
                text = str(result[0]).strip()
                # Clean the text
                if text.startswith('[') and text.endswith(']'):
                    text = text[1:-1].strip("'\"")
                text = text.replace('&#39;', "'")
                logger.info(f"Transcription result: {text}")
                return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        
        return ""

    def set_transcription(self, text: str):
        """Set the transcription result to return during pipeline processing."""
        self.transcription_result = text
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
        
        # Send transcription when we see the first audio frame
        if hasattr(frame, 'audio') and frame.audio is not None and not self.transcription_sent:
            if self.transcription_result and hasattr(self, '_sink') and self._sink:
                logger.info(f"Sending transcription: {self.transcription_result}")
                transcription_frame = TranscriptionFrame(
                    self.transcription_result, 
                    "", 
                    int(time.time() * 1000)
                )
                await self._sink.queue_frame(transcription_frame, direction)
                self.transcription_sent = True
        
        # Forward all frames downstream
        if hasattr(self, '_sink') and self._sink:
            await self._sink.queue_frame(frame, direction)
        
        return frame

    async def cleanup(self):
        """Cleanup method for pipeline shutdown."""
        pass
    
    def can_generate_metrics(self) -> bool:
        """Check if this service can generate performance metrics."""
        return False


# Alias for backward compatibility with config files
NeMoBatchSTT = NeMoBatchSTTService
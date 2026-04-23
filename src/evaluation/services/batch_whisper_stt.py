# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
Batch-enabled Whisper STT service for evaluation framework.
Extends Pipecat's WhisperSTTService with batch transcription capabilities.
"""
from typing import Optional, Any
from loguru import logger

import whisper
from pipecat.frames.frames import TranscriptionFrame, Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.whisper.stt import WhisperSTTService


class BatchWhisperSTTService(WhisperSTTService):
    """
    Extended Whisper STT service with batch transcription support.

    Inherits from Pipecat's WhisperSTTService and adds methods for
    batch processing of complete audio files. This enables evaluation
    workflows that need to pre-transcribe files before pipeline execution.

    The service operates in two modes:
    1. Streaming mode: Normal Pipecat pipeline processing
    2. Batch mode: Pre-transcribe file and emit stored result

    Attributes:
        _stored_transcription: Pre-transcribed text for batch mode
        _batch_mode: Flag indicating if service is in batch mode
        _whisper_model: Loaded Whisper model instance
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize batch-enabled Whisper service.

        Args:
            *args: Arguments passed to parent WhisperSTTService
            **kwargs: Keyword arguments passed to parent WhisperSTTService
        """
        super().__init__(*args, **kwargs)
        self._stored_transcription: Optional[str] = None
        self._batch_mode: bool = False
        self._whisper_model: Optional[Any] = None

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe complete audio file using Whisper.

        Loads the Whisper model if not already loaded and transcribes
        the entire audio file. This method is called before pipeline
        execution for batch evaluation workflows.

        Args:
            audio_path: Absolute path to audio file for transcription

        Returns:
            Transcribed text from the audio file

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If Whisper model fails to load or transcribe
        """
        logger.info(f"Batch transcribing {audio_path} with Whisper")

        try:
            # Always use fresh Whisper model to avoid Pipecat conflicts
            model_name = getattr(self, '_model_name', 'large-v2')
            logger.info(f"Loading fresh Whisper model: {model_name}")

            whisper_model = whisper.load_model(model_name)
            result = whisper_model.transcribe(audio_path)

            # Debug the result type and structure
            logger.info(f"Whisper result type: {type(result)}")
            logger.info(f"Whisper result: {result}")

            # Handle different result formats
            if isinstance(result, dict) and "text" in result:
                transcription = result["text"].strip()
            elif isinstance(result, tuple) and len(result) > 0:
                # If it's a tuple, try to get text from first element
                transcription = str(result[0]).strip()
            else:
                # Fallback: convert to string
                transcription = str(result).strip()

            logger.info(f"Whisper transcription: '{transcription}'")
            return transcription

        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path}: {e}")
            raise RuntimeError(f"Whisper transcription failed: {e}") from e

    def set_transcription(self, text: str) -> None:
        """
        Store transcription for batch mode pipeline execution.

        Sets the service to batch mode and stores the pre-transcribed
        text. During pipeline execution, this stored text will be
        emitted as a TranscriptionFrame instead of processing audio.

        Args:
            text: Pre-transcribed text to store and emit during pipeline
        """
        self._stored_transcription = text
        self._batch_mode = True
        logger.info(f"Stored transcription for batch mode: '{text}'")

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ) -> None:
        """
        Process pipeline frames with batch mode support.

        In batch mode, emits the stored transcription when audio processing
        would normally begin. In streaming mode, delegates to parent class
        for normal Whisper processing.

        Args:
            frame: Pipeline frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        # In batch mode, emit stored transcription on first audio frame
        if (self._batch_mode and
                self._stored_transcription and
                hasattr(frame, 'audio')):

            logger.info("Emitting stored batch transcription")
            transcription_frame = TranscriptionFrame(
                text=self._stored_transcription,
                user_id="",
                timestamp=""
            )
            await self.push_frame(transcription_frame, direction)

            # Clear batch mode to prevent duplicate emissions
            self._batch_mode = False
            return

        # Normal streaming processing
        await self.push_frame(frame, direction)

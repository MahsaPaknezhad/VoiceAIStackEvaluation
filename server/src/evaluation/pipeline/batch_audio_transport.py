"""
Batch audio transport for evaluation pipeline.
Handles batch processing of audio files through voice evaluation pipelines.
"""

import asyncio
import uuid
from loguru import logger
from typing import Optional, Any, List
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSAudioRawFrame,
    TextFrame,
    TTSStoppedFrame,
    EndFrame
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BatchAudioInput(FrameProcessor):
    """
    Input processor that sends pre-recorded audio data in chunks.

    Handles streaming of pre-recorded audio files through the evaluation
    pipeline by breaking audio into manageable chunks and coordinating
    with VAD analyzers for speech detection. Manages timing and sequencing
    to ensure proper pipeline initialization and processing.

    Key responsibilities:
    - Stream audio data in timed chunks for optimal STT processing
    - Coordinate VAD start/stop signals for speech detection
    - Add silence padding to ensure complete transcription
    - Manage dynamic wait times based on STT model requirements

    Attributes:
        _audio_data: Raw audio data as bytes to stream
        _sample_rate: Audio sample rate in Hz
        _sent: Flag to prevent duplicate audio streaming
        _vad_analyzer: Optional VAD analyzer for speech detection
        _stt_model: STT model identifier for timing optimization
    """

    def __init__(
            self,
            audio_data: bytes,
            sample_rate: int,
            vad_analyzer: Optional[Any] = None,
            stt_model: str = "small",
            sleep_time: float = 0.05
    ) -> None:
        """
        Initialize batch audio input processor.

        Args:
            audio_data: Raw audio data as bytes to stream through pipeline
            sample_rate: Audio sample rate in Hz (e.g., 16000, 44100)
            vad_analyzer: Optional VAD analyzer for speech detection.
                         Will be initialized with sample rate if provided.
            stt_model: STT model identifier for processing optimization.
                      Used to determine appropriate wait times.
                      Defaults to "small".
            sleep_time: Global sleep time. Defaults to 0.1
        """
        super().__init__()
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._sent = False
        self._vad_analyzer = vad_analyzer
        self._stt_model = stt_model
        self.sleep_time = sleep_time

        # Initialize VAD with sample rate if provided
        if self._vad_analyzer:
            self._vad_analyzer.set_sample_rate(sample_rate)

    async def _handle_start_frame(
            self,
            frame: Frame,
            direction: FrameDirection
    ) -> None:
        """
        Handle StartFrame by initializing audio streaming sequence.

        Args:
            frame: StartFrame to process
            direction: Frame processing direction
        """
        await self.push_frame(frame, direction)

        if not self._sent:
            self._sent = True
            await asyncio.sleep(self.sleep_time)  # Let StartFrame propagate
            await self._stream_audio_sequence(direction)

    async def _stream_audio_sequence(self, direction: FrameDirection) -> None:
        """
        Execute complete audio streaming sequence with VAD coordination.

        Args:
            direction: Frame processing direction
        """
        await self._send_vad_start(direction)
        await self._stream_audio_chunks(direction)
        await self._stream_silence_padding(direction)
        await self._send_vad_stop(direction)
        await self._wait_for_processing()

    async def _send_vad_start(self, direction: FrameDirection) -> None:
        """
        Send VAD start signal if analyzer is configured.

        Args:
            direction: Frame processing direction
        """
        if self._vad_analyzer:
            await self.push_frame(UserStartedSpeakingFrame(), direction)

    async def _stream_audio_chunks(
            self,
            direction: FrameDirection) -> None:
        """
        Stream audio data in 100ms chunks for optimal STT processing.

        Args:
            direction: Frame processing direction
        """
        chunk_size = int(self._sample_rate * 0.1 * 2)
        total_chunks = len(self._audio_data) // chunk_size
        logger.info(
            f"Sending {len(self._audio_data)} bytes of audio in "
            f"{total_chunks} chunks"
        )

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
            await asyncio.sleep(self.sleep_time)

    async def _stream_silence_padding(
            self,
            direction: FrameDirection,
            silence_duration: float = 2.0,
            num_channels: int = 1) -> None:
        """
        Add silence padding to ensure complete transcription.

        Args:
            direction: Frame processing direction
            silence_duration: How long the silence padding should be.
                Defaults to 2.0.
            num_channels: Number of audio channels. Defaults to 1.
        """
        silence_bytes = int(self._sample_rate * silence_duration * 2)
        chunk_size = int(self._sample_rate * 0.1 * 2)
        silence_chunk = b'\x00' * silence_bytes

        for i in range(0, len(silence_chunk), chunk_size):
            chunk = silence_chunk[i:i+chunk_size]
            frame = AudioRawFrame(
                audio=chunk,
                sample_rate=self._sample_rate,
                num_channels=num_channels
            )
            frame.id = str(uuid.uuid4())
            await self.push_frame(frame, direction)
            await asyncio.sleep(self.sleep_time)

    async def _send_vad_stop(self, direction: FrameDirection) -> None:
        """
        Send VAD stop signal if analyzer is configured.

        Args:
            direction: Frame processing direction
        """
        if self._vad_analyzer:
            await self.push_frame(UserStoppedSpeakingFrame(), direction)

    async def _wait_for_processing(
            self,
            wait_time: float = 30.0,
            whisper_wait_time: float = 30.0,
            transcribe_wait_time: float = 10.0
    ) -> None:
        """
        Wait for STT processing with model-specific timing.

        Args:
            wait_time: Default wait time for small/medium models in seconds
            whisper_wait_time: Wait time for large Whisper models in seconds
        """
        if "large" in self._stt_model.lower():
            wait_time = whisper_wait_time
        if 'transcribe' in self._stt_model.lower():
            wait_time = transcribe_wait_time
        logger.info(
            f"All audio chunks sent, waiting {wait_time}s for "
            f"{self._stt_model} transcription..."
        )
        await asyncio.sleep(wait_time)

    async def process_frame(
            self,
            frame: Frame,
            direction: FrameDirection
    ) -> None:
        """
        Process pipeline frames and stream audio data on StartFrame.

        Args:
            frame: Pipeline frame to process
            direction: Frame processing direction (upstream/downstream)
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._handle_start_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class BatchAudioOutput(FrameProcessor):
    """
    Output processor that collects TTS audio chunks during pipeline execution.

    Monitors pipeline frames to detect TTS audio generation and collects
    the generated audio data for evaluation purposes. Uses frame type
    detection to start and stop collection at appropriate times.

    Attributes:
        _output_list: List to store collected TTS audio chunks
        _collecting_tts: Flag indicating if TTS collection is active
        sample_rate: Audio sample rate, updated from first TTS frame
    """

    def __init__(self, output_list: List[bytes]) -> None:
        """
        Initialize output processor with collection list.

        Args:
            output_list: List to store collected TTS audio chunks as bytes
        """
        super().__init__()
        self._output_list = output_list
        self._collecting_tts = False
        logger.debug(
            "BatchAudioOutput initialized with _collecting_tts = "
            f"{self._collecting_tts}"
        )
        self.sample_rate = 16000  # Default, will be updated from first frame
        self._chunk_counter = 0

    def reset(self) -> None:
        """Reset the output processor state for new file processing."""
        self._collecting_tts = False
        logger.debug(
            f"BatchAudioOutput reset: _collecting_tts = {self._collecting_tts}"
        )
        self._chunk_counter = 0
        self._output_list.clear()

    async def process_frame(
            self,
            frame: Frame,
            direction: FrameDirection
    ) -> None:
        """
        Process pipeline frames and collect TTS audio when active.

        Args:
            frame: Pipeline frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        # Start collecting when we see TTS audio (after TextFrame processing)
        if isinstance(frame, EndFrame):
            logger.info("EndFrame detected - pipeline should be ending")
        elif isinstance(frame, TTSAudioRawFrame) and self._collecting_tts:
            self._chunk_counter += 1
            logger.info(
                f"Collecting TTS audio chunk #{self._chunk_counter}: "
                f"{len(frame.audio)} bytes"
            )
            self.sample_rate = frame.sample_rate  # Get actual sample rate
            self._output_list.append(frame.audio)
        elif isinstance(frame, TextFrame):
            # TextFrame indicates TTS is about to start
            self._collecting_tts = True
            logger.info(
                "TextFrame detected, starting TTS collection - "
                f"Setting _collecting_tts to {self._collecting_tts}"
            )
        elif isinstance(frame, TTSStoppedFrame):
            self._collecting_tts = False
            logger.info(
                "TTSStoppedFrame detected, stopping TTS collection - "
                f"Setting _collecting_tts to {self._collecting_tts}"
            )
            logger.info(f"Total TTS chunks collected: {self._chunk_counter}")

            # Send EndFrame to complete pipeline
            logger.info("Sending EndFrame to complete pipeline")
            await self.push_frame(EndFrame(), direction)

        await self.push_frame(frame, direction)


class BatchAudioTransport(BaseTransport):
    """
    Transport for batch evaluation that processes audio files.

    Handles batch processing of pre-recorded audio files through voice
    evaluation pipelines. Creates input and output processors to manage
    audio data flow and collect generated TTS audio for evaluation.

    Key responsibilities:
    - Process pre-recorded audio files in batch mode
    - Coordinate input audio streaming and output audio collection
    - Manage VAD analyzer integration for speech detection
    - Provide interface for retrieving generated TTS audio

    Attributes:
        _audio_data: Raw audio data as bytes
        _sample_rate: Audio sample rate in Hz
        _output_audio: List to collect generated TTS audio chunks
        _params: Transport parameters including VAD configuration
        _stt_model: STT model identifier for processing optimization
        _input: Input processor for sending audio data
        _output: Output processor for collecting TTS audio
    """

    def __init__(
            self,
            audio_data: bytes,
            sample_rate: int,
            params: Optional[TransportParams] = None,
            stt_model: str = "small",
            batch_audio_input: Optional[BatchAudioInput] = None,
            batch_audio_output: Optional[BatchAudioOutput] = None
    ) -> None:
        """
        Initialize batch audio transport with audio data and configuration.

        Args:
            audio_data: Raw audio data as bytes to process
            sample_rate: Audio sample rate in Hz (e.g., 16000, 44100)
            params: Transport parameters including VAD analyzer configuration.
                   Defaults to empty TransportParams if not provided.
            stt_model: STT model identifier for processing optimization.
                      Used to determine appropriate wait times.
                      Defaults to "small".
            batch_audio_input: Optional BatchAudioInput instance to use as
                input processor. If not provided, will create new instance.
            batch_audio_output: Optional BatchAudioOutput instance to use as
                output processor. If not provided, will create new instance.
        """
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._output_audio: List[bytes] = []
        self._params = params or TransportParams()
        self._stt_model = stt_model

        # Create input/output processors BEFORE calling super().__init__()
        vad_analyzer = self._params.vad_analyzer if self._params else None

        # Always create fresh instances
        # (ignore passed parameters to ensure clean state)
        self._input = BatchAudioInput(
            self._audio_data,
            self._sample_rate,
            vad_analyzer,
            stt_model
        ) or batch_audio_input

        self._output = (
            BatchAudioOutput(self._output_audio) or batch_audio_output
        )

        super().__init__(
            input_name="BatchAudioInput",
            output_name="BatchAudioOutput"
        )

    def input(self) -> Any:
        """
        Return input processor that sends audio data to pipeline.

        Returns:
            BatchAudioInput processor instance for streaming audio data
        """
        return self._input

    def output(self) -> Any:
        """
        Return output processor that collects generated TTS audio.

        Returns:
            BatchAudioOutput processor instance for collecting TTS audio
        """
        return self._output

    def get_output_audio(self) -> bytes:
        """
        Get collected TTS audio data from pipeline execution.

        Concatenates all collected TTS audio chunks into a single
        bytes object for saving or further processing.

        Returns:
            Complete TTS audio data as bytes, or empty bytes if no audio
                generated
        """
        logger.info(f"Getting output audio: {len(self._output_audio)} chunks")
        result = b''.join(self._output_audio) if self._output_audio else b''
        logger.info(f"Output audio size: {len(result)} bytes")
        return result

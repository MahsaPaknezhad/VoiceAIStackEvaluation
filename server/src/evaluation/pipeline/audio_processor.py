"""
Audio processor for voice evaluation framework.
Handles audio file processing and pipeline execution.
"""

from typing import Dict, Optional, Any, Type
import wave
import numpy as np
import librosa
from loguru import logger

from src.evaluation.models import PipelineComponents, ExecutionResults
from src.evaluation.pipeline.pipeline_builder import PipelineBuilder
from src.evaluation.pipeline.pipeline_executor import PipelineExecutor


class AudioProcessor:
    """
    Processes audio files and orchestrates pipeline execution.

    This class handles the complete audio processing workflow including
    file loading, format conversion, resampling, and pipeline execution
    coordination. It serves as the main interface between raw audio files
    and the voice evaluation pipeline.

    Key responsibilities:
    - Load and validate audio files
    - Resample audio to meet STT service requirements
    - Coordinate pipeline building and execution
    - Manage audio data throughout the processing lifecycle

    Attributes:
        stt_config: STT service configuration for audio requirements
        sample_rate: Current audio sample rate in Hz
        audio_data: Raw audio data as bytes
        pipeline_builder: Builder instance for creating pipelines
        pipeline_executor: Executor instance for running pipelines
    """

    def __init__(
            self,
            stt_config: Optional[Dict[str, Any]] = None,
            pipeline_builder: Type[PipelineBuilder] = PipelineBuilder,
            pipeline_executor: Type[PipelineExecutor] = PipelineExecutor
    ) -> None:
        """
        Initialize audio processor with configuration and dependencies.

        Args:
            stt_config: STT service configuration containing audio
                requirements, model settings, and service parameters.
                Defaults to empty dict.
            pipeline_builder: PipelineBuilder class for creating pipelines
            pipeline_executor: PipelineExecutor class for running pipelines
        """
        self.stt_config = stt_config or {}
        self.sample_rate: Optional[int] = None
        self.audio_data: Optional[bytes] = None
        self.pipeline_builder = pipeline_builder(stt_config=self.stt_config)
        self.pipeline_executor = pipeline_executor()

    def load_audio_file(self, audio_path: str) -> None:
        """
        Load audio file and extract metadata.

        Reads WAV audio file, extracts sample rate and raw audio data,
        and logs audio characteristics for debugging. Stores audio data
        and metadata in instance variables for subsequent processing.

        Args:
            audio_path: Path to WAV audio file to load

        Raises:
            FileNotFoundError: If audio file doesn't exist
            wave.Error: If file is not a valid WAV file
        """
        with wave.open(audio_path, 'rb') as wf:
            self.sample_rate = wf.getframerate()
            self.audio_data = wf.readframes(wf.getnframes())
            duration_seconds = len(self.audio_data) / (self.sample_rate * 2)
            logger.info(f"Audio sample rate: {self.sample_rate}Hz")
            logger.info(f"Audio data size: {len(self.audio_data)} bytes")
            logger.info(f"Audio duration: {duration_seconds:.2f} seconds")

    def resample_if_needed(self) -> None:
        """
        Resample audio to meet STT service requirements.

        Checks STT configuration for audio requirements and resamples
        the loaded audio if the current sample rate doesn't match
        the required rates. Uses librosa for high-quality resampling.

        The resampling process:
        1. Convert bytes to float32 array
        2. Resample using librosa
        3. Convert back to int16 bytes
        4. Update sample rate
        """
        if not self.stt_config.get("audio_requirements"):
            return
        audio_reqs = self.stt_config["audio_requirements"]
        required_rates = audio_reqs.get("sample_rates", [])
        target_rate = audio_reqs.get("resample_to")

        if required_rates and \
                self.sample_rate not in required_rates and target_rate:
            logger.info(
                f"Resampling audio from {self.sample_rate} Hz "
                f"to {target_rate} Hz"
            )
            audio_array = np.frombuffer(
                self.audio_data, dtype=np.int16
            ).astype(np.float32) / 32768.0
            audio_array = librosa.resample(
                audio_array,
                orig_sr=self.sample_rate,
                target_sr=target_rate)
            self.audio_data = (
                audio_array * 32768.0
            ).astype(np.int16).tobytes()
            self.sample_rate = target_rate

    def process_audio_file(self, audio_path: str) -> None:
        """
        Complete audio file processing workflow.

        Orchestrates the full audio processing pipeline including
        file loading and conditional resampling. This is the main
        entry point for preparing audio files for pipeline execution.

        Args:
            audio_path: Path to audio file to process
        """
        self.load_audio_file(audio_path)
        self.resample_if_needed()
        logger.info(f'Successfully Retrieved Audio File {audio_path}')

    def build_pipeline(
            self,
            stt_service: Any,
            tts_service: Any,
            llm_service: Any
    ) -> PipelineComponents:
        """
        Build complete pipeline using PipelineBuilder.

        Delegates pipeline construction to the PipelineBuilder instance,
        passing the processed audio data and service instances.

        Args:
            stt_service: Speech-to-text service instance
            tts_service: Text-to-speech service instance
            llm_service: Large language model service instance

        Returns:
            PipelineComponents containing transport, pipeline, and collectors
        """
        return self.pipeline_builder.build_pipeline(
            self.audio_data,
            self.sample_rate,
            stt_service,
            tts_service,
            llm_service
        )

    async def execute_pipeline(
            self,
            pipeline_components: PipelineComponents,
            stt_service: Any,
            audio_path: str,
            stt_config: Optional[Dict[str, Any]] = None
    ) -> ExecutionResults:
        """
        Execute pipeline using PipelineExecutor.

        Delegates pipeline execution to the PipelineExecutor instance,
        handling the complete execution workflow and result collection.

        Args:
            pipeline_components: Complete pipeline components
            stt_service: STT service instance for batch preparation
            audio_path: Path to audio file for batch STT services
            stt_config: STT configuration for execution parameters

        Returns:
            ExecutionResults containing transcriptions, responses, and timing
        """
        return await self.pipeline_executor.execute_pipeline(
            pipeline_components, stt_service, audio_path, stt_config
        )

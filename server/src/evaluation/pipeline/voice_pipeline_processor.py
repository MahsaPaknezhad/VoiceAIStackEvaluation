"""
Voice pipeline processor for evaluation framework.
Core processing logic for voice assistant evaluation.
"""

import os
from typing import Dict, Any, Optional
from loguru import logger

from src.core.agent_builder import build_conversation_agent
from src.core.llm_processor import StrandsAgentsProcessor

from src.evaluation.config.configuration_manager import ConfigurationManager
from src.evaluation.factories.stt_factory import STTServiceFactory
from src.evaluation.factories.tts_factory import TTSServiceFactory
from src.evaluation.pipeline.audio_processor import AudioProcessor
from src.evaluation.services.service_manager import ServiceManager
from src.evaluation.results.results_collector import ResultCollector
from src.evaluation.models import PipelineResult


class VoicePipelineProcessor:
    """
    Core processor for voice pipeline evaluation.

    Handles the complete processing workflow for individual audio files
    including service creation, pipeline execution, and result collection.
    Separated from runner/orchestrator concerns for clean architecture.

    Key responsibilities:
    - Create STT/TTS services using factory pattern
    - Process audio files through complete voice pipeline
    - Coordinate service lifecycle and cleanup
    - Collect and return structured results

    Attributes:
        config_manager: Configuration manager for loading service configs
        stt_factory: Factory for creating STT service instances
        tts_factory: Factory for creating TTS service instances
        stt_config: STT service configuration dictionary
        tts_config: TTS service configuration dictionary
    """

    def __init__(
            self,
            stt_config: Optional[Dict[str, Any]] = None,
            tts_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize processor with service configurations.

        Args:
            stt_config: STT service configuration dictionary
            tts_config: TTS service configuration dictionary
        """
        self.config_manager = ConfigurationManager()
        self.stt_factory = STTServiceFactory(self.config_manager)
        self.tts_factory = TTSServiceFactory(self.config_manager)
        self.stt_config = stt_config
        self.tts_config = tts_config

    def _create_stt_service(self) -> Any:
        """
        Create STT service instance using factory pattern.

        Returns:
            Configured STT service instance based on loaded configuration
        """
        return self.stt_factory.create_service(self.stt_config)

    def _create_tts_service(self) -> Any:
        """
        Create TTS service instance using factory pattern.

        Returns:
            Configured TTS service instance based on loaded configuration
        """
        return self.tts_factory.create_service(self.tts_config)

    async def process_audio_file(
            self,
            audio_path: str,
            question_id: str,
            ground_truth: str
    ) -> PipelineResult:
        """
        Process single audio file through complete voice pipeline.

        Orchestrates the full processing workflow including audio
        preprocessing, service creation, pipeline execution,
        and result collection.

        Args:
            audio_path: Path to audio file to process
            question_id: Unique identifier for the question/audio pair
            ground_truth: Expected transcription for evaluation

        Returns:
            PipelineResult containing transcription, response, timing, metadata

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        logger.info(f"=== PROCESSING FILE: {question_id} ===")
        logger.info(f"Audio file path: {audio_path}")
        logger.info(f"File exists: {os.path.exists(audio_path)}")

        # Process audio
        audio_processor = AudioProcessor(self.stt_config)
        audio_processor.process_audio_file(audio_path)

        # Create services
        stt = self._create_stt_service()
        tts = self._create_tts_service()
        agent = build_conversation_agent(
            model_id="au.anthropic.claude-haiku-4-5-20251001-v1:0",
            tts_service=tts)
        llm = StrandsAgentsProcessor(agent=agent)

        # Setup pipeline
        pipeline_components = audio_processor.build_pipeline(stt, tts, llm)

        # Execute pipeline
        execution_results = await audio_processor.execute_pipeline(
            pipeline_components, stt, audio_path, self.stt_config
        )

        # Cleanup services
        service_manager = ServiceManager()
        await service_manager.cleanup_services(stt, tts)

        logger.info("Pipeline processing complete, continuing...")

        # Collect result
        result_collector = ResultCollector(self.stt_config, self.tts_config)
        result = result_collector.collect_result(
            execution_results,
            pipeline_components,
            question_id,
            audio_path,
            ground_truth
        )

        return result

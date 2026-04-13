import os
from loguru import logger
from typing import Dict, Any, Optional
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from .base_factory import BaseServiceFactory
from src.evaluation.services.batch_whisper_stt import BatchWhisperSTTService


class STTServiceFactory(BaseServiceFactory):
    """
    Factory for creating Speech-to-Text service instances.

    Supports multiple STT providers including:
    - Deepgram Nova models
    - AWS Transcribe
    - OpenAI Whisper
    - AssemblyAI
    - Speechmatics
    - Gladia

    Handles provider-specific configuration, authentication, and service
    instantiation.
    """

    def create_service(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create STT service instance from configuration.

        If no config provided, returns default Deepgram Nova-3 service.

        Args:
            config: STT service configuration containing:
                - module: Python module path
                (e.g., 'pipecat.services.deepgram.stt')
                - class: Service class name (e.g., 'DeepgramSTTService')
                - stt_service_id: Service identifier for API key resolution
                - config: Service-specific parameters

        Returns:
            Configured STT service instance ready for pipeline integration

        Raises:
            ImportError: If specified module/class cannot be imported
            ValueError: If required API key is missing
        """
        if not config:
            return DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    model="nova-3",
                    language="en",
                    smart_format=True)
            )

        return self._create_standard_service(config)

    def _create_standard_service(self, config: Dict[str, Any]) -> Any:
        """
        Create STT service using standard instantiation pattern.

        Args:
            config: Service configuration dictionary

        Returns:
            Configured STT service instance
        """
        module = __import__(config["module"], fromlist=[config["class"]])
        service_class = getattr(module, config["class"])

        # Use batch-enabled Whisper service for evaluation
        if config["class"] == "WhisperSTTService":
            service_class = BatchWhisperSTTService
            logger.info(
                "Using BatchWhisperSTTService for batch transcription support"
            )

        service_config = self.config_manager.substitute_env_vars(
            config.get("config", {})
        )

        service_id = config.get("stt_service_id", "")
        provider = service_id.split('_')[0].upper()

        if self._needs_api_key(provider, config["module"]):
            api_key = self._get_api_key_for_provider(service_id)
            logger.info(f"STT API key provided: {'Yes' if api_key else 'No'}")
            return service_class(api_key=api_key, **service_config)
        else:
            logger.info("STT service does not require API key")
            return service_class(**service_config)

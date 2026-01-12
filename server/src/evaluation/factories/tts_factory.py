import os
from loguru import logger
from typing import Dict, Any, Optional
from src.core.nvidia.livekit_tts_adapter import LiveKitTTSAdapter
from tts import DeepgramTTSService
from .base_factory import BaseServiceFactory


class TTSServiceFactory(BaseServiceFactory):
    """
    Factory for creating Text-to-Speech service instances.

    Supports multiple TTS providers including:
    - Deepgram Aura models
    - AWS Polly
    - ElevenLabs
    - Cartesia
    - PlayHT
    - LMNT
    - Rime
    - NVIDIA Riva (via LiveKit adapter)

    Handles provider-specific configuration, authentication, and service
    instantiation.
    Special handling for LiveKit-based services and NVIDIA adapters.
    """

    def create_service(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create TTS service instance from configuration.

        If no config provided, returns default Deepgram Aura service.

        Args:
            config: TTS service configuration containing:
                - module: Python module path (e.g., 'tts.DeepgramTTSService')
                - class: Service class name (e.g., 'DeepgramTTSService')
                - tts_service_id: Service identifier for API key resolution
                - config: Service-specific parameters (voice, model, etc.)

        Returns:
            Configured TTS service instance ready for pipeline integration

        Raises:
            ImportError: If specified module/class cannot be imported
            ValueError: If required API key is missing
        """
        if not config:
            return DeepgramTTSService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                voice="aura-2-delia-en"
            )

        # Handle LiveKit special cases
        if "livekit" in config["module"]:
            return self._create_livekit_service(config)

        # Standard service creation
        return self._create_standard_service(config)

    def _create_livekit_service(self, config: Dict[str, Any]) -> Any:
        """
        Create LiveKit adapter services for real-time TTS.

        Special handling for NVIDIA services that require LiveKit adapters
        for WebRTC streaming integration.

        Args:
            config: LiveKit service configuration

        Returns:
            LiveKit adapter or standard service instance
        """
        module_name = config["module"]
        service_config = self.config_manager.substitute_env_vars(
            config.get("config", {})
        )

        if "nvidia" in module_name:
            return LiveKitTTSAdapter(**service_config)
        else:
            # Fallback to standard service creation for other LiveKit services
            module = __import__(config["module"], fromlist=[config["class"]])
            service_class = getattr(module, config["class"])
            return service_class(**service_config)

    def _create_standard_service(self, config: Dict[str, Any]) -> Any:
        """
        Create TTS service using standard instantiation pattern.

        Handles API key injection for services that require authentication.
        AWS and NVIDIA services use credential-based auth instead of API keys.

        Args:
            config: Service configuration dictionary

        Returns:
            Configured TTS service instance
        """
        module = __import__(config["module"], fromlist=[config["class"]])
        service_class = getattr(module, config["class"])
        service_config = self.config_manager.substitute_env_vars(
            config.get("config", {})
        )

        service_id = config.get("tts_service_id", "")
        provider = service_id.split('_')[0].upper()

        logger.info(f"Creating TTS service: {service_class.__name__}")
        logger.info(f"TTS Service ID: {service_id}")
        logger.info(f"TTS Config passed to Pipecat: {service_config}")

        if self._needs_api_key(provider, config["module"]):
            api_key = self._get_api_key_for_provider(service_id)
            logger.info(f"TTS API key provided: {'Yes' if api_key else 'No'}")
            return service_class(api_key=api_key, **service_config)
        else:
            logger.info("TTS service does not require API key")
            return service_class(**service_config)

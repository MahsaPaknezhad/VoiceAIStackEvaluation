# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

import os
from abc import ABC, abstractmethod
from typing import Dict
from src.evaluation.config.configuration_manager import ConfigurationManager


class BaseServiceFactory(ABC):
    """
    Abstract factory for creating AI service instances.

    Provides common functionality for STT and TTS service creation including:
    - Configuration management integration
    - API key resolution from environment variables
    - Provider-specific service instantiation logic

    Args:
        config_manager: Configuration manager for loading and processing
        service configs
    """

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    @abstractmethod
    def create_service(self, config: Dict = None):
        """
        Create service instance from configuration.

        Args:
            config: Service configuration dictionary containing module, class,
            and parameters

        Returns:
            Configured service instance ready for use in pipeline
        """
        pass

    def _get_api_key_for_provider(self, service_id: str) -> str:
        """
        Resolve API key from environment based on service provider.

        Maps service IDs to environment variable names:
        - deepgram_nova -> DEEPGRAM_API_KEY
        - openai_whisper -> OPENAI_API_KEY

        Args:
            service_id: Service identifier (e.g., 'deepgram_nova',
            'aws_transcribe')

        Returns:
            API key from environment variable or empty string if not found
        """
        provider = service_id.split('_')[0].upper() if service_id else ''
        return os.getenv(f"{provider}_API_KEY")

    def _needs_api_key(self, provider: str, module_name: str) -> bool:
        """
        Determine if service requires API key authentication.

        AWS and NVIDIA services use different auth mechanisms.
        Riva services use gRPC authentication.

        Args:
            provider: Provider name (e.g., 'DEEPGRAM', 'AWS', 'NVIDIA')
            module_name: Python module name for the service

        Returns:
            True if service requires API key, False otherwise
        """
        return provider not in {"AWS", "NVIDIA"} and "riva" not in module_name

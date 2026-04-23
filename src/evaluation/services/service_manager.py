# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
Service manager for voice evaluation framework.
Handles service lifecycle management and cleanup.
"""

import asyncio
from typing import Any
from loguru import logger

from pipecat.frames.frames import EndFrame


class ServiceManager:
    """
    Manages lifecycle and cleanup of voice evaluation services.

    This class provides centralized management for STT and TTS service
    cleanup, handling the graceful shutdown of various service types
    with appropriate error handling for different connection types.

    The cleanup process ensures proper resource deallocation and
    connection termination for both streaming and batch services.
    """

    async def cleanup_services(
            self,
            stt_service: Any,
            tts_service: Any
    ) -> None:
        """
        Clean shutdown of all voice services.

        Orchestrates the cleanup of both STT and TTS services in sequence,
        allowing time for proper resource deallocation. Handles cleanup
        errors gracefully to prevent hanging processes.

        Args:
            stt_service: Speech-to-text service instance to cleanup
            tts_service: Text-to-speech service instance to cleanup
        """
        await self._cleanup_stt_service(stt_service)
        await self._cleanup_tts_service(tts_service)
        await asyncio.sleep(0.1)  # Give time for cleanup

    async def _cleanup_stt_service(self, stt_service: Any) -> None:
        """
        Clean shutdown of STT service.

        Attempts to disconnect STT service if it supports the disconnect
        method. Handles expected disconnection errors gracefully as
        these are common during normal shutdown procedures.

        Args:
            stt_service: STT service instance to cleanup
        """
        try:
            if hasattr(stt_service, 'disconnect'):
                await stt_service.disconnect()
        except Exception as e:
            logger.debug(f"STT disconnect error (expected): {e}")

    async def _cleanup_tts_service(self, tts_service: Any) -> None:
        """
        Clean shutdown of TTS service.

        Attempts to stop TTS service with EndFrame if it supports the
        stop method. Handles WebSocket close errors specially for
        services like Cartesia that use WebSocket connections.

        Args:
            tts_service: TTS service instance to cleanup
        """
        try:
            if hasattr(tts_service, 'stop'):
                await tts_service.stop(EndFrame())
        except Exception as e:
            # Ignore WebSocket close errors for Cartesia
            if "sent 1000 (OK); then received 1000 (OK)" in str(e):
                pass  # Normal WebSocket close
            else:
                logger.error(f"TTS stop error: {e}")

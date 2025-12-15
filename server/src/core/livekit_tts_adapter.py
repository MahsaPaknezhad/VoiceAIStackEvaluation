import asyncio
import uuid
import os
from typing import Optional
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, TTSAudioRawFrame
from loguru import logger
from livekit.plugins import nvidia

class LiveKitTTSAdapter(FrameProcessor):
    def __init__(self, server: str, voice: str, use_ssl: bool = False):
        super().__init__()
        
        # Handle environment variable substitution
        if isinstance(server, str) and server.startswith('${') and '}' in server:
            # Extract environment variable name and any suffix
            if ':' in server:
                # Handle ${VAR}:port format
                env_part, suffix = server.split(':', 1)
                env_var = env_part[2:-1]  # Remove ${ and }
                env_value = os.getenv(env_var)
                if env_value:
                    server = f"{env_value}:{suffix}"
                else:
                    logger.error(f"Environment variable {env_var} not found")
            else:
                # Handle ${VAR} format
                env_var = server[2:-1]  # Remove ${ and }
                env_value = os.getenv(env_var)
                if env_value:
                    server = env_value
                else:
                    logger.error(f"Environment variable {env_var} not found")
        
        logger.info(f"Creating NVIDIA TTS with server: {server}, voice: {voice}, use_ssl: {use_ssl}")
        
        self.tts = nvidia.TTS(
            server=server,
            voice=voice,
            use_ssl=use_ssl
        )
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            # Let TextFrame pass through first to trigger collection
            await self.push_frame(frame, direction)
            try:
                # Generate audio using LiveKit NVIDIA TTS streaming
                audio_stream = self.tts.stream()
                audio_stream.push_text(frame.text)
                audio_stream.end_input()
                
                audio_data = b''
                async for event in audio_stream:
                    if hasattr(event, 'frame') and hasattr(event.frame, 'data'):
                        audio_data += event.frame.data
                
                await audio_stream.aclose()
                # Convert to Pipecat audio frame
                tts_frame = TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.tts.sample_rate,
                    num_channels=self.tts.num_channels
                )
                # Add frame ID for Pipecat observers
                tts_frame.id = str(uuid.uuid4())
                await self.push_frame(tts_frame, direction)
            except Exception as e:
                logger.error(f"TTS error: {e}")
        else:
            await self.push_frame(frame, direction)
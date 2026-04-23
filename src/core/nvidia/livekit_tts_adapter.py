import asyncio
import uuid
import os
from typing import Optional
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, TTSAudioRawFrame, LLMTextFrame
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
        
        # Initialize buffering for improved streaming
        self._text_buffer = []
        self._processing_tts = False
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, (TextFrame, LLMTextFrame)):
            # Let TextFrame pass through first to trigger collection
            await self.push_frame(frame, direction)
            
            # Buffer text chunks for better TTS processing
            import re
            cleaned_text = re.sub(r'\s+', ' ', frame.text.strip())
            self._text_buffer.append(cleaned_text)
            
            # Process TTS only at sentence boundaries for natural prosody
            full_text = ''.join(self._text_buffer)
            should_process = any(punct in cleaned_text for punct in ['.', '!', '?'])
            
            if should_process and not self._processing_tts:
                await self._process_tts_chunk(full_text, direction)
                self._text_buffer = []
        else:
            await self.push_frame(frame, direction)
    
    async def _process_tts_chunk(self, text, direction):
        """Process a chunk of text through TTS with better audio handling."""
        if not text.strip():
            return
            
        self._processing_tts = True
        try:
            # Generate audio using LiveKit NVIDIA TTS streaming
            audio_stream = self.tts.stream()
            audio_stream.push_text(text)
            audio_stream.end_input()
            
            # Collect audio with proper streaming
            audio_chunks = []
            async for event in audio_stream:
                audio_chunk = None
                if hasattr(event, 'frame') and hasattr(event.frame, 'data'):
                    audio_chunk = event.frame.data
                elif hasattr(event, 'data'):
                    audio_chunk = event.data
                elif hasattr(event, 'audio'):
                    audio_chunk = event.audio
                elif hasattr(event, 'frame') and hasattr(event.frame, 'audio'):
                    audio_chunk = event.frame.audio
                
                if audio_chunk:
                    # Convert to bytes and store
                    if isinstance(audio_chunk, bytes):
                        audio_chunks.append(audio_chunk)
                    elif hasattr(audio_chunk, 'tobytes'):
                        audio_chunks.append(audio_chunk.tobytes())
            
            await audio_stream.aclose()
            
            # Combine and smooth audio chunks for natural flow
            if audio_chunks:
                sample_rate = getattr(self.tts, 'sample_rate', 24000)
                num_channels = getattr(self.tts, 'num_channels', 1)
                
                # Concatenate all chunks into one smooth audio segment
                combined_audio = b''.join(audio_chunks)
                
                if len(combined_audio) > 0:
                    # Add brief silence padding for natural sentence breaks
                    silence_samples = int(0.2 * sample_rate * 2)  # 200ms silence, 2 bytes per sample
                    silence = b'\x00' * silence_samples
                    
                    # Create single frame with padding
                    final_audio = combined_audio + silence
                    
                    tts_frame = TTSAudioRawFrame(
                        audio=final_audio,
                        sample_rate=sample_rate,
                        num_channels=num_channels
                    )
                    tts_frame.id = str(uuid.uuid4())
                    await self.push_frame(tts_frame, direction)
                        
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self._processing_tts = False
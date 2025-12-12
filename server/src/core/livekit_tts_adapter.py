import asyncio
from typing import Optional
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, TTSAudioRawFrame
from livekit.plugins import nvidia

class LiveKitTTSAdapter(FrameProcessor):
    def __init__(self, server: str, voice: str, use_ssl: bool = False):
        super().__init__()
        self.tts = nvidia.TTS(
            server=server,
            voice=voice,
            use_ssl=use_ssl
        )
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
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
                await self.push_frame(tts_frame, direction)
            except Exception as e:
                print(f"TTS error: {e}")
        else:
            await self.push_frame(frame, direction)
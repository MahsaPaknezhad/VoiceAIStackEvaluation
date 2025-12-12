#!/usr/bin/env python3
"""
Quick test script for TTS services without running full pipeline
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

async def test_tts_service(config_path: str, test_text: str = "Hello, this is a test of the TTS service."):
    """Test a TTS service configuration"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Testing {config['tts_service_name']}")
    
    try:
        # Import service dynamically
        module_name = config["module"]
        class_name = config["class"]
        service_config = config.get("config", {})
        
        logger.info(f"Importing {class_name} from {module_name}")
        
        if "livekit" in module_name:
            from src.core.livekit_tts_adapter import LiveKitTTSAdapter
            tts = LiveKitTTSAdapter(**service_config)
        else:
            module = __import__(module_name, fromlist=[class_name])
            service_class = getattr(module, class_name)
            
            # Get API key if needed
            api_key = None
            if "deepgram" in module_name:
                api_key = os.getenv("DEEPGRAM_API_KEY")
            elif "nvidia" in module_name:
                api_key = os.getenv("NVIDIA_API_KEY")
            
            if api_key:
                tts = service_class(api_key=api_key, **service_config)
            else:
                tts = service_class(**service_config)
        
        logger.info(f"✅ TTS service created successfully")
        logger.info(f"Config: {service_config}")
        
        # Test synthesis if it's a LiveKit adapter
        if hasattr(tts, 'tts'):
            try:
                logger.info("Using LiveKit TTS stream method")
                audio_stream = tts.tts.stream()
                
                # Debug stream object
                logger.info(f"Stream type: {type(audio_stream)}")
                logger.info(f"Stream methods: {[m for m in dir(audio_stream) if not m.startswith('_')]}")
                
                # Use correct methods
                audio_stream.push_text(test_text)
                audio_stream.end_input()
                
                audio_data = b''
                chunk_count = 0
                async for event in audio_stream:
                    if hasattr(event, 'frame') and hasattr(event.frame, 'data'):
                        audio_data += event.frame.data
                        chunk_count += 1
                
                await audio_stream.aclose()
                logger.info(f"✅ TTS synthesis test completed, audio length: {len(audio_data)} bytes")
                    
            except Exception as e:
                logger.error(f"❌ TTS synthesis failed: {e}")
        else:
            logger.info("ℹ️  Direct synthesis test not available for this service type")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TTS test failed: {e}")
        return False

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test TTS service configuration')
    parser.add_argument('--config', required=True, help='TTS config file path')
    parser.add_argument('--text', default="Hello, this is a test of the TTS service.", help='Test text')
    
    args = parser.parse_args()
    
    success = await test_tts_service(args.config, args.text)
    
    if success:
        print("✅ TTS test completed successfully")
    else:
        print("❌ TTS test failed")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
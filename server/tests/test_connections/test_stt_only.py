#!/usr/bin/env python3
"""
Quick test script for STT services without running full pipeline
"""
import asyncio
import json
import os
import wave
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

async def test_stt_service(config_path: str, audio_path: str):
    """Test an STT service configuration"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Testing {config['stt_service_name']}")
    
    try:
        # Import service dynamically
        module_name = config["module"]
        class_name = config["class"]
        service_config = config.get("config", {})
        
        logger.info(f"Importing {class_name} from {module_name}")
        
        module = __import__(module_name, fromlist=[class_name])
        service_class = getattr(module, class_name)
        
        # Create service
        if "deepgram" in module_name:
            from pipecat.services.deepgram.stt import LiveOptions
            stt = service_class(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(**service_config)
            )
        elif "aws" in module_name:
            stt = service_class(
                aws_access_key_id=None,
                aws_secret_access_key=None,
                **service_config
            )
        else:
            stt = service_class(**service_config)
        
        logger.info(f"✅ STT service created successfully")
        logger.info(f"Config: {service_config}")
        
        # Load audio file
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            logger.info(f"Loaded audio: {len(audio_data)} bytes at {sample_rate}Hz")
        
        logger.info("ℹ️  STT service created - full transcription requires pipeline integration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ STT test failed: {e}")
        return False

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test STT service configuration')
    parser.add_argument('--config', required=True, help='STT config file path')
    parser.add_argument('--audio', required=True, help='Audio file path')
    
    args = parser.parse_args()
    
    success = await test_stt_service(args.config, args.audio)
    
    if success:
        print("✅ STT test completed successfully")
    else:
        print("❌ STT test failed")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
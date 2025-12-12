#!/usr/bin/env python3
"""
Test the exact evaluation pipeline flow for AWS Transcribe
"""
import asyncio
import json
import os
import wave
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

async def test_evaluation_pipeline():
    """Test the exact same flow as voice_pipeline_evaluator.py"""
    
    # Load AWS Transcribe config
    config_path = "evaluation_data/stt_bot_configs/aws_transcribe_config.json"
    with open(config_path, 'r') as f:
        stt_config = json.load(f)
    
    logger.info(f"Testing with config: {stt_config}")
    
    try:
        # Import the service class dynamically (same as evaluator)
        module_name = stt_config["module"]
        class_name = stt_config["class"]
        service_config = stt_config.get("config", {})
        
        logger.info(f"Importing {class_name} from {module_name}")
        
        module = __import__(module_name, fromlist=[class_name])
        service_class = getattr(module, class_name)
        
        # Create service exactly like the evaluator does
        if "aws" in module_name:
            import boto3
            session = boto3.Session()
            credentials = session.get_credentials()
            
            logger.info("Creating AWS service with explicit credentials from boto3 session")
            stt = service_class(
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
                **service_config
            )
        else:
            stt = service_class(**service_config)
        
        logger.info("✅ STT service created successfully")
        logger.info(f"Service type: {type(stt)}")
        
        # Test with a small pipeline
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.frames.frames import StartFrame, EndFrame
        from src.transport.batch_audio_transport import EvaluationTransport
        
        # Create a small audio file for testing
        sample_rate = 16000
        duration = 1  # 1 second
        import numpy as np
        audio_data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, sample_rate * duration)) * 32767).astype(np.int16).tobytes()
        
        transport = EvaluationTransport(audio_data, sample_rate)
        
        # Simple pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            transport.output()
        ])
        
        task = PipelineTask(pipeline, params=PipelineParams())
        
        logger.info("Testing pipeline execution...")
        
        await task.queue_frames([
            StartFrame(),
            EndFrame()
        ])
        
        runner = PipelineRunner()
        await runner.run(task)
        
        logger.info("✅ Pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_evaluation_pipeline())
    
    if success:
        print("✅ Evaluation pipeline test passed!")
    else:
        print("❌ Evaluation pipeline test failed!")
        exit(1)
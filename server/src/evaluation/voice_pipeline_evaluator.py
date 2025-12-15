"""
Run the Q&A bot on VoiceAssistant-Eval dataset audio files.
Collects STT outputs and bot responses for evaluation.
"""

import json
import os
import asyncio
import time
import random
from typing import Dict, List
from loguru import logger
import argparse
from pathlib import Path
import re

# Import Pipecat components
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.frames.frames import EndFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator
from dotenv import load_dotenv
import wave
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.agent_builder import build_conversation_agent
from src.core.llm_processor import StrandsAgentsProcessor
from tts import DeepgramTTSService
from src.evaluation.audio_quality_analyzer import VoiceQualityEvaluator
from src.transport.batch_audio_transport import EvaluationTransport

# frame processing
from src.evaluation.frame_processor import (
    TimingCollector,
    STTTimingProcessor,
    TTSTimingProcessor,
    STTCollector,
    LLMCollector
)

load_dotenv(override=True)


class VoiceAssistantRunner:
    def __init__(self, dataset_path: str, audio_dir: str, stt_config: str = None, 
                 tts_config: str = None, evaluate_voice_quality: bool = True, 
                 use_llm_judge: bool = False):
        self.dataset_path = dataset_path
        self.audio_dir = audio_dir
        self.evaluate_voice_quality = evaluate_voice_quality
        self.dataset = self._load_dataset()
        
        # Load configs
        self.stt_config = self._load_config(stt_config) if stt_config else None
        self.tts_config = self._load_config(tts_config) if tts_config else None
        
        if evaluate_voice_quality:
            self.voice_evaluator = VoiceQualityEvaluator(use_llm_judge=use_llm_judge)
            self.use_llm_judge = use_llm_judge
    
    def _load_config(self, config_path: str) -> Dict:
        """Load service config from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            logger.info(f'Loading in config {config_path}')
            json_file = json.load(f)
            return json_file
    
    def _create_stt_service(self):
        """Create STT service from config"""
        logger.info("Creating STT service...")
        if not self.stt_config:
            # Default
            return DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(model="nova-3", language="en", smart_format=True)
            )
        
        # Import the service class dynamically
        module_name = self.stt_config["module"]
        class_name = self.stt_config["class"]
        
        module = __import__(module_name, fromlist=[class_name])
        service_class = getattr(module, class_name)
        
        # Get config params
        config = self.stt_config.get("config", {})
        
        # Create service
        if "deepgram" in module_name:
            return service_class(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(**config)
            )
        elif "openai" in module_name:
            return service_class(
                api_key=os.getenv("OPENAI_API_KEY"),
                **config
            )
        elif "aws" in module_name:
            return service_class(**config)
        else:
            return service_class(**config)
    
    def _create_tts_service(self):
        """Create TTS service from config"""
        logger.info("Creating TTS service...")
        if not self.tts_config:
            # Default
            return DeepgramTTSService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                voice="aura-2-delia-en"
            )

        # Import the service class dynamically
        module_name = self.tts_config["module"]
        class_name = self.tts_config["class"]
        
        logger.info(f"Importing {class_name} from {module_name}")
        
        try:
            module = __import__(module_name, fromlist=[class_name])
            service_class = getattr(module, class_name)
        except Exception as e:
            logger.error(f"Failed to import {class_name} from {module_name}: {e}")
            raise
        
        # Get config params and substitute environment variables
        config = self.tts_config.get("config", {})
        
        # Substitute environment variables in config values
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]  # Remove ${ and }
                if ':' in env_var:  # Handle ${VAR}:port format
                    env_var, suffix = env_var.split(':', 1)
                    env_value = os.getenv(env_var)
                    if env_value:
                        config[key] = f"{env_value}:{suffix}"
                else:
                    env_value = os.getenv(env_var)
                    if env_value:
                        config[key] = env_value
        
        # Determine API key based on service
        api_key = None
        if "deepgram" in module_name:
            api_key = os.getenv("DEEPGRAM_API_KEY")
        elif "openai" in module_name:
            api_key = os.getenv("OPENAI_API_KEY")
        elif "elevenlabs" in module_name:
            api_key = os.getenv("ELEVENLABS_API_KEY")
        elif "cartesia" in module_name:
            api_key = os.getenv("CARTESIA_API_KEY")
        elif "nvidia" in module_name or "riva" in module_name:
            api_key = os.getenv("NVIDIA_API_KEY")
        elif "fish" in module_name:
            api_key = os.getenv("FISH_AUDIO_API_KEY")
        elif "lmnt" in module_name:
            api_key = os.getenv("LMNT_API_KEY")
        elif "playht" in module_name:
            api_key = os.getenv("PLAYHT_API_KEY")
        elif "rime" in module_name:
            api_key = os.getenv("RIME_API_KEY")
        
        logger.info(f"Using API key: {'***' if api_key else 'None'}")
        logger.info(f"Final config after env substitution: {config}")
        
        try:
            if "livekit" in module_name:
                if "nvidia" in module_name:
                    from src.core.livekit_tts_adapter import LiveKitTTSAdapter
                    return LiveKitTTSAdapter(**config)
                elif "patakeet" in module_name:
                    from src.core.livekit_patakeet_adapter import LiveKitPatakeetAdapter
                    return LiveKitPatakeetAdapter(**config)
                else:
                    # Generic LiveKit adapter fallback
                    from src.core.livekit_tts_adapter import LiveKitTTSAdapter
                    return LiveKitTTSAdapter(**config)
            elif "nvidia" in module_name or "riva" in module_name:
                # NVIDIA services don't use api_key parameter
                return service_class(**config)
            elif api_key:
                return service_class(api_key=api_key, **config)
            else:
                return service_class(**config)
        except Exception as e:
            logger.error(f"Failed to create {class_name} with config {config}: {e}")
            raise
        
    def _load_dataset(self) -> Dict:
        """Load the evaluation dataset"""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    async def process_audio_file(self, audio_path: str, question_id: str) -> Dict:
        """
        Process a single audio file through the bot pipeline.
        
        Returns:
            Dict with stt_output, bot_response, and latencies
        """
        logger.info(f"Processing {question_id}: {audio_path}")
        
        # Get the ground truth transcript from dataset (for WER comparison)
        question_data = next((q for q in self.dataset['questions'] if q['id'] == question_id), None)
        if not question_data:
            raise ValueError(f"Question {question_id} not found in dataset")
        
        ground_truth = question_data['text']
        # Read audio file
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            logger.info(f"File {audio_path} loaded with sample rate {sample_rate}")
        
        transport = EvaluationTransport(audio_data, sample_rate)
        
        # Create services
        stt = self._create_stt_service()
        tts = self._create_tts_service()
        logger.info('Successfully created STT and TTS Service')
        agent = build_conversation_agent(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", tts_service=tts)
        llm = StrandsAgentsProcessor(agent=agent)
        
        # Setup context
        #context = AWSBedrockLLMContext()
        context = OpenAILLMContext()
        tma_in = LLMUserContextAggregator(context=context)
        tma_out = LLMAssistantContextAggregator(context=context)
        
        # Collectors
        stt_texts = []
        llm_texts = []
        
        # Start frame processing
        timing_collector = TimingCollector()
        stt_timing_start = STTTimingProcessor(timing_collector)
        tts_timing_end = TTSTimingProcessor(timing_collector)
        stt_collector = STTCollector(timing_collector, stt_texts)
        llm_collector = LLMCollector(timing_collector, llm_texts)
        logger.info('Successfully completed video frame processing')
        
        # Build pipeline: STT -> context -> LLM -> context -> TTS
        pipeline = Pipeline([
            transport.input(),
            stt_timing_start,
            stt,
            stt_collector,
            tma_in,
            llm,
            llm_collector,
            #tma_out,
            tts,
            tts_timing_end,
            transport.output()
        ])
        logger.info('Pipeline build complete')
        
        task = PipelineTask(pipeline, params=PipelineParams())
        
        # Process audio
        start_time = time.time()
        
        await task.queue_frames([
            StartFrame(),
            EndFrame()
        ])
        
        # Run pipeline with error handling
        runner = PipelineRunner()
        try:
            await runner.run(task)
        except Exception as pipeline_error:
            # Re-raise the error to be caught by outer retry logic
            raise pipeline_error
        
        total_latency = (time.time() - start_time) * 1000
        
        # Collect results
        stt_output = " ".join(stt_texts)
        llm_response = " ".join(llm_texts)
        
        # Save TTS audio from transport
        tts_audio_path = None
        output_audio = transport.get_output_audio()
        print(f"DEBUG: Output audio length: {len(output_audio) if output_audio else 0}")
        if output_audio:
            # Create experiment-specific directory
            stt_id = self.stt_config.get("stt_service_id") if self.stt_config else "default_stt"
            tts_id = self.tts_config.get("tts_service_id") if self.tts_config else "default_tts"
            experiment_name = f"{stt_id}_{tts_id}"
            output_dir = os.path.join("evaluation_output", "tts_audio", experiment_name)
            os.makedirs(output_dir, exist_ok=True)
            tts_audio_path = os.path.join(output_dir, f"{question_id}_response.wav")
            
            # Get the actual sample rate from the transport output processor
            sample_rate = transport.output().sample_rate if hasattr(transport.output(), 'sample_rate') else 16000
            
            with wave.open(tts_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(output_audio)
            print(f"DEBUG: Saved TTS audio to {tts_audio_path} at {sample_rate}Hz")
        else:
            print("DEBUG: No output audio collected")
        # Calculate latencies
        stt_latency = timing_collector.get_stt_latency_ms()
        print(f'TTS START TIME: {timing_collector.tts_start_time}')
        print(f'TTS END TIME: {timing_collector.tts_end_time}')
        tts_latency = timing_collector.get_tts_latency_ms()
        
        result = {
            "question_id": question_id,
            "audio_file": audio_path,
            "stt_output": stt_output,
            "ground_truth": ground_truth,
            "llm_response": llm_response,
            "tts_audio_path": tts_audio_path,
            "stt_latency_ms": round(stt_latency, 2) if stt_latency is not None else None,
            "tts_latency_ms": round(tts_latency, 2) if tts_latency is not None else None,
            "total_latency_ms": round(total_latency, 2) if total_latency is not None else None
        }
        
        # Add voice quality metrics if enabled
        if self.evaluate_voice_quality and tts_audio_path:
            try:
                if self.use_llm_judge:
                    voice_metrics = await self.voice_evaluator.evaluate_async(tts_audio_path, result["bot_response"])
                else:
                    voice_metrics = self.voice_evaluator.evaluate(tts_audio_path)
                
                result["voice_quality"] = voice_metrics
            except Exception as e:
                logger.error(f"Voice quality evaluation failed: {e}")
        
        return result
    
    
    async def run_all(self, output_path: str = None) -> List[Dict]:
        """Run bot on all audio files in dataset"""
        results = []
        total_files = len(self.dataset['questions'])
        processed_count = 0
        error_count = 0
        skipped_count = 0
        
        logger.info(f"Starting evaluation of {total_files} audio files")
        
        for i, question in enumerate(self.dataset['questions']):
            question_id = question['id']
            audio_file = question['audio_file']
            audio_path = os.path.join(self.audio_dir, audio_file)
            
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                skipped_count += 1
                continue
            
            # Retry logic for Bedrock failures
            max_retries = 3
            result = None
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {question_id}")
                    result = await self.process_audio_file(audio_path, question_id)
                    
                    # Check if TTS actually worked (if TTS config is provided)
                    if self.tts_config and result.get("tts_audio_path") is None:
                        # TTS was expected but failed
                        result["status"] = "failed"
                        result["error"] = "TTS failed - no audio generated"
                        error_count += 1
                    else:
                        result["status"] = "success"
                        processed_count += 1
                    
                    results.append(result)
                    
                    # Save results after each sample
                    if output_path:
                        self.save_results(results, output_path)
                    if attempt > 0:
                        logger.info(f"Retry successful for {question_id}")
                    break  # Success, exit retry loop
                        
                except Exception as e:
                    error_str = str(e).lower()
                    is_bedrock_error = any([
                        "serviceunavailableexception" in error_str,
                        "bedrock is unable to process" in error_str,
                        "throttlingexception" in error_str,
                        "rate limit" in error_str,
                        "too many requests" in error_str,
                        "service temporarily unavailable" in error_str,
                        "eventstreamError" in error_str,
                        "conversestream operation" in error_str,
                        "botocore.exceptions.eventstreamerror" in error_str
                    ])
                    
                    if attempt < max_retries - 1:  # Retry all errors, not just Bedrock ones
                        wait_time = (3 ** (attempt + 1)) + random.uniform(0, 1)  # Add jitter
                        error_type = "Bedrock" if is_bedrock_error else "General"
                        logger.warning(f"{error_type} error on {question_id} (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Error processing {question_id} (final attempt): {e}")
                        error_count += 1
                        results.append({
                            "question_id": question_id,
                            "audio_file": audio_path,
                            "stt_output": "",
                            "llm_response": "",
                            "error": str(e),
                            "status": "failed"
                        })
                        
                        # Save results even on error
                        if output_path:
                            self.save_results(results, output_path)
                        break
            
            # Pause between items (regardless of success/failure)
            if i < len(self.dataset['questions']) - 1:  # Don't pause after last item
                logger.info("Pausing 3 seconds to avoid rate limiting...")
                await asyncio.sleep(3)
        
        # Log final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total files in dataset: {total_files}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed with errors: {error_count}")
        logger.info(f"Skipped (file not found): {skipped_count}")
        success_rate = (processed_count/(processed_count + error_count))*100 if (processed_count + error_count) > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"{'='*60}")
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Calculate summary statistics
        successful = len([r for r in results if r.get('status') == 'success'])
        failed = len([r for r in results if r.get('status') == 'failed'])
        
        # Add metadata about STT and TTS models
        output_data = {
            "stt_model": self.stt_config.get("stt_service_name") if self.stt_config else None,
            "stt_service_id": self.stt_config.get("stt_service_id") if self.stt_config else None,
            "tts_model": self.tts_config.get("tts_service_name") if self.tts_config else None,
            "tts_service_id": self.tts_config.get("tts_service_id") if self.tts_config else None,
            "summary": {
                "total_files": len(results),
                "successful": successful,
                "failed": failed,
                "skipped": len(results) - successful - failed,
                "success_rate": round((successful/len(results))*100, 1) if results else 0
            },
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Run bot on VoiceAssistant-Eval dataset')
    parser.add_argument('--dataset', default='evaluation_data/voiceassistant_eval/voiceassistant_eval_dataset.json',
                       help='Path to dataset JSON')
    parser.add_argument('--audio-dir', default='evaluation_data/voiceassistant_eval/audio_input',
                       help='Directory containing audio files')
    parser.add_argument('--output', default='evaluation_data/voiceassistant_eval/bot_results.json',
                       help='Output path for bot results')
    parser.add_argument('--stt-config', help='STT service config (e.g., evaluation_data/bot_configs/deepgram_nova3_config.json)')
    parser.add_argument('--tts-config', help='TTS service config (e.g., evaluation_data/tts_bot_configs/deepgram_aura_config.json)')
    parser.add_argument('--voice-quality', action='store_true',
                       help='Enable voice quality evaluation')
    parser.add_argument('--llm-judge', action='store_true',
                       help='Use LLM judge for voice quality (slower)')
    
    args = parser.parse_args()
    
    runner = VoiceAssistantRunner(
        args.dataset, 
        args.audio_dir,
        stt_config=args.stt_config,
        tts_config=args.tts_config,
        evaluate_voice_quality=args.voice_quality,
        use_llm_judge=args.llm_judge
    )
    
    logger.info("Running bot on all audio files...")
    results = await runner.run_all(args.output)
    
    runner.save_results(results, args.output)
    
    # Count successful vs failed results
    successful = len([r for r in results if r.get('status') == 'success'])
    failed = len([r for r in results if r.get('status') == 'failed'])
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(results))*100:.1f}%" if results else "No results")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
    print("\nNext step: Run evaluation with:")
    print(f"  python evaluate_voiceassistant.py --results {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

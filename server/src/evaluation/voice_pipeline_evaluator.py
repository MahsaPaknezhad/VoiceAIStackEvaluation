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
import warnings
warnings.filterwarnings("ignore", message="Dangling tasks detected")
import argparse
from pathlib import Path
import re

# Import Pipecat components
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import AudioRawFrame, EndFrame, StartFrame, TranscriptionFrame, TextFrame, LLMFullResponseStartFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.aws.llm import AWSBedrockLLMContext
from pipecat.services.aws.llm import AWSBedrockLLMContext
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

# Suppress Pipecat cleanup warnings
logger.disable("pipecat.pipeline.task")


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
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _substitute_env_vars(self, config: Dict) -> Dict:
        """Substitute environment variables in config values"""
        result = config.copy()
        for key, value in result.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]  # Remove ${ and }
                if ':' in env_var:  # Handle ${VAR}:port format
                    env_var, suffix = env_var.split(':', 1)
                    env_value = os.getenv(env_var)
                    if env_value:
                        result[key] = f"{env_value}:{suffix}"
                else:
                    env_value = os.getenv(env_var)
                    if env_value:
                        result[key] = env_value
        return result
    
    def _create_stt_service(self):
        """Create STT service from config"""
        if not self.stt_config:
            # Default
            return DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(model="nova-3", language="en", smart_format=True)
            )
        
        # Import the service class dynamically
        module_name = self.stt_config["module"]
        class_name = self.stt_config["class"]
        
        # Special handling for NVIDIA/LiveKit STT services
        if "livekit" in module_name and "nvidia" in module_name:
            from src.core.livekit_stt_adapter import LiveKitSTTAdapter
            config = self.stt_config.get("config", {})
            return LiveKitSTTAdapter(**config)
        
        module = __import__(module_name, fromlist=[class_name])
        service_class = getattr(module, class_name)
        
        # Get config params and substitute environment variables
        config = self._substitute_env_vars(self.stt_config.get("config", {}))
        
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
            # AWS services use default credential chain
            return service_class(**config)
        else:
            return service_class(**config)
    
    def _create_tts_service(self):
        """Create TTS service from config"""
        if not self.tts_config:
            # Default
            return DeepgramTTSService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                voice="aura-2-delia-en"
            )
        
        # Import the service class dynamically
        module_name = self.tts_config["module"]
        class_name = self.tts_config["class"]
        
        module = __import__(module_name, fromlist=[class_name])
        service_class = getattr(module, class_name)
        
        # Get config params and substitute environment variables
        config = self._substitute_env_vars(self.tts_config.get("config", {}))
        
        # Determine API key based on service
        api_key = None
        if "deepgram" in module_name:
            return service_class(api_key=os.getenv("DEEPGRAM_API_KEY"), **config)
        elif "openai" in module_name:
            return service_class(api_key=os.getenv("OPENAI_API_KEY"), **config)
        elif "elevenlabs" in module_name:
            return service_class(api_key=os.getenv("ELEVENLABS_API_KEY"), **config)
        elif "cartesia" in module_name:
            return service_class(api_key=os.getenv("CARTESIA_API_KEY"), **config)
        elif "riva" in module_name:
            return service_class(api_key=os.getenv("RIVA_API_KEY"), **config)
        elif "fish" in module_name:
            return service_class(api_key=os.getenv("FISH_AUDIO_API_KEY"), **config)
        elif "lmnt" in module_name:
            return service_class(api_key=os.getenv("LMNT_API_KEY"), **config)
        elif "playht" in module_name:
            return service_class(api_key=os.getenv("PLAYHT_API_KEY"), **config)
        elif "rime" in module_name:
            api_key = os.getenv("RIME_API_KEY")
        elif "groq" in module_name:
            api_key = os.getenv("GROQ_API_KEY")
        
        logger.info(f"Using API key: {'***' if api_key else 'None'}")
        logger.info(f"Final config after env substitution: {config}")
        
        try:
            if "livekit" in module_name:
                if "nvidia" in module_name:
                    from src.core.nvidia.livekit_tts_adapter import LiveKitTTSAdapter
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
            elif "aws" in module_name:
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
    
    def _extract_final_response(self, full_text: str) -> str:
        """Extract the final/most complete response from concatenated LLM outputs.
        
        Strategy: Split on common response patterns and take the last complete response.
        """
        if not full_text:
            return ""
        
        # Common patterns that indicate start of new responses
        response_starters = [
            "I notice your question",
            "I'd be happy to help", 
            "I'm not quite sure",
            "I see your question",
            "I'm still not",
            "I'm getting closer",
            "I appreciate",
            "I still can't",
            "I'm not able"
        ]
        
        # Find all potential response boundaries
        boundaries = [0]  # Start of text
        for starter in response_starters:
            pos = 0
            while True:
                pos = full_text.find(starter, pos)
                if pos == -1:
                    break
                boundaries.append(pos)
                pos += 1
        
        # Sort boundaries and split text
        boundaries = sorted(set(boundaries))
        responses = []
        
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(full_text)
            response = full_text[start:end].strip()
            if response:
                responses.append(response)
        
        if not responses:
            return full_text.strip()
        
        # Return the longest response (most complete)
        final_response = max(responses, key=len)
        print(f"DEBUG: Found {len(responses)} responses, selected longest ({len(final_response)} chars)")
        
        return final_response
    
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
        
        # Handle audio resampling if required by STT config
        if self.stt_config and self.stt_config.get("audio_requirements"):
            audio_reqs = self.stt_config["audio_requirements"]
            required_rates = audio_reqs.get("sample_rates", [])
            target_rate = audio_reqs.get("resample_to")
            
            if required_rates and sample_rate not in required_rates and target_rate:
                import librosa
                import numpy as np
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_rate)
                audio_data = (audio_array * 32768.0).astype(np.int16).tobytes()
                sample_rate = target_rate
        
        # Always use VAD for batch evaluation to detect speech end
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.transports.base_transport import TransportParams
        
        vad_analyzer = SileroVADAnalyzer(params=VADParams(stop_secs=1.0))  # 1 second silence = speech end
        params = TransportParams(audio_in_enabled=True, vad_analyzer=vad_analyzer)
        
        transport = EvaluationTransport(
            audio_data, 
            sample_rate, 
            params=params,
            stt_model=self.stt_config.get("config", {}).get("model", "default") if self.stt_config else "default"
        )
        
        # Create services
        stt = self._create_stt_service()
        
        # For batch STT, transcribe the file first
        if hasattr(stt, 'transcribe_file'):
            transcription = stt.transcribe_file(audio_path)
            stt.set_transcription(transcription)
        
        tts = self._create_tts_service()
        agent = build_conversation_agent(model_id="au.anthropic.claude-haiku-4-5-20251001-v1:0", tts_service=tts)
        llm = StrandsAgentsProcessor(agent=agent)
        
        # Setup context with VAD-based aggregation (no timeout, wait for speech end)
        context = AWSBedrockLLMContext()
        from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
        user_params = LLMUserAggregatorParams(aggregation_timeout=None)  # Disable timeout, use VAD
        tma_in = LLMUserContextAggregator(context=context, params=user_params)
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

        
        # Build pipeline: STT -> Context -> LLM -> TTS (bypass output context aggregator)
        pipeline = Pipeline([
            transport.input(),
            stt_timing_start,
            stt,
            stt_collector,
            tma_in,  # Convert TranscriptionFrame to OpenAILLMContextFrame
            llm,
            llm_collector,
            # tma_out,  # Skip output aggregator - TextFrames go directly to TTS
            tts,
            tts_timing_end,
            transport.output()
        ])
        
        # Get pipeline params from config (with defaults)
        pipeline_params_config = self.stt_config.get("pipeline_params", {}) if self.stt_config else {}
        pipeline_params = PipelineParams(
            allow_interruptions=pipeline_params_config.get("allow_interruptions", True),
            enable_metrics=pipeline_params_config.get("enable_metrics", False),
            enable_usage_metrics=pipeline_params_config.get("enable_usage_metrics", False),
            report_only_initial_ttfb=pipeline_params_config.get("report_only_initial_ttfb", True)
        )
        
        task = PipelineTask(pipeline, params=pipeline_params)
        
        # Process audio
        start_time = time.time()
        
        await task.queue_frames([StartFrame(), EndFrame()])
        
        # Run pipeline in background
        runner = PipelineRunner()
        run_task = asyncio.create_task(runner.run(task))
        
        # Wait for VAD-triggered LLM and TTS to complete
        pipeline_timeout = self.stt_config.get("pipeline_timeout", 18.0) if self.stt_config else 18.0
        await asyncio.sleep(pipeline_timeout)
        
        # Force TTS disconnect if configured
        force_tts_stop = self.tts_config.get("force_stop_on_cancel", False) if self.tts_config else False
        if force_tts_stop:
            try:
                await tts.stop(EndFrame())
            except:
                pass
        
        # Cancel task
        await task.cancel()
        
        # Wait for run_task to complete cancellation with config-driven timeout
        # cleanup_timeout = self.stt_config.get("cleanup_timeout", 2.0) if self.stt_config else 2.0
        # try:
        #     await asyncio.wait_for(run_task, timeout=cleanup_timeout)
        # except (asyncio.TimeoutError, asyncio.CancelledError):
        #     pass
        
        logger.info("Pipeline processing complete, continuing...")
        
        total_latency = (time.time() - start_time) * 1000
        
        # Collect results
        stt_output = " ".join(stt_texts)
        llm_response = "".join(llm_texts)
        
        print(f"DEBUG: LLM response length: {len(llm_response)}")
        print(f"DEBUG: LLM response: {llm_response[:200]}...")
        
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
                    voice_metrics = await self.voice_evaluator.evaluate_with_llm_judge(tts_audio_path, result["bot_response"])
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
        
        # Write atomically to prevent corruption
        temp_path = output_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        os.rename(temp_path, output_path)
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
    
    # Cancel background tasks but not the main task
    try:
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if task != current_task and not task.done():
                task.cancel()
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())

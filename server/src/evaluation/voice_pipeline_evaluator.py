"""
Run the Q&A bot on VoiceAssistant-Eval dataset audio files.
Collects STT outputs and bot responses for evaluation.
"""

# Standard library imports
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

# Third-party imports
import librosa
import numpy as np
import wave
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

# Pipecat imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    StartFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMUserAggregatorParams
)
from pipecat.services.aws.llm import AWSBedrockLLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.transports.base_transport import TransportParams

from src.core.agent_builder import build_conversation_agent
from src.core.llm_processor import StrandsAgentsProcessor
from src.evaluation.frame_processor import (
    TimingCollector, STTTimingProcessor, TTSTimingProcessor,
    STTCollector, LLMCollector
)
from src.evaluation.models import PipelineResult
from src.transport.batch_audio_transport import EvaluationTransport
from tts import DeepgramTTSService

# Configuration
warnings.filterwarnings("ignore", message="Dangling tasks detected")
load_dotenv(override=True)
logger.disable("pipecat.pipeline.task")


# Pydantic Data Classes
class PipelineCollectors(BaseModel):
    """Container for pipeline data collectors."""
    timing: Any  # TimingCollector
    stt_texts: List[str] = []
    llm_texts: List[str] = []


class PipelineComponents(BaseModel):
    """Container for pipeline components."""
    transport: Any
    pipeline: Any
    collectors: PipelineCollectors


class PipelineConfig(BaseModel):
    """Pipeline configuration parameters."""
    allow_interruptions: bool = True
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    report_only_initial_ttfb: bool = True
    timeout: float = 18.0


class ExecutionResults(BaseModel):
    """Results from pipeline execution."""
    stt_output: str
    llm_response: str
    stt_latency_ms: Optional[float] = None
    tts_latency_ms: Optional[float] = None
    total_latency_ms: float
    output_audio: Optional[bytes] = None


class EvaluationSummary(BaseModel):
    """Summary statistics for evaluation results."""
    total_files: int
    successful: int
    failed: int
    skipped: int
    success_rate: float


class EvaluationOutput(BaseModel):
    """Complete evaluation output structure."""
    stt_model: Optional[str] = None
    stt_service_id: Optional[str] = None
    tts_model: Optional[str] = None
    tts_service_id: Optional[str] = None
    summary: EvaluationSummary
    results: List[PipelineResult]


class ConfigurationManager:
    """
    Manages loading and processing of service configuration files.

    Responsibilities:
    - Load JSON configuration files
    - Substitute environment variables in config values
    - Validate configuration structure
    """

    def __init__(self):
        """Initialize the configuration manager."""
        pass

    def load_config(self, config_path: str) -> Dict:
        """
        Load service configuration from JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            Dict containing the loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file contains invalid JSON
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Configuration file not found: {config_path}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in config file {config_path}: {e}"
            )

    def substitute_env_vars(self, config: Dict) -> Dict:
        """
        Substitute environment variables in configuration values.

        Supports formats:
        - ${VAR_NAME} - Simple variable substitution
        - ${VAR_NAME}:suffix - Variable with suffix (e.g., for ports)

        Args:
            config: Configuration dictionary to process

        Returns:
            Dict with environment variables substituted
        """
        result = config.copy()
        for key, value in result.items():
            if isinstance(value, str) and \
                    value.startswith('${') and value.endswith('}'):
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


class BaseServiceFactory(ABC):
    """Abstract base class for service factories."""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    @abstractmethod
    def create_service(self, config: Dict = None):
        """Create service from configuration."""
        pass

    def _get_api_key_for_provider(self, service_id: str) -> str:
        """
        Get API key based on service_id: deepgram_aura -> DEEPGRAM_API_KEY
        """
        provider = service_id.split('_')[0].upper() if service_id else ''
        return os.getenv(f"{provider}_API_KEY")

    def _needs_api_key(self, provider: str, module_name: str) -> bool:
        """Check if service needs API key."""
        return provider not in {"AWS", "NVIDIA"} and "riva" not in module_name


class TTSServiceFactory(BaseServiceFactory):
    """Factory for creating TTS services."""

    def create_service(self, config: Dict = None):
        """Create TTS service from configuration."""
        if not config:
            return DeepgramTTSService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                voice="aura-2-delia-en"
            )

        # Handle LiveKit special cases
        if "livekit" in config["module"]:
            return self._create_livekit_service(config)

        # Standard service creation
        return self._create_standard_service(config)

    def _create_livekit_service(self, config: Dict):
        """Create LiveKit adapter services."""
        module_name = config["module"]
        service_config = self.config_manager.substitute_env_vars(
            config.get("config", {})
        )

        if "nvidia" in module_name:
            from src.core.nvidia.livekit_tts_adapter import LiveKitTTSAdapter
            return LiveKitTTSAdapter(**service_config)
        else:
            # Fallback to standard service creation for other LiveKit services
            module = __import__(config["module"], fromlist=[config["class"]])
            service_class = getattr(module, config["class"])
            return service_class(**service_config)

    def _create_standard_service(self, config: Dict):
        """Create standard TTS service."""
        module = __import__(config["module"], fromlist=[config["class"]])
        service_class = getattr(module, config["class"])
        service_config = self.config_manager.substitute_env_vars(
            config.get("config", {})
        )

        service_id = config.get("tts_service_id", "")
        provider = service_id.split('_')[0].upper()

        if self._needs_api_key(provider, config["module"]):
            api_key = self._get_api_key_for_provider(service_id)
            return service_class(api_key=api_key, **service_config)
        else:
            return service_class(**service_config)


class STTServiceFactory(BaseServiceFactory):
    """Factory for creating STT services."""

    def create_service(self, config: Dict = None):
        """Create STT service from configuration."""
        if not config:
            return DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    model="nova-3",
                    language="en",
                    smart_format=True)
            )

        # Standard service creation
        return self._create_standard_service(config)

    def _create_standard_service(self, config: Dict):
        """Create standard STT service."""
        module = __import__(config["module"], fromlist=[config["class"]])
        service_class = getattr(module, config["class"])
        service_config = self.config_manager.substitute_env_vars(
            config.get("config", {})
        )

        service_id = config.get("stt_service_id", "")
        provider = service_id.split('_')[0].upper()

        if self._needs_api_key(provider, config["module"]):
            api_key = self._get_api_key_for_provider(service_id)
            return service_class(api_key=api_key, **service_config)
        else:
            return service_class(**service_config)


class PipelineBuilder:
    """
    Handles pipeline construction and component assembly.

    Responsibilities:
    - Create pipeline components (transport, collectors, aggregators)
    - Assemble complete pipeline from services
    """

    def __init__(self, stt_config: Dict = None):
        """Initialize pipeline builder."""
        self.stt_config = stt_config or {}

    def create_transport(self, audio_data: bytes, sample_rate: int) -> Any:
        """Create evaluation transport with VAD."""
        vad_analyzer = SileroVADAnalyzer(params=VADParams(stop_secs=1.0))
        params = TransportParams(
            audio_in_enabled=True,
            vad_analyzer=vad_analyzer)

        return EvaluationTransport(
            audio_data,
            sample_rate,
            params=params,
            stt_model=self.stt_config.get("config", {}).get("model", "default")
        )

    def create_collectors(self) -> PipelineCollectors:
        """Create pipeline data collectors."""
        return PipelineCollectors(
            timing=TimingCollector(),
            stt_texts=[],
            llm_texts=[]
        )

    def create_context_aggregator(self) -> Any:
        """Create LLM context aggregator based on STT type."""
        context = AWSBedrockLLMContext()
        stt_service_id = self.stt_config.get('stt_service_id', '').lower()
        is_whisper = 'whisper' in stt_service_id

        if is_whisper:
            user_params = LLMUserAggregatorParams(aggregation_timeout=5.0)
            logger.info("Using timeout-based aggregation for Whisper (5s)")
        else:
            user_params = LLMUserAggregatorParams(aggregation_timeout=None)
            logger.info("Using VAD-based aggregation for streaming STT")

        return LLMUserContextAggregator(context=context, params=user_params)

    def build_pipeline(
            self,
            audio_data: bytes,
            sample_rate: int,
            stt_service,
            tts_service,
            llm_service) -> PipelineComponents:
        """Build complete pipeline with all components."""
        transport = self.create_transport(audio_data, sample_rate)
        collectors = self.create_collectors()
        context_aggregator = self.create_context_aggregator()

        pipeline = Pipeline([
            transport.input(),
            STTTimingProcessor(collectors.timing),
            stt_service,
            STTCollector(collectors.timing, collectors.stt_texts),
            context_aggregator,
            llm_service,
            LLMCollector(collectors.timing, collectors.llm_texts),
            tts_service,
            TTSTimingProcessor(collectors.timing),
            transport.output()
        ])

        return PipelineComponents(
            transport=transport,
            pipeline=pipeline,
            collectors=collectors
        )


class PipelineExecutor:
    """
    Handles pipeline execution and result collection.

    Responsibilities:
    - Execute pipeline with proper configuration
    - Handle batch STT setup
    - Collect and format execution results
    """

    def _prepare_batch_stt(self, stt_service, audio_path: str) -> None:
        """Handle batch STT transcription setup."""
        if hasattr(stt_service, 'transcribe_file'):
            transcription = stt_service.transcribe_file(audio_path)
            stt_service.set_transcription(transcription)

    def _create_pipeline_config(
            self,
            stt_config: Dict = None) -> PipelineConfig:
        """Create pipeline configuration from STT config."""
        if not stt_config:
            return PipelineConfig()

        pipeline_params = stt_config.get("pipeline_params", {})
        return PipelineConfig(
            allow_interruptions=pipeline_params.get(
                "allow_interruptions", True
            ),
            enable_metrics=pipeline_params.get("enable_metrics", False),
            enable_usage_metrics=pipeline_params.get(
                "enable_usage_metrics", False
            ),
            report_only_initial_ttfb=pipeline_params.get(
                "report_only_initial_ttfb", True
            ),
            timeout=stt_config.get("pipeline_timeout", 18.0)
        )

    async def _run_pipeline(
            self,
            pipeline_components: PipelineComponents,
            config: PipelineConfig) -> float:
        """Execute the pipeline and return total latency."""
        pipeline_params = PipelineParams(
            allow_interruptions=config.allow_interruptions,
            enable_metrics=config.enable_metrics,
            enable_usage_metrics=config.enable_usage_metrics,
            report_only_initial_ttfb=config.report_only_initial_ttfb
        )

        task = PipelineTask(
            pipeline_components.pipeline,
            params=pipeline_params)
        start_time = time.time()

        await task.queue_frames([StartFrame(), EndFrame()])
        runner = PipelineRunner()
        await asyncio.create_task(runner.run(task))
        await asyncio.sleep(config.timeout)
        await task.cancel()
        await asyncio.sleep(0.1)

        return (time.time() - start_time) * 1000

    async def execute_pipeline(
            self,
            pipeline_components: PipelineComponents,
            stt_service,
            audio_path: str,
            stt_config: Dict = None) -> ExecutionResults:
        """Execute pipeline and return structured results."""
        self._prepare_batch_stt(stt_service, audio_path)
        config = self._create_pipeline_config(stt_config)
        total_latency = await self._run_pipeline(pipeline_components, config)

        collectors = pipeline_components.collectors
        timing = collectors.timing
        transport = pipeline_components.transport

        return ExecutionResults(
            stt_output=" ".join(collectors.stt_texts),
            llm_response="".join(collectors.llm_texts),
            stt_latency_ms=timing.get_stt_latency_ms(),
            tts_latency_ms=timing.get_tts_latency_ms(),
            total_latency_ms=total_latency,
            output_audio=transport.get_output_audio()
        )


class AudioProcessor:
    """
    Handles audio file processing and pipeline execution.

    Responsibilities:
    - Load and preprocess audio files
    - Set up and execute Pipecat pipeline
    - Collect timing and output data
    """

    def __init__(
            self,
            stt_config: Dict = None,
            pipeline_builder=PipelineBuilder,
            pipeline_executor=PipelineExecutor):
        """Initialize audio processor."""
        self.stt_config = stt_config or {}
        self.sample_rate = None
        self.audio_data = None
        self.pipeline_builder = pipeline_builder(stt_config=self.stt_config)
        self.pipeline_executor = PipelineExecutor()

    def load_audio_file(self, audio_path: str) -> None:
        """Load audio file and store in instance variables."""
        with wave.open(audio_path, 'rb') as wf:
            self.sample_rate = wf.getframerate()
            self.audio_data = wf.readframes(wf.getnframes())
            duration_seconds = len(self.audio_data) / (self.sample_rate * 2)
            logger.info(f"Audio sample rate: {self.sample_rate}Hz")
            logger.info(f"Audio data size: {len(self.audio_data)} bytes")
            logger.info(f"Audio duration: {duration_seconds:.2f} seconds")

    def resample_if_needed(self) -> None:
        """Resample audio if required by STT config."""
        if not self.stt_config.get("audio_requirements"):
            return

        audio_reqs = self.stt_config["audio_requirements"]
        required_rates = audio_reqs.get("sample_rates", [])
        target_rate = audio_reqs.get("resample_to")

        if required_rates and \
                self.sample_rate not in required_rates and target_rate:
            logger.info(
                f"Resampling audio from {self.sample_rate}Hz "
                f"to {target_rate}Hz"
            )
            audio_array = np.frombuffer(
                self.audio_data, dtype=np.int16
            ).astype(np.float32) / 32768.0
            audio_array = librosa.resample(
                audio_array,
                orig_sr=self.sample_rate,
                target_sr=target_rate)
            self.audio_data = (
                audio_array * 32768.0
            ).astype(np.int16).tobytes()
            self.sample_rate = target_rate

    def process_audio_file(self, audio_path: str) -> None:
        """Process audio file completely."""
        self.load_audio_file(audio_path)
        self.resample_if_needed()

    def create_transport(self) -> Any:
        """Create evaluation transport with VAD."""
        vad_analyzer = SileroVADAnalyzer(params=VADParams(stop_secs=1.0))
        params = TransportParams(
            audio_in_enabled=True, vad_analyzer=vad_analyzer
        )

        return EvaluationTransport(
            self.audio_data,
            self.sample_rate,
            params=params,
            stt_model=self.stt_config.get("config", {}).get("model", "default")
        )

    def create_collectors(self) -> PipelineCollectors:
        """Create pipeline data collectors."""
        return PipelineCollectors(
            timing=TimingCollector(),
            stt_texts=[],
            llm_texts=[]
        )

    def create_context_aggregator(self) -> Any:
        """Create LLM context aggregator based on STT type."""
        context = AWSBedrockLLMContext()
        stt_service_id = self.stt_config.get('stt_service_id', '').lower()
        is_whisper = 'whisper' in stt_service_id

        if is_whisper:
            user_params = LLMUserAggregatorParams(aggregation_timeout=5.0)
            logger.info("Using timeout-based aggregation for Whisper (5s)")
        else:
            user_params = LLMUserAggregatorParams(aggregation_timeout=None)
            logger.info("Using VAD-based aggregation for streaming STT")

        return LLMUserContextAggregator(context=context, params=user_params)

    def build_pipeline(
            self,
            stt_service,
            tts_service,
            llm_service) -> PipelineComponents:
        """Build complete pipeline using PipelineBuilder."""
        return self.pipeline_builder.build_pipeline(
            self.audio_data,
            self.sample_rate,
            stt_service,
            tts_service,
            llm_service
        )

    async def execute_pipeline(
            self,
            pipeline_components: PipelineComponents,
            stt_service,
            audio_path: str,
            stt_config: Dict = None) -> ExecutionResults:
        """Execute pipeline using PipelineExecutor."""
        return await self.pipeline_executor.execute_pipeline(
            pipeline_components, stt_service, audio_path, stt_config
        )


class ServiceManager:
    """
    Handles service lifecycle management.

    Responsibilities:
    - Clean shutdown of STT/TTS services
    - Handle service-specific cleanup logic
    """

    async def cleanup_services(self, stt_service, tts_service) -> None:
        """Clean shutdown of all services."""
        await self._cleanup_stt_service(stt_service)
        await self._cleanup_tts_service(tts_service)
        await asyncio.sleep(0.1)  # Give time for cleanup

    async def _cleanup_stt_service(self, stt_service) -> None:
        """Clean shutdown of STT service."""
        try:
            if hasattr(stt_service, 'disconnect'):
                await stt_service.disconnect()
        except Exception as e:
            logger.debug(f"STT disconnect error (expected): {e}")

    async def _cleanup_tts_service(self, tts_service) -> None:
        """Clean shutdown of TTS service."""
        try:
            if hasattr(tts_service, 'stop'):
                await tts_service.stop(EndFrame())
        except Exception as e:
            # Ignore WebSocket close errors for Cartesia
            if "sent 1000 (OK); then received 1000 (OK)" in str(e):
                pass  # Normal WebSocket close
            else:
                logger.debug(f"TTS stop error: {e}")


class ResultCollector:
    """
    Handles result collection and file operations.

    Responsibilities:
    - Format execution results into PipelineResult
    - Save TTS audio files
    - Handle experiment directory creation
    """

    def __init__(self, stt_config: Dict = None, tts_config: Dict = None):
        """Initialize result collector."""
        self.stt_config = stt_config or {}
        self.tts_config = tts_config or {}

    def _save_tts_audio(
            self,
            output_audio: bytes,
            question_id: str,
            pipeline_components: PipelineComponents) -> Optional[str]:
        """Save TTS audio file and return path."""
        if not output_audio:
            return None

        # Create experiment-specific directory
        stt_id = self.stt_config.get("stt_service_id", "default_stt")
        tts_id = self.tts_config.get("tts_service_id", "default_tts")
        experiment_name = f"{stt_id}_{tts_id}"
        output_dir = os.path.join(
            "evaluation_output", "tts_audio", experiment_name
        )
        os.makedirs(output_dir, exist_ok=True)
        tts_audio_path = os.path.join(
            output_dir, f"{question_id}_response.wav"
        )

        # Get sample rate from transport
        sample_rate = getattr(
            pipeline_components.transport.output(), 'sample_rate', 16000
        )

        with wave.open(tts_audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(output_audio)

        return tts_audio_path

    def collect_result(
            self,
            execution_results: ExecutionResults,
            pipeline_components: PipelineComponents,
            question_id: str,
            audio_path: str,
            ground_truth: str) -> PipelineResult:
        """Collect and format final pipeline result."""
        # Save TTS audio
        tts_audio_path = self._save_tts_audio(
            execution_results.output_audio, question_id, pipeline_components
        )
        stt_latency = execution_results.stt_latency_ms
        tts_latency = execution_results.tts_latency_ms

        return PipelineResult(
            question_id=question_id,
            audio_file=audio_path,
            stt_output=execution_results.stt_output,
            ground_truth=ground_truth,
            llm_response=execution_results.llm_response,
            tts_audio_path=tts_audio_path,
            stt_latency_ms=(
                round(stt_latency, 2) if stt_latency is not None else None
            ),
            tts_latency_ms=(
                round(tts_latency, 2) if tts_latency is not None else None
            ),
            total_latency_ms=round(execution_results.total_latency_ms, 2)
        )

    def save_results(self, results: List[PipelineResult], output_path: str):
        """Save results to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Calculate summary statistics
        successful = len([r for r in results if r.status == 'success'])
        failed = len([r for r in results if r.status == 'failed'])
        skipped = len(results) - successful - failed

        summary = EvaluationSummary(
            total_files=len(results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            success_rate=round(
                (successful/len(results))*100, 1
            ) if results else 0
        )

        output_data = EvaluationOutput(
            stt_model=self.stt_config.get("stt_service_name"),
            stt_service_id=self.stt_config.get("stt_service_id"),
            tts_model=self.tts_config.get("tts_service_name"),
            tts_service_id=self.tts_config.get("tts_service_id"),
            summary=summary,
            results=results
        )

        # Write atomically to prevent corruption
        temp_path = output_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(output_data.model_dump(), f, indent=2)

        logger.info(f"Results saved to {output_path}")


class EvaluationOrchestrator:
    """
    Orchestrates the evaluation process across all dataset items.

    Responsibilities:
    - Manage evaluation loop
    - Handle progress tracking
    - Coordinate file processing
    - Manage result collection and saving
    """

    def __init__(self, runner, output_path: str = None):
        self.runner = runner
        self.output_path = output_path
        self.results = []
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        self.result_collector = ResultCollector(
            runner.stt_config, runner.tts_config
        )

    async def run_evaluation(self) -> List[PipelineResult]:
        """Run evaluation on all dataset items."""
        total_files = len(self.runner.dataset['questions'])
        logger.info(f"Starting evaluation of {total_files} audio files")

        for i, question in enumerate(self.runner.dataset['questions']):
            await self._process_question(question, i)
            await self._pause_if_needed(i, total_files)

        self._log_summary(total_files)
        return self.results

    async def _process_question(self, question: Dict, index: int) -> None:
        """Process a single question from the dataset."""
        question_id = question['id']
        audio_file = question['audio_file']
        audio_path = os.path.join(self.runner.audio_dir, audio_file)

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            self.skipped_count += 1
            return

        try:
            result = await self.retry_operation(
                lambda: self.runner.process_audio_file(
                    audio_path,
                    question_id),
                question_id
            )
            result, is_success = self._process_result(result)

            if is_success:
                self.processed_count += 1
            else:
                self.error_count += 1

            self.results.append(result)

            if self.output_path:
                self.result_collector.save_results(
                    self.results,
                    self.output_path
                )

        except Exception as e:
            logger.error(
                f"Error processing {question_id} (final attempt): {e}"
            )
            self.error_count += 1
            self.results.append(PipelineResult(
                question_id=question_id,
                audio_file=audio_path,
                stt_output="",
                llm_response="",
                error=str(e),
                status="failed"
            ))

            if self.output_path:
                self.runner.save_results(self.results, self.output_path)

    async def _pause_if_needed(self, index: int, total_files: int) -> None:
        """Pause between items to avoid rate limiting."""
        if index < total_files - 1:
            logger.info("Pausing 3 seconds to avoid rate limiting...")
            await asyncio.sleep(3)

    def _log_summary(self, total_files: int) -> None:
        """Log final evaluation summary."""
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total files in dataset: {total_files}")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed with errors: {self.error_count}")
        logger.info(f"Skipped (file not found): {self.skipped_count}")
        success_rate = (
            self.processed_count/(self.processed_count + self.error_count)
        )*100 if (self.processed_count + self.error_count) > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"{'='*60}")

    @staticmethod
    async def retry_operation(
            operation,
            question_id: str,
            max_retries: int = 3):
        """Execute operation with retry logic and exponential backoff."""
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retry attempt {attempt + 1}/"
                        f"{max_retries} for {question_id}"
                    )

                result = await operation()

                if attempt > 0:
                    logger.info(f"Retry successful for {question_id}")

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (3 ** (attempt + 1)) + random.uniform(0, 1)
                    logger.warning(
                        f"Error on {question_id} (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)

        raise last_error

    def _process_result(
            self,
            result: PipelineResult) -> tuple[PipelineResult, bool]:
        """Process result and return (result, is_success)."""
        if self.runner.tts_config and result.tts_audio_path is None:
            result.status = "failed"
            result.error = "TTS failed - no audio generated"
            return result, False
        else:
            result.status = "success"
            return result, True


class VoiceAssistantRunner:
    def __init__(
            self,
            dataset_path: str,
            audio_dir: str,
            stt_config: str = None,
            tts_config: str = None):
        self.dataset_path = dataset_path
        self.audio_dir = audio_dir

        # Use configuration manager
        self.config_manager = ConfigurationManager()
        self.stt_factory = STTServiceFactory(self.config_manager)
        self.tts_factory = TTSServiceFactory(self.config_manager)
        self.dataset = self._load_dataset()

        # Load configs using the manager
        self.stt_config = self.config_manager.load_config(stt_config) if \
            stt_config else None
        self.tts_config = self.config_manager.load_config(tts_config) if \
            tts_config else None

    def _create_stt_service(self):
        """Create STT service using factory."""
        return self.stt_factory.create_service(self.stt_config)

    def _create_tts_service(self):
        """Create TTS service using factory."""
        return self.tts_factory.create_service(self.tts_config)

    def _load_dataset(self) -> Dict:
        """Load the evaluation dataset"""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    async def process_audio_file(
            self,
            audio_path: str,
            question_id: str) -> PipelineResult:
        """
        Process a single audio file through the bot pipeline.

        Returns:
            Dict with stt_output, bot_response, and latencies
        """
        logger.info(f"=== PROCESSING FILE: {question_id} ===")
        logger.info(f"Audio file path: {audio_path}")
        logger.info(f"File exists: {os.path.exists(audio_path)}")
        logger.info(f"Processing {question_id}: {audio_path}")

        # Get the ground truth transcript from dataset (for WER comparison)
        question_data = next(
            (q for q in self.dataset['questions'] if q['id'] == question_id),
            None
        )
        if not question_data:
            raise ValueError(f"Question {question_id} not found in dataset")

        ground_truth = question_data['text']

        # Process audio
        audio_processor = AudioProcessor(self.stt_config)
        audio_processor.process_audio_file(audio_path)

        # Create services
        stt = self._create_stt_service()
        tts = self._create_tts_service()
        agent = build_conversation_agent(
            model_id="au.anthropic.claude-haiku-4-5-20251001-v1:0",
            tts_service=tts)
        llm = StrandsAgentsProcessor(agent=agent)

        # Setup pipeline
        pipeline_components = audio_processor.build_pipeline(stt, tts, llm)

        # Execute pipeline using AudioProcessor
        execution_results = await audio_processor.execute_pipeline(
            pipeline_components, stt, audio_path, self.stt_config
        )

        # Create service manager and cleanup
        service_manager = ServiceManager()
        await service_manager.cleanup_services(stt, tts)

        logger.info("Pipeline processing complete, continuing...")
        # Create result collector
        result_collector = ResultCollector(self.stt_config, self.tts_config)

        # Collect final result
        result = result_collector.collect_result(
            execution_results,
            pipeline_components,
            question_id,
            audio_path,
            ground_truth
        )

        return result

    async def run_all(self, output_path: str = None) -> List[PipelineResult]:
        """Run bot on all audio files in dataset"""
        orchestrator = EvaluationOrchestrator(self, output_path)
        return await orchestrator.run_evaluation()


async def main():
    parser = argparse.ArgumentParser(
        description='Run bot on VoiceAssistant-Eval dataset'
    )
    parser.add_argument('--dataset', default='evaluation_data/voiceassistant_eval/voiceassistant_eval_dataset.json',
                       help='Path to dataset JSON')
    parser.add_argument('--audio-dir', default='evaluation_data/voiceassistant_eval/audio_input',
                       help='Directory containing audio files')
    parser.add_argument('--output', default='evaluation_data/voiceassistant_eval/bot_results.json',
                       help='Output path for bot results')
    parser.add_argument('--stt-config', help='STT service config (e.g., evaluation_data/bot_configs/deepgram_nova3_config.json)')
    parser.add_argument('--tts-config', help='TTS service config (e.g., evaluation_data/tts_bot_configs/deepgram_aura_config.json)')

    args = parser.parse_args()

    runner = VoiceAssistantRunner(
        args.dataset,
        args.audio_dir,
        stt_config=args.stt_config,
        tts_config=args.tts_config,
    )
    
    logger.info("Running bot on all audio files...")
    results = await runner.run_all(args.output)
    
    runner.save_results(results, args.output)
    
    # Count successful vs failed results
    successful = len([r for r in results if r.status == 'success'])
    failed = len([r for r in results if r.status == 'failed'])
    
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

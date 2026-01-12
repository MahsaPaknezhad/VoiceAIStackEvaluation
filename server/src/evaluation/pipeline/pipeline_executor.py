"""
Pipeline executor for voice evaluation framework.
Handles pipeline execution and result collection.
"""

import asyncio
import time
from typing import Dict, Optional, Any
from loguru import logger
from pipecat.frames.frames import EndFrame, StartFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

from src.evaluation.models import (
    PipelineComponents,
    PipelineConfig,
    ExecutionResults
)


class PipelineExecutor:
    """
    Executes voice evaluation pipelines and collects results.

    This class handles the complete pipeline execution lifecycle including
    batch STT preparation, pipeline configuration, execution timing, and
    result collection. It manages the asynchronous execution of Pipecat
    pipelines with proper error handling and cleanup.

    The execution flow:
    1. Prepare batch STT transcription (if applicable)
    2. Create pipeline configuration from STT settings
    3. Execute pipeline with timing measurements
    4. Collect and format execution results
    """

    def _prepare_batch_stt(self, stt_service: Any, audio_path: str) -> None:
        """
        Prepare batch STT transcription for non-streaming services.

        For batch STT services (like Whisper), pre-transcribes the audio
        file and sets the transcription on the service before pipeline
        execution. Streaming services are unaffected.

        Args:
            stt_service: STT service instance that may support batch
            transcription audio_path: Path to audio file for batch
            transcription
        """
        if hasattr(stt_service, 'transcribe_file'):
            transcription = stt_service.transcribe_file(audio_path)
            stt_service.set_transcription(transcription)

    def _create_pipeline_config(
            self,
            stt_config: Optional[Dict[str, Any]] = None
    ) -> PipelineConfig:
        """
        Create pipeline configuration from STT service settings.

        Extracts pipeline parameters from STT configuration including
        interruption handling, metrics collection, and timeout settings.
        Uses sensible defaults if configuration is not provided.

        Args:
            stt_config: STT service configuration containing pipeline
            parameters

        Returns:
            PipelineConfig instance with execution parameters
        """
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
            config: PipelineConfig
    ) -> float:
        """
        Execute the pipeline and measure total latency.

        Runs the complete pipeline with proper frame queuing, task management,
        and timing measurements. Handles pipeline cancellation and cleanup
        after execution timeout.

        Args:
            pipeline_components: Complete pipeline components including
            processors
            config: Pipeline configuration with execution parameters

        Returns:
            Total pipeline execution latency in milliseconds
        """
        pipeline_params = PipelineParams(
            allow_interruptions=config.allow_interruptions,
            enable_metrics=config.enable_metrics,
            enable_usage_metrics=config.enable_usage_metrics,
            report_only_initial_ttfb=config.report_only_initial_ttfb
        )
        logger.info('Starting Pipeline')
        task = PipelineTask(
            pipeline_components.pipeline,
            params=pipeline_params)
        start_time = time.time()

        await task.queue_frames([StartFrame(), EndFrame()])
        logger.info("StartFrame and EndFrame queued")

        runner = PipelineRunner()
        logger.info("About to run pipeline task")

        run_start = time.time()
        await asyncio.create_task(runner.run(task))
        run_end = time.time()
        logger.info(f"Pipeline runner completed in {run_end - run_start:.2f}s")

        sleep_start = time.time()
        await asyncio.sleep(config.timeout)
        sleep_end = time.time()
        logger.info(f"Sleep completed in {sleep_end - sleep_start:.2f}s")

        cancel_start = time.time()
        await task.cancel()
        cancel_end = time.time()
        logger.info(f"Task cancelled in {cancel_end - cancel_start:.2f}s")
        await asyncio.sleep(0.05)

        logger.info('Finished Pipeline')
        return (time.time() - start_time) * 1000

    async def execute_pipeline(
            self,
            pipeline_components: PipelineComponents,
            stt_service: Any,
            audio_path: str,
            stt_config: Optional[Dict[str, Any]] = None
    ) -> ExecutionResults:
        """
        Execute complete pipeline and return structured results.

        Orchestrates the full pipeline execution including batch STT
        preparation, configuration creation, pipeline execution,
        and result collection.
        Extracts timing data, transcriptions, and generated audio.

        Args:
            pipeline_components: Complete pipeline with transport and
            collectors
            stt_service: Speech-to-text service instance
            audio_path: Path to input audio file
            stt_config: STT service configuration for pipeline parameters

        Returns:
            ExecutionResults containing transcriptions, responses, timing, and
            audio
        """
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

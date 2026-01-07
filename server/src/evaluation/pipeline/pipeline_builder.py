"""
Pipeline builder for voice evaluation framework.
Handles pipeline construction and component assembly.
"""

from typing import Dict, Any, Optional
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMUserAggregatorParams
)
from pipecat.services.aws.llm import AWSBedrockLLMContext
from pipecat.transports.base_transport import TransportParams

from src.evaluation.frame_processor import (
    TimingCollector,
    STTTimingProcessor,
    TTSTimingProcessor,
    STTCollector,
    LLMCollector
)
from src.evaluation.models import PipelineCollectors, PipelineComponents
from src.transport.batch_audio_transport import EvaluationTransport


class PipelineBuilder:
    """
    Constructs and assembles voice evaluation pipelines.

    This class handles the creation of pipeline components including audio
    transport, data collectors, context aggregators, and the complete
    pipeline assembly with proper ordering of processors.

    The pipeline flow:
    Input -> STT Timing -> STT Service -> STT Collector -> Context Aggregator
    -> LLM Service -> LLM Collector -> TTS Service -> TTS Timing -> Output

    Attributes:
        stt_config: Configuration dictionary for STT service settings
    """

    def __init__(self, stt_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize pipeline builder with STT configuration.

        Args:
            stt_config: STT service configuration containing model settings,
                       service ID, and pipeline parameters.
                       Defaults to empty dict.
        """
        self.stt_config = stt_config or {}

    def create_transport(
            self,
            audio_data: bytes,
            sample_rate: int
    ) -> EvaluationTransport:
        """
        Create evaluation transport with Voice Activity Detection.

        Sets up audio transport with Silero VAD analyzer for detecting
        speech segments and managing audio input/output flow.

        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate in Hz

        Returns:
            Configured EvaluationTransport instance with VAD enabled
        """
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
        """
        Create pipeline data collectors for metrics and outputs.

        Initializes collectors for timing measurements, STT transcriptions,
        and LLM responses that will be populated during pipeline execution.

        Returns:
            PipelineCollectors instance with timing collector and empty text
            lists
        """
        return PipelineCollectors(
            timing=TimingCollector(),
            stt_texts=[],
            llm_texts=[]
        )

    def create_context_aggregator(self) -> LLMUserContextAggregator:
        """
        Create LLM context aggregator based on STT service type.

        Configures aggregation strategy based on STT service:
        - Whisper services: Timeout-based aggregation (5 seconds)
        - Streaming services: VAD-based aggregation

        Returns:
            Configured LLMUserContextAggregator for managing conversation
            context
        """
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
            stt_service: Any,
            tts_service: Any,
            llm_service: Any
    ) -> PipelineComponents:
        """
        Build complete pipeline with all components and processors.

        Assembles the full voice evaluation pipeline with proper processor
        ordering and data flow. Creates transport, collectors, and context
        aggregator, then connects them in the correct sequence.

        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate in Hz
            stt_service: Speech-to-text service instance
            tts_service: Text-to-speech service instance
            llm_service: Large language model service instance

        Returns:
            PipelineComponents containing transport, pipeline, and collectors
        """
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

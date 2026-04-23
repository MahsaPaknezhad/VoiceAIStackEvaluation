# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

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

from src.evaluation.models import TimingCollector
from src.evaluation.pipeline.frame_processors import (
    STTTimingProcessor,
    TTSTimingProcessor,
    STTCollector,
    LLMCollector
)
from src.evaluation.models import PipelineCollectors, PipelineComponents
from src.evaluation.pipeline.batch_audio_transport import BatchAudioTransport


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
        self._is_whisper = 'whisper' in self.stt_config.get(
            'stt_service_id', ''
        ).lower()

    def create_transport(
            self,
            audio_data: bytes,
            sample_rate: int,
            vad_analyzer_class: type = SileroVADAnalyzer,
            vad_params: Optional[VADParams] = None,
            transport_params: Optional[TransportParams] = None
    ) -> BatchAudioTransport:
        """
        Create evaluation transport with Voice Activity Detection.

        Sets up audio transport with Silero VAD analyzer for detecting
        speech segments and managing audio input/output flow.

        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate in Hz
            vad_analyzer_class: VAD analyzer class to use.
                Defaults to SileroVADAnalyzer
            vad_params: VAD parameters. Defaults to VADParams(stop_secs=1.0)
            transport_params: Transport parameters.
                Defaults to audio_in_enabled=True with VAD

        Returns:
            Configured BatchAudioTransport instance with VAD enabled
        """
        if vad_params is None:
            vad_params = VADParams(stop_secs=1.0)

        vad_analyzer = vad_analyzer_class(params=vad_params)

        if transport_params is None:
            transport_params = TransportParams(
                audio_in_enabled=True,
                vad_analyzer=vad_analyzer
            )
        else:
            transport_params.vad_analyzer = vad_analyzer

        return BatchAudioTransport(
            audio_data,
            sample_rate,
            params=transport_params,
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

    def create_context_aggregator(
            self,
            aggregation_timeout: Optional[float] = 5.0
    ) -> LLMUserContextAggregator:
        """
        Create LLM context aggregator

        Args:
            aggregation_timeout: Base timeout value. Default to None,
                meaning the pipeline will use VAD-based aggregation
        Returns:
            Configured LLMUserContextAggregator for managing conversation
            context
        """
        context = AWSBedrockLLMContext()

        logger.info(
            f"Adding timeout of {aggregation_timeout} -> "
            "If None, will revert to VAD-based aggregation"
        )
        user_params = LLMUserAggregatorParams(
            aggregation_timeout=aggregation_timeout
        )
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

        Pipeline Flow:
        1. transport.input() - Streams audio chunks with VAD coordination
        2. STTTimingProcessor - Captures STT start timing on first audio frame
        3. stt_service - Converts audio to text (AWS Transcribe, Whisper, etc.)
        4. STTCollector - Collects transcriptions and marks STT end timing
        5. context_aggregator - Aggregates STT output for LLM context
        6. llm_service - Generates response text (Bedrock Claude, OpenAI, etc.)
        7. LLMCollector - Collects/cleans LLM text and marks TTS start timing
        8. tts_service - Converts text to speech (AWS Polly, ElevenLabs, etc.)
        9. TTSTimingProcessor - Captures TTS end timing on first audio output
        10. transport.output() - Collects generated TTS audio for evaluation

        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate in Hz
            stt_service: Speech-to-text service instance
            tts_service: Text-to-speech service instance
            llm_service: Large language model service instance

        Returns:
            PipelineComponents containing transport, pipeline, and collectors
        """
        logger.info('Starting Pipeline Build')
        collectors = self.create_collectors()
        context_aggregator = self.create_context_aggregator(
            aggregation_timeout=self.stt_config.get("aggregation_timeout", 3.0)
        )
        transport = self.create_transport(audio_data, sample_rate)

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
        logger.info("Finalised pipeline build")
        return PipelineComponents(
            transport=transport,
            pipeline=pipeline,
            collectors=collectors
        )

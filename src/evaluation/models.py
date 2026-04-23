# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
Pydantic models for voice evaluation framework.
Provides type safety and validation for evaluation data structures.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# =============================================================================
# Base Mixins and Common Functionality
# =============================================================================

class LatencyMixin(BaseModel):
    """Common latency measurements."""
    stt_latency_ms: Optional[float] = Field(
        default=None, ge=0,
        description="STT processing latency in milliseconds"
    )
    tts_latency_ms: Optional[float] = Field(
        default=None, ge=0,
        description="TTS generation latency in milliseconds"
    )
    total_latency_ms: Optional[float] = Field(
        default=None, ge=0,
        description="End-to-end latency in milliseconds"
    )


class BaseResult(BaseModel):
    """Base class for evaluation results."""
    question_id: str = Field(description="Unique question identifier")
    stt_output: str = Field(
        default="", description="Speech-to-text transcription output"
    )
    llm_response: str = Field(
        default="", description="LLM generated response text"
    )
    tts_audio_path: Optional[str] = Field(
        default=None, description="Path to generated TTS audio file"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )


class ServiceIdentificationMixin(BaseModel):
    """Common service identification fields."""
    stt_model: Optional[str] = Field(
        default=None, description="STT model name")
    stt_service_id: Optional[str] = Field(
        default=None, description="STT service identifier")
    tts_model: Optional[str] = Field(
        default=None, description="TTS model name")
    tts_service_id: Optional[str] = Field(
        default=None, description="TTS service identifier")


# =============================================================================
# Pipeline Processing Models
# =============================================================================

class PipelineResult(BaseResult, LatencyMixin):
    """Result from voice pipeline processing."""
    audio_file: str = Field(description="Path to input audio file")
    ground_truth: str = Field(
        default="", description="Expected correct transcription"
    )
    status: str = Field(
        default="success",
        description="Processing status: success or failed"
    )


class PipelineCollectors(BaseModel):
    """Container for pipeline data collectors."""
    timing: Any = Field(
        description="Timing collector for latency measurements"
    )
    stt_texts: List[str] = Field(
        default_factory=list,
        description="Collected STT transcription texts"
    )
    llm_texts: List[str] = Field(
        default_factory=list,
        description="Collected LLM response texts"
    )


class TimingCollector(BaseModel):
    """
    Centralized timing data collector for pipeline performance metrics.

    Tracks start/end times for STT and TTS processing stages to calculate
    latency metrics for evaluation. Provides methods to compute processing
    durations in milliseconds for performance analysis.

    Attributes:
        stt_start_time: Timestamp when STT processing begins
        stt_end_time: Timestamp when STT processing completes
        tts_start_time: Timestamp when TTS processing begins
        tts_end_time: Timestamp when TTS processing completes
    """

    stt_start_time: Optional[float] = None
    stt_end_time: Optional[float] = None
    tts_start_time: Optional[float] = None
    tts_end_time: Optional[float] = None

    def get_stt_latency_ms(self) -> Optional[float]:
        """Calculate STT processing latency in milliseconds."""
        if self.stt_start_time is not None and self.stt_end_time is not None:
            return (self.stt_end_time - self.stt_start_time) * 1000
        return None

    def get_tts_latency_ms(self) -> Optional[float]:
        """Calculate TTS processing latency in milliseconds."""
        if self.tts_start_time is not None and self.tts_end_time is not None:
            return (self.tts_end_time - self.tts_start_time) * 1000
        return None


class PipelineComponents(BaseModel):
    """Container for pipeline components."""
    transport: Any = Field(
        description="Audio transport component for input/output"
    )
    pipeline: Any = Field(description="Pipecat pipeline instance")
    collectors: PipelineCollectors = Field(
        description="Data collectors for metrics and outputs"
    )


class PipelineConfig(BaseModel):
    """Pipeline configuration parameters."""
    allow_interruptions: bool = Field(
        default=True, description="Allow pipeline interruptions"
    )
    enable_metrics: bool = Field(
        default=False, description="Enable pipeline metrics collection"
    )
    enable_usage_metrics: bool = Field(
        default=False, description="Enable usage metrics tracking"
    )
    report_only_initial_ttfb: bool = Field(
        default=True, description="Report only initial time-to-first-byte"
    )
    timeout: float = Field(
        default=18.0, description="Pipeline timeout in seconds"
    )


class ExecutionResults(LatencyMixin):
    """Results from pipeline execution."""
    stt_output: str = Field(
        description="Speech-to-text transcription result"
    )
    llm_response: str = Field(description="LLM generated response")
    output_audio: Optional[bytes] = Field(
        default=None, description="Generated audio output as bytes"
    )

    # Override total_latency_ms to be required
    total_latency_ms: float = Field(
        description="Total pipeline execution latency in milliseconds"
    )


class PipelineEvaluationSummary(BaseModel):
    """Summary statistics for pipeline evaluation results."""
    total_files: int = Field(description="Total number of files processed")
    successful: int = Field(
        description="Number of successfully processed files")
    failed: int = Field(description="Number of files that failed processing")
    skipped: int = Field(description="Number of files skipped due to errors")
    success_rate: float = Field(
        description="Success rate as percentage (0-100)")


class EvaluationOutput(ServiceIdentificationMixin):
    """Complete evaluation output structure."""
    summary: PipelineEvaluationSummary = Field(
        description="Aggregated evaluation statistics"
    )
    results: List[PipelineResult] = Field(
        description="Individual pipeline processing results"
    )


# =============================================================================
# Quality Assessment Models
# =============================================================================

class WERResults(BaseModel):
    """WER evaluation results."""
    wer_score: float = Field(
        ge=0, le=100, description="Word Error Rate percentage")
    reference: str = Field(description="Ground truth text")
    hypothesis: str = Field(description="Predicted text")


class ResponseQualityResults(BaseModel):
    """Alias for JudgeScores to maintain naming consistency."""
    correctness: float = Field(
        default=0.0, ge=0, le=10, description="Factual accuracy score (0-10)")
    relevance: float = Field(
        default=0.0, ge=0, le=10,
        description="Question relevance score (0-10)")
    completeness: float = Field(
        default=0.0, ge=0, le=10,
        description="Response completeness score (0-10)")
    clarity: float = Field(
        default=0.0, ge=0, le=10,
        description="Communication clarity score (0-10)")
    overall: float = Field(
        default=0.0, ge=0, le=10, description="Overall quality score (0-10)")
    reasoning: str = Field(
        default="", description="Judge reasoning explanation")


class JudgeScores(BaseModel):
    """LLM judge evaluation scores for response quality."""
    correctness: float = Field(
        default=0.0, ge=0, le=10, description="Factual accuracy score (0-10)"
    )
    relevance: float = Field(
        default=0.0, ge=0, le=10, description="Question relevance score (0-10)"
    )
    completeness: float = Field(
        default=0.0, ge=0, le=10,
        description="Response completeness score (0-10)"
    )
    clarity: float = Field(
        default=0.0, ge=0, le=10,
        description="Communication clarity score (0-10)"
    )
    overall: float = Field(
        default=0.0, ge=0, le=10, description="Overall quality score (0-10)"
    )
    reasoning: str = Field(
        default="", description="Judge reasoning explanation"
    )


class VoiceQuality(BaseModel):
    """Voice quality metrics from NISQA and SpeechMetrics."""
    # NISQA metrics
    nisqa_mos: Optional[float] = Field(
        default=None, ge=0, le=5, description="Mean Opinion Score (1-5)"
    )
    nisqa_noisiness: Optional[float] = Field(
        default=None, ge=0, le=5, description="Noisiness score (1-5)"
    )
    nisqa_coloration: Optional[float] = Field(
        default=None, ge=0, le=5, description="Coloration score (1-5)"
    )
    nisqa_discontinuity: Optional[float] = Field(
        default=None, ge=0, le=5, description="Discontinuity score (1-5)"
    )
    nisqa_loudness: Optional[float] = Field(
        default=None, ge=0, le=5, description="Loudness score (1-5)"
    )
    overall_quality: Optional[float] = Field(
        default=None, ge=0, le=5, description="Overall NISQA quality (1-5)"
    )

    # SpeechMetrics
    mosnet_score: Optional[float] = Field(
        default=None, description="MOSNet quality score"
    )
    srmr_score: Optional[float] = Field(
        default=None,
        description="Speech-to-Reverberation Modulation Ratio"
    )

    # LLM judge voice quality (if enabled)
    llm_fluency: Optional[float] = Field(
        default=None, ge=0, le=10, description="LLM fluency score (0-10)"
    )
    llm_naturalness: Optional[float] = Field(
        default=None, ge=0, le=10,
        description="LLM naturalness score (0-10)"
    )
    llm_tone: Optional[float] = Field(
        default=None, ge=0, le=10, description="LLM tone score (0-10)"
    )
    llm_overall: Optional[float] = Field(
        default=None, ge=0, le=10, description="LLM overall score (0-10)"
    )
    llm_reasoning: Optional[str] = Field(
        default=None, description="LLM voice quality reasoning"
    )


# =============================================================================
# Evaluation Results and Statistics
# =============================================================================

class EvaluationResult(BaseResult, LatencyMixin):
    """Single question evaluation result."""
    category: str = Field(default="", description="Question category")
    ground_truth: str = Field(default="", description="Expected transcription")
    wer: float = Field(
        default=100.0, ge=0, le=100,
        description="Word Error Rate percentage"
    )
    judge_scores: JudgeScores = Field(
        default_factory=JudgeScores, description="LLM judge evaluation"
    )
    voice_quality: Optional[VoiceQuality] = Field(
        default=None, description="Voice quality metrics"
    )


class CategoryStats(LatencyMixin):
    """Statistics for a question category."""
    count: int = Field(ge=0, description="Number of questions in category")
    avg_wer: float = Field(
        ge=0, le=100, description="Average WER for category"
    )
    avg_score: float = Field(
        ge=0, le=10, description="Average judge score for category"
    )

    # Rename inherited fields to match category stats naming
    avg_stt_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Average STT latency",
        alias="stt_latency_ms"
    )
    avg_tts_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Average TTS latency",
        alias="tts_latency_ms"
    )
    avg_total_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Average total latency",
        alias="total_latency_ms"
    )


class EvaluationSummary(LatencyMixin):
    """Summary statistics for evaluation run."""
    average_wer: float = Field(
        ge=0, le=100, description="Average Word Error Rate"
    )
    average_overall_score: float = Field(
        ge=0, le=10, description="Average judge overall score"
    )
    by_category: Dict[str, CategoryStats] = Field(
        default_factory=dict, description="Per-category statistics"
    )

    # Rename inherited fields to match summary naming
    average_stt_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Average STT latency",
        alias="stt_latency_ms"
    )
    average_tts_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Average TTS latency",
        alias="tts_latency_ms"
    )
    average_total_latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Average total latency",
        alias="total_latency_ms"
    )


class EvaluationReport(ServiceIdentificationMixin):
    """Complete evaluation report."""
    timestamp: str = Field(description="Evaluation timestamp")
    dataset: str = Field(description="Dataset file path")
    total_questions: int = Field(
        ge=0, description="Total number of questions"
    )
    evaluations: List[EvaluationResult] = Field(
        description="Individual evaluation results"
    )
    summary: EvaluationSummary = Field(description="Aggregated statistics")


# =============================================================================
# Configuration Models
# =============================================================================

class WERConfig(BaseModel):
    """
    Configuration for WER calculation.

    Attributes:
        return_percentage: Whether to return WER as percentage (0-100)
                          or decimal (0-1)
        handle_empty_strings: How to handle empty reference or hypothesis
    """
    return_percentage: bool = Field(
        default=True,
        description="Return WER as percentage instead of decimal"
    )
    handle_empty_strings: bool = Field(
        default=True,
        description="Return 1.0 (100%) for empty strings"
    )


class JudgeConfig(BaseModel):
    """
    Configuration for LLM judge service.

    Attributes:
        model_id: Bedrock model identifier for judge agent
        region_name: AWS region for Bedrock service
        max_retries: Maximum retry attempts for failed evaluations
        system_prompt: System prompt for judge agent
    """
    model_id: str = Field(
        default="au.anthropic.claude-haiku-4-5-20251001-v1:0",
        description="Bedrock model ID for judge evaluation"
    )
    region_name: str = Field(
        default="ap-southeast-2",
        description="AWS region for Bedrock service"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed evaluations"
    )
    system_prompt: str = Field(
        default="",  # Add default empty string
        description="System prompt for judge agent"
    )


class EvaluatorConfig(BaseModel):
    """
    Configuration for voice assistant evaluator.

    Attributes:
        dataset_path: Path to evaluation dataset JSON file
        audio_dir: Directory containing audio files
        evaluate_voice_quality: Whether to evaluate voice quality metrics
        max_retries: Maximum retry attempts for failed evaluations
        wer_config: Configuration for WER calculation
        judge_config: Configuration for LLM judge
    """
    dataset_path: str = Field(
        description="Path to evaluation dataset JSON file"
    )
    audio_dir: str = Field(description="Directory containing audio files")
    evaluate_voice_quality: bool = Field(
        default=True,
        description="Whether to evaluate voice quality metrics"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed evaluations"
    )
    wer_config: Optional[WERConfig] = Field(
        default=None,
        description="Configuration for WER calculation"
    )
    judge_config: Optional[JudgeConfig] = Field(
        default=None,
        description="Configuration for LLM judge"
    )


# =============================================================================
# Audio Quality Analysis Models
# =============================================================================

class AudioMetrics(BaseModel):
    """Comprehensive audio feature metrics extracted from librosa analysis."""

    # Fluency metrics
    duration: float = Field(
        ge=0, description="Audio duration in seconds"
    )
    pitch_mean: float = Field(
        ge=0, description="Mean fundamental frequency (Hz)"
    )
    pitch_std: float = Field(
        ge=0, description="Pitch standard deviation"
    )
    pitch_cv: float = Field(ge=0, description="Pitch coefficient of variation")
    speech_rate: float = Field(ge=0, description="Speech rate (onsets/second)")
    primary_tempo: float = Field(ge=0, description="Primary tempo (BPM)")
    energy_mean: float = Field(ge=0, description="Mean RMS energy")
    energy_consistency: float = Field(
        ge=0, le=1, description="Energy consistency (0-1)"
    )

    # Naturalness metrics
    spectral_centroid: float = Field(
        ge=0, description="Spectral centroid (Hz)"
    )
    mfcc_coefficients: List[float] = Field(description="MFCC coefficients")
    spectral_rolloff: float = Field(ge=0, description="Spectral rolloff (Hz)")

    # Tone metrics
    spectral_contrast: float = Field(ge=0, description="Spectral contrast")
    zero_crossing_rate: float = Field(ge=0, description="Zero crossing rate")
    harmonic_noise_ratio: float = Field(
        description="Harmonic-to-noise ratio (dB)"
    )

    # Overall quality metrics
    chroma: float = Field(ge=0, le=1, description="Chroma feature")
    tonnetz: float = Field(description="Tonnetz harmonic relationships")
    dynamic_range: float = Field(ge=0, description="Dynamic range")
    spectral_flatness: float = Field(
        ge=0, le=1, description="Spectral flatness"
    )


class NISQAConfig(BaseModel):
    """Configuration for NISQA evaluation."""
    model_path: str = Field(description="Path to NISQA model weights")
    sample_rate: int = Field(default=48000, description="Required sample rate")
    max_segments: int = Field(default=2000, description="Max audio segments")
    batch_size: int = Field(default=1, description="Batch size for processing")


class NISQAResults(BaseModel):
    """NISQA evaluation results with validation."""
    nisqa_mos: float = Field(
        ge=0, le=5, description="Mean Opinion Score (0-5)"
    )
    nisqa_noisiness: float = Field(
        ge=0, le=5, description="Noisiness score (0-5)"
    )
    nisqa_coloration: float = Field(
        ge=0, le=5, description="Coloration score (0-5)"
    )
    nisqa_discontinuity: float = Field(
        ge=0, le=5, description="Discontinuity score (0-5)"
    )
    nisqa_loudness: float = Field(
        ge=0, le=5, description="Loudness score (0-5)"
    )


class SpeechMetricsConfig(BaseModel):
    """Configuration for SpeechMetrics evaluation."""
    window_size: float = Field(
        default=0.75, description="Analysis window size"
    )
    enable_mosnet: bool = Field(default=True, description="Enable MOSNet")
    enable_srmr: bool = Field(default=True, description="Enable SRMR")


class SpeechMetricsResults(BaseModel):
    """SpeechMetrics evaluation results with validation."""
    mosnet_score: float = Field(
        ge=0, le=5, description="MOSNet quality score (0-5)"
    )
    srmr_score: float = Field(
        ge=-10, le=30, description="SRMR quality score (dB)"
    )


class LLMJudgeConfig(BaseModel):
    """Configuration for LLM-based voice quality evaluation."""
    model_id: str = Field(
        default="au.anthropic.claude-haiku-4-5-20251001-v1:0",
        description="Bedrock model ID"
    )
    region_name: str = Field(
        default="ap-southeast-2",
        description="AWS region"
    )
    temperature: float = Field(
        default=0.3, ge=0, le=1,
        description="LLM temperature"
    )
    max_tokens: int = Field(
        default=500, ge=1,
        description="Maximum response tokens"
    )


class LLMJudgeResults(BaseModel):
    """LLM voice judge evaluation results with validation."""
    llm_fluency: float = Field(
        ge=0, le=10, description="Fluency score (0-10)"
    )
    llm_naturalness: float = Field(
        ge=0, le=10, description="Naturalness score (0-10)"
    )
    llm_tone: float = Field(
        ge=0, le=10, description="Tone quality score (0-10)"
    )
    llm_overall: float = Field(
        ge=0, le=10, description="Overall quality score (0-10)"
    )
    llm_reasoning: str = Field(description="LLM explanation of scores")


class AudioQualityConfig(BaseModel):
    """Complete configuration for audio quality evaluation."""
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    enable_nisqa: bool = Field(default=True, description="Enable NISQA")
    enable_speechmetrics: bool = Field(
        default=True, description="Enable SpeechMetrics"
    )
    enable_llm_judge: bool = Field(
        default=False, description="Enable LLM judge"
    )

    nisqa_config: Optional[NISQAConfig] = Field(default=None)
    speechmetrics_config: Optional[SpeechMetricsConfig] = Field(default=None)
    llm_judge_config: Optional[LLMJudgeConfig] = Field(default=None)

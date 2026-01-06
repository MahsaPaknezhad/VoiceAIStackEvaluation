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
    mos: Optional[float] = Field(
        default=None, ge=0, le=5, description="Mean Opinion Score (1-5)"
    )
    noisiness: Optional[float] = Field(
        default=None, ge=0, le=5, description="Noisiness score (1-5)"
    )
    coloration: Optional[float] = Field(
        default=None, ge=0, le=5, description="Coloration score (1-5)"
    )
    discontinuity: Optional[float] = Field(
        default=None, ge=0, le=5, description="Discontinuity score (1-5)"
    )
    loudness: Optional[float] = Field(
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
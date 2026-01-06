"""
Pydantic models for voice evaluation framework.
Provides type safety and validation for evaluation data structures.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class JudgeScores(BaseModel):
    """LLM judge evaluation scores for response quality."""
    correctness: float = Field(default=0.0, ge=0, le=10, description="Factual accuracy score (0-10)")
    relevance: float = Field(default=0.0, ge=0, le=10, description="Question relevance score (0-10)")
    completeness: float = Field(default=0.0, ge=0, le=10, description="Response completeness score (0-10)")
    clarity: float = Field(default=0.0, ge=0, le=10, description="Communication clarity score (0-10)")
    overall: float = Field(default=0.0, ge=0, le=10, description="Overall quality score (0-10)")
    reasoning: str = Field(default="", description="Judge reasoning explanation")


class VoiceQuality(BaseModel):
    """Voice quality metrics from NISQA and SpeechMetrics."""
    # NISQA metrics
    mos: Optional[float] = Field(default=None, ge=0, le=5, description="Mean Opinion Score (1-5)")
    noisiness: Optional[float] = Field(default=None, ge=0, le=5, description="Noisiness score (1-5)")
    coloration: Optional[float] = Field(default=None, ge=0, le=5, description="Coloration score (1-5)")
    discontinuity: Optional[float] = Field(default=None, ge=0, le=5, description="Discontinuity score (1-5)")
    loudness: Optional[float] = Field(default=None, ge=0, le=5, description="Loudness score (1-5)")
    overall_quality: Optional[float] = Field(default=None, ge=0, le=5, description="Overall NISQA quality (1-5)")
    
    # SpeechMetrics
    mosnet_score: Optional[float] = Field(default=None, description="MOSNet quality score")
    srmr_score: Optional[float] = Field(default=None, description="Speech-to-Reverberation Modulation Ratio")
    
    # LLM judge voice quality (if enabled)
    llm_fluency: Optional[float] = Field(default=None, ge=0, le=10, description="LLM fluency score (0-10)")
    llm_naturalness: Optional[float] = Field(default=None, ge=0, le=10, description="LLM naturalness score (0-10)")
    llm_tone: Optional[float] = Field(default=None, ge=0, le=10, description="LLM tone score (0-10)")
    llm_overall: Optional[float] = Field(default=None, ge=0, le=10, description="LLM overall score (0-10)")
    llm_reasoning: Optional[str] = Field(default=None, description="LLM voice quality reasoning")


class TimingMetrics(BaseModel):
    """Latency and timing measurements."""
    stt_latency_ms: Optional[float] = Field(default=None, ge=0, description="STT processing latency in milliseconds")
    tts_latency_ms: Optional[float] = Field(default=None, ge=0, description="TTS generation latency in milliseconds")
    total_latency_ms: Optional[float] = Field(default=None, ge=0, description="End-to-end latency in milliseconds")


class EvaluationResult(BaseModel):
    """Single question evaluation result."""
    question_id: str = Field(description="Unique question identifier")
    category: str = Field(default="", description="Question category")
    ground_truth: str = Field(default="", description="Expected transcription")
    stt_output: str = Field(default="", description="Actual STT transcription")
    wer: float = Field(default=100.0, ge=0, le=100, description="Word Error Rate percentage")
    llm_response: str = Field(default="", description="LLM response text")
    judge_scores: JudgeScores = Field(default_factory=JudgeScores, description="LLM judge evaluation")
    voice_quality: Optional[VoiceQuality] = Field(default=None, description="Voice quality metrics")
    timing: TimingMetrics = Field(default_factory=TimingMetrics, description="Latency measurements")
    tts_audio_path: Optional[str] = Field(default=None, description="Path to generated TTS audio")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")


class CategoryStats(BaseModel):
    """Statistics for a question category."""
    count: int = Field(ge=0, description="Number of questions in category")
    avg_wer: float = Field(ge=0, le=100, description="Average WER for category")
    avg_score: float = Field(ge=0, le=10, description="Average judge score for category")
    avg_stt_latency_ms: Optional[float] = Field(default=None, ge=0, description="Average STT latency")
    avg_tts_latency_ms: Optional[float] = Field(default=None, ge=0, description="Average TTS latency")


class EvaluationSummary(BaseModel):
    """Summary statistics for evaluation run."""
    average_wer: float = Field(ge=0, le=100, description="Average Word Error Rate")
    average_overall_score: float = Field(ge=0, le=10, description="Average judge overall score")
    average_stt_latency_ms: Optional[float] = Field(default=None, ge=0, description="Average STT latency")
    average_tts_latency_ms: Optional[float] = Field(default=None, ge=0, description="Average TTS latency")
    average_total_latency_ms: Optional[float] = Field(default=None, ge=0, description="Average total latency")
    by_category: Dict[str, CategoryStats] = Field(default_factory=dict, description="Per-category statistics")


class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    timestamp: str = Field(description="Evaluation timestamp")
    dataset: str = Field(description="Dataset file path")
    total_questions: int = Field(ge=0, description="Total number of questions")
    stt_model: Optional[str] = Field(default=None, description="STT model name")
    stt_service_id: Optional[str] = Field(default=None, description="STT service identifier")
    tts_model: Optional[str] = Field(default=None, description="TTS model name")
    tts_service_id: Optional[str] = Field(default=None, description="TTS service identifier")
    evaluations: List[EvaluationResult] = Field(description="Individual evaluation results")
    summary: EvaluationSummary = Field(description="Aggregated statistics")


class PipelineResult(BaseModel):
    """Result from voice pipeline processing."""
    question_id: str
    audio_file: str
    stt_output: str = ""
    ground_truth: str = ""
    llm_response: str = ""
    tts_audio_path: Optional[str] = None
    stt_latency_ms: Optional[float] = None
    tts_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    status: str = "success"  # success, failed
    error: Optional[str] = None
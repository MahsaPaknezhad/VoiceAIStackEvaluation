"""
Result collector for voice evaluation framework.
Handles result collection and file operations.
"""

import json
import os
import wave
from typing import Dict, List, Optional, Any
from loguru import logger

from src.evaluation.models import (
    PipelineResult,
    PipelineComponents,
    ExecutionResults,
    PipelineEvaluationSummary,
    EvaluationOutput
)


class ResultCollector:
    """
    Collects and manages voice evaluation results and artifacts.

    This class handles the complete result collection workflow including
    execution result formatting, TTS audio file saving, experiment
    directory management, and JSON result serialization.

    Key responsibilities:
    - Format execution results into structured PipelineResult objects
    - Save generated TTS audio files with proper naming conventions
    - Create experiment-specific directory structures
    - Serialize results to JSON with atomic writes
    - Calculate and aggregate evaluation statistics

    Attributes:
        stt_config: STT service configuration for result metadata
        tts_config: TTS service configuration for result metadata
    """

    def __init__(
            self,
            stt_config: Optional[Dict[str, Any]] = None,
            tts_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize result collector with service configurations.

        Args:
            stt_config: STT service configuration containing service IDs
                       and model names for result metadata.
                       Defaults to empty dict.
            tts_config: TTS service configuration containing service IDs
                       and model names for result metadata.
                       Defaults to empty dict.
        """
        self.stt_config = stt_config or {}
        self.tts_config = tts_config or {}

    def _save_tts_audio(
            self,
            output_audio: Optional[bytes],
            question_id: str,
            pipeline_components: PipelineComponents
    ) -> Optional[str]:
        """
        Save TTS audio file to experiment-specific directory.

        Creates a directory structure based on STT and TTS service IDs
        and saves the generated audio with proper WAV formatting.
        Uses sample rate from the pipeline transport for correct playback.

        Args:
            output_audio: Generated TTS audio data as bytes, may be None
            question_id: Unique identifier for the question/audio pair
            pipeline_components: Pipeline components containing transport info

        Returns:
            Path to saved audio file, or None if no audio was generated
        """
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
            ground_truth: str
    ) -> PipelineResult:
        """
        Collect and format final pipeline result.

        Combines execution results with metadata to create a complete
        PipelineResult object. Handles TTS audio saving and latency
        rounding for consistent result formatting.

        Args:
            execution_results: Raw execution results from pipeline
            pipeline_components: Pipeline components for audio saving
            question_id: Unique identifier for the question
            audio_path: Path to input audio file
            ground_truth: Expected transcription for comparison

        Returns:
            Complete PipelineResult with all metadata and results
        """
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

    def save_results(
            self,
            results: List[PipelineResult],
            output_path: str
    ) -> None:
        """
        Save evaluation results to JSON file with statistics.

        Calculates summary statistics, creates evaluation output structure,
        and writes results to JSON with atomic file operations to prevent
        corruption during concurrent access.

        Args:
            results: List of pipeline results to save
            output_path: Path where JSON results should be saved
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Calculate summary statistics
        successful = len([r for r in results if r.status == 'success'])
        failed = len([r for r in results if r.status == 'failed'])
        skipped = len(results) - successful - failed

        summary = PipelineEvaluationSummary(
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

        os.rename(temp_path, output_path)
        logger.info(f"Results saved to {output_path}")

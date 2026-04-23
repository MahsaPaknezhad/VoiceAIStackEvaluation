# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
Main voice assistant evaluator orchestrating the complete evaluation workflow.
"""

import json
import asyncio
import random
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
from datetime import datetime

from src.evaluation.audio_quality_analyzer import VoiceQualityEvaluator
from src.evaluation.models import (
    JudgeScores, EvaluationResult, EvaluationReport,
    EvaluationSummary, CategoryStats, EvaluatorConfig
)
from src.evaluation.factories.quality_evaluator_factory import (
    QualityEvaluatorFactory
)


class VoiceAssistantEvaluator:
    """
    Main orchestrator for voice assistant evaluation workflow.

    Coordinates WER calculation, response evaluation, and voice quality
    assessment to provide comprehensive evaluation of voice assistant
    performance across multiple metrics.

    Attributes:
        config: Evaluator configuration settings
        dataset: Loaded evaluation dataset
        wer_calculator: Calculator for Word Error Rate metrics
        response_evaluator: Evaluator for response quality
        voice_evaluator: Optional voice quality evaluator
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        """
        Initialize voice assistant evaluator with configuration.

        Args:
            config: Configuration settings for evaluation
        """
        self.config = config
        self.dataset = self._load_dataset()

        # Initialize evaluators with unified factory
        factory = QualityEvaluatorFactory()
        self.wer_evaluator = factory.create_wer_evaluator(config.wer_config)
        self.response_evaluator = factory.create_response_evaluator(
            config.judge_config
        )

        if config.evaluate_voice_quality:
            self.voice_evaluator = VoiceQualityEvaluator(use_llm_judge=True)
        else:
            self.voice_evaluator = None

    async def initialize(self) -> None:
        """Initialize all evaluators asynchronously."""
        await self.wer_evaluator.initialize()
        await self.response_evaluator.initialize()
        if self.voice_evaluator:
            await self.voice_evaluator.initialize()

    def _load_dataset(self) -> Dict:
        """
        Load evaluation dataset from JSON file.

        Returns:
            Dictionary containing dataset questions and metadata

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If dataset file is not valid JSON
        """
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, 'r') as f:
            return json.load(f)

    async def evaluate_single(
        self,
        question_id: str,
        category: str,
        stt_output: str,
        ground_truth: str,
        llm_response: str,
        stt_latency: Optional[float] = None,
        tts_latency: Optional[float] = None,
        total_latency: Optional[float] = None,
        tts_audio_path: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single question with comprehensive metrics.

        Args:
            question_id: Unique identifier for the question
            category: Question category for analysis
            stt_output: Speech-to-text transcription output
            ground_truth: Expected correct transcription
            llm_response: AI assistant's response
            stt_latency: STT processing latency in milliseconds
            tts_latency: TTS generation latency in milliseconds
            total_latency: Total end-to-end latency in milliseconds
            tts_audio_path: Path to generated TTS audio file

        Returns:
            EvaluationResult containing all evaluation metrics
        """
        # Calculate WER for STT accuracy
        wer_result = await self.wer_evaluator.evaluate(
            ground_truth, stt_output
        )
        # Evaluate response quality using LLM judge
        judge_scores = await self.response_evaluator.evaluate(
            stt_output, llm_response
        )

        # Evaluate voice quality using:
        # - LLM
        # - Librosa
        # - Nisqa
        # - Speechmetrics
        voice_quality = None
        if self.config.evaluate_voice_quality and tts_audio_path \
                and self.voice_evaluator:
            voice_quality = await self.voice_evaluator.evaluate(tts_audio_path)

        return EvaluationResult(
            question_id=question_id,
            category=category,
            ground_truth=ground_truth,
            stt_output=stt_output,
            wer=wer_result.wer_score,
            llm_response=llm_response,
            judge_scores=judge_scores,
            voice_quality=voice_quality,
            stt_latency_ms=stt_latency,
            tts_latency_ms=tts_latency,
            total_latency_ms=total_latency,
            tts_audio_path=tts_audio_path
        )

    def _load_results_file(self, results_file: str) -> Optional[Dict]:
        """
        Load and validate a JSON results file containing evaluation data.

        Reads the specified results file, validates it's not empty, and parses
        the JSON content. Handles common file reading and JSON parsing errors
        gracefully with appropriate logging.

        Args:
            results_file: Path to the JSON results file to load

        Returns:
            Dictionary containing parsed JSON data, or None if loading failed

        """
        try:
            with open(results_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"Results file {results_file} is empty")
                return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error reading results file {results_file}: {e}")
            return None

    def _extract_results_data(self, results_data: Dict) -> tuple[Dict, List]:
        """
        Extract metadata and results from loaded evaluation data.

        Handles both new format (dict with 'results' key and metadata) and
        legacy format (direct list of results). Extracts model configuration
        metadata when available for comprehensive evaluation reporting.

        Args:
            results_data: Dictionary containing evaluation results and
            optional metadata

        Returns:
            Tuple containing:
                - metadata: Dict with STT/TTS model information
                    (empty if not available)
                - results: List of individual evaluation results
        """
        if isinstance(results_data, dict) and 'results' in results_data:
            metadata = {
                'stt_model': results_data.get('stt_model'),
                'stt_service_id': results_data.get('stt_service_id'),
                'tts_model': results_data.get('tts_model'),
                'tts_service_id': results_data.get('tts_service_id')
            }
            return metadata, results_data['results']
        else:
            # Old format - just a list
            return {}, results_data

    async def _process_single_question(
        self,
        question: Dict,
        results: List[Dict]
    ) -> Optional[EvaluationResult]:
        """
        Process a single evaluation question with robust retry logic.

        Finds the corresponding result for a question, evaluates it using the
        configured metrics (WER, LLM judge, voice quality), and handles
        failures with exponential backoff retry for transient errors like API
        rate limits.

        Args:
            question: Dictionary containing question data with 'id' and
                'category'
            results: List of result dictionaries to search for matching
                question

        Returns:
            EvaluationResult with comprehensive metrics, or None if processing
            failed after all retry attempts or if no matching result found
        """

        question_id = question['id']

        # Find corresponding result
        result = next(
            (r for r in results if r['question_id'] == question_id), None
        )
        if not result:
            logger.warning(f"No result found for {question_id}")
            return None

        logger.info(f"Evaluating {question_id}...")

        # Retry logic for robust evaluation
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retry attempt {attempt + 1}/"
                        f"{self.config.max_retries} for {question_id}"
                    )

                evaluation = await self.evaluate_single(
                    question_id,
                    question.get('category', ''),
                    result.get('stt_output', ''),
                    result.get('ground_truth', ''),
                    result.get('llm_response', ''),
                    result.get('stt_latency_ms'),
                    result.get('tts_latency_ms'),
                    result.get('total_latency_ms'),
                    result.get('tts_audio_path')
                )

                if attempt > 0:
                    logger.info(f"Retry successful for {question_id}")
                return evaluation

            except Exception as e:
                if self._should_retry(e, attempt):
                    wait_time = self._calculate_wait_time(attempt)
                    logger.warning(
                        f"Error on {question_id} (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Error evaluating {question_id} "
                        f"(final attempt): {e}"
                    )
                    return self._create_failed_evaluation(
                        question_id, question, result, str(e)
                    )

        return None

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should trigger a retry."""
        if attempt >= self.config.max_retries - 1:
            return False

        error_str = str(error).lower()
        bedrock_errors = [
            "serviceunavailableexception",
            "bedrock is unable to process",
            "throttlingexception",
            "rate limit",
            "too many requests",
            "service temporarily unavailable",
            "eventstreamError",
            "conversestream operation"
        ]

        return any(err in error_str for err in bedrock_errors)

    def _calculate_wait_time(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        return (3 ** (attempt + 1)) + random.uniform(0, 1)

    def _create_failed_evaluation(
        self,
        question_id: str,
        question: Dict,
        result: Dict,
        error_msg: str
    ) -> EvaluationResult:
        """Create evaluation result for failed processing."""
        return EvaluationResult(
            question_id=question_id,
            category=question.get('category', ''),
            ground_truth=result.get('ground_truth', ''),
            stt_output=result.get('stt_output', ''),
            wer=100.0,  # Max error for failed evaluation
            llm_response=result.get('llm_response', ''),
            judge_scores=JudgeScores(
                reasoning=f"Evaluation failed: {error_msg}"
            ),
            error=error_msg
        )

    def _log_evaluation_summary(
            self,
            evaluations: List[EvaluationResult],
            summary: EvaluationSummary) -> None:
        """Log detailed evaluation summary."""
        # Calculate latency averages
        stt_latencies = [
            e.stt_latency_ms for e in evaluations if e.stt_latency_ms
        ]
        tts_latencies = [
            e.tts_latency_ms for e in evaluations if e.tts_latency_ms
        ]
        total_latencies = [
            e.total_latency_ms for e in evaluations if e.total_latency_ms
        ]

        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total questions: {len(evaluations)}")
        print(f"Average WER: {summary.average_wer:.2f}%")
        print(f"Average Overall Score: {summary.average_overall_score:.2f}/10")

        if stt_latencies:
            print(
                "Average STT Latency: "
                f"{sum(stt_latencies)/len(stt_latencies):.2f}ms"
            )
        if tts_latencies:
            print(
                "Average TTS Latency: "
                f"{sum(tts_latencies)/len(tts_latencies):.2f}ms"
            )
        if total_latencies:
            print(
                "Average Total Latency: "
                f"{sum(total_latencies)/len(total_latencies):.2f}ms"
            )

        print("\nBy Category:")
        for category, stats in summary.by_category.items():
            stt_avg = (
                f", STT={int(sum(stt_latencies)/len(stt_latencies))}ms"
                if stt_latencies else ""
            )
            tts_avg = (
                f", TTS={int(sum(tts_latencies)/len(tts_latencies))}ms"
                if tts_latencies else ""
            )
            print(
                f"  {category}: WER={stats.avg_wer:.2f}%, "
                f"Score={stats.avg_score:.2f}/10{stt_avg}{tts_avg}"
            )

        print("="*80)

    def _create_evaluation_report(
            self,
            evaluations: List[EvaluationResult],
            metadata: Dict) -> EvaluationReport:
        """Create comprehensive evaluation report."""
        total_questions = len(evaluations)
        avg_wer = (
            sum(e.wer for e in evaluations) /
            total_questions if evaluations else 0
        )
        avg_score = (
            sum(e.judge_scores.overall for e in evaluations) /
            total_questions if evaluations else 0
        )

        stt_latencies = [
            e.stt_latency_ms for e in evaluations
            if e.stt_latency_ms is not None
        ]
        tts_latencies = [
            e.tts_latency_ms for e in evaluations
            if e.tts_latency_ms is not None
        ]
        total_latencies = [
            e.total_latency_ms for e in evaluations
            if e.total_latency_ms is not None
        ]

        avg_stt_latency = (
            sum(stt_latencies) / len(stt_latencies) if stt_latencies else None
        )
        avg_tts_latency = (
            sum(tts_latencies) / len(tts_latencies) if tts_latencies else None
        )
        avg_total_latency = (
            sum(total_latencies) / len(total_latencies)
            if total_latencies else None
        )

        # Calculate category statistics
        by_category = {}
        for category in set(e.category for e in evaluations if e.category):
            cat_evals = [e for e in evaluations if e.category == category]

            # Calculate category latency averages
            cat_stt_latencies = [
                e.stt_latency_ms for e in cat_evals
                if e.stt_latency_ms is not None
            ]
            cat_tts_latencies = [
                e.tts_latency_ms for e in cat_evals
                if e.tts_latency_ms is not None
            ]
            cat_total_latencies = [
                e.total_latency_ms for e in cat_evals
                if e.total_latency_ms is not None
            ]

            cat_avg_stt = (
                sum(cat_stt_latencies) / len(cat_stt_latencies)
                if cat_stt_latencies else None
            )
            cat_avg_tts = (
                sum(cat_tts_latencies) / len(cat_tts_latencies)
                if cat_tts_latencies else None
            )
            cat_avg_total = (
                sum(cat_total_latencies) / len(cat_total_latencies)
                if cat_total_latencies else None
            )

            by_category[category] = CategoryStats(
                count=len(cat_evals),
                avg_wer=sum(e.wer for e in cat_evals) / len(cat_evals),
                avg_score=(
                    sum(e.judge_scores.overall for e in cat_evals) /
                    len(cat_evals)
                ),
                stt_latency_ms=cat_avg_stt,      # Use alias name
                tts_latency_ms=cat_avg_tts,      # Use alias name
                total_latency_ms=cat_avg_total   # Use alias name
            )

        summary = EvaluationSummary(
            average_wer=avg_wer,
            average_overall_score=avg_score,
            by_category=by_category,
            stt_latency_ms=avg_stt_latency,      # Use alias name
            tts_latency_ms=avg_tts_latency,      # Use alias name
            total_latency_ms=avg_total_latency   # Use alias name
        )

        self._log_evaluation_summary(evaluations, summary)

        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            dataset=self.config.dataset_path,
            total_questions=total_questions,
            evaluations=evaluations,
            summary=summary,
            **metadata
        )

    async def run_evaluation(
            self,
            results_file: str) -> Optional[EvaluationReport]:
        """
        Run comprehensive evaluation on pre-recorded results.

        Args:
            results_file: Path to JSON file with STT outputs and responses

        Returns:
            EvaluationReport with complete analysis or None if failed
        """
        # Initialize evaluators first
        await self.initialize()

        # Load results with error handling
        results_data = self._load_results_file(results_file)
        if results_data is None:
            return None

        # Extract metadata and results
        metadata, results = self._extract_results_data(results_data)

        logger.info("Starting evaluation...")

        # Process all evaluations
        evaluations = []
        for i, question in enumerate(self.dataset['questions']):
            if i > 0:  # Not the first question
                logger.info("Pausing 3 seconds to avoid rate limiting...")
                await asyncio.sleep(3)

            evaluation = await self._process_single_question(question, results)
            if evaluation:
                evaluations.append(evaluation)

        # Create evaluation report (implementation depends on existing models)
        # This would need to be implemented based on your
        # EvaluationReport model
        return self._create_evaluation_report(evaluations, metadata)

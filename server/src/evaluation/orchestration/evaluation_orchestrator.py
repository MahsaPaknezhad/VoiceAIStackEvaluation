"""
Evaluation orchestrator for voice evaluation framework.
Orchestrates the evaluation process across all dataset items.
"""

import asyncio
import os
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from loguru import logger

from src.evaluation.models import PipelineResult
from src.evaluation.results.results_collector import ResultCollector


class EvaluationOrchestrator:
    """
    Orchestrates voice evaluation across entire datasets.

    This class manages the complete evaluation workflow including dataset
    iteration, progress tracking, error handling with retries, result
    collection, and incremental saving. It coordinates between the runner,
    result collector, and provides comprehensive logging and statistics.

    Key responsibilities:
    - Iterate through dataset questions with progress tracking
    - Handle file processing with retry logic and exponential backoff
    - Coordinate result collection and incremental saving
    - Provide comprehensive evaluation statistics and logging
    - Manage rate limiting between requests

    Attributes:
        runner: Voice assistant runner instance for processing
        output_path: Path for saving incremental results
        results: Accumulated evaluation results
        processed_count: Number of successfully processed items
        error_count: Number of items that failed processing
        skipped_count: Number of items skipped due to missing files
        result_collector: Instance for collecting and saving results
    """

    def __init__(
            self,
            runner: Any,
            output_path: Optional[str] = None
    ) -> None:
        """
        Initialize evaluation orchestrator.

        Args:
            runner: Voice assistant runner instance with dataset and configs
            output_path: Optional path for incremental result saving
        """
        self.runner = runner
        self.output_path = output_path
        self.results: List[PipelineResult] = []
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        self.result_collector = ResultCollector(
            runner.stt_config, runner.tts_config
        )

    async def run_evaluation(self) -> List[PipelineResult]:
        """
        Run evaluation on all dataset items.

        Orchestrates the complete evaluation process including dataset
        iteration, progress tracking, and final summary logging.

        Returns:
            List of PipelineResult objects for all processed items
        """
        total_files = len(self.runner.dataset['questions'])
        logger.info(f"Starting evaluation of {total_files} audio files")

        for i, question in enumerate(self.runner.dataset['questions']):
            await self._process_question(question, i)
            await self._pause_if_needed(i, total_files)

        self._log_summary(total_files)
        return self.results

    async def _process_question(
            self,
            question: Dict[str, Any],
            index: int
    ) -> None:
        """
        Process a single question from the dataset.

        Handles file validation, processing with retry logic, result
        classification, and incremental saving. Creates failed results
        for missing files or processing errors.

        Args:
            question: Question dictionary containing id and audio_file
            index: Question index for progress tracking
        """
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
                self.result_collector.save_results(
                    self.results,
                    self.output_path
                )

    async def _pause_if_needed(self, index: int, total_files: int) -> None:
        """
        Pause between items to avoid rate limiting.

        Adds a delay between processing items to prevent overwhelming
        external services with too many concurrent requests.

        Args:
            index: Current item index
            total_files: Total number of files to process
        """
        if index < total_files - 1:
            logger.info("Pausing 3 seconds to avoid rate limiting...")
            await asyncio.sleep(3)

    def _log_summary(self, total_files: int) -> None:
        """
        Log final evaluation summary with statistics.

        Provides comprehensive statistics including success rates,
        error counts, and processing summary in a formatted display.

        Args:
            total_files: Total number of files in the dataset
        """
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
            operation: Callable[[], Any],
            question_id: str,
            max_retries: int = 3
    ) -> Any:
        """
        Execute operation with retry logic and exponential backoff.

        Implements robust retry mechanism with exponential backoff and
        jitter to handle transient failures in voice service processing.

        Args:
            operation: Async callable to execute with retries
            question_id: Question ID for logging purposes
            max_retries: Maximum number of retry attempts

        Returns:
            Result from successful operation execution

        Raises:
            Exception: Last exception if all retries are exhausted
        """
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
            result: PipelineResult
    ) -> Tuple[PipelineResult, bool]:
        """
        Process result and classify success status.

        Evaluates pipeline result to determine if processing was successful
        based on TTS configuration and audio generation requirements.

        Args:
            result: Pipeline result to evaluate

        Returns:
            Tuple of (processed_result, is_success_boolean)
        """
        if self.runner.tts_config and result.tts_audio_path is None:
            result.status = "failed"
            result.error = "TTS failed - no audio generated"
            return result, False
        else:
            result.status = "success"
            return result, True

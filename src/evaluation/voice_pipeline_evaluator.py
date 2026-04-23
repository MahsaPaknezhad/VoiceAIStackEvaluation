# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
Voice assistant evaluation pipeline entry point.
Orchestrates voice assistant evaluation across datasets with multiple AI
services.
"""

# Standard library imports
import argparse
import asyncio
import json
import warnings
from typing import Dict, List, Optional, Any

# Third-party imports
from dotenv import load_dotenv
from loguru import logger

from src.evaluation.config.configuration_manager import ConfigurationManager
from src.evaluation.models import PipelineResult
from src.evaluation.results.results_collector import ResultCollector
from src.evaluation.pipeline.voice_pipeline_processor import (
    VoicePipelineProcessor
)
from src.evaluation.orchestration.evaluation_orchestrator import (
    EvaluationOrchestrator
)

# Configuration
warnings.filterwarnings("ignore", message="Dangling tasks detected")
load_dotenv(override=True)
logger.disable("pipecat.pipeline.task")


class VoiceAssistantRunner:
    """
    Main runner for voice assistant evaluation pipeline.

    CLI interface and dataset manager for voice assistant evaluation.
    Delegates core processing to VoicePipelineProcessor for clean separation.

    Key responsibilities:
    - Load and manage evaluation datasets
    - Extract ground truth data for evaluation
    - Delegate processing to VoicePipelineProcessor
    - Coordinate with evaluation orchestrator for batch processing

    Attributes:
        dataset_path: Path to evaluation dataset JSON file
        audio_dir: Directory containing audio files for evaluation
        config_manager: Configuration manager for loading service configs
        dataset: Loaded evaluation dataset dictionary
        stt_config: STT service configuration dictionary
        tts_config: TTS service configuration dictionary
    """

    def __init__(
            self,
            dataset_path: str,
            audio_dir: str,
            stt_config: Optional[str] = None,
            tts_config: Optional[str] = None
    ) -> None:
        """
        Initialize voice assistant runner with dataset and service configs.

        Args:
            dataset_path: Path to JSON file containing evaluation dataset
            audio_dir: Directory path containing audio files referenced in
                dataset
            stt_config: Optional path to STT service configuration file
            tts_config: Optional path to TTS service configuration file
        """
        self.dataset_path = dataset_path
        self.audio_dir = audio_dir

        # Use configuration manager
        self.config_manager = ConfigurationManager()
        self.dataset = self._load_dataset()

        # Load configs using the manager
        self.stt_config = self.config_manager.load_config(stt_config) if \
            stt_config else None
        self.tts_config = self.config_manager.load_config(tts_config) if \
            tts_config else None

    def _load_dataset(self) -> Dict[str, Any]:
        """
        Load evaluation dataset from JSON file.

        Returns:
            Dictionary containing dataset questions and metadata

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If dataset file is not valid JSON
        """
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    async def run_all(
            self,
            output_path: Optional[str] = None
    ) -> List[PipelineResult]:
        """
        Run evaluation on all audio files in the dataset.

        Creates processor instance and delegates to EvaluationOrchestrator
        for managing the complete evaluation workflow.

        Args:
            output_path: Optional path for saving incremental results

        Returns:
            List of PipelineResult objects for all processed items
        """
        processor = VoicePipelineProcessor(self.stt_config, self.tts_config)
        orchestrator = EvaluationOrchestrator(
            processor, self.dataset, self.audio_dir, output_path
        )
        return await orchestrator.run_evaluation()


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure command-line argument parser.

    Sets up argument parser with all required and optional parameters
    for running voice assistant evaluation including dataset paths,
    service configurations, and output settings.

    Returns:
        Configured ArgumentParser instance with all evaluation parameters
    """
    parser = argparse.ArgumentParser(
        description='Run voice assistant evaluation on dataset audio files'
    )
    parser.add_argument(
        '--dataset',
        default='data/voiceassistant_eval/'
        'voiceassistant_eval_dataset.json',
        help='Path to evaluation dataset JSON file'
    )
    parser.add_argument(
        '--audio-dir',
        default='data/voiceassistant_eval/audio_input',
        help='Directory containing audio files referenced in dataset'
    )
    parser.add_argument(
        '--output',
        default='data/voiceassistant_eval/bot_results.json',
        help='Output path for evaluation results JSON file'
    )
    parser.add_argument(
        '--stt-config',
        help='Path to STT service configuration file '
             '(e.g., data/stt_bot_configs/'
             'deepgram_nova3_config.json)'
    )
    parser.add_argument(
        '--tts-config',
        help='Path to TTS service configuration file '
             '(e.g., data/tts_bot_configs/'
             'deepgram_aura_config.json)'
    )
    return parser


def print_final_summary(
        results: List[PipelineResult],
        output_path: str
) -> None:
    """
    Print comprehensive evaluation summary with statistics.

    Displays formatted summary including success rates, error counts,
    and next steps for further analysis. Provides clear visual separation
    and actionable information for users.

    Args:
        results: List of pipeline results from evaluation
        output_path: Path where results were saved
    """
    successful = len([r for r in results if r.status == 'success'])
    failed = len([r for r in results if r.status == 'failed'])

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    success_rate = (successful/len(results))*100 if results else 0
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    print("\nNext step: Run evaluation with:")
    print(f"  python evaluate_voiceassistant.py --results {output_path}")


def cleanup_background_tasks() -> None:
    """
    Cancel background asyncio tasks to prevent hanging processes.

    Identifies and cancels all pending asyncio tasks except the current
    main task to ensure clean shutdown. Handles exceptions gracefully
    to prevent shutdown errors.
    """
    try:
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if task != current_task and not task.done():
                task.cancel()
    except Exception:
        pass


async def main() -> None:
    """
    Main entry point for voice assistant evaluation pipeline.

    Orchestrates the complete evaluation workflow including argument parsing,
    runner initialization, evaluation execution, result saving, and cleanup.
    Provides comprehensive logging and error handling.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    runner = VoiceAssistantRunner(
        args.dataset,
        args.audio_dir,
        stt_config=args.stt_config,
        tts_config=args.tts_config,
    )

    logger.info("Starting voice assistant evaluation on all audio files...")
    results = await runner.run_all(args.output)

    result_collector = ResultCollector(runner.stt_config, runner.tts_config)
    result_collector.save_results(results, args.output)

    print_final_summary(results, args.output)
    cleanup_background_tasks()


if __name__ == "__main__":
    asyncio.run(main())

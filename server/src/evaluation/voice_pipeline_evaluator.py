"""
Run the Q&A bot on VoiceAssistant-Eval dataset audio files.
Collects STT outputs and bot responses for evaluation.
"""

# Standard library imports
import argparse
import asyncio
import json
import os
import warnings
from typing import Dict, List

# Third-party imports
from dotenv import load_dotenv
from loguru import logger

from src.core.agent_builder import build_conversation_agent
from src.core.llm_processor import StrandsAgentsProcessor

from src.evaluation.config.configuration_manager import ConfigurationManager
from src.evaluation.factories.stt_factory import STTServiceFactory
from src.evaluation.factories.tts_factory import TTSServiceFactory
from src.evaluation.pipeline.audio_processor import AudioProcessor
from src.evaluation.services.service_manager import ServiceManager
from src.evaluation.results.results_collector import ResultCollector
from src.evaluation.orchestration.evaluation_orchestrator import (
    EvaluationOrchestrator
)

from src.evaluation.models import (
    PipelineResult,
)

# Configuration
warnings.filterwarnings("ignore", message="Dangling tasks detected")
load_dotenv(override=True)
logger.disable("pipecat.pipeline.task")


class VoiceAssistantRunner:
    def __init__(
            self,
            dataset_path: str,
            audio_dir: str,
            stt_config: str = None,
            tts_config: str = None):
        self.dataset_path = dataset_path
        self.audio_dir = audio_dir

        # Use configuration manager
        self.config_manager = ConfigurationManager()
        self.stt_factory = STTServiceFactory(self.config_manager)
        self.tts_factory = TTSServiceFactory(self.config_manager)
        self.dataset = self._load_dataset()

        # Load configs using the manager
        self.stt_config = self.config_manager.load_config(stt_config) if \
            stt_config else None
        self.tts_config = self.config_manager.load_config(tts_config) if \
            tts_config else None

    def _create_stt_service(self):
        """Create STT service using factory."""
        return self.stt_factory.create_service(self.stt_config)

    def _create_tts_service(self):
        """Create TTS service using factory."""
        return self.tts_factory.create_service(self.tts_config)

    def _load_dataset(self) -> Dict:
        """Load the evaluation dataset"""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    async def process_audio_file(
            self,
            audio_path: str,
            question_id: str) -> PipelineResult:
        """
        Process a single audio file through the bot pipeline.

        Returns:
            Dict with stt_output, bot_response, and latencies
        """
        logger.info(f"=== PROCESSING FILE: {question_id} ===")
        logger.info(f"Audio file path: {audio_path}")
        logger.info(f"File exists: {os.path.exists(audio_path)}")
        logger.info(f"Processing {question_id}: {audio_path}")

        # Get the ground truth transcript from dataset (for WER comparison)
        question_data = next(
            (q for q in self.dataset['questions'] if q['id'] == question_id),
            None
        )
        if not question_data:
            raise ValueError(f"Question {question_id} not found in dataset")

        ground_truth = question_data['text']

        # Process audio
        audio_processor = AudioProcessor(self.stt_config)
        audio_processor.process_audio_file(audio_path)

        # Create services
        stt = self._create_stt_service()
        tts = self._create_tts_service()
        agent = build_conversation_agent(
            model_id="au.anthropic.claude-haiku-4-5-20251001-v1:0",
            tts_service=tts)
        llm = StrandsAgentsProcessor(agent=agent)

        # Setup pipeline
        pipeline_components = audio_processor.build_pipeline(stt, tts, llm)

        # Execute pipeline using AudioProcessor
        execution_results = await audio_processor.execute_pipeline(
            pipeline_components, stt, audio_path, self.stt_config
        )

        # Create service manager and cleanup
        service_manager = ServiceManager()
        await service_manager.cleanup_services(stt, tts)

        logger.info("Pipeline processing complete, continuing...")
        # Create result collector
        result_collector = ResultCollector(self.stt_config, self.tts_config)

        # Collect final result
        result = result_collector.collect_result(
            execution_results,
            pipeline_components,
            question_id,
            audio_path,
            ground_truth
        )

        return result

    async def run_all(self, output_path: str = None) -> List[PipelineResult]:
        """Run bot on all audio files in dataset"""
        orchestrator = EvaluationOrchestrator(self, output_path)
        return await orchestrator.run_evaluation()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Run bot on VoiceAssistant-Eval dataset'
    )
    parser.add_argument(
        '--dataset',
        default='evaluation_data/voiceassistant_eval/'
        'voiceassistant_eval_dataset.json',
        help='Path to dataset JSON'
    )
    parser.add_argument(
        '--audio-dir',
        default='evaluation_data/voiceassistant_eval/audio_input',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output',
        default='evaluation_data/voiceassistant_eval/bot_results.json',
        help='Output path for bot results'
    )
    parser.add_argument(
        '--stt-config',
        help='STT service config (e.g., evaluation_data/bot_configs/'
        'deepgram_nova3_config.json)'
    )
    parser.add_argument(
        '--tts-config',
        help='TTS service config (e.g., evaluation_data/tts_bot_configs/'
        'deepgram_aura_config.json)'
    )
    return parser


def print_final_summary(
        results: List[PipelineResult],
        output_path: str) -> None:
    """Print final evaluation summary."""
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
    """Cancel background tasks but not the main task."""
    try:
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if task != current_task and not task.done():
                task.cancel()
    except Exception:
        pass


async def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    runner = VoiceAssistantRunner(
        args.dataset,
        args.audio_dir,
        stt_config=args.stt_config,
        tts_config=args.tts_config,
    )

    logger.info("Running bot on all audio files...")
    results = await runner.run_all(args.output)

    result_collector = ResultCollector(runner.stt_config, runner.tts_config)
    result_collector.save_results(results, args.output)

    print_final_summary(results, args.output)
    cleanup_background_tasks()


if __name__ == "__main__":
    asyncio.run(main())

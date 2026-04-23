# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
Evaluation script entry point for VoiceAssistant-Eval dataset.
Measures STT accuracy (WER) and response quality using LLM-as-judge.
"""

import asyncio
import argparse
import json

from src.evaluation.models import EvaluatorConfig, WERConfig, JudgeConfig
from src.evaluation.orchestration.voice_assistant_evaluator import (
    VoiceAssistantEvaluator
)


async def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description='Voice Assistant Evaluation')
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to evaluation dataset JSON'
    )
    parser.add_argument(
        '--results',
        required=True,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output evaluation JSON file'
    )
    parser.add_argument(
        '--audio-dir',
        default='',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--no-voice-quality',
        action='store_true',
        help='Skip voice quality evaluation'
    )

    args = parser.parse_args()

    config = EvaluatorConfig(
        dataset_path=args.dataset,
        audio_dir=args.audio_dir,
        evaluate_voice_quality=not args.no_voice_quality,
        wer_config=WERConfig(),
        judge_config=JudgeConfig()
    )

    evaluator = VoiceAssistantEvaluator(config)
    report = await evaluator.run_evaluation(args.results)

    if report:
        # Save report to output file

        with open(args.output, 'w') as f:
            json.dump(report.model_dump(), f, indent=2)
        print(
            f"Evaluation completed: {report.summary.average_wer:.2f}% WER, "
            f"{report.summary.average_overall_score:.2f} avg score"
        )
        print(f"Results saved to: {args.output}")
    else:
        print("Evaluation failed")

if __name__ == "__main__":
    asyncio.run(main())

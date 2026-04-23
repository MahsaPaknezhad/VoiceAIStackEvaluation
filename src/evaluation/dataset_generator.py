# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

"""
Generate evaluation dataset from VoiceAssistant-Eval dataset.
Uses the MathLLMs/VoiceAssistant-Eval dataset from HuggingFace.
"""

import json
import os
from datetime import datetime
from typing import Dict, List
from loguru import logger
from datasets import load_dataset


# Available splits in VoiceAssistant-Eval
AVAILABLE_SPLITS = [
    'listening_general',
    'listening_music',
    'listening_sound',
    'listening_speech',
    'speaking_assistant',
    'speaking_emotion',
    'speaking_instruction_following',
    'speaking_multi_round',
    'speaking_reasoning',
    'speaking_robustness',
    'speaking_roleplay',
    'speaking_safety',
    'viewing_multi_discipline'
]


def save_audio_to_file(audio_bytes, output_file):
    """Save audio bytes to file."""
    with open(output_file, "wb") as f:
        f.write(audio_bytes)
    logger.info(f"Saved audio to {output_file}")


def generate_evaluation_dataset(
    splits: List[str] = None,
    max_samples_per_split: int = 10,
    output_dir: str = "data/voiceassistant_eval",
    min_audio_duration: float = 5.0  # Minimum duration in seconds
):
    """
    Generate evaluation dataset from VoiceAssistant-Eval.
    
    Args:
        splits: List of splits to use. If None, uses a default subset.
        max_samples_per_split: Maximum number of samples to take from each split.
        output_dir: Directory to save the evaluation data.
        min_audio_duration: Minimum audio duration in seconds to include.
    """
    
    if splits is None:
        # Use a diverse subset by default
        splits = [
            'listening_general',
            'listening_speech',
            'speaking_assistant',
            'speaking_instruction_following',
            'speaking_reasoning'
        ]
    
    logger.info(f"Generating evaluation dataset from {len(splits)} splits")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio_input")
    os.makedirs(audio_dir, exist_ok=True)
    
    all_questions = []
    audio_files = []
    
    for split in splits:
        logger.info(f"Loading split: {split}")
        
        try:
            # Load the dataset split
            data = load_dataset("MathLLMs/VoiceAssistant-Eval", split)
            
            # Get test split
            test_data = data["test"]
            
            # Take up to max_samples_per_split
            num_samples = min(len(test_data), max_samples_per_split)
            
            logger.info(f"Processing {num_samples} samples from {split}")
            
            samples_added = 0
            for i in range(len(test_data)):
                if samples_added >= max_samples_per_split:
                    break
                    
                sample = test_data[i]
                
                # Extract audio
                user_audio = sample.get("user_audio_0")
                if user_audio is None:
                    logger.warning(f"No audio found for sample {i} in {split}")
                    continue
                
                # Check audio duration (assuming 16kHz, 16-bit mono)
                audio_duration = len(user_audio) / (16000 * 2)  # bytes / (sample_rate * bytes_per_sample)
                
                if audio_duration < min_audio_duration:
                    logger.debug(f"Skipping sample {i} in {split}: duration {audio_duration:.2f}s < {min_audio_duration}s")
                    continue
                
                # Generate unique ID
                question_id = f"{split}_{samples_added}"
                
                logger.info(f"Adding sample {question_id}: duration {audio_duration:.2f}s")
                
                # Save audio file
                audio_filename = f"{question_id}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)
                save_audio_to_file(user_audio, audio_path)
                audio_files.append(audio_filename)
                
                # Extract expected text (transcript) from extra field
                expected_text = ""
                if 'extra' in sample and sample['extra']:
                    transcripts = sample['extra'].get('user_audio_transcripts', [])
                    if transcripts and len(transcripts) > 0:
                        expected_text = transcripts[0]
                
                # Extract expected response from ref_answers
                expected_response = ""
                if 'ref_answers' in sample and sample['ref_answers']:
                    expected_response = sample['ref_answers'][0] if isinstance(sample['ref_answers'], list) else sample['ref_answers']
                
                # Determine category from split name
                if split.startswith("listening_"):
                    category = split.replace("listening_", "")
                elif split.startswith("speaking_"):
                    category = split.replace("speaking_", "")
                elif split.startswith("viewing_"):
                    category = split.replace("viewing_", "")
                else:
                    category = split
                
                # Create question entry
                question_entry = {
                    "id": question_id,
                    "text": expected_text,  # The transcript of what's in the audio
                    "expected_answer": expected_response,  # The expected answer to the question
                    "category": category,
                    "split": split,
                    "audio_file": audio_filename,
                    "metadata": {
                        "original_index": i,
                        "has_audio": True
                    }
                }
                
                all_questions.append(question_entry)
                samples_added += 1
                
        except Exception as e:
            logger.error(f"Error processing split {split}: {e}")
            continue
    
    # Create dataset JSON
    dataset = {
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "source": "MathLLMs/VoiceAssistant-Eval",
            "total_questions": len(all_questions),
            "splits_used": splits,
            "max_samples_per_split": max_samples_per_split,
            "description": "Evaluation dataset generated from VoiceAssistant-Eval",
            "audio_location": audio_dir
        },
        "questions": all_questions
    }
    
    # Save dataset
    dataset_path = os.path.join(output_dir, "voiceassistant_eval_dataset.json")
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Saved dataset to {dataset_path}")
    
    # Create evaluation template
    template = create_evaluation_template(all_questions, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("VOICEASSISTANT-EVAL DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"\nDataset saved to: {dataset_path}")
    print(f"Audio files saved to: {audio_dir}")
    print(f"\nTotal questions: {len(all_questions)}")
    print(f"Total audio files: {len(audio_files)}")
    
    print("\nQuestions by category:")
    category_counts = {}
    for q in all_questions:
        cat = q["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Audio files are ready in: " + audio_dir)
    print("2. Use these files with the evaluation framework:")
    print("   - Single STT: python evaluation_dataset.py analyze <template>")
    print("   - Multi-STT: python multi_stt_evaluation.py")
    print("   - Multi-TTS: python multi_tts_evaluation.py")
    print("   - Combined: python combined_evaluation.py")
    print("\n3. Template file created: " + template)
    print("="*80 + "\n")
    
    return dataset_path, template


def create_evaluation_template(questions: List[Dict], output_dir: str) -> str:
    """Create evaluation template for the generated questions."""
    
    template = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "source": "VoiceAssistant-Eval",
            "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "stt_service": "deepgram-nova-3",
            "tts_service": "deepgram-aura-2-delia",
            "judge_model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        },
        "results": []
    }
    
    for q in questions:
        template["results"].append({
            "question_id": q["id"],
            "question_text": q["text"],
            "question_category": q["category"],
            "expected_answer": q["expected_answer"],
            "audio_file": q["audio_file"],
            "transcription": "",  # Fill this after STT
            "agent_response": "",  # Fill this after agent responds
            "wer_metrics": {},
            "validity_evaluation": {}
        })
    
    template_path = os.path.join(output_dir, "voiceassistant_eval_template.json")
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created evaluation template: {template_path}")
    return template_path


def list_available_splits():
    """List all available splits in the VoiceAssistant-Eval dataset."""
    print("\n" + "="*80)
    print("AVAILABLE SPLITS IN VOICEASSISTANT-EVAL")
    print("="*80)
    print("\nListening Tasks:")
    for split in AVAILABLE_SPLITS:
        if split.startswith("listening_"):
            print(f"  - {split}")
    
    print("\nSpeaking Tasks:")
    for split in AVAILABLE_SPLITS:
        if split.startswith("speaking_"):
            print(f"  - {split}")
    
    print("\nViewing Tasks:")
    for split in AVAILABLE_SPLITS:
        if split.startswith("viewing_"):
            print(f"  - {split}")
    
    print("\n" + "="*80)
    print("USAGE")
    print("="*80)
    print("\n# Generate with default splits (5 splits, 10 samples each):")
    print("python generate_eval_from_dataset.py")
    
    print("\n# Generate with specific splits:")
    print("python generate_eval_from_dataset.py --splits listening_general,speaking_assistant")
    
    print("\n# Generate with more samples per split:")
    print("python generate_eval_from_dataset.py --max-samples 20")
    
    print("\n# Generate all splits:")
    print("python generate_eval_from_dataset.py --all")
    print("="*80 + "\n")


async def analyze_voiceassistant_results(
    results_file: str,
    output_file: str = None
):
    """Analyze VoiceAssistant-Eval results using the standard evaluation framework."""
    from evaluation_dataset import analyze_results
    
    logger.info(f"Analyzing VoiceAssistant-Eval results: {results_file}")
    
    # Use the standard analysis function
    analyzed_file = await analyze_results(results_file, output_file)
    
    # Load and print summary
    with open(analyzed_file, 'r') as f:
        data = json.load(f)
    
    summary = data.get("summary", {})
    
    print("\n" + "="*80)
    print("VOICEASSISTANT-EVAL ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {analyzed_file}")
    print(f"\nTotal Questions: {summary.get('total_questions', 0)}")
    print(f"Average WER: {summary.get('average_wer', 0)}%")
    print(f"Validity Rate: {summary.get('validity_rate', 0)}%")
    print(f"Average Score: {summary.get('average_overall_score', 0)}/10")
    
    print("\nBy Category:")
    for cat, stats in summary.get("by_category", {}).items():
        print(f"  {cat}:")
        print(f"    Valid: {stats['valid']}/{stats['total']} ({stats['validity_rate']}%)")
        print(f"    Avg Score: {stats['average_score']}/10")
        print(f"    Avg WER: {stats['average_wer']}%")
    
    print("="*80 + "\n")
    
    return analyzed_file


if __name__ == "__main__":
    import sys
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Generate evaluation dataset from VoiceAssistant-Eval")
    parser.add_argument(
        "--splits",
        type=str,
        help="Comma-separated list of splits to use"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples per split (default: 10)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all available splits"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available splits and exit"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        help="Analyze existing results file"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_splits()
    elif args.analyze:
        asyncio.run(analyze_voiceassistant_results(args.analyze))
    else:
        # Determine which splits to use
        splits = None
        if args.all:
            splits = AVAILABLE_SPLITS
        elif args.splits:
            splits = args.splits.split(",")
        
        # Generate dataset
        try:
            generate_evaluation_dataset(
                splits=splits,
                max_samples_per_split=args.max_samples
            )
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            print("\nNote: Make sure you have the required packages installed:")
            print("  pip install datasets torchaudio soundfile")
            sys.exit(1)

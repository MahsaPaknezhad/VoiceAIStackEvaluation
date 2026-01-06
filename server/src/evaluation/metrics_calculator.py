"""
Evaluation script for VoiceAssistant-Eval dataset.
Measures STT accuracy (WER) and response quality using LLM-as-judge.
"""

import json
import os
import asyncio
import random
from typing import Dict, List
from jiwer import wer
from datetime import datetime
from strands import Agent
from strands.models import BedrockModel
from loguru import logger
import argparse
from audio_quality_analyzer import VoiceQualityEvaluator
from .models import (
    JudgeScores, VoiceQuality, TimingMetrics, EvaluationResult,
    CategoryStats, EvaluationSummary, EvaluationReport, PipelineResult
)


class VoiceAssistantEvaluator:
    def __init__(self, dataset_path: str, audio_dir: str, evaluate_voice_quality: bool = True):
        self.dataset_path = dataset_path
        self.audio_dir = audio_dir
        self.dataset = self._load_dataset()
        self.judge_agent = self._create_judge_agent()
        self.evaluate_voice_quality = evaluate_voice_quality
        
        if evaluate_voice_quality:
            self.voice_evaluator = VoiceQualityEvaluator(use_llm_judge=True)
        
    def _load_dataset(self) -> Dict:
        """Load the evaluation dataset"""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def _create_judge_agent(self) -> Agent:
        """Create LLM judge agent"""
        model = BedrockModel(
            model_id="au.anthropic.claude-haiku-4-5-20251001-v1:0",
            region_name="ap-southeast-2"
        )
        return Agent(
            name="EvaluationJudge",
            model=model,
            system_prompt="""You are an expert evaluator assessing AI assistant responses.
            
Your task is to evaluate if the assistant's response appropriately answers the user's question.
Consider:
- Correctness: Is the information accurate?
- Relevance: Does it address the question asked?
- Completeness: Does it cover the key points?
- Clarity: Is it well-explained?

Provide scores (0-10) for each criterion and an overall score.
Return ONLY valid JSON with this structure:
{
    "correctness": <0-10>,
    "relevance": <0-10>,
    "completeness": <0-10>,
    "clarity": <0-10>,
    "overall": <0-10>,
    "reasoning": "<brief explanation>"
}"""
        )
    
    async def evaluate_response(self, question: str, llm_response: str) -> JudgeScores:
        """Use LLM to judge response quality with retry logic"""
        prompt = f"""Question: {question}

Response: {llm_response}

Evaluate the actual response."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await self.judge_agent.invoke_async(prompt)
                response = result.output if hasattr(result, 'output') else str(result)
                
                # Remove markdown code blocks if present
                response = response.strip()
                if response.startswith('```'):
                    # Remove opening ```json or ```
                    response = response.split('\n', 1)[1] if '\n' in response else response[3:]
                    # Remove closing ```
                    if response.endswith('```'):
                        response = response.rsplit('```', 1)[0]
                
                # Parse JSON from response
                result_json = json.loads(response.strip())
                return JudgeScores.model_validate(result_json)
            except Exception as e:
                error_str = str(e).lower()
                is_bedrock_error = any([
                    "serviceunavailableexception" in error_str,
                    "bedrock is unable to process" in error_str,
                    "throttlingexception" in error_str,
                    "rate limit" in error_str,
                    "too many requests" in error_str,
                    "service temporarily unavailable" in error_str,
                    "eventstreamError" in error_str,
                    "conversestream operation" in error_str,
                    "botocore.exceptions.eventstreamerror" in error_str
                ])
                
                if attempt < max_retries - 1:
                    wait_time = (3 ** (attempt + 1)) + random.uniform(0, 1)
                    error_type = "Bedrock" if is_bedrock_error else "General"
                    logger.warning(f"{error_type} error in judge evaluation (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Judge evaluation failed after {max_retries} attempts: {e}")
                    return JudgeScores(reasoning=f"Evaluation failed: {str(e)}")
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        if not reference or not hypothesis:
            return 1.0
        return wer(reference, hypothesis) * 100
    
    async def evaluate_single(self, question_id: str, category: str, stt_output: str, ground_truth: str, llm_response: str, 
                             stt_latency: float = None, tts_latency: float = None, 
                             total_latency: float = None, tts_audio_path: str = None) -> EvaluationResult:

        """Evaluate a single question"""
        question = stt_output
        
        # Calculate WER for STT
        wer_score = self.calculate_wer(ground_truth, stt_output)
        
        # Evaluate response quality
        judge_scores = await self.evaluate_response(question, llm_response)
        
        # Create timing metrics
        timing = TimingMetrics(
            stt_latency_ms=stt_latency,
            tts_latency_ms=tts_latency,
            total_latency_ms=total_latency
        )
        
        result = EvaluationResult(
            question_id=question_id,
            category=category,
            ground_truth=ground_truth,
            stt_output=stt_output,
            wer=wer_score,
            llm_response=llm_response,
            judge_scores=judge_scores,
            timing=timing,
            tts_audio_path=tts_audio_path
        )
        
        # Add voice quality metrics if enabled and TTS audio available
        if self.evaluate_voice_quality and tts_audio_path:
            try:
                # Get LLM judge evaluation
                llm_voice_metrics = await self.voice_evaluator.evaluate_with_llm_judge(tts_audio_path, llm_response)
                
                # Get NISQA and speechmetrics evaluation
                technical_voice_metrics = self.voice_evaluator.evaluate(tts_audio_path)
                
                # Combine both sets of metrics
                voice_metrics = {**llm_voice_metrics, **technical_voice_metrics}
                result.voice_quality = VoiceQuality.model_validate(voice_metrics)
            except Exception as e:
                logger.error(f"Voice quality evaluation failed for {question_id}: {e}")
            
        return result
    
    async def run_evaluation(self, results_file: str) -> EvaluationReport:
        """
        Run evaluation on pre-recorded results.
        
        Args:
            results_file: Path to JSON file with STT outputs and bot responses
        """
        # Load results with error handling
        try:
            with open(results_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"Results file {results_file} is empty")
                results_data = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error reading results file {results_file}: {e}")
            print("File may be corrupted or still being written. Please try again.")
            return None
        
        # Extract model metadata if present
        if isinstance(results_data, dict) and 'results' in results_data:
            stt_model = results_data.get('stt_model')
            stt_service_id = results_data.get('stt_service_id')
            tts_model = results_data.get('tts_model')
            tts_service_id = results_data.get('tts_service_id')
            results = results_data['results']
        else:
            # Old format - just a list
            stt_model = None
            stt_service_id = None
            tts_model = None
            tts_service_id = None
            results = results_data
        
        evaluations = []
        
        for i, question in enumerate(self.dataset['questions']):
            question_id = question['id']
            
            # Find corresponding result
            result = next((r for r in results if r['question_id'] == question_id), None)
            if not result:
                logger.warning(f"No result found for {question_id}")
                continue
            
            logger.info(f"Evaluating {question_id}...")
            
            # Retry logic for Bedrock failures
            max_retries = 3
            eval_result = None
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {question_id}")
                    eval_result = await self.evaluate_single(
                        question_id,
                        question.get('category',''),
                        result.get('stt_output', ''),
                        result.get('ground_truth', ''),
                        result.get('llm_response', ''),
                        result.get('stt_latency_ms'),
                        result.get('tts_latency_ms'),
                        result.get('total_latency_ms'),
                        result.get('tts_audio_path')
                    )
                    evaluations.append(eval_result)
                    if attempt > 0:
                        logger.info(f"Retry successful for {question_id}")
                    break  # Success, exit retry loop
                        
                except Exception as e:
                    error_str = str(e).lower()
                    is_bedrock_error = any([
                        "serviceunavailableexception" in error_str,
                        "bedrock is unable to process" in error_str,
                        "throttlingexception" in error_str,
                        "rate limit" in error_str,
                        "too many requests" in error_str,
                        "service temporarily unavailable" in error_str,
                        "eventstreamError" in error_str,
                        "conversestream operation" in error_str
                    ])
                    
                    if is_bedrock_error and attempt < max_retries - 1:
                        wait_time = (3 ** (attempt + 1)) + random.uniform(0, 1)  # Add jitter
                        logger.warning(f"Bedrock error on {question_id} (attempt {attempt + 1}), retrying in {wait_time}s: {error_str}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Error evaluating {question_id} (final attempt): {e}")
                        # Add failed evaluation with error info
                        eval_result = EvaluationResult(
                            question_id=question_id,
                            category=question.get('category',''),
                            ground_truth=result.get('ground_truth', ''),
                            stt_output=result.get('stt_output', ''),
                            wer=100.0,  # Max error for failed evaluation
                            llm_response=result.get('llm_response', ''),
                            judge_scores=JudgeScores(reasoning=f"Evaluation failed: {str(e)}"),
                            error=str(e)
                        )
                        evaluations.append(eval_result)
                        break
            
            # Add pause between evaluations to avoid rate limiting
            if i < len(self.dataset['questions']) - 1:  # Don't pause after last item
                logger.info("Pausing 3 seconds to avoid rate limiting...")
                await asyncio.sleep(3)
  
        # Calculate summary statistics
        summary = self._calculate_summary(evaluations)
        
        # Create evaluation report
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            dataset=self.dataset_path,
            total_questions=len(evaluations),
            stt_model=stt_model,
            stt_service_id=stt_service_id,
            tts_model=tts_model,
            tts_service_id=tts_service_id,
            evaluations=evaluations,
            summary=summary
        )
        
        return report
    
    def _calculate_summary(self, evaluations: List[EvaluationResult]) -> EvaluationSummary:
        """Calculate summary statistics using Pydantic models"""
        if not evaluations:
            return EvaluationSummary(average_wer=0, average_overall_score=0)
        
        avg_wer = sum(e.wer for e in evaluations) / len(evaluations)
        avg_overall = sum(e.judge_scores.overall for e in evaluations) / len(evaluations)
        
        # Calculate average latencies if available - filter out None values
        stt_latencies = [e.timing.stt_latency_ms for e in evaluations if e.timing.stt_latency_ms is not None]
        tts_latencies = [e.timing.tts_latency_ms for e in evaluations if e.timing.tts_latency_ms is not None]
        total_latencies = [e.timing.total_latency_ms for e in evaluations if e.timing.total_latency_ms is not None]
        
        summary = EvaluationSummary(
            average_wer=avg_wer,
            average_overall_score=avg_overall
        )
        
        if stt_latencies:
            summary.average_stt_latency_ms = sum(stt_latencies) / len(stt_latencies)
        if tts_latencies:
            summary.average_tts_latency_ms = sum(tts_latencies) / len(tts_latencies)
        if total_latencies:
            summary.average_total_latency_ms = sum(total_latencies) / len(total_latencies)
        
        # Group by category
        by_category = {}
        for eval in evaluations:
            cat = eval.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(eval)
        
        category_stats = {}
        for cat, evals in by_category.items():
            cat_stt = [e.timing.stt_latency_ms for e in evals if e.timing.stt_latency_ms is not None]
            cat_tts = [e.timing.tts_latency_ms for e in evals if e.timing.tts_latency_ms is not None]
            
            stats = CategoryStats(
                count=len(evals),
                avg_wer=sum(e.wer for e in evals) / len(evals),
                avg_score=sum(e.judge_scores.overall for e in evals) / len(evals)
            )
            
            if cat_stt:
                stats.avg_stt_latency_ms = sum(cat_stt) / len(cat_stt)
            if cat_tts:
                stats.avg_tts_latency_ms = sum(cat_tts) / len(cat_tts)
                
            category_stats[cat] = stats
        
        summary.by_category = category_stats
        return summary
    
    def save_results(self, results: EvaluationReport, output_path: str):
        """Save evaluation results using Pydantic model"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results.model_dump(), f, indent=2, default=float)
        logger.info(f"Results saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Evaluate VoiceAssistant-Eval dataset')
    parser.add_argument('--dataset', default='evaluation_data/voiceassistant_eval/voiceassistant_eval_dataset.json',
                       help='Path to dataset JSON')
    parser.add_argument('--audio-dir', default='evaluation_data/voiceassistant_eval/audio_input',
                       help='Directory containing audio files')
    parser.add_argument('--results', required=True,
                       help='Path to results JSON file with STT outputs and bot responses')
    parser.add_argument('--output', default='evaluation_data/voiceassistant_eval/evaluation_results.json',
                       help='Output path for evaluation results')
    parser.add_argument('--voice-quality', action='store_true', default=True,
                       help='Enable voice quality evaluation')
    
    args = parser.parse_args()
    
    evaluator = VoiceAssistantEvaluator(args.dataset, args.audio_dir, args.voice_quality)
    
    logger.info("Starting evaluation...")
    results = await evaluator.run_evaluation(args.results)
    
    if results is None:
        print("Evaluation failed due to corrupted or empty results file.")
        print("Please re-run the voice pipeline evaluator to generate new results.")
        return
    
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total questions: {results.total_questions}")
    print(f"Average WER: {results.summary.average_wer:.2f}%")
    print(f"Average Overall Score: {results.summary.average_overall_score:.2f}/10")
    
    if results.summary.average_stt_latency_ms:
        print(f"Average STT Latency: {results.summary.average_stt_latency_ms:.2f}ms")
    if results.summary.average_tts_latency_ms:
        print(f"Average TTS Latency: {results.summary.average_tts_latency_ms:.2f}ms")
    if results.summary.average_total_latency_ms:
        print(f"Average Total Latency: {results.summary.average_total_latency_ms:.2f}ms")
    
    print("\nBy Category:")
    for cat, stats in results.summary.by_category.items():
        print(f"  {cat}: WER={stats.avg_wer:.2f}%, Score={stats.avg_score:.2f}/10", end="")
        if stats.avg_stt_latency_ms:
            print(f", STT={stats.avg_stt_latency_ms:.0f}ms", end="")
        if stats.avg_tts_latency_ms:
            print(f", TTS={stats.avg_tts_latency_ms:.0f}ms", end="")
        print()
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

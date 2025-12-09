"""
Evaluation script for VoiceAssistant-Eval dataset.
Measures STT accuracy (WER) and response quality using LLM-as-judge.
"""

import json
import os
import asyncio
from typing import Dict, List
from jiwer import wer
from datetime import datetime
from strands import Agent
from strands.models import BedrockModel
from loguru import logger
import argparse


class VoiceAssistantEvaluator:
    def __init__(self, dataset_path: str, audio_dir: str):
        self.dataset_path = dataset_path
        self.audio_dir = audio_dir
        self.dataset = self._load_dataset()
        self.judge_agent = self._create_judge_agent()
        
    def _load_dataset(self) -> Dict:
        """Load the evaluation dataset"""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def _create_judge_agent(self) -> Agent:
        """Create LLM judge agent"""
        model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
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
    
    async def evaluate_response(self, question: str, expected_answer: str, actual_response: str) -> Dict:
        """Use LLM to judge response quality"""
        prompt = f"""Question: {question}

Expected Answer: {expected_answer}

Actual Response: {actual_response}

Evaluate the actual response."""

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
            return result_json
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {
                "correctness": 0,
                "relevance": 0,
                "completeness": 0,
                "clarity": 0,
                "overall": 0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        if not reference or not hypothesis:
            return 1.0
        return wer(reference, hypothesis) * 100
    
    async def evaluate_single(self, question_data: Dict, stt_output: str, bot_response: str, 
                             stt_latency: float = None, tts_latency: float = None, 
                             total_latency: float = None) -> Dict:
        """Evaluate a single question"""
        question_id = question_data['id']
        expected_text = question_data['text']
        expected_answer = question_data['expected_answer']
        
        # Calculate WER for STT
        wer_score = self.calculate_wer(expected_text, stt_output)
        
        # Evaluate response quality
        judge_scores = await self.evaluate_response(expected_text, expected_answer, bot_response)
        
        result = {
            "question_id": question_id,
            "category": question_data['category'],
            "expected_text": expected_text,
            "stt_output": stt_output,
            "wer": wer_score,
            "expected_answer": expected_answer,
            "bot_response": bot_response,
            "judge_scores": judge_scores
        }
        
        # Add latency metrics if available
        if stt_latency is not None:
            result["stt_latency_ms"] = stt_latency
        if tts_latency is not None:
            result["tts_latency_ms"] = tts_latency
        if total_latency is not None:
            result["total_latency_ms"] = total_latency
            
        return result
    
    async def run_evaluation(self, results_file: str) -> Dict:
        """
        Run evaluation on pre-recorded results.
        
        Args:
            results_file: Path to JSON file with STT outputs and bot responses
        """
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        evaluations = []
        
        for question in self.dataset['questions']:
            question_id = question['id']
            
            # Find corresponding result
            result = next((r for r in results if r['question_id'] == question_id), None)
            if not result:
                logger.warning(f"No result found for {question_id}")
                continue
            
            logger.info(f"Evaluating {question_id}...")
            
            eval_result = await self.evaluate_single(
                question,
                result.get('stt_output', ''),
                result.get('bot_response', ''),
                result.get('stt_latency_ms'),
                result.get('tts_latency_ms'),
                result.get('total_latency_ms')
            )
            
            evaluations.append(eval_result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(evaluations)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_path,
            "total_questions": len(evaluations),
            "evaluations": evaluations,
            "summary": summary
        }
    
    def _calculate_summary(self, evaluations: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        if not evaluations:
            return {}
        
        avg_wer = sum(e['wer'] for e in evaluations) / len(evaluations)
        avg_overall = sum(e['judge_scores']['overall'] for e in evaluations) / len(evaluations)
        
        # Calculate average latencies if available
        stt_latencies = [e['stt_latency_ms'] for e in evaluations if 'stt_latency_ms' in e]
        tts_latencies = [e['tts_latency_ms'] for e in evaluations if 'tts_latency_ms' in e]
        total_latencies = [e['total_latency_ms'] for e in evaluations if 'total_latency_ms' in e]
        
        summary = {
            "average_wer": avg_wer,
            "average_overall_score": avg_overall
        }
        
        if stt_latencies:
            summary["average_stt_latency_ms"] = sum(stt_latencies) / len(stt_latencies)
        if tts_latencies:
            summary["average_tts_latency_ms"] = sum(tts_latencies) / len(tts_latencies)
        if total_latencies:
            summary["average_total_latency_ms"] = sum(total_latencies) / len(total_latencies)
        
        # Group by category
        by_category = {}
        for eval in evaluations:
            cat = eval['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(eval)
        
        category_stats = {}
        for cat, evals in by_category.items():
            cat_stt = [e['stt_latency_ms'] for e in evals if 'stt_latency_ms' in e]
            cat_tts = [e['tts_latency_ms'] for e in evals if 'tts_latency_ms' in e]
            
            stats = {
                "count": len(evals),
                "avg_wer": sum(e['wer'] for e in evals) / len(evals),
                "avg_score": sum(e['judge_scores']['overall'] for e in evals) / len(evals)
            }
            
            if cat_stt:
                stats["avg_stt_latency_ms"] = sum(cat_stt) / len(cat_stt)
            if cat_tts:
                stats["avg_tts_latency_ms"] = sum(cat_tts) / len(cat_tts)
                
            category_stats[cat] = stats
        
        summary["by_category"] = category_stats
        return summary
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
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
    
    args = parser.parse_args()
    
    evaluator = VoiceAssistantEvaluator(args.dataset, args.audio_dir)
    
    logger.info("Starting evaluation...")
    results = await evaluator.run_evaluation(args.results)
    
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total questions: {results['total_questions']}")
    print(f"Average WER: {results['summary']['average_wer']:.2f}%")
    print(f"Average Overall Score: {results['summary']['average_overall_score']:.2f}/10")
    
    if 'average_stt_latency_ms' in results['summary']:
        print(f"Average STT Latency: {results['summary']['average_stt_latency_ms']:.2f}ms")
    if 'average_tts_latency_ms' in results['summary']:
        print(f"Average TTS Latency: {results['summary']['average_tts_latency_ms']:.2f}ms")
    if 'average_total_latency_ms' in results['summary']:
        print(f"Average Total Latency: {results['summary']['average_total_latency_ms']:.2f}ms")
    
    print("\nBy Category:")
    for cat, stats in results['summary']['by_category'].items():
        print(f"  {cat}: WER={stats['avg_wer']:.2f}%, Score={stats['avg_score']:.2f}/10", end="")
        if 'avg_stt_latency_ms' in stats:
            print(f", STT={stats['avg_stt_latency_ms']:.0f}ms", end="")
        if 'avg_tts_latency_ms' in stats:
            print(f", TTS={stats['avg_tts_latency_ms']:.0f}ms", end="")
        print()
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

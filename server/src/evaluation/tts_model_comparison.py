"""
Compare TTS models using voice quality metrics.
Evaluates fluency, naturalness, tone, and latency for each TTS service.
"""

import json
import os
import asyncio
from typing import Dict, List
from datetime import datetime
from loguru import logger
import argparse
from voice_quality_evaluator import VoiceQualityEvaluator


# TTS services to evaluate
TTS_SERVICES = {
    "deepgram_aura": "Deepgram Aura",
    "openai_tts": "OpenAI TTS",
    "openai_tts_hd": "OpenAI TTS HD",
    "elevenlabs": "ElevenLabs",
    "cartesia": "Cartesia",
    "playht": "PlayHT",
    "aws_polly": "AWS Polly",
    "azure_tts": "Azure TTS",
    "google_tts": "Google TTS",
    "lmnt": "LMNT",
    "rime": "Rime",
    "fish_audio": "Fish Audio",
    "groq_tts": "Groq TTS",
    "speechmatics": "Speechmatics",
    "nvidia_riva": "NVIDIA Riva"
}


class TTSModelComparator:
    
    def __init__(self, audio_output_dir: str, use_llm_judge: bool = False):
        self.audio_output_dir = audio_output_dir
        self.evaluator = VoiceQualityEvaluator(use_llm_judge=use_llm_judge)
        self.use_llm_judge = use_llm_judge
    
    async def evaluate_tts_model(self, model_name: str, audio_files: List[str], 
                                 transcripts: Dict[str, str] = None) -> Dict:
        """
        Evaluate a single TTS model across multiple audio samples.
        
        Args:
            model_name: Name of TTS model
            audio_files: List of audio file paths
            transcripts: Optional dict mapping filename to transcript
            
        Returns:
            Dict with aggregated metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        evaluations = []
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                logger.warning(f"Audio file not found: {audio_file}")
                continue
            
            try:
                # Get transcript if available
                transcript = ""
                if transcripts:
                    filename = os.path.basename(audio_file)
                    transcript = transcripts.get(filename, "")
                
                # Evaluate
                if self.use_llm_judge:
                    metrics = await self.evaluator.evaluate_async(audio_file, transcript)
                else:
                    metrics = self.evaluator.evaluate(audio_file)
                
                metrics["audio_file"] = os.path.basename(audio_file)
                evaluations.append(metrics)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {audio_file}: {e}")
        
        # Aggregate metrics
        if not evaluations:
            return {"model": model_name, "error": "No valid evaluations"}
        
        aggregated = self._aggregate_metrics(model_name, evaluations)
        return aggregated
    
    def _aggregate_metrics(self, model_name: str, evaluations: List[Dict]) -> Dict:
        """Aggregate metrics across all samples"""
        
        result = {
            "model": model_name,
            "sample_count": len(evaluations),
            "fluency": {
                "avg_score": self._avg([e["fluency"]["score"] for e in evaluations]),
                "avg_speaking_rate": self._avg([e["fluency"]["speaking_rate_sps"] for e in evaluations]),
                "avg_silence_ratio": self._avg([e["fluency"]["silence_ratio"] for e in evaluations]),
                "avg_energy_consistency": self._avg([e["fluency"]["energy_consistency"] for e in evaluations])
            },
            "naturalness": {
                "avg_score": self._avg([e["naturalness"]["score"] for e in evaluations]),
                "avg_pitch_mean": self._avg([e["naturalness"]["pitch_mean_hz"] for e in evaluations]),
                "avg_pitch_variation": self._avg([e["naturalness"]["pitch_variation"] for e in evaluations]),
                "avg_spectral_centroid": self._avg([e["naturalness"]["spectral_centroid_hz"] for e in evaluations])
            },
            "tone": {
                "avg_score": self._avg([e["tone"]["score"] for e in evaluations]),
                "avg_clarity": self._avg([e["tone"]["clarity"] for e in evaluations]),
                "avg_smoothness": self._avg([e["tone"]["smoothness"] for e in evaluations]),
                "avg_hnr": self._avg([e["tone"]["hnr_db"] for e in evaluations])
            },
            "overall_quality": self._avg([e["overall_quality"] for e in evaluations])
        }
        
        # Add LLM scores if available
        if self.use_llm_judge and "llm_overall" in evaluations[0]:
            result["llm_scores"] = {
                "avg_fluency": self._avg([e.get("llm_fluency", 0) for e in evaluations]),
                "avg_naturalness": self._avg([e.get("llm_naturalness", 0) for e in evaluations]),
                "avg_tone": self._avg([e.get("llm_tone", 0) for e in evaluations]),
                "avg_overall": self._avg([e.get("llm_overall", 0) for e in evaluations])
            }
        
        result["detailed_evaluations"] = evaluations
        
        return result
    
    def _avg(self, values: List[float]) -> float:
        """Calculate average, handling None values"""
        valid = [v for v in values if v is not None and v > 0]
        return round(sum(valid) / len(valid), 2) if valid else 0.0
    
    async def compare_all_models(self, audio_dir_pattern: str = None) -> Dict:
        """
        Compare all TTS models.
        
        Args:
            audio_dir_pattern: Pattern like "evaluation_data/tts_output/{model}/"
            
        Returns:
            Dict with comparison results
        """
        results = []
        
        for model_id, model_name in TTS_SERVICES.items():
            # Find audio files for this model
            if audio_dir_pattern:
                model_audio_dir = audio_dir_pattern.format(model=model_id)
            else:
                model_audio_dir = os.path.join(self.audio_output_dir, model_id)
            
            if not os.path.exists(model_audio_dir):
                logger.warning(f"Audio directory not found for {model_name}: {model_audio_dir}")
                continue
            
            # Get all audio files
            audio_files = [
                os.path.join(model_audio_dir, f) 
                for f in os.listdir(model_audio_dir) 
                if f.endswith('.wav')
            ]
            
            if not audio_files:
                logger.warning(f"No audio files found for {model_name}")
                continue
            
            # Evaluate model
            model_results = await self.evaluate_tts_model(model_name, audio_files)
            results.append(model_results)
        
        # Create comparison summary
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(results),
            "models": results,
            "ranking": self._rank_models(results)
        }
        
        return comparison
    
    def _rank_models(self, results: List[Dict]) -> Dict:
        """Rank models by different metrics"""
        
        rankings = {
            "by_overall_quality": sorted(
                results, 
                key=lambda x: x.get("overall_quality", 0), 
                reverse=True
            ),
            "by_fluency": sorted(
                results,
                key=lambda x: x["fluency"]["avg_score"],
                reverse=True
            ),
            "by_naturalness": sorted(
                results,
                key=lambda x: x["naturalness"]["avg_score"],
                reverse=True
            ),
            "by_tone": sorted(
                results,
                key=lambda x: x["tone"]["avg_score"],
                reverse=True
            )
        }
        
        # Simplify to just model names and scores
        simplified = {}
        for metric, ranked in rankings.items():
            simplified[metric] = [
                {"model": r["model"], "score": r.get("overall_quality", r["fluency"]["avg_score"])}
                for r in ranked[:5]  # Top 5
            ]
        
        return simplified
    
    def save_results(self, results: Dict, output_path: str):
        """Save comparison results to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Compare TTS models with voice quality metrics')
    parser.add_argument('--audio-dir', default='evaluation_data/tts_output',
                       help='Base directory containing TTS model outputs')
    parser.add_argument('--output', default='evaluation_data/tts_comparison_results.json',
                       help='Output path for comparison results')
    parser.add_argument('--llm', action='store_true',
                       help='Use LLM judge for additional evaluation')
    
    args = parser.parse_args()
    
    comparator = TTSModelComparator(args.audio_dir, use_llm_judge=args.llm)
    
    logger.info("Starting TTS model comparison...")
    results = await comparator.compare_all_models()
    
    comparator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("TTS MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"Total models evaluated: {results['total_models']}")
    print("\nTop 5 by Overall Quality:")
    for i, model in enumerate(results['ranking']['by_overall_quality'][:5], 1):
        print(f"  {i}. {model['model']}: {model['score']:.2f}/10")
    print("\nTop 5 by Fluency:")
    for i, model in enumerate(results['ranking']['by_fluency'][:5], 1):
        print(f"  {i}. {model['model']}: {model['score']:.2f}/10")
    print("\nTop 5 by Naturalness:")
    for i, model in enumerate(results['ranking']['by_naturalness'][:5], 1):
        print(f"  {i}. {model['model']}: {model['score']:.2f}/10")
    print("\nTop 5 by Tone:")
    for i, model in enumerate(results['ranking']['by_tone'][:5], 1):
        print(f"  {i}. {model['model']}: {model['score']:.2f}/10")
    print("="*80)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

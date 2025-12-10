#!/usr/bin/env python3
"""
Visualization script for STT/TTS evaluation results.
Plots STT latency vs WER and TTS latency vs quality scores.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['grid.alpha'] = 0.3

def load_evaluation_results(results_dir: str) -> List[Dict]:
    """Load all evaluation result files."""
    results = []
    pattern = os.path.join(results_dir, "*_evaluation.json")
    
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                filename = os.path.basename(filepath)
                
                # Use service IDs from the data if available, otherwise parse filename
                if 'stt_service_id' in data and 'tts_service_id' in data:
                    data['stt_model'] = data['stt_service_id']
                    data['tts_model'] = data['tts_service_id']
                else:
                    # Fallback: parse from filename
                    parts = filename.replace('_evaluation.json', '').split('_')
                    stt_parts = []
                    tts_parts = []
                    found_tts = False
                    
                    for i, part in enumerate(parts):
                        if part in ['deepgram', 'openai', 'groq', 'elevenlabs', 'speechmatics', 
                                    'assemblyai', 'gladia', 'cartesia', 'fish', 'lmnt', 'playht', 
                                    'rime', 'nvidia', 'riva', 'aura', 'tts', 'polly', 'audio', 'aws', 'transcribe']:
                            if not found_tts and len(stt_parts) > 0:
                                found_tts = True
                            if found_tts:
                                tts_parts.append(part)
                            else:
                                stt_parts.append(part)
                        else:
                            if found_tts:
                                tts_parts.append(part)
                            else:
                                stt_parts.append(part)
                    
                    data['stt_model'] = '_'.join(stt_parts) if stt_parts else 'unknown'
                    data['tts_model'] = '_'.join(tts_parts) if tts_parts else 'unknown'
                
                data['filename'] = filename
                results.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return results

def extract_metrics(results: List[Dict]) -> Tuple[Dict, Dict]:
    """Extract STT and TTS metrics from results."""
    stt_metrics = {}
    tts_metrics = {}
    
    for result in results:
        stt_model = result['stt_model']
        tts_model = result['tts_model']
        summary = result.get('summary', {})
        
        # STT metrics
        avg_wer = summary.get('average_wer', None)
        avg_latency = summary.get('average_total_latency_ms', None)
        
        # Calculate average STT latency from individual evaluations
        stt_latencies = []
        tts_latencies = []
        scores = []
        
        for eval_item in result.get('evaluations', []):
            if eval_item.get('stt_latency_ms') is not None:
                stt_latencies.append(eval_item['stt_latency_ms'])
            if eval_item.get('tts_latency_ms') is not None:
                tts_latencies.append(eval_item['tts_latency_ms'])
            judge_scores = eval_item.get('judge_scores', {})
            if judge_scores.get('overall') is not None:
                scores.append(judge_scores['overall'])
        
        avg_stt_latency = np.mean(stt_latencies) if stt_latencies else None
        avg_tts_latency = np.mean(tts_latencies) if tts_latencies else None
        avg_score = np.mean(scores) if scores else summary.get('average_overall_score', None)
        
        # Store STT metrics
        if stt_model not in stt_metrics:
            stt_metrics[stt_model] = {'wer': [], 'latency': []}
        if avg_wer is not None and avg_stt_latency is not None:
            stt_metrics[stt_model]['wer'].append(avg_wer)
            stt_metrics[stt_model]['latency'].append(avg_stt_latency)
        
        # Store TTS metrics
        if tts_model not in tts_metrics:
            tts_metrics[tts_model] = {'score': [], 'latency': []}
        if avg_score is not None and avg_tts_latency is not None:
            tts_metrics[tts_model]['score'].append(avg_score)
            tts_metrics[tts_model]['latency'].append(avg_tts_latency)
    
    # Average across multiple runs
    for model in stt_metrics:
        stt_metrics[model]['wer'] = np.mean(stt_metrics[model]['wer'])
        stt_metrics[model]['latency'] = np.mean(stt_metrics[model]['latency'])
    
    for model in tts_metrics:
        tts_metrics[model]['score'] = np.mean(tts_metrics[model]['score'])
        tts_metrics[model]['latency'] = np.mean(tts_metrics[model]['latency'])
    
    return stt_metrics, tts_metrics

def plot_stt_metrics(stt_metrics: Dict, output_path: str):
    """Plot STT latency vs WER."""
    # Filter for AWS and Deepgram models
    filtered_metrics = {k: v for k, v in stt_metrics.items() 
                       if 'aws' in k.lower() or 'deepgram' in k.lower()}
    
    if not filtered_metrics:
        print("No AWS or Deepgram STT metrics found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separate AWS and Deepgram
    aws_models = {k: v for k, v in filtered_metrics.items() if 'aws' in k.lower()}
    deepgram_models = {k: v for k, v in filtered_metrics.items() if 'deepgram' in k.lower()}
    
    # Plot AWS models
    if aws_models:
        latencies = [v['latency'] for v in aws_models.values()]
        wers = [v['wer'] for v in aws_models.values()]
        labels = [k.replace('_', ' ').title() for k in aws_models.keys()]
        ax.scatter(latencies, wers, s=250, alpha=0.8, marker='o', 
                  label='AWS Transcribe', color='#FF9900', edgecolors='black', linewidth=2)
        for i, label in enumerate(labels):
            ax.annotate(label, (latencies[i], wers[i]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=13, fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Plot Deepgram models
    if deepgram_models:
        latencies = [v['latency'] for v in deepgram_models.values()]
        wers = [v['wer'] for v in deepgram_models.values()]
        labels = [k.replace('_', ' ').title() for k in deepgram_models.keys()]
        ax.scatter(latencies, wers, s=250, alpha=0.8, marker='s', 
                  label='Deepgram', color='#13EF93', edgecolors='black', linewidth=2)
        for i, label in enumerate(labels):
            ax.annotate(label, (latencies[i], wers[i]), 
                       xytext=(10, -15), textcoords='offset points',
                       fontsize=13, fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel('Average STT Latency (ms)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Average Word Error Rate (WER)', fontweight='bold', fontsize=16)
    ax.set_title('STT Performance: Latency vs Accuracy', fontweight='bold', fontsize=18, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.legend(frameon=True, shadow=True, loc='best', fontsize=14, markerscale=1.2)
    
    # Add optimal region annotation
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.4, linewidth=2)
    ax.text(ax.get_xlim()[1] * 0.95, 0.1, 'Target WER', 
           verticalalignment='bottom', horizontalalignment='right',
           fontsize=12, alpha=0.7, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved STT plot to {output_path}")
    plt.close()

def plot_tts_metrics(tts_metrics: Dict, output_path: str):
    """Plot TTS latency vs quality score."""
    if not tts_metrics:
        print("No TTS metrics found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for different TTS providers
    colors = plt.cm.tab20(np.linspace(0, 1, len(tts_metrics)))
    
    for i, (model, metrics) in enumerate(sorted(tts_metrics.items())):
        latency = metrics['latency']
        score = metrics['score']
        label = model.replace('_', ' ').title()
        
        ax.scatter(latency, score, s=250, alpha=0.8, 
                  color=colors[i], edgecolors='black', linewidth=2,
                  label=label)
        ax.annotate(label, (latency, score), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=13, fontweight='bold', alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel('Average TTS Latency (ms)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Average Quality Score (0-10)', fontweight='bold', fontsize=16)
    ax.set_title('TTS Performance: Latency vs Quality', fontweight='bold', fontsize=18, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_ylim(bottom=0, top=10.5)
    
    # Add quality threshold
    ax.axhline(y=7.0, color='green', linestyle='--', alpha=0.4, linewidth=2)
    ax.text(ax.get_xlim()[1] * 0.95, 7.0, 'Good Quality', 
           verticalalignment='bottom', horizontalalignment='right',
           fontsize=12, alpha=0.7, style='italic', fontweight='bold')
    
    # Legend outside plot if too many models
    if len(tts_metrics) > 8:
        ax.legend(frameon=True, shadow=True, loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=13, markerscale=1.2)
    else:
        ax.legend(frameon=True, shadow=True, loc='best', fontsize=14, markerscale=1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved TTS plot to {output_path}")
    plt.close()

def main():
    # Paths
    script_dir = Path(__file__).parent
    server_dir = script_dir.parent
    results_dir = server_dir / "evaluation_output" / "small"
    output_dir = script_dir
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print(f"Loading evaluation results from {results_dir}...")
    results = load_evaluation_results(str(results_dir))
    print(f"Loaded {len(results)} evaluation results")
    
    if not results:
        print("No evaluation results found!")
        return
    
    # Extract metrics
    print("Extracting metrics...")
    stt_metrics, tts_metrics = extract_metrics(results)
    
    print(f"Found {len(stt_metrics)} STT models and {len(tts_metrics)} TTS models")
    
    # Generate plots
    print("Generating plots...")
    plot_stt_metrics(stt_metrics, str(output_dir / "stt_latency_vs_wer.png"))
    plot_tts_metrics(tts_metrics, str(output_dir / "tts_latency_vs_quality.png"))
    
    print("\nVisualization complete!")
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Scientific visualization of STT/TTS evaluation results.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Professional scientific style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

def load_evaluation_results(results_dir: str) -> List[Dict]:
    """Load all evaluation result files."""
    results = []
    pattern = os.path.join(results_dir, "*_evaluation.json")
    
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                filename = os.path.basename(filepath)
                
                if 'stt_service_id' in data and 'tts_service_id' in data:
                    data['stt_model'] = data['stt_service_id']
                    data['tts_model'] = data['tts_service_id']
                else:
                    parts = filename.replace('_evaluation.json', '').split('_')
                    stt_parts = []
                    tts_parts = []
                    found_tts = False
                    
                    for part in parts:
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
    """Extract STT and TTS metrics with statistics."""
    stt_metrics = {}
    tts_metrics = {}
    
    for result in results:
        stt_model = result['stt_model']
        tts_model = result['tts_model']
        
        stt_latencies = []
        tts_latencies = []
        scores = []
        wers = []
        
        for eval_item in result.get('evaluations', []):
            if eval_item.get('stt_latency_ms') is not None:
                stt_latencies.append(eval_item['stt_latency_ms'])
            if eval_item.get('tts_latency_ms') is not None:
                tts_latencies.append(eval_item['tts_latency_ms'])
            if eval_item.get('wer') is not None:
                wers.append(eval_item['wer'])
            judge_scores = eval_item.get('judge_scores', {})
            if judge_scores.get('overall') is not None:
                scores.append(judge_scores['overall'])
        
        # STT metrics
        if stt_model not in stt_metrics:
            stt_metrics[stt_model] = {
                'latency': [], 'wer': [],
                'latency_std': [], 'wer_std': []
            }
        
        if stt_latencies and wers:
            stt_metrics[stt_model]['latency'].append(np.mean(stt_latencies))
            stt_metrics[stt_model]['wer'].append(np.mean(wers))
            stt_metrics[stt_model]['latency_std'].append(np.std(stt_latencies))
            stt_metrics[stt_model]['wer_std'].append(np.std(wers))
        
        # TTS metrics
        if tts_model not in tts_metrics:
            tts_metrics[tts_model] = {
                'latency': [], 'score': [],
                'latency_std': [], 'score_std': []
            }
        
        if tts_latencies and scores:
            tts_metrics[tts_model]['latency'].append(np.mean(tts_latencies))
            tts_metrics[tts_model]['score'].append(np.mean(scores))
            tts_metrics[tts_model]['latency_std'].append(np.std(tts_latencies))
            tts_metrics[tts_model]['score_std'].append(np.std(scores))
    
    # Aggregate
    for model in stt_metrics:
        stt_metrics[model] = {
            'latency': np.mean(stt_metrics[model]['latency']),
            'wer': np.mean(stt_metrics[model]['wer']),
            'latency_std': np.mean(stt_metrics[model]['latency_std']),
            'wer_std': np.mean(stt_metrics[model]['wer_std'])
        }
    
    for model in tts_metrics:
        tts_metrics[model] = {
            'latency': np.mean(tts_metrics[model]['latency']),
            'score': np.mean(tts_metrics[model]['score']),
            'latency_std': np.mean(tts_metrics[model]['latency_std']),
            'score_std': np.mean(tts_metrics[model]['score_std'])
        }
    
    return stt_metrics, tts_metrics

def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """Draw confidence ellipse."""
    cov = np.array([[x**2, 0], [0, y**2]])
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(0, 0)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_stt_metrics(stt_metrics: Dict, output_path: str):
    """Scientific plot of STT performance."""
    filtered = {k: v for k, v in stt_metrics.items() 
                if 'aws' in k.lower() or 'deepgram' in k.lower()}
    
    if not filtered:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define colors and markers
    provider_styles = {
        'aws': {'color': '#232F3E', 'marker': 'o', 'label': 'AWS Transcribe'},
        'deepgram': {'color': '#00A67E', 'marker': 's', 'label': 'Deepgram'}
    }
    
    plotted_providers = set()
    
    for model, metrics in filtered.items():
        provider = 'aws' if 'aws' in model.lower() else 'deepgram'
        style = provider_styles[provider]
        
        x = metrics['latency']
        y = metrics['wer']
        x_std = metrics['latency_std']
        y_std = metrics['wer_std']
        
        # Error ellipse
        ellipse = Ellipse((x, y), width=x_std*2, height=y_std*2,
                         facecolor=style['color'], alpha=0.15, 
                         edgecolor=style['color'], linewidth=1, linestyle='--')
        ax.add_patch(ellipse)
        
        # Data point
        label = style['label'] if provider not in plotted_providers else None
        ax.scatter(x, y, s=120, marker=style['marker'], 
                  color=style['color'], edgecolors='white', linewidth=1.5,
                  label=label, zorder=3, alpha=0.9)
        
        plotted_providers.add(provider)
        
        # Model label - use service ID directly
        ax.annotate(model, (x, y), xytext=(8, 8), 
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8, linewidth=0.5))
    
    ax.set_xlabel('Latency (ms)', fontweight='normal')
    ax.set_ylabel('Word Error Rate', fontweight='normal')
    ax.set_title('Speech-to-Text Performance Analysis', fontweight='bold', pad=15)
    
    # Add note about error ellipses
    ax.text(0.02, 0.98, 'Ellipses show ±1 standard deviation', 
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='gray', alpha=0.9, linewidth=0.5))
    
    ax.legend(loc='best', frameon=True, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_tts_metrics(tts_metrics: Dict, output_path: str):
    """Scientific plot of TTS performance."""
    if not tts_metrics:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(tts_metrics)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (model, metrics) in enumerate(sorted(tts_metrics.items())):
        x = metrics['latency']
        y = metrics['score']
        x_std = metrics['latency_std']
        y_std = metrics['score_std']
        
        # Error ellipse
        ellipse = Ellipse((x, y), width=x_std*2, height=y_std*2,
                         facecolor=colors[i], alpha=0.15,
                         edgecolor=colors[i], linewidth=1, linestyle='--')
        ax.add_patch(ellipse)
        
        # Data point
        marker = markers[i % len(markers)]
        ax.scatter(x, y, s=120, marker=marker, color=colors[i],
                  edgecolors='white', linewidth=1.5, label=model,
                  zorder=3, alpha=0.9)
        
        # Model label - use service ID directly
        ax.annotate(model, (x, y), xytext=(8, 8),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8, linewidth=0.5))
    
    # Add note about error ellipses
    ax.text(0.02, 0.98, 'Ellipses show ±1 standard deviation', 
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='gray', alpha=0.9, linewidth=0.5))
    
    ax.set_xlabel('Latency (ms)', fontweight='normal')
    ax.set_ylabel('Quality Score (0-10)', fontweight='normal')
    ax.set_title('Text-to-Speech Performance Analysis', fontweight='bold', pad=15)
    ax.set_ylim(0, 10.5)
    ax.legend(loc='best', frameon=True, edgecolor='gray', ncol=1)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    script_dir = Path(__file__).parent
    server_dir = script_dir.parent
    results_dir = server_dir / "evaluation_output" / "small"
    output_dir = script_dir
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from {results_dir}...")
    results = load_evaluation_results(str(results_dir))
    print(f"Loaded {len(results)} evaluation results")
    
    if not results:
        print("No results found")
        return
    
    print("Extracting metrics...")
    stt_metrics, tts_metrics = extract_metrics(results)
    print(f"STT models: {len(stt_metrics)}, TTS models: {len(tts_metrics)}")
    
    print("Generating plots...")
    plot_stt_metrics(stt_metrics, str(output_dir / "stt_latency_vs_wer.png"))
    plot_tts_metrics(tts_metrics, str(output_dir / "tts_latency_vs_quality.png"))
    
    print(f"\nComplete. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()

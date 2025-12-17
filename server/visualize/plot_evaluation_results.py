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
                                    'rime', 'nvidia', 'riva', 'aura', 'tts', 'polly', 'audio', 'aws', 'transcribe', 'magpie']:
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

def extract_metrics(results: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    """Extract STT and TTS metrics with statistics."""
    stt_metrics = {}
    tts_metrics = {}
    quality_metrics = {}
    
    for result in results:
        stt_model = result['stt_model']
        tts_model = result['tts_model']
        
        stt_latencies = []
        tts_latencies = []
        scores = []
        wers = []
        fluency = []
        tone = []
        naturalness = []
        llm_fluency = []
        llm_tone = []
        llm_naturalness = []
        
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
            
            voice_quality = eval_item.get('voice_quality', {})
            if voice_quality.get('fluency', {}).get('score') is not None:
                fluency.append(voice_quality['fluency']['score'])
            if voice_quality.get('tone', {}).get('score') is not None:
                tone.append(voice_quality['tone']['score'])
            if voice_quality.get('naturalness', {}).get('score') is not None:
                naturalness.append(voice_quality['naturalness']['score'])
            if voice_quality.get('llm_fluency') is not None:
                llm_fluency.append(voice_quality['llm_fluency'])
            if voice_quality.get('llm_tone') is not None:
                llm_tone.append(voice_quality['llm_tone'])
            if voice_quality.get('llm_naturalness') is not None:
                llm_naturalness.append(voice_quality['llm_naturalness'])
        
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
        
        # Quality metrics
        if tts_model not in quality_metrics:
            quality_metrics[tts_model] = {
                'fluency': [], 'tone': [], 'naturalness': [],
                'llm_fluency': [], 'llm_tone': [], 'llm_naturalness': [],
                'fluency_std': [], 'tone_std': [], 'naturalness_std': [],
                'llm_fluency_std': [], 'llm_tone_std': [], 'llm_naturalness_std': []
            }
        
        if fluency:
            quality_metrics[tts_model]['fluency'].append(np.mean(fluency))
            quality_metrics[tts_model]['fluency_std'].append(np.std(fluency))
        if tone:
            quality_metrics[tts_model]['tone'].append(np.mean(tone))
            quality_metrics[tts_model]['tone_std'].append(np.std(tone))
        if naturalness:
            quality_metrics[tts_model]['naturalness'].append(np.mean(naturalness))
            quality_metrics[tts_model]['naturalness_std'].append(np.std(naturalness))
        if llm_fluency:
            quality_metrics[tts_model]['llm_fluency'].append(np.mean(llm_fluency))
            quality_metrics[tts_model]['llm_fluency_std'].append(np.std(llm_fluency))
        if llm_tone:
            quality_metrics[tts_model]['llm_tone'].append(np.mean(llm_tone))
            quality_metrics[tts_model]['llm_tone_std'].append(np.std(llm_tone))
        if llm_naturalness:
            quality_metrics[tts_model]['llm_naturalness'].append(np.mean(llm_naturalness))
            quality_metrics[tts_model]['llm_naturalness_std'].append(np.std(llm_naturalness))
    
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
    
    for model in quality_metrics:
        quality_metrics[model] = {
            'fluency': np.mean(quality_metrics[model]['fluency']) if quality_metrics[model]['fluency'] else None,
            'tone': np.mean(quality_metrics[model]['tone']) if quality_metrics[model]['tone'] else None,
            'naturalness': np.mean(quality_metrics[model]['naturalness']) if quality_metrics[model]['naturalness'] else None,
            'llm_fluency': np.mean(quality_metrics[model]['llm_fluency']) if quality_metrics[model]['llm_fluency'] else None,
            'llm_tone': np.mean(quality_metrics[model]['llm_tone']) if quality_metrics[model]['llm_tone'] else None,
            'llm_naturalness': np.mean(quality_metrics[model]['llm_naturalness']) if quality_metrics[model]['llm_naturalness'] else None,
            'fluency_std': np.mean(quality_metrics[model]['fluency_std']) if quality_metrics[model]['fluency_std'] else 0,
            'tone_std': np.mean(quality_metrics[model]['tone_std']) if quality_metrics[model]['tone_std'] else 0,
            'naturalness_std': np.mean(quality_metrics[model]['naturalness_std']) if quality_metrics[model]['naturalness_std'] else 0,
            'llm_fluency_std': np.mean(quality_metrics[model]['llm_fluency_std']) if quality_metrics[model]['llm_fluency_std'] else 0,
            'llm_tone_std': np.mean(quality_metrics[model]['llm_tone_std']) if quality_metrics[model]['llm_tone_std'] else 0,
            'llm_naturalness_std': np.mean(quality_metrics[model]['llm_naturalness_std']) if quality_metrics[model]['llm_naturalness_std'] else 0
        }
    
    return stt_metrics, tts_metrics, quality_metrics

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
        
        # Smaller error ellipse
        ellipse = Ellipse((x, y), width=x_std*1.2, height=y_std*1.2,
                         facecolor=style['color'], alpha=0.1, 
                         edgecolor=style['color'], linewidth=1.5, linestyle='--')
        ax.add_patch(ellipse)
        
        # Data point
        label = style['label'] if provider not in plotted_providers else None
        ax.scatter(x, y, s=150, marker=style['marker'], 
                  color=style['color'], edgecolors='white', linewidth=2,
                  label=label, zorder=3, alpha=0.9)
        
        plotted_providers.add(provider)
        
        # Model label
        ax.annotate(model, (x, y), xytext=(10, 10), 
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='gray', alpha=0.95, linewidth=0.8))
    
    ax.set_xlabel('Latency (ms)', fontweight='normal', fontsize=13)
    ax.set_ylabel('Word Error Rate', fontweight='normal', fontsize=13)
    ax.set_title('Speech-to-Text Performance Analysis', fontweight='bold', pad=15, fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    
    # Add note about error ellipses
    ax.text(0.02, 0.98, 'Ellipses show ±1 standard deviation', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='gray', alpha=0.95, linewidth=0.8))
    
    ax.legend(loc='best', frameon=True, edgecolor='gray', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_tts_metrics(tts_metrics: Dict, output_path: str):
    """Scientific plot of TTS performance."""
    if not tts_metrics:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Provider-specific colors and markers
    provider_styles = {
        'nvidia': {'color': '#76b900', 'marker': '^', 'label': 'NVIDIA Riva'},
        'riva': {'color': '#76b900', 'marker': '^', 'label': 'NVIDIA Riva'},
        'deepgram': {'color': '#00A67E', 'marker': 's', 'label': 'Deepgram'},
        'aws': {'color': '#232F3E', 'marker': 'o', 'label': 'AWS Polly'},
        'elevenlabs': {'color': '#9467bd', 'marker': 'D', 'label': 'ElevenLabs'},
        'cartesia': {'color': '#ff7f0e', 'marker': 'v', 'label': 'Cartesia'},
        'openai': {'color': '#2ca02c', 'marker': 'p', 'label': 'OpenAI'}
    }
    
    # Fallback colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, len(tts_metrics)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    plotted_providers = set()
    
    for i, (model, metrics) in enumerate(sorted(tts_metrics.items())):
        x = metrics['latency']
        y = metrics['score']
        x_std = metrics['latency_std']
        y_std = metrics['score_std']
        
        # Determine provider and style
        provider = None
        for p in provider_styles.keys():
            if p in model.lower():
                provider = p
                break
        
        if provider and provider in provider_styles:
            style = provider_styles[provider]
            color = style['color']
            marker = style['marker']
            provider_label = style['label'] if provider not in plotted_providers else None
            plotted_providers.add(provider)
        else:
            color = colors[i]
            marker = markers[i % len(markers)]
            provider_label = model
        
        # Smaller error ellipse
        ellipse = Ellipse((x, y), width=x_std*1.2, height=y_std*1.2,
                         facecolor=color, alpha=0.1,
                         edgecolor=color, linewidth=1.5, linestyle='--')
        ax.add_patch(ellipse)
        
        # Data point
        ax.scatter(x, y, s=150, marker=marker, color=color,
                  edgecolors='white', linewidth=2, label=provider_label,
                  zorder=3, alpha=0.9)
        
        # Model label
        ax.annotate(model, (x, y), xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='gray', alpha=0.95, linewidth=0.8))
    
    # Add note about error ellipses
    ax.text(0.02, 0.98, 'Ellipses show ±1 standard deviation', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='gray', alpha=0.95, linewidth=0.8))
    
    ax.set_xlabel('Latency (ms)', fontweight='normal', fontsize=13)
    ax.set_ylabel('Quality Score (0-10)', fontweight='normal', fontsize=13)
    ax.set_title('Text-to-Speech Performance Analysis', fontweight='bold', pad=15, fontsize=14)
    ax.set_ylim(0, 10.5)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(loc='best', frameon=True, edgecolor='gray', ncol=1, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_quality_metrics(quality_metrics: Dict, output_dir: str):
    """Plot quality metrics: fluency, tone, naturalness and LLM variants."""
    metrics_to_plot = [
        ('fluency', 'Fluency', 'Acoustic Analysis'),
        ('tone', 'Tone', 'Acoustic Analysis'),
        ('naturalness', 'Naturalness', 'Acoustic Analysis'),
        ('llm_fluency', 'Fluency', 'LLM Evaluation'),
        ('llm_tone', 'Tone', 'LLM Evaluation'),
        ('llm_naturalness', 'Naturalness', 'LLM Evaluation')
    ]
    
    # Provider-specific colors
    provider_colors = {
        'aws_polly': '#1f77b4',
        'deepgram_aura': '#17becf',
        'elevenlabs': '#9467bd',
        'cartesia': '#ff7f0e',
        'openai': '#2ca02c',
        'nvidia_riva': '#76b900',
        'nvidia': '#76b900',
        'riva': '#76b900'
    }
    
    for metric_key, metric_label, method in metrics_to_plot:
        filtered = {k: v for k, v in quality_metrics.items() if v.get(metric_key) is not None}
        if not filtered:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        models = sorted(filtered.keys())
        values = [filtered[m][metric_key] for m in models]
        stds = [filtered[m].get(f'{metric_key}_std', 0) for m in models]
        
        # Assign distinct colors per service
        colors = []
        for m in models:
            if 'nvidia' in m.lower() or 'riva' in m.lower():
                colors.append(provider_colors['nvidia_riva'])
            else:
                colors.append(provider_colors.get(m, provider_colors.get(m.split('_')[0] + '_' + m.split('_')[-1], '#7f7f7f')))
        
        bars = ax.barh(models, values, xerr=stds, color=colors, alpha=0.2, 
                      edgecolor='white', linewidth=2.5, capsize=6, 
                      error_kw={'ecolor': '#333333', 'elinewidth': 2})
        
        ax.set_xlabel('Score (0-10)', fontweight='normal', fontsize=13)
        ax.set_ylabel('TTS Service', fontweight='normal', fontsize=13)
        ax.set_title(f'{metric_label} Assessment ({method})', fontweight='bold', pad=15, fontsize=14)
        ax.set_xlim(0, 10.5)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
        ax.axvline(x=5, color='#666666', linestyle=':', linewidth=1.5, alpha=0.6, zorder=0)
        
        # Add statistical note
        ax.text(0.98, 0.02, 'Error bars: ±1 SD', 
               transform=ax.transAxes, fontsize=10, 
               horizontalalignment='right', verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='gray', alpha=0.95, linewidth=0.8))
        
        plt.tight_layout()
        filename = f"tts_{metric_key}.png"
        plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_dir}/{filename}")
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
    
    # Debug: Print detected models
    if results:
        stt_models = set(r.get('stt_model', 'unknown') for r in results)
        tts_models = set(r.get('tts_model', 'unknown') for r in results)
        print(f"Detected STT models: {sorted(stt_models)}")
        print(f"Detected TTS models: {sorted(tts_models)}")
    
    if not results:
        print("No results found")
        return
    
    print("Extracting metrics...")
    stt_metrics, tts_metrics, quality_metrics = extract_metrics(results)
    print(f"STT models: {len(stt_metrics)}, TTS models: {len(tts_metrics)}")
    
    print("Generating plots...")
    plot_stt_metrics(stt_metrics, str(output_dir / "stt_latency_vs_wer.png"))
    plot_tts_metrics(tts_metrics, str(output_dir / "tts_latency_vs_quality.png"))
    plot_quality_metrics(quality_metrics, str(output_dir))
    
    print(f"\nComplete. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()

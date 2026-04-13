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

def extract_metrics(results: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    """Extract STT, TTS, and combination metrics with statistics."""
    stt_metrics = {}
    tts_metrics = {}
    quality_metrics = {}
    combination_metrics = {}
    
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
                'mos': [], 'noisiness': [], 'coloration': [], 'discontinuity': [], 'loudness': [],
                'mosnet_score': [], 'srmr_score': [],
                'fluency_std': [], 'tone_std': [], 'naturalness_std': [],
                'llm_fluency_std': [], 'llm_tone_std': [], 'llm_naturalness_std': [],
                'mos_std': [], 'noisiness_std': [], 'coloration_std': [], 'discontinuity_std': [], 'loudness_std': [],
                'mosnet_score_std': [], 'srmr_score_std': []
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
        
        # NISQA metrics
        mos_scores = []
        noisiness_scores = []
        coloration_scores = []
        discontinuity_scores = []
        loudness_scores = []
        
        # SpeechMetrics
        mosnet_scores = []
        srmr_scores = []
        
        for eval_item in result.get('evaluations', []):
            voice_quality = eval_item.get('voice_quality', {})
            if voice_quality.get('mos') is not None:
                mos_scores.append(voice_quality['mos'])
            if voice_quality.get('noisiness') is not None:
                noisiness_scores.append(voice_quality['noisiness'])
            if voice_quality.get('coloration') is not None:
                coloration_scores.append(voice_quality['coloration'])
            if voice_quality.get('discontinuity') is not None:
                discontinuity_scores.append(voice_quality['discontinuity'])
            if voice_quality.get('loudness') is not None:
                loudness_scores.append(voice_quality['loudness'])
            if voice_quality.get('mosnet_score') is not None:
                mosnet_scores.append(voice_quality['mosnet_score'])
            if voice_quality.get('srmr_score') is not None:
                srmr_scores.append(voice_quality['srmr_score'])
        
        if mos_scores:
            quality_metrics[tts_model]['mos'].append(np.mean(mos_scores))
            quality_metrics[tts_model]['mos_std'].append(np.std(mos_scores))
        if noisiness_scores:
            quality_metrics[tts_model]['noisiness'].append(np.mean(noisiness_scores))
            quality_metrics[tts_model]['noisiness_std'].append(np.std(noisiness_scores))
        if coloration_scores:
            quality_metrics[tts_model]['coloration'].append(np.mean(coloration_scores))
            quality_metrics[tts_model]['coloration_std'].append(np.std(coloration_scores))
        if discontinuity_scores:
            quality_metrics[tts_model]['discontinuity'].append(np.mean(discontinuity_scores))
            quality_metrics[tts_model]['discontinuity_std'].append(np.std(discontinuity_scores))
        if loudness_scores:
            quality_metrics[tts_model]['loudness'].append(np.mean(loudness_scores))
            quality_metrics[tts_model]['loudness_std'].append(np.std(loudness_scores))
        if mosnet_scores:
            quality_metrics[tts_model]['mosnet_score'].append(np.mean(mosnet_scores))
            quality_metrics[tts_model]['mosnet_score_std'].append(np.std(mosnet_scores))
        if srmr_scores:
            quality_metrics[tts_model]['srmr_score'].append(np.mean(srmr_scores))
            quality_metrics[tts_model]['srmr_score_std'].append(np.std(srmr_scores))
    
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
            'mos': np.mean(quality_metrics[model]['mos']) if quality_metrics[model]['mos'] else None,
            'noisiness': np.mean(quality_metrics[model]['noisiness']) if quality_metrics[model]['noisiness'] else None,
            'coloration': np.mean(quality_metrics[model]['coloration']) if quality_metrics[model]['coloration'] else None,
            'discontinuity': np.mean(quality_metrics[model]['discontinuity']) if quality_metrics[model]['discontinuity'] else None,
            'loudness': np.mean(quality_metrics[model]['loudness']) if quality_metrics[model]['loudness'] else None,
            'mosnet_score': np.mean(quality_metrics[model]['mosnet_score']) if quality_metrics[model]['mosnet_score'] else None,
            'srmr_score': np.mean(quality_metrics[model]['srmr_score']) if quality_metrics[model]['srmr_score'] else None,
            'fluency_std': np.mean(quality_metrics[model]['fluency_std']) if quality_metrics[model]['fluency_std'] else 0,
            'tone_std': np.mean(quality_metrics[model]['tone_std']) if quality_metrics[model]['tone_std'] else 0,
            'naturalness_std': np.mean(quality_metrics[model]['naturalness_std']) if quality_metrics[model]['naturalness_std'] else 0,
            'llm_fluency_std': np.mean(quality_metrics[model]['llm_fluency_std']) if quality_metrics[model]['llm_fluency_std'] else 0,
            'llm_tone_std': np.mean(quality_metrics[model]['llm_tone_std']) if quality_metrics[model]['llm_tone_std'] else 0,
            'llm_naturalness_std': np.mean(quality_metrics[model]['llm_naturalness_std']) if quality_metrics[model]['llm_naturalness_std'] else 0,
            'mos_std': np.mean(quality_metrics[model]['mos_std']) if quality_metrics[model]['mos_std'] else 0,
            'noisiness_std': np.mean(quality_metrics[model]['noisiness_std']) if quality_metrics[model]['noisiness_std'] else 0,
            'coloration_std': np.mean(quality_metrics[model]['coloration_std']) if quality_metrics[model]['coloration_std'] else 0,
            'discontinuity_std': np.mean(quality_metrics[model]['discontinuity_std']) if quality_metrics[model]['discontinuity_std'] else 0,
            'loudness_std': np.mean(quality_metrics[model]['loudness_std']) if quality_metrics[model]['loudness_std'] else 0,
            'mosnet_score_std': np.mean(quality_metrics[model]['mosnet_score_std']) if quality_metrics[model]['mosnet_score_std'] else 0,
            'srmr_score_std': np.mean(quality_metrics[model]['srmr_score_std']) if quality_metrics[model]['srmr_score_std'] else 0
        }
    
    # Extract combination metrics
    for result in results:
        stt_model = result['stt_model']
        tts_model = result['tts_model']
        combo_key = f"{stt_model}+{tts_model}"
        
        if combo_key not in combination_metrics:
            combination_metrics[combo_key] = {
                'stt_model': stt_model,
                'tts_model': tts_model,
                'stt_latency': [], 'tts_latency': [], 'total_latency': [],
                'wer': [], 'overall_score': [],
                'stt_latency_std': [], 'tts_latency_std': [], 'total_latency_std': [],
                'wer_std': [], 'overall_score_std': []
            }
        
        stt_latencies = []
        tts_latencies = []
        total_latencies = []
        wers = []
        scores = []
        
        for eval_item in result.get('evaluations', []):
            if eval_item.get('stt_latency_ms') is not None:
                stt_latencies.append(eval_item['stt_latency_ms'])
            if eval_item.get('tts_latency_ms') is not None:
                tts_latencies.append(eval_item['tts_latency_ms'])
            if eval_item.get('total_latency_ms') is not None:
                total_latencies.append(eval_item['total_latency_ms'])
            if eval_item.get('wer') is not None:
                wers.append(eval_item['wer'])
            judge_scores = eval_item.get('judge_scores', {})
            if judge_scores.get('overall') is not None:
                scores.append(judge_scores['overall'])
        
        if stt_latencies:
            combination_metrics[combo_key]['stt_latency'].append(np.mean(stt_latencies))
            combination_metrics[combo_key]['stt_latency_std'].append(np.std(stt_latencies))
        if tts_latencies:
            combination_metrics[combo_key]['tts_latency'].append(np.mean(tts_latencies))
            combination_metrics[combo_key]['tts_latency_std'].append(np.std(tts_latencies))
        if total_latencies:
            combination_metrics[combo_key]['total_latency'].append(np.mean(total_latencies))
            combination_metrics[combo_key]['total_latency_std'].append(np.std(total_latencies))
        if wers:
            combination_metrics[combo_key]['wer'].append(np.mean(wers))
            combination_metrics[combo_key]['wer_std'].append(np.std(wers))
        if scores:
            combination_metrics[combo_key]['overall_score'].append(np.mean(scores))
            combination_metrics[combo_key]['overall_score_std'].append(np.std(scores))
    
    # Aggregate combination metrics
    for combo in combination_metrics:
        combination_metrics[combo] = {
            'stt_model': combination_metrics[combo]['stt_model'],
            'tts_model': combination_metrics[combo]['tts_model'],
            'stt_latency': np.mean(combination_metrics[combo]['stt_latency']) if combination_metrics[combo]['stt_latency'] else None,
            'tts_latency': np.mean(combination_metrics[combo]['tts_latency']) if combination_metrics[combo]['tts_latency'] else None,
            'total_latency': np.mean(combination_metrics[combo]['total_latency']) if combination_metrics[combo]['total_latency'] else None,
            'wer': np.mean(combination_metrics[combo]['wer']) if combination_metrics[combo]['wer'] else None,
            'overall_score': np.mean(combination_metrics[combo]['overall_score']) if combination_metrics[combo]['overall_score'] else None,
            'stt_latency_std': np.mean(combination_metrics[combo]['stt_latency_std']) if combination_metrics[combo]['stt_latency_std'] else 0,
            'tts_latency_std': np.mean(combination_metrics[combo]['tts_latency_std']) if combination_metrics[combo]['tts_latency_std'] else 0,
            'total_latency_std': np.mean(combination_metrics[combo]['total_latency_std']) if combination_metrics[combo]['total_latency_std'] else 0,
            'wer_std': np.mean(combination_metrics[combo]['wer_std']) if combination_metrics[combo]['wer_std'] else 0,
            'overall_score_std': np.mean(combination_metrics[combo]['overall_score_std']) if combination_metrics[combo]['overall_score_std'] else 0
        }
    
    return stt_metrics, tts_metrics, quality_metrics, combination_metrics

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
    filtered = stt_metrics
    
    if not filtered:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define colors and markers
    provider_styles = {
        'aws': {'color': '#232F3E', 'marker': 'o', 'label': 'AWS Transcribe'},
        'deepgram': {'color': '#00A67E', 'marker': 's', 'label': 'Deepgram'},
        'whisper': {'color': '#FF6B35', 'marker': '^', 'label': 'Whisper'}
    }
    
    plotted_providers = set()
    
    for model, metrics in filtered.items():
        if 'whisper' in model.lower():
            provider = 'whisper'
        elif 'aws' in model.lower() or 'transcribe' in model.lower():
            provider = 'aws'
        elif 'deepgram' in model.lower():
            provider = 'deepgram'
        else:
            provider = 'aws'
        
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
    
    # Use tab10 colormap for consistency
    models = sorted(tts_metrics.items())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (model, metrics) in enumerate(models):
        x = metrics['latency']
        y = metrics['score']
        x_std = metrics['latency_std']
        y_std = metrics['score_std']
        
        color = colors[i]
        marker = markers[i % len(markers)]
        
        # Smaller error ellipse
        ellipse = Ellipse((x, y), width=x_std*1.2, height=y_std*1.2,
                         facecolor=color, alpha=0.1,
                         edgecolor=color, linewidth=1.5, linestyle='--')
        ax.add_patch(ellipse)
        
        # Data point
        ax.scatter(x, y, s=150, marker=marker, color=color,
                  edgecolors='white', linewidth=2, label=model,
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
    """Plot quality metrics: fluency, tone, naturalness, LLM variants."""
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
        
        bars = ax.barh(models, values, xerr=stds, color=colors, alpha=0.3, 
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

def plot_combination_matrix(combination_metrics: Dict, output_dir: str):
    """Plot STT+TTS combination performance matrix."""
    if not combination_metrics:
        return
    
    # Create performance matrix heatmap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get unique STT and TTS models
    stt_models = sorted(set(combo['stt_model'] for combo in combination_metrics.values()))
    tts_models = sorted(set(combo['tts_model'] for combo in combination_metrics.values()))
    
    # Create matrices for different metrics
    wer_matrix = np.full((len(stt_models), len(tts_models)), np.nan)
    score_matrix = np.full((len(stt_models), len(tts_models)), np.nan)
    stt_latency_matrix = np.full((len(stt_models), len(tts_models)), np.nan)
    tts_latency_matrix = np.full((len(stt_models), len(tts_models)), np.nan)
    
    for combo_key, metrics in combination_metrics.items():
        stt_idx = stt_models.index(metrics['stt_model'])
        tts_idx = tts_models.index(metrics['tts_model'])
        
        if metrics['wer'] is not None:
            wer_matrix[stt_idx, tts_idx] = metrics['wer']
        if metrics['overall_score'] is not None:
            score_matrix[stt_idx, tts_idx] = metrics['overall_score']
        if metrics['stt_latency'] is not None:
            stt_latency_matrix[stt_idx, tts_idx] = metrics['stt_latency']
        if metrics['tts_latency'] is not None:
            tts_latency_matrix[stt_idx, tts_idx] = metrics['tts_latency']
    
    # Plot WER heatmap
    im1 = ax1.imshow(wer_matrix, cmap='Reds', aspect='auto')
    ax1.set_title('Word Error Rate by STT+TTS Combination', fontweight='bold', pad=15)
    ax1.set_xlabel('TTS Service', fontweight='normal')
    ax1.set_ylabel('STT Service', fontweight='normal')
    ax1.set_xticks(range(len(tts_models)))
    ax1.set_yticks(range(len(stt_models)))
    ax1.set_xticklabels(tts_models, rotation=45, ha='right')
    ax1.set_yticklabels(stt_models)
    plt.colorbar(im1, ax=ax1, label='WER')
    
    # Add text annotations
    for i in range(len(stt_models)):
        for j in range(len(tts_models)):
            if not np.isnan(wer_matrix[i, j]):
                ax1.text(j, i, f'{wer_matrix[i, j]:.1f}', ha='center', va='center', 
                        color='white' if wer_matrix[i, j] > np.nanmean(wer_matrix) else 'black')
    
    # Plot Overall Score heatmap
    im2 = ax2.imshow(score_matrix, cmap='Greens', aspect='auto')
    ax2.set_title('Overall Quality Score by STT+TTS Combination', fontweight='bold', pad=15)
    ax2.set_xlabel('TTS Service', fontweight='normal')
    ax2.set_ylabel('STT Service', fontweight='normal')
    ax2.set_xticks(range(len(tts_models)))
    ax2.set_yticks(range(len(stt_models)))
    ax2.set_xticklabels(tts_models, rotation=45, ha='right')
    ax2.set_yticklabels(stt_models)
    plt.colorbar(im2, ax=ax2, label='Score (0-10)')
    
    for i in range(len(stt_models)):
        for j in range(len(tts_models)):
            if not np.isnan(score_matrix[i, j]):
                ax2.text(j, i, f'{score_matrix[i, j]:.1f}', ha='center', va='center',
                        color='white' if score_matrix[i, j] < np.nanmean(score_matrix) else 'black')
    
    # Plot STT Latency heatmap
    im3 = ax3.imshow(stt_latency_matrix, cmap='Blues', aspect='auto')
    ax3.set_title('STT Latency by STT+TTS Combination', fontweight='bold', pad=15)
    ax3.set_xlabel('TTS Service', fontweight='normal')
    ax3.set_ylabel('STT Service', fontweight='normal')
    ax3.set_xticks(range(len(tts_models)))
    ax3.set_yticks(range(len(stt_models)))
    ax3.set_xticklabels(tts_models, rotation=45, ha='right')
    ax3.set_yticklabels(stt_models)
    plt.colorbar(im3, ax=ax3, label='Latency (ms)')
    
    for i in range(len(stt_models)):
        for j in range(len(tts_models)):
            if not np.isnan(stt_latency_matrix[i, j]):
                ax3.text(j, i, f'{stt_latency_matrix[i, j]:.0f}', ha='center', va='center',
                        color='white' if stt_latency_matrix[i, j] > np.nanmean(stt_latency_matrix) else 'black')
    
    # Plot TTS Latency heatmap
    im4 = ax4.imshow(tts_latency_matrix, cmap='Purples', aspect='auto')
    ax4.set_title('TTS Latency by STT+TTS Combination', fontweight='bold', pad=15)
    ax4.set_xlabel('TTS Service', fontweight='normal')
    ax4.set_ylabel('STT Service', fontweight='normal')
    ax4.set_xticks(range(len(tts_models)))
    ax4.set_yticks(range(len(stt_models)))
    ax4.set_xticklabels(tts_models, rotation=45, ha='right')
    ax4.set_yticklabels(stt_models)
    plt.colorbar(im4, ax=ax4, label='Latency (ms)')
    
    for i in range(len(stt_models)):
        for j in range(len(tts_models)):
            if not np.isnan(tts_latency_matrix[i, j]):
                ax4.text(j, i, f'{tts_latency_matrix[i, j]:.0f}', ha='center', va='center',
                        color='white' if tts_latency_matrix[i, j] > np.nanmean(tts_latency_matrix) else 'black')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combination_performance_matrix.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/combination_performance_matrix.png")
    plt.close()

def plot_combination_scatter(combination_metrics: Dict, output_dir: str):
    """Plot simplified STT+TTS combination scatter plots."""
    if not combination_metrics:
        return
    
    # Define distinct symbols and colors by provider
    stt_colors = {
        'aws_transcribe': '#1f77b4',  # Blue
        'deepgram_nova2': '#ff7f0e',  # Orange  
        'deepgram_nova3': '#2ca02c',  # Green
        'whisper_large': '#d62728',   # Red
        'whisper_small': '#9467bd',   # Purple
        'whisper_turbo': '#8c564b',   # Brown
        'nvidia_parakeet': '#e377c2', # Pink
        'assemblyai': '#7f7f7f',      # Gray
        'gladia': '#bcbd22'           # Olive
    }
    
    tts_symbols = {
        'aws_polly': 'o',
        'cartesia': 's', 
        'deepgram': '^',
        'groq': 'D',
        'elevenlabs': 'v',
        'openai': '<',
        'nvidia': '>',
        'lmnt': 'p',
        'playht': '*',
        'rime': 'h'
    }
    
    # Plot 1: Accuracy vs Quality Trade-off
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for combo_key, metrics in combination_metrics.items():
        if metrics['wer'] is None or metrics['overall_score'] is None:
            continue
        
        # Determine STT color based on specific model
        stt_model = metrics['stt_model'].lower()
        color = '#17becf'  # default cyan
        for stt_key, stt_color in stt_colors.items():
            if stt_key.replace('_', '') in stt_model.replace('_', ''):
                color = stt_color
                break
        
        # Determine TTS symbol
        tts_model = metrics['tts_model']
        symbol = 'o'  # default
        for tts_key, tts_symbol in tts_symbols.items():
            if tts_key in tts_model.lower():
                symbol = tts_symbol
                break
        
        combo_label = f"{metrics['stt_model']}+{metrics['tts_model']}"
        
        x = metrics['wer']
        y = metrics['overall_score']
        x_std = metrics['wer_std']
        y_std = metrics['overall_score_std']
        
        # Error ellipse
        ellipse = Ellipse((x, y), width=x_std*1.2, height=y_std*1.2,
                         facecolor=color, alpha=0.1, 
                         edgecolor=color, linewidth=1.5, linestyle='--')
        ax.add_patch(ellipse)
        
        # Data point with distinct symbol
        ax.scatter(x, y, s=150, marker=symbol, color=color, edgecolors='white', linewidth=2, 
                  alpha=0.8, label=combo_label)
    
    ax.set_xlabel('Word Error Rate (%)', fontweight='normal', fontsize=13)
    ax.set_ylabel('Overall Quality Score (0-10)', fontweight='normal', fontsize=13)
    ax.set_title('Accuracy vs Quality Trade-off by STT+TTS Combination', fontweight='bold', pad=15, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add note about error ellipses
    ax.text(0.02, 0.98, 'Ellipses show ±1 standard deviation', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='gray', alpha=0.95, linewidth=0.8))
    
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combination_accuracy_vs_quality.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/combination_accuracy_vs_quality.png")
    plt.close()
    
    # Plot 2: Speed vs Quality Trade-off
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for combo_key, metrics in combination_metrics.items():
        if metrics['total_latency'] is None or metrics['overall_score'] is None:
            continue
        
        # Determine STT color based on specific model
        stt_model = metrics['stt_model'].lower()
        color = '#17becf'  # default cyan
        for stt_key, stt_color in stt_colors.items():
            if stt_key.replace('_', '') in stt_model.replace('_', ''):
                color = stt_color
                break
        
        # Determine TTS symbol
        tts_model = metrics['tts_model']
        symbol = 'o'  # default
        for tts_key, tts_symbol in tts_symbols.items():
            if tts_key in tts_model.lower():
                symbol = tts_symbol
                break
        
        combo_label = f"{metrics['stt_model']}+{metrics['tts_model']}"
        
        x = metrics['total_latency']
        y = metrics['overall_score']
        x_std = metrics['total_latency_std']
        y_std = metrics['overall_score_std']
        
        # Error ellipse
        ellipse = Ellipse((x, y), width=x_std*1.2, height=y_std*1.2,
                         facecolor=color, alpha=0.1,
                         edgecolor=color, linewidth=1.5, linestyle='--')
        ax.add_patch(ellipse)
        
        # Data point with distinct symbol
        ax.scatter(x, y, s=150, marker=symbol, color=color, edgecolors='white', linewidth=2,
                  alpha=0.8, label=combo_label)
    
    ax.set_xlabel('Total Latency (ms)', fontweight='normal', fontsize=13)
    ax.set_ylabel('Overall Quality Score (0-10)', fontweight='normal', fontsize=13)
    ax.set_title('Speed vs Quality Trade-off by STT+TTS Combination', fontweight='bold', pad=15, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add note about error ellipses
    ax.text(0.02, 0.98, 'Ellipses show ±1 standard deviation', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='gray', alpha=0.95, linewidth=0.8))
    
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combination_speed_vs_quality.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/combination_speed_vs_quality.png")
    plt.close()

def plot_metric_comparison(data: Dict, metric_key: str, title: str, xlabel: str, output_path: str, xlim=None):
    """Generic bar chart comparing models on a single metric."""
    filtered = {k: v for k, v in data.items() if v.get(metric_key) is not None}
    if not filtered:
        return
    
    fig, ax = plt.subplots(figsize=(10, max(7, len(filtered) * 0.4)))
    
    models = sorted(filtered.keys())
    values = [filtered[m][metric_key] for m in models]
    stds = [filtered[m].get(f'{metric_key}_std', 0) for m in models]
    
    # Check if this is STT data based on title
    if 'STT' in title:
        # STT-specific colors matching stt_latency_vs_wer.png
        stt_colors = {
            'aws': '#232F3E',
            'deepgram': '#00A67E', 
            'whisper': '#FF6B35'
        }
        
        colors = []
        for model in models:
            if 'whisper' in model.lower():
                colors.append(stt_colors['whisper'])
            elif 'aws' in model.lower() or 'transcribe' in model.lower():
                colors.append(stt_colors['aws'])
            elif 'deepgram' in model.lower():
                colors.append(stt_colors['deepgram'])
            else:
                colors.append('#7f7f7f')
    else:
        # Use tab10 colormap for TTS models
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    bars = ax.barh(models, values, xerr=stds, color=colors, alpha=0.3, 
                  edgecolor='white', linewidth=2, capsize=5,
                  error_kw={'ecolor': '#333333', 'elinewidth': 1.5})
    
    ax.set_xlabel(xlabel, fontweight='normal', fontsize=13)
    ax.set_ylabel('Model', fontweight='normal', fontsize=13)
    ax.set_title(title, fontweight='bold', pad=15, fontsize=14)
    if xlim:
        ax.set_xlim(xlim)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent
    results_dir = repo_dir / "evaluation_output" 
    output_dir = script_dir / "plots"
    
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
    stt_metrics, tts_metrics, quality_metrics, combination_metrics = extract_metrics(results)
    print(f"STT models: {len(stt_metrics)}, TTS models: {len(tts_metrics)}, Combinations: {len(combination_metrics)}")
    
    print("Generating plots...")
    plot_stt_metrics(stt_metrics, str(output_dir / "stt_latency_vs_wer.png"))
    plot_tts_metrics(tts_metrics, str(output_dir / "tts_latency_vs_quality.png"))
    plot_quality_metrics(quality_metrics, str(output_dir))
    
    # Generate combination analysis plots
    print("Generating combination analysis plots...")
    plot_combination_matrix(combination_metrics, str(output_dir))
    plot_combination_scatter(combination_metrics, str(output_dir))
    
    # Individual metric comparisons
    print("Generating metric comparison plots...")
    plot_metric_comparison(stt_metrics, 'latency', 'STT Latency Comparison', 'Latency (ms)', 
                          str(output_dir / "stt_latency.png"))
    plot_metric_comparison(stt_metrics, 'wer', 'STT Word Error Rate Comparison', 'WER', 
                          str(output_dir / "stt_wer.png"))
    plot_metric_comparison(tts_metrics, 'latency', 'TTS Latency Comparison', 'Latency (ms)', 
                          str(output_dir / "tts_latency.png"))
    plot_metric_comparison(tts_metrics, 'score', 'TTS Quality Score Comparison', 'Score (0-10)', 
                          str(output_dir / "tts_score.png"), xlim=(0, 10.5))
    plot_metric_comparison(quality_metrics, 'fluency', 'Fluency Comparison (Acoustic)', 'Score (0-10)', 
                          str(output_dir / "tts_fluency.png"), xlim=(0, 10.5))
    plot_metric_comparison(quality_metrics, 'tone', 'Tone Comparison (Acoustic)', 'Score (0-10)', 
                          str(output_dir / "tts_tone.png"), xlim=(0, 10.5))
    plot_metric_comparison(quality_metrics, 'naturalness', 'Naturalness Comparison (Acoustic)', 'Score (0-10)', 
                          str(output_dir / "tts_naturalness.png"), xlim=(0, 10.5))
    plot_metric_comparison(quality_metrics, 'llm_fluency', 'Fluency Comparison (LLM)', 'Score (0-10)', 
                          str(output_dir / "tts_llm_fluency.png"), xlim=(0, 10.5))
    plot_metric_comparison(quality_metrics, 'llm_tone', 'Tone Comparison (LLM)', 'Score (0-10)', 
                          str(output_dir / "tts_llm_tone.png"), xlim=(0, 10.5))
    plot_metric_comparison(quality_metrics, 'llm_naturalness', 'Naturalness Comparison (LLM)', 'Score (0-10)', 
                          str(output_dir / "tts_llm_naturalness.png"), xlim=(0, 10.5))
    
    # NISQA metrics
    plot_metric_comparison(quality_metrics, 'mos', 'Mean Opinion Score (NISQA)', 'MOS (1-5)', 
                          str(output_dir / "tts_mos.png"), xlim=(1, 5))
    plot_metric_comparison(quality_metrics, 'noisiness', 'Noisiness (NISQA)', 'Score (1-5)', 
                          str(output_dir / "tts_noisiness.png"), xlim=(1, 5))
    plot_metric_comparison(quality_metrics, 'coloration', 'Coloration (NISQA)', 'Score (1-5)', 
                          str(output_dir / "tts_coloration.png"), xlim=(1, 5))
    plot_metric_comparison(quality_metrics, 'discontinuity', 'Discontinuity (NISQA)', 'Score (1-5)', 
                          str(output_dir / "tts_discontinuity.png"), xlim=(1, 5))
    plot_metric_comparison(quality_metrics, 'loudness', 'Loudness (NISQA)', 'Score (1-5)', 
                          str(output_dir / "tts_loudness.png"), xlim=(1, 5))
    
    # SpeechMetrics
    plot_metric_comparison(quality_metrics, 'mosnet_score', 'MOSNet Score (SpeechMetrics)', 'Score (1-5)', 
                          str(output_dir / "tts_mosnet.png"), xlim=(1, 5))
    plot_metric_comparison(quality_metrics, 'srmr_score', 'SRMR Score (SpeechMetrics)', 'SRMR (dB)', 
                          str(output_dir / "tts_srmr.png"))
    
    print(f"\nComplete. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()

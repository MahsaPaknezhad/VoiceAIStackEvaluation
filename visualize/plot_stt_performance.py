#!/usr/bin/env python3
# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT
"""Plot STT latency vs WER with broken axis for local vs API-based models."""

import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DATA_DIR = 'output/eval-results/evaluation_output/evaluation'
OUTPUT_PATH = 'visualize/plots/stt_latency_vs_wer.png'

PROVIDER_STYLES = {
    'aws': {'color': '#3498db', 'marker': 'o', 'label': 'AWS Transcribe'},
    'deepgram': {'color': '#00A67E', 'marker': 's', 'label': 'Deepgram'},
    'whisper': {'color': '#FF6B35', 'marker': '^', 'label': 'Whisper'},
    'assemblyai': {'color': '#9467bd', 'marker': 'D', 'label': 'AssemblyAI'},
    'nvidia': {'color': '#76b900', 'marker': 'v', 'label': 'NVIDIA Parakeet'},
}

LABEL_OFFSETS = {
    'nvidia_parakeet': (15, -30),
    'whisper_small': (15, 25),
    'whisper_turbo': (15, -25),
    'whisper_large_v2': (15, 25),
    'assemblyai': (15, 30),
    'aws_transcribe': (15, 25),
    'deepgram_nova2': (-80, -30),
    'deepgram_nova3': (-80, 25),
}


def get_provider(model):
    m = model.lower()
    if 'whisper' in m: return 'whisper'
    if 'aws' in m or 'transcribe' in m: return 'aws'
    if 'deepgram' in m: return 'deepgram'
    if 'assemblyai' in m: return 'assemblyai'
    if 'nvidia' in m or 'parakeet' in m or 'riva' in m: return 'nvidia'
    return 'aws'


def normalize_model_name(name):
    mapping = {
        'AWS Transcribe': 'aws_transcribe',
        'Deepgram Nova 2': 'deepgram_nova2',
        'Deepgram Nova 3': 'deepgram_nova3',
        'AssemblyAI': 'assemblyai',
        'Whisper Small (Local)': 'whisper_small',
        'Whisper Turbo (Local)': 'whisper_turbo',
        'Whisper Large-v2 (Local)': 'whisper_large_v2',
        'NVIDIA Riva STT': 'nvidia_parakeet',
    }
    return mapping.get(name, name.lower().replace(' ', '_').replace('-', '_'))


def load_metrics():
    latencies = defaultdict(list)
    wers = defaultdict(list)
    
    for jf in glob.glob(f'{DATA_DIR}/**/*.json', recursive=True):
        with open(jf) as f:
            d = json.load(f)
        stt = d.get('stt_model') or d.get('stt_service_id', '').replace('_streaming', '')
        if not stt:
            continue
        stt = normalize_model_name(stt)
        for ev in d.get('evaluations', []):
            lat = ev.get('stt_latency_ms')
            wer = ev.get('wer')
            if lat is not None and wer is not None:
                latencies[stt].append(lat)
                wers[stt].append(wer)  # Already in percentage
    
    metrics = {}
    for model in latencies:
        metrics[model] = {
            'latency': np.mean(latencies[model]),
            'latency_std': np.std(latencies[model]),
            'wer': np.mean(wers[model]),
            'wer_std': np.std(wers[model]),
        }
    return metrics


def plot(metrics, output_path):
    all_models = dict(metrics)
    
    if not all_models:
        print("No data to plot")
        return
    
    xlim = (0, 35000)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    plotted = set()
    for model, m in sorted(all_models.items()):
        x, y = m['latency'], m['wer']
        
        provider = get_provider(model)
        style = PROVIDER_STYLES.get(provider, PROVIDER_STYLES['aws'])
        
        # WER SD - vertical error bar lines
        y_bottom = max(y - m['wer_std'], 0)
        y_top = y + m['wer_std']
        ax.vlines(x, y_bottom, y_top, color='black', linewidth=1.5, alpha=0.7, zorder=2)
        cap_width = (xlim[1] - xlim[0]) * 0.008
        ax.hlines(y_bottom, x - cap_width, x + cap_width, color='black', linewidth=1.5, alpha=0.7, zorder=2)
        ax.hlines(y_top, x - cap_width, x + cap_width, color='black', linewidth=1.5, alpha=0.7, zorder=2)
        
        # Latency SD - horizontal gradient bar
        lat_std = m['latency_std']
        bar_height = 1.5
        n_layers = 50
        for i in range(n_layers):
            frac = (i + 1) / n_layers
            alpha = 0.15 * (1 - frac)
            left = max(x - frac * lat_std, xlim[0])
            right = min(x + frac * lat_std, xlim[1])
            if left < right:
                ax.fill_betweenx([y - bar_height/2, y + bar_height/2], left, right,
                                 color=style['color'], alpha=alpha, zorder=1, linewidth=0)
        
        label = style['label'] if provider not in plotted else None
        ax.scatter(x, y, s=180, marker=style['marker'], color=style['color'],
                  edgecolors='black', linewidth=1.5, label=label, zorder=10)
        plotted.add(provider)
        
        ox, oy = LABEL_OFFSETS.get(model, (12, 14))
        ax.annotate(model, (x, y), xytext=(ox, oy), textcoords='offset points', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=style['color'],
                            alpha=0.95, linewidth=1.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8), zorder=11)
    
    ax.set_xlim(xlim)
    ax.set_ylim(0, 45)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.tick_params(axis='both', labelsize=13)
    
    ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000])
    ax.set_xticklabels(['0', '5k', '10k', '15k', '20k', '25k', '30k', '35k'])
    
    ax.set_ylabel('Word Error Rate (%)', fontsize=15)
    ax.set_xlabel('Latency (ms)', fontsize=15)
    fig.suptitle('Speech-to-Text Performance', fontweight='bold', fontsize=16, y=0.98)
    
    # Background shading for streaming vs batch regions
    BATCH_MODELS = {'nvidia', 'whisper'}
    batch_lats = [m['latency'] for mod, m in all_models.items() if get_provider(mod) in BATCH_MODELS]
    stream_lats = [m['latency'] for mod, m in all_models.items() if get_provider(mod) not in BATCH_MODELS]
    if stream_lats and batch_lats:
        boundary = (max(stream_lats) + min(batch_lats)) / 2
        ax.axvspan(xlim[0], boundary, alpha=0.08, color='#3498db', zorder=0)
        ax.axvspan(boundary, xlim[1], alpha=0.08, color='gray', zorder=0)
        ax.text(boundary * 0.35, 43, 'Streaming', fontsize=14, fontweight='bold',
                ha='center', color='#2471a3')
        ax.text((boundary + xlim[1]) / 2, 43, 'Batch Processing', fontsize=14, fontweight='bold',
                ha='center', color='gray')
    
    # Legend
    handles, labels = [], []
    for p in ['aws', 'assemblyai', 'deepgram', 'nvidia', 'whisper']:
        if p in [get_provider(m) for m in all_models]:
            s = PROVIDER_STYLES[p]
            handles.append(Line2D([0], [0], marker=s['marker'], color='w',
                                 markerfacecolor=s['color'], markersize=10))
            labels.append(s['label'])
    handles.append(Line2D([0, 0], [0, 1], color='black', lw=1.5, marker='_', markersize=8, markevery=[0, 1]))
    labels.append('WER ±1 SD (vertical)')
    handles.append(Patch(facecolor='gray', alpha=0.4))
    labels.append('Latency ±1 SD (horizontal)')
    ax.legend(handles, labels, loc='upper right', frameon=True, edgecolor='gray', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    metrics = load_metrics()
    plot(metrics, OUTPUT_PATH)

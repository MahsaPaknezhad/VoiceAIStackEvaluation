#!/usr/bin/env python3
"""Plot STT latency vs WER with broken axis for local vs API-based models."""

import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DATA_DIR = 'lily-data/evaluation_output/evaluation'
OUTPUT_PATH = 'visualize/plots/stt_latency_vs_wer.png'

PROVIDER_STYLES = {
    'aws': {'color': '#3498db', 'marker': 'o', 'label': 'AWS Transcribe'},
    'deepgram': {'color': '#00A67E', 'marker': 's', 'label': 'Deepgram'},
    'whisper': {'color': '#FF6B35', 'marker': '^', 'label': 'Whisper'},
    'assemblyai': {'color': '#9467bd', 'marker': 'D', 'label': 'AssemblyAI'},
    'nvidia': {'color': '#76b900', 'marker': 'v', 'label': 'NVIDIA Parakeet'},
}

LABEL_OFFSETS = {
    'nvidia_parakeet': (-60, 25),
    'whisper_small': (15, 30),
    'whisper_turbo': (-60, -25),
    'whisper_large_v2': (15, -25),
    'assemblyai': (15, 20),
    'aws_transcribe': (15, -30),
    'deepgram_nova2': (-25, -35),
    'deepgram_nova3': (-85, 25),
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
    LOCAL_THRESHOLD = 100
    api_models = {m: v for m, v in metrics.items() if v['latency'] >= LOCAL_THRESHOLD}
    local_models = {m: v for m, v in metrics.items() if v['latency'] < LOCAL_THRESHOLD}
    all_models = {**api_models, **local_models}
    
    if not all_models:
        print("No data to plot")
        return
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8), sharey=True,
                                             gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.05})
    
    for ax_obj, xlim in [(ax_left, (0.15, 0.6)), (ax_right, (1500, 9500))]:
        plotted = set()
        for model, m in sorted(all_models.items()):
            x, y = m['latency'], m['wer']
            if x < xlim[0] * 0.5 or x > xlim[1] * 2:
                continue
            
            provider = get_provider(model)
            style = PROVIDER_STYLES.get(provider, PROVIDER_STYLES['aws'])
            
            # WER SD - vertical error bar lines (like boxplot whiskers)
            y_bottom = max(y - m['wer_std'], 0)
            y_top = y + m['wer_std']
            ax_obj.vlines(x, y_bottom, y_top, color='black', linewidth=1.5, alpha=0.7, zorder=2)
            # Horizontal caps at ends
            cap_width = (xlim[1] - xlim[0]) * 0.015
            ax_obj.hlines(y_bottom, x - cap_width, x + cap_width, color='black', linewidth=1.5, alpha=0.7, zorder=2)
            ax_obj.hlines(y_top, x - cap_width, x + cap_width, color='black', linewidth=1.5, alpha=0.7, zorder=2)
            
            # Latency SD - horizontal gradient bar (darker at center, fades to edges)
            lat_std = m['latency_std']
            bar_height = 1.5
            n_layers = 50
            for i in range(n_layers):
                frac = (i + 1) / n_layers
                alpha = 0.15 * (1 - frac)
                left = x - frac * lat_std
                right = x + frac * lat_std
                # Clip to axis limits
                left = max(left, xlim[0])
                right = min(right, xlim[1])
                if left < right:
                    ax_obj.fill_betweenx([y - bar_height/2, y + bar_height/2], left, right,
                                         color=style['color'], alpha=alpha, zorder=1, linewidth=0)
            
            label = style['label'] if provider not in plotted else None
            ax_obj.scatter(x, y, s=180, marker=style['marker'], color=style['color'],
                          edgecolors='black', linewidth=1.5, label=label, zorder=10)
            plotted.add(provider)
            
            ox, oy = LABEL_OFFSETS.get(model, (12, 14))
            ax_obj.annotate(model, (x, y), xytext=(ox, oy), textcoords='offset points', fontsize=11,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray',
                                    alpha=0.95, linewidth=0.8),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.8), zorder=11)
        
        ax_obj.set_xlim(xlim)
        ax_obj.set_ylim(0, 45)
        ax_obj.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax_obj.tick_params(axis='both', labelsize=13)
    
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.tick_params(left=False)
    
    # Add 3 dots between the two panels
    fig.text(0.436, 0.5, '···', fontsize=20, ha='center', va='center', color='gray')
    
    ax_left.set_xticks([0.2, 0.3, 0.4, 0.5])
    ax_left.set_xticklabels(['0.2', '0.3', '0.4', '0.5'])
    ax_right.set_xticks([2000, 4000, 6000, 8000])
    ax_right.set_xticklabels(['2000', '4000', '6000', '8000'])
    
    ax_left.set_ylabel('Word Error Rate (%)', fontsize=15)
    fig.text(0.5, 0.01, 'Latency (ms)', ha='center', fontsize=15)
    fig.suptitle('Speech-to-Text Performance', fontweight='bold', fontsize=16, y=0.98)
    
    ax_left.text(0.5, 0.95, 'Local', transform=ax_left.transAxes, fontsize=14,
                color='black', ha='center', fontweight='bold')
    ax_right.text(0.3, 0.95, 'API-Based', transform=ax_right.transAxes, fontsize=14,
                 color='black', ha='center', fontweight='bold')
    
    # Background shading for regions
    ax_left.axvspan(ax_left.get_xlim()[0], ax_left.get_xlim()[1], alpha=0.08, color='#76b900', zorder=0)
    ax_right.axvspan(ax_right.get_xlim()[0], ax_right.get_xlim()[1], alpha=0.08, color='#232F3E', zorder=0)
    
    # Legend
    handles, labels = [], []
    for p in ['aws', 'assemblyai', 'deepgram', 'nvidia', 'whisper']:
        if p in [get_provider(m) for m in all_models]:
            s = PROVIDER_STYLES[p]
            handles.append(Line2D([0], [0], marker=s['marker'], color='w',
                                 markerfacecolor=s['color'], markersize=10))
            labels.append(s['label'])
    # Custom legend handle for vertical error bar with caps
    from matplotlib.collections import LineCollection
    handles.append(Line2D([0, 0], [0, 1], color='black', lw=1.5, marker='_', markersize=8, markevery=[0, 1]))
    labels.append('WER ±1 SD (vertical)')
    handles.append(Patch(facecolor='gray', alpha=0.4))
    labels.append('Latency ±1 SD (horizontal)')
    ax_right.legend(handles, labels, loc='upper right', frameon=True, edgecolor='gray', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 0, 0])
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    metrics = load_metrics()
    plot(metrics, OUTPUT_PATH)

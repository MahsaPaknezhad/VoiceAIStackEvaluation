# Voice Assistant Evaluation Pipeline

A comprehensive evaluation framework for voice assistants that measures speech-to-text accuracy, response quality, and latency metrics across different AI service combinations.

## Overview

This project provides a complete pipeline for building, testing, and evaluating voice assistants with multiple AI service providers. It enables systematic comparison of different STT/TTS/LLM combinations to optimize for accuracy, latency, and voice quality.

![Implementation Pipeline](assets/diagram.jpg)

**Implementation Pipeline:** Overview of the voice pipeline evaluation framework. Audio inputs and configuration files are processed through a Pipecat-based pipeline (STT, LLM, TTS), followed by a four-phase evaluation comprising voice quality assessment (NISQA, SpeechMetrics, LLM Voice Judge, Audio LLM Judge), response quality scoring (correctness, relevance, completeness, clarity), transcription quality via Word Error Rate, and end-to-end latency measurement across each pipeline stage.

> **Pre-computed results:** Evaluation results and generated TTS audio samples are available on Hugging Face:
> [voice-ai-stack-evaluation](https://huggingface.co/datasets/MahsaPak/voice-ai-stack-evaluation)

## Architecture

### Core Frameworks
- **Pipecat** - Voice agent orchestration and real-time audio processing
- **Strands Agents** - Multi-agent conversation management and tool calling
- **FastAPI** - Backend API server with WebSocket support

### AI Services
- **Speech-to-Text (STT)**: AWS Transcribe, Deepgram Nova-3, Speechmatics, AssemblyAI, Gladia
- **Text-to-Speech (TTS)**: AWS Polly, Deepgram Aura-2, ElevenLabs, Cartesia, PlayHT, LMNT, Rime
- **Large Language Models (LLM)**: Amazon Bedrock (Claude 3.5 Haiku), OpenAI, Groq

## Quick Start

### Prerequisites
- Python 3.8+
- ffmpeg (required for audio quality metrics)
- AWS account with Transcribe/Polly access
- API keys for desired AI services (see `.env.example`)

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Required keys:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` - For AWS services
- `DEEPGRAM_API_KEY` - For Deepgram STT/TTS (optional)
- `OPENAI_API_KEY` - For OpenAI services (optional)
- `GROQ_API_KEY` - For Groq LLM services (optional)
- `CARTESIA_API_KEY` - For Cartesia TTS (optional)
- Additional keys for other providers (optional)

### 2. Setup

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fix numpy compatibility in srmrpy
chmod +x ./scripts/fix_srmrpy.sh
./scripts/fix_srmrpy.sh

# Download Whisper models (required for local STT)
python -c "import whisper; whisper.load_model('turbo'); whisper.load_model('large'); whisper.load_model('small')"
```

## Evaluation

### Running Evaluations

Evaluate AWS Transcribe + Polly pipeline:
```bash
./scripts/test_aws_transcribe_polly.sh
```

### NVIDIA Services

For NVIDIA AI services setup and deployment instructions, see [src/core/nvidia/README.md](src/core/nvidia/README.md).

### Evaluation Metrics

The framework measures:

**Speech Recognition Accuracy**
- Word Error Rate (WER)
- Character Error Rate (CER)
- Transcription confidence scores

![STT Latency vs WER](assets/stt_latency_vs_wer.png)

**STT Latency vs WER:** Comparison of transcription speed across evaluated STT models. (Left) shows the comparison of streaming models where DeepGram Nova 3 had the lowest latency of 3,342±836ms. (Right) shows the comparison of batch models where Whisper Large had the highest latency of 28,463±7,797ms.

**Response Quality** (LLM-as-judge scoring)
- Correctness - Factual accuracy
- Relevance - Alignment with user intent
- Completeness - Coverage of required information
- Clarity - Communication effectiveness

**Performance Metrics**
- STT latency (time to first word, total transcription time)
- LLM processing time
- TTS generation latency
- End-to-end response time

**Voice Quality**
- Audio clarity and naturalness
- Prosody and intonation
- Speech rate and rhythm

![TTS Quality Across Evaluation Methods](assets/cross-method-radar.png)

**TTS Quality Across Evaluation Methods:** Normalised to a 0–1 scale. NISQA MOS clusters all providers between 0.90–0.95, while LLM Holistic and Librosa Naturalness show the widest spread across providers.

### Full Composite Ranking

Composite ranking of STT×TTS service combinations across accuracy, latency, and voice quality metrics. Scores are weighted averages of min-max normalized values (higher is better). Weights: WER 30%, Latency 30%, LLM Judge 10%, Voice LLM 10%, MiniCPM metrics 20%.

The MiniCPM voice-quality columns (mcNat/mcNoi/mcLod) are produced by the fine-tuned MiniCPM-o regression scorer (`score_audio_minicpm_reg.py`, `best_full` checkpoint), which maps the audio-conditioned final-token hidden state to three continuous 1–10 scores. This regression-head scorer replaces the earlier generative/JSON scorer; because it is trained with a mean-squared-error objective it exhibits mild regression toward the mean, so its scores span a narrower band than the previous generative values.

| Rk | STT | TTS | Score | WER% | Judge | TotLat | VcLLM | mcNat | mcNoi | mcLod | STTms | TTSms |
|----|-----|-----|-------|------|-------|--------|-------|-------|-------|-------|-------|-------|
| 1 | nvidia_parakeet | deepgram_aura | 0.827 | 15.9 | 8.70 | 26817 | 4.38 | 6.0 | 4.96 | 6.63 | 7964 | 779 |
| 2 | whisper_small | deepgram_aura | 0.783 | 17.7 | 8.68 | 25739 | 4.39 | 5.9 | 4.96 | 6.66 | 11180 | 732 |
| 3 | aws_transcribe | deepgram_aura | 0.765 | 18.6 | 8.61 | 23000 | 4.41 | 5.9 | 5.07 | 6.65 | 5009 | 411 |
| 4 | nvidia_parakeet | cartesia | 0.751 | 15.9 | 8.70 | 20584 | 3.20 | 6.5 | 6.04 | 6.14 | 8092 | 433 |
| 5 | whisper_turbo | deepgram_aura | 0.743 | 18.1 | 8.80 | 34398 | 4.39 | 5.9 | 5.00 | 6.65 | 18705 | 775 |
| 6 | nvidia_parakeet | groq | 0.743 | 15.9 | 8.71 | 27635 | 3.30 | 6.8 | 6.07 | 6.56 | 8014 | 1716 |
| 7 | whisper_large | deepgram_aura | 0.725 | 16.3 | 8.72 | 41813 | 4.47 | 5.9 | 5.00 | 6.63 | 28167 | 761 |
| 8 | whisper_small | groq | 0.700 | 17.7 | 8.73 | 26708 | 3.26 | 6.7 | 6.03 | 6.54 | 11076 | 1770 |
| 9 | whisper_small | cartesia | 0.696 | 17.6 | 8.68 | 22528 | 3.24 | 6.5 | 6.00 | 6.13 | 10108 | 432 |
| 10 | aws_transcribe | aws_polly | 0.678 | 18.6 | 8.62 | 12472 | 2.02 | 4.1 | 5.71 | 7.07 | 5024 | 302 |
| 11 | aws_transcribe | groq | 0.672 | 18.8 | 8.56 | 20573 | 3.22 | 6.7 | 6.05 | 6.51 | 5008 | 1655 |
| 12 | deepgram_nova3 | deepgram_aura | 0.668 | 21.5 | 8.57 | 25697 | 4.41 | 6.0 | 5.11 | 6.70 | 3329 | 399 |
| 13 | deepgram_nova2 | deepgram_aura | 0.655 | 21.4 | 8.52 | 25527 | 4.40 | 5.9 | 5.08 | 6.67 | 3391 | 403 |
| 14 | assemblyai | deepgram_aura | 0.649 | 22.1 | 8.55 | 25075 | 4.42 | 5.9 | 5.09 | 6.65 | 4280 | 409 |
| 15 | aws_transcribe | cartesia | 0.642 | 19.1 | 8.46 | 17530 | 3.18 | 6.5 | 5.97 | 6.12 | 5197 | 485 |
| 16 | aws_transcribe | nvidia_magpie | 0.638 | 18.6 | 8.55 | 13491 | 2.17 | 6.4 | 6.07 | 5.89 | 5034 | 899 |
| 17 | whisper_large | groq | 0.633 | 16.2 | 8.73 | 43398 | 3.23 | 6.8 | 6.06 | 6.52 | 28533 | 2008 |
| 18 | whisper_turbo | groq | 0.624 | 18.4 | 8.74 | 35785 | 3.24 | 6.8 | 6.07 | 6.53 | 19510 | 1677 |
| 19 | nvidia_parakeet | aws_polly | 0.592 | 15.9 | 8.69 | 38669 | 2.01 | 4.1 | 5.76 | 7.05 | 8052 | 261 |
| 20 | deepgram_nova3 | cartesia | 0.588 | 21.5 | 8.59 | 19840 | 3.19 | 6.5 | 6.03 | 6.16 | 3338 | 450 |
| 21 | nvidia_parakeet | nvidia_magpie | 0.586 | 15.9 | 8.74 | 38672 | 2.15 | 6.5 | 6.05 | 5.86 | 8042 | 793 |
| 22 | deepgram_nova2 | groq | 0.575 | 21.5 | 8.58 | 26072 | 3.23 | 6.8 | 6.09 | 6.56 | 3325 | 1429 |
| 23 | whisper_turbo | cartesia | 0.569 | 19.6 | 8.50 | 27697 | 3.22 | 6.5 | 5.98 | 6.10 | 17521 | 426 |
| 24 | deepgram_nova2 | cartesia | 0.568 | 21.5 | 8.56 | 22362 | 3.22 | 6.5 | 6.01 | 6.15 | 3346 | 662 |
| 25 | deepgram_nova3 | groq | 0.567 | 21.7 | 8.58 | 26426 | 3.23 | 6.8 | 6.10 | 6.56 | 3352 | 1524 |
| 26 | assemblyai | groq | 0.560 | 22.0 | 8.56 | 25470 | 3.23 | 6.7 | 6.05 | 6.55 | 4181 | 1523 |
| 27 | whisper_small | aws_polly | 0.547 | 17.5 | 8.70 | 41021 | 2.01 | 4.0 | 5.69 | 7.13 | 10969 | 278 |
| 28 | whisper_small | nvidia_magpie | 0.510 | 17.6 | 8.66 | 40985 | 2.12 | 6.4 | 6.04 | 5.86 | 10935 | 778 |
| 29 | assemblyai | cartesia | 0.486 | 22.8 | 8.26 | 22310 | 3.20 | 6.5 | 5.98 | 6.11 | 3945 | 450 |
| 30 | whisper_large | aws_polly | 0.476 | 16.2 | 8.79 | 59349 | 2.01 | 4.1 | 5.72 | 7.09 | 29356 | 279 |
| 31 | whisper_turbo | aws_polly | 0.475 | 18.3 | 8.73 | 48580 | 2.01 | 4.1 | 5.75 | 7.08 | 18466 | 271 |
| 32 | whisper_turbo | nvidia_magpie | 0.465 | 18.0 | 8.76 | 49661 | 2.14 | 6.5 | 6.04 | 5.88 | 19567 | 802 |
| 33 | whisper_large | nvidia_magpie | 0.454 | 16.1 | 8.74 | 58237 | 2.17 | 6.4 | 6.06 | 5.87 | 28224 | 754 |
| 34 | assemblyai | nvidia_magpie | 0.449 | 18.3 | 8.47 | 41755 | 2.21 | 6.3 | 6.02 | 5.88 | 2487 | 815 |
| 35 | deepgram_nova2 | aws_polly | 0.415 | 21.4 | 8.60 | 42278 | 2.02 | 4.1 | 5.71 | 7.13 | 3350 | 269 |
| 36 | whisper_large | cartesia | 0.406 | 21.9 | 8.12 | 38066 | 3.22 | 6.5 | 5.99 | 6.11 | 28000 | 447 |
| 37 | deepgram_nova3 | aws_polly | 0.404 | 21.6 | 8.57 | 42254 | 2.00 | 4.0 | 5.68 | 7.13 | 3336 | 263 |
| 38 | deepgram_nova3 | nvidia_magpie | 0.388 | 21.2 | 8.53 | 42257 | 2.21 | 6.4 | 6.00 | 5.91 | 3353 | 832 |
| 39 | deepgram_nova2 | nvidia_magpie | 0.384 | 21.3 | 8.55 | 42287 | 2.17 | 6.4 | 6.00 | 5.86 | 3395 | 835 |
| 40 | assemblyai | aws_polly | 0.347 | 22.9 | 8.47 | 42402 | 2.01 | 4.0 | 5.70 | 7.09 | 4320 | 278 |

> The `gladia_nvidia_magpie` combination is omitted from this ranking: no TTS audio was generated for it (STT/LLM/voice-quality metrics are unavailable), so it cannot be scored on the voice-quality dimensions.

### Output

Results are saved in `output/` (gitignored due to size):
- `eval-results/` - Full evaluation run data
- `scoring_output/` - Audio quality scoring results
- `plots/` - Generated charts and visualizations
- `tts_audio/` - Generated audio files for quality review
- `*_results.json` / `*_evaluation.json` - Per-experiment metrics

## Development

### Core Components

**Voice Bot Core**
- `src/core/voice_bot.py` - Bot orchestration and pipeline management
- `src/core/agent_builder.py` - Strands agent configuration
- `src/core/llm_processor.py` - LLM integration and prompt management

**Evaluation Pipeline Architecture**
- `src/evaluation/voice_pipeline_evaluator.py` - Main evaluation entry point and runner
- `src/evaluation/config/configuration_manager.py` - Configuration loading and management
- `src/evaluation/factories/` - Service factory pattern for STT/TTS creation
- `src/evaluation/pipeline/` - Pipeline construction, execution, and audio processing
- `src/evaluation/services/service_manager.py` - Service lifecycle management
- `src/evaluation/results/results_collector.py` - Result collection and file operations
- `src/evaluation/orchestration/evaluation_orchestrator.py` - Evaluation workflow orchestration
- `src/evaluation/models.py` - Pydantic data models for type safety
- `src/evaluation/metrics_calculator.py` - Metric computation
- `src/evaluation/audio_quality_analyzer.py` - Voice quality assessment

**Audio Quality Scoring & Fine-tuning (MiniCPM-o)**
- `src/audio_scoring/finetune_audio_scorer_reg.py` - LoRA + regression-head fine-tune of MiniCPM-o-4.5 on human-labelled audio (naturalness, noisiness, loudness). Trains the adapter and a linear head jointly under a Huber loss, with a fixed-seed 70/15/15 train/dev/test split and early stopping on dev mean |Pearson r|.
- `src/audio_scoring/score_audio_minicpm_reg.py` - Scores TTS audio with the trained LoRA + regression head, writing results into each evaluation JSON under `voice_quality.minicpm_finetuned`.
- `data/trained_models/best_full/` - Selected checkpoint (adapter + `reg_head.pt`) and its held-out `test_correlation.json` (Pearson r = 0.71/0.71/0.58 for naturalness/noisiness/loudness on n=76).

Reproduce scoring for all TTS audio referenced by the evaluation JSONs:

```bash
python src/audio_scoring/score_audio_minicpm_reg.py --batch \
    --eval-dir output/eval-results/evaluation_output/evaluation \
    --audio-dir output/tts_audio \
    --lora-path data/trained_models/best_full
```

**Visualization**
- `visualize/rank_combinations.py` - Computes the composite STT×TTS ranking (weights: WER 30%, Latency 30%, LLM Judge 10%, Voice LLM 10%, MiniCPM 20%).
- `visualize/plot_cross_method_radar.py` - Renders the cross-method TTS quality radar (NISQA, MOSNet, Librosa Naturalness, LLM Holistic, MiniCPM Baseline/Fine-tuned).
- `visualize/weight_sensitivity.py` - Re-ranks combinations under five weighting profiles (default, accuracy-first, latency-first, voice-quality-first, equal-thirds) to test robustness of the ranking to the weight choice (paper Appendix "Weight Sensitivity").

```bash
# Regenerate the composite ranking
python visualize/rank_combinations.py \
    --eval-dir output/eval-results/evaluation_output/evaluation

# Regenerate the cross-method radar figure
python visualize/plot_cross_method_radar.py \
    --eval-dir output/eval-results/evaluation_output/evaluation \
    --out output/plots/cross-method-radar.png

# Reproduce the weight-sensitivity analysis (add --latex for the paper table)
python visualize/weight_sensitivity.py \
    --eval-dir output/eval-results/evaluation_output/evaluation
```

### Adding New AI Services

1. Add API key to `.env`
2. Update service configuration in `src/core/voice_bot.py`
3. Create evaluation config in `data/`
4. Add evaluation script in `scripts/`

### Running Tests

```bash
pytest tests/
```

### Agent Configuration

Modify `src/core/agent_builder.py` to customize:
- System prompts
- Tool definitions
- Conversation flow
- Context management

## Troubleshooting

**Poor STT accuracy**
- Verify audio quality (16kHz, mono recommended)
- Check microphone permissions
- Try different STT providers

**High latency**
- Check network connectivity
- Consider regional endpoints (set `AWS_REGION`)
- Use streaming STT/TTS services

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

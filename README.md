# Voice Assistant Evaluation Pipeline

A comprehensive evaluation framework for voice assistants that measures speech-to-text accuracy, response quality, and latency metrics across different AI service combinations.

![Architecture Diagram](assets/diagram.jpg)

## Overview

This project provides a complete pipeline for building, testing, and evaluating voice assistants with multiple AI service providers. It enables systematic comparison of different STT/TTS/LLM combinations to optimize for accuracy, latency, and voice quality.

## Architecture

### Core Frameworks
- **Pipecat** - Voice agent orchestration and real-time audio processing
- **Strands Agents** - Multi-agent conversation management and tool calling
- **FastAPI** - Backend API server with WebSocket support

### AI Services
- **Speech-to-Text (STT)**: AWS Transcribe, Deepgram Nova-3, Speechmatics, AssemblyAI, Gladia
- **Text-to-Speech (TTS)**: AWS Polly, Deepgram Aura-2, ElevenLabs, Cartesia, PlayHT, LMNT, Rime
- **Large Language Models (LLM)**: Amazon Bedrock (Claude 3.5 Haiku), OpenAI, Groq

## Project Structure

```
├── src/
│   ├── core/                  # Core voice bot logic
│   │   ├── voice_bot.py
│   │   ├── agent_builder.py
│   │   ├── llm_processor.py
│   │   ├── tts.py
│   │   ├── custom_rtvi_observer.py
│   │   └── nvidia/            # NVIDIA AI services
│   ├── evaluation/            # Evaluation framework
│   │   ├── config/            # Configuration management
│   │   ├── factories/         # Service factory pattern
│   │   ├── pipeline/          # Pipeline components
│   │   ├── services/          # Service management
│   │   ├── results/           # Result collection
│   │   ├── orchestration/     # Evaluation orchestration
│   │   ├── metrics/           # Quality metrics (NISQA, WER, etc.)
│   │   ├── voice_pipeline_evaluator.py
│   │   ├── models.py
│   │   ├── metrics_calculator.py
│   │   ├── audio_quality_analyzer.py
│   │   └── dataset_generator.py
│   └── transport/             # Audio transport layers
├── data/                      # Input data (gitignored)
│   ├── stt_bot_configs/       # STT service configurations
│   ├── tts_bot_configs/       # TTS service configurations
│   ├── voiceassistant_eval_new/  # Test datasets and audio input
│   ├── training_dataset/      # Training data for audio scoring
│   └── trained_models/        # Trained model weights
├── output/                    # Output data (gitignored)
│   ├── eval-results/          # Full evaluation run results
│   ├── scoring_output/        # Audio quality scoring results
│   ├── plots/                 # Generated visualizations
│   ├── tts_audio/             # Generated TTS audio files
│   └── *.json                 # Per-experiment result/evaluation files
├── scripts/                   # Evaluation shell scripts
├── visualize/                 # Plotting and visualization scripts
├── audio_quality_scoring/     # Audio quality scoring tools
├── tests/                     # Unit and integration tests
├── rank_combinations.py       # STT+TTS combination ranking
├── requirements.txt
├── .env.example
└── .gitignore
```

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

### Output

Results are saved in `output/`:
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

# Voice Assistant Evaluation Pipeline

A comprehensive evaluation framework for voice assistants that measures speech-to-text accuracy, response quality, and latency metrics across different AI service combinations.

![](demo.png)

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
├── client/                 # Frontend web application
├── server/                 # Backend services
│   ├── src/
│   │   ├── core/          # Core voice bot logic
│   │   │   ├── voice_bot.py         # Main bot implementation
│   │   │   ├── agent_builder.py     # Agent configuration
│   │   │   └── llm_processor.py     # LLM integration
│   │   ├── evaluation/    # Evaluation framework
│   │   │   ├── voice_pipeline_evaluator.py  # Main evaluator
│   │   │   ├── metrics_calculator.py        # WER, latency metrics
│   │   │   ├── audio_quality_analyzer.py    # Voice quality analysis
│   │   │   └── dataset_downloader.py        # Test data management
│   │   └── transport/     # Audio transport layers
│   │       ├── batch_audio_transport.py     # Batch processing
│   │       └── realtime_transport.py        # Real-time streaming
│   ├── scripts/           # Evaluation scripts
│   ├── evaluation_data/   # Test datasets and configurations
│   ├── evaluation_output/ # Results, metrics, and TTS audio
│   ├── main_server.py     # FastAPI server entry point
│   └── requirements.txt   # Python dependencies
└── .nab_venv/             # Python virtual environment
```

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- AWS account with Transcribe/Polly access
- API keys for desired AI services (see `.env.example`)

### 1. Environment Setup

```bash
# Clone the repository
cd nab-eba

# Copy environment template
cp server/.env.example server/.env

# Edit .env and add your API keys
nano server/.env
```

Required keys:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` - For AWS services
- `DEEPGRAM_API_KEY` - For Deepgram STT/TTS (optional)
- Additional keys for other providers (optional)

### 2. Backend Setup

```bash
cd server/

# Create virtual environment (recommended)
python -m venv .nab_venv
source .nab_venv/bin/activate  # On Windows: .nab_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Whisper models (required for local STT)
python -c "import whisper; whisper.load_model('turbo'); whisper.load_model('large'); whisper.load_model('small')"

# Start the server
python main_server.py
```

Server runs on `http://localhost:8765`

### 3. Frontend Setup

```bash
# Open new terminal
cd client/

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs on `http://localhost:5173`

## Evaluation

### Running Evaluations

Evaluate AWS Transcribe + Polly pipeline:
```bash
cd server/
./scripts/evaluate_aws_pipeline.sh
```

Run comprehensive evaluation across all configured models:
```bash
cd server/
./scripts/evaluate_all_models.sh
```

### NVIDIA Services

For NVIDIA AI services setup and deployment instructions, see [server/src/core/nvidia/README.md](server/src/core/nvidia/README.md).


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

Results are saved in `server/evaluation_output/`:
- `metrics_summary.json` - Aggregated metrics
- `detailed_results.json` - Per-utterance analysis
- `tts_audio/` - Generated audio files for quality review
- `visualizations/` - Charts and graphs

## Development

### Core Components

**Server Entry Point**
- `main_server.py` - FastAPI application with WebSocket endpoints

**Voice Bot Core**
- `src/core/voice_bot.py` - Bot orchestration and pipeline management
- `src/core/agent_builder.py` - Strands agent configuration
- `src/core/llm_processor.py` - LLM integration and prompt management

**Evaluation Pipeline**
- `src/evaluation/voice_pipeline_evaluator.py` - Main evaluation orchestrator
- `src/evaluation/metrics_calculator.py` - Metric computation
- `src/evaluation/audio_quality_analyzer.py` - Voice quality assessment
- `src/evaluation/frame_processor.py` - Audio frame processing

**Transport Layers**
- `src/transport/batch_audio_transport.py` - Batch audio processing
- `src/transport/realtime_transport.py` - Real-time streaming

### Adding New AI Services

1. Add API key to `server/.env`
2. Update service configuration in `src/core/voice_bot.py`
3. Create evaluation config in `evaluation_data/`
4. Add evaluation script in `scripts/`

### Running Tests

```bash
cd server/
pytest tests/
```

## Configuration

### STT/TTS Configuration

Create JSON configs in `evaluation_data/` to define service combinations:

```json
{
  "stt_service": "aws_transcribe",
  "tts_service": "aws_polly",
  "llm_service": "bedrock_claude",
  "test_dataset": "common_voice"
}
```

### Agent Configuration

Modify `src/core/agent_builder.py` to customize:
- System prompts
- Tool definitions
- Conversation flow
- Context management

## Troubleshooting

**Server won't start**
- Check all required API keys are set in `.env`
- Verify Python dependencies: `pip install -r requirements.txt`
- Check port 8765 is available

**Poor STT accuracy**
- Verify audio quality (16kHz, mono recommended)
- Check microphone permissions
- Try different STT providers

**High latency**
- Check network connectivity
- Consider regional endpoints (set `AWS_REGION`)
- Use streaming STT/TTS services

## License

[Add your license information]

## Contributing

[Add contribution guidelines]

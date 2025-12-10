# Voice Assistant Evaluation Pipeline

A comprehensive evaluation framework for voice assistants that measures speech-to-text accuracy, response quality, and latency metrics across different AI service combinations.

![](demo.png)

## Architecture

This system uses the following frameworks and AI services:

- **Pipecat** - Voice agent orchestration framework
- **Strands Agents** - Agentic conversation management
- **Amazon Bedrock** - Anthropic Claude 3.5 Haiku LLM
- **AWS Transcribe** - Speech-to-Text service
- **AWS Polly** - Text-to-Speech service
- **Deepgram Nova-3** - Alternative STT model
- **Deepgram Aura-2** - Alternative TTS model

## Project Structure

```
├── client/                 # Frontend application
├── server/                 # Backend services
│   ├── src/
│   │   ├── core/          # Core application logic
│   │   ├── evaluation/    # Evaluation and testing tools
│   │   └── transport/     # Audio transport layers
│   ├── scripts/           # Evaluation and deployment scripts
│   ├── evaluation_data/   # Test datasets and configurations
│   └── evaluation_output/ # Evaluation results and TTS audio
└── .nab_venv/             # Python virtual environment
```

## Setup Instructions

### Environment Setup

1. Copy the environment example file and add your API keys:
```bash
cp server/.env.example server/.env
```

2. Edit `server/.env` and add your API keys for:
   - AWS credentials (for Transcribe/Polly)
   - Deepgram API key (optional)
   - Anthropic API key (for Claude)

### Start the Backend Server

```bash
cd server/
pip install -r requirements.txt
python main_server.py
```

### Start the Frontend Client

Open a new terminal and run:
```bash
cd client/
npm install
npm run dev
```

Then open your browser to http://localhost:5173/

## Evaluation

### Run Voice Assistant Evaluation

To evaluate the system with AWS Transcribe + Polly:
```bash
cd server/
./scripts/evaluate_aws_pipeline.sh
```

To run all evaluations (multiple STT/TTS combinations):
```bash
cd server/
./scripts/evaluate_all_models.sh
```

### Evaluation Metrics

The system measures:
- **Word Error Rate (WER)** - Speech recognition accuracy
- **Response Quality** - LLM judge scoring on correctness, relevance, completeness, clarity
- **Latency Metrics** - STT, TTS, and total processing times
- **Voice Quality** - Audio output assessment

Results are saved in `server/evaluation_output/` with detailed metrics, analysis, and TTS audio files.

## Development

### Core Components

- `src/core/main_server.py` - Main application server
- `src/core/voice_bot.py` - Voice assistant bot logic
- `src/core/agent_builder.py` - Conversation management
- `src/evaluation/voice_pipeline_evaluator.py` - Evaluation pipeline
- `src/transport/batch_audio_transport.py` - Audio processing transport

### Adding New Evaluations

1. Create STT/TTS config files in `evaluation_data/`
2. Add evaluation script in `scripts/`
3. Update evaluation pipeline in `src/evaluation/`

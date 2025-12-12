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

### Running NVIDIA Evaluations Locally

#### Prerequisites
- NGC API Key generated via [NVIDIA Dashboard](https://catalog.ngc.nvidia.com/)
- AWS Account
- AWS CLI

#### Step 1: Launch the EC2 Instance
**Example Configuration:**
- AMI: ami-0208bd0f236e7f67a (Deep Learning OSS Nvidia Driver AMI GPU TensorFlow 2.18 - Ubuntu 22.04)
- Instance type: g5.xlarge
- Key pair: Proceed without a key pair (using Session Manager)
- IAM instance profile: ec2-nvidia-voice-coach-access
- Security group: Create or select security group with:
  - Port 22: Your IP (for SSH if needed)
  - Port 9000: Your VPC CIDR or client instance security group
  - Port 50051: Your VPC CIDR or client instance security group
- Storage: 150 GB gp3, encrypted, delete on termination

**Important:** Use Ubuntu 22.04 AMI, not Ubuntu 24.04, for better driver stability.

#### Step 2: Deploy Docker Version of NVIDIA Magpie

Run the following steps in order:

1. **Connect to the instance**
```bash
aws ssm start-session --target i-YOUR-INSTANCE-ID --region ap-southeast-2 --profile your-profile-name
```
```bash
sudo su ubuntu
cd ~
```

2. **Check NVIDIA is deployed on the instance**
```bash
nvidia-smi
```

3. **Install NVIDIA Container Toolkit**

   Install Dependencies:
```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends curl gnupg2
```

   Configure Repository:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

   Install Toolkit:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

   Configure & restart Docker:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

   Test GPU access on Docker:
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

4. **Connect to Magpie Docker deployment**

   Set up NGC Key:
```bash
export NGC_API_KEY=your_api_key_here
echo "export NGC_API_KEY=your_api_key_here" >> ~/.bashrc
```

   Login to NGC registry:
```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

   Run NVIDIA Magpie TTS:
```bash
docker run -d --rm --name=magpie-tts-multilingual \
    --runtime=nvidia \
    --gpus all \
    --shm-size=8GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -e NIM_HTTP_API_PORT=9000 \
    -e NIM_GRPC_API_PORT=50051 \
    -p 9000:9000 \
    -p 50051:50051 \
    nvcr.io/nim/nvidia/magpie-tts-multilingual:latest
```

   **Note:**
   - The first deployment can take up to an hour
   - You might see error messages about ONNX or IR - ignore these and wait for the deployment

5. **Check deployment has finished**

   In your EC2 instance:
```bash
curl -X 'GET' 'http://localhost:9000/v1/health/ready'
```

   When you receive `{"status":"ready"}`, the service is operational.

#### Step 3: Connect to Local Deployment

Add NVIDIA private IP to `.env` file to connect.


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

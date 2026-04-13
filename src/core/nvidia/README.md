# NVIDIA Services Setup

This guide covers setting up NVIDIA AI services for the voice assistant evaluation pipeline.

## Services Overview

- **STT (Speech-to-Text)**: NVIDIA NeMo Parakeet - High-accuracy ASR model for audio transcription
- **TTS (Text-to-Speech)**: NVIDIA Magpie - Multilingual neural TTS for audio generation

## NVIDIA STT (Speech-to-Text) - NeMo Parakeet

### Local Installation

Install NeMo dependencies:
```bash
pip install lightning lhotse einops sentencepiece pyannote.audio webdataset editdistance IPython
```

### Configuration

The pipeline uses NVIDIA's Parakeet model via NeMo toolkit for speech recognition:

**Model**: `nvidia/parakeet-tdt-0.6b-v3`
- High-accuracy multilingual ASR
- Optimized for conversational speech
- Supports batch and streaming processing

### Batch Processing Architecture

For evaluation scenarios, the pipeline uses batch processing instead of streaming:

```python
# Configuration in evaluation_data/stt_bot_configs/nvidia_riva_config.json
{
  "stt_service": "NeMoBatchSTT",
  "model_name": "nvidia/parakeet-tdt-0.6b-v3",
  "tts_service": "livekit_tts",
  "llm_service": "bedrock_claude"
}
```

**Key Features**:
- Pre-processes entire audio files before pipeline execution
- Avoids Pipecat timing issues with EndFrame delivery
- Better reliability for single-turn Q&A evaluation
- Eliminates need for context aggregators in batch scenarios

### Audio Processing

**Input Requirements**:
- Sample rate: 16kHz recommended
- Format: WAV, mono channel
- Duration: Up to 30 seconds per utterance

**Processing Flow**:
1. Audio file loaded and preprocessed
2. NeMo model performs transcription
3. Results injected into pipeline as UserTranscriptionMessage
4. Pipeline continues with LLM processing

### Troubleshooting

**Import Errors**:
```bash
# If NeMo import fails, install missing dependencies
pip install omegaconf hydra-core
```

**Memory Issues**:
- Use smaller batch sizes for large audio files
- Consider GPU memory limits with concurrent processing

**Model Download**:
- First run downloads model weights (~600MB)
- Subsequent runs use cached model

## NVIDIA TTS (Text-to-Speech) - Magpie

### Running NVIDIA TTS Evaluations Locally

### Prerequisites
- NGC API Key generated via [NVIDIA Dashboard](https://catalog.ngc.nvidia.com/)
- AWS Account
- AWS CLI

### Step 1: Launch the EC2 Instance
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

### Step 2: Deploy Docker Version of NVIDIA Magpie

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

### Step 3: Connect to Local Deployment

Add NVIDIA private IP to `.env` file to connect.
# Voice Assistant Evaluation Metrics

## Overview
This evaluation framework measures voice assistant performance across accuracy, quality, and latency dimensions using both objective metrics and AI-based subjective evaluation.

## Metrics Categories

### 🎯 STT (Speech-to-Text) Metrics
- **WER (Word Error Rate)**: Percentage of incorrectly transcribed words (0-100%, lower = better)
- **STT Latency**: Time from audio input to transcription output (milliseconds)

### 🤖 LLM Response Quality (LLM-as-Judge)
- **Correctness**: Factual accuracy (0-10 scale)
- **Relevance**: Alignment with user question (0-10 scale)
- **Completeness**: Coverage of key information (0-10 scale)
- **Clarity**: Communication effectiveness (0-10 scale)
- **Overall**: Combined response quality (0-10 scale)

### 🔊 TTS Voice Quality (Objective - Librosa)

#### Fluency Metrics
- **Speaking Rate**: Syllables per second (optimal: 3-5 sps)
- **Silence Ratio**: Proportion of pauses (0-1, lower = better)
- **Energy Consistency**: Audio energy stability (0-1, higher = better)
- **Fluency Score**: Combined fluency rating (0-10 scale)

#### Naturalness Metrics
- **Pitch Mean**: Average fundamental frequency (Hz)
- **Pitch Variation**: Coefficient of variation (optimal: 10-20%)
- **Pitch Range**: Max - min pitch (Hz)
- **Spectral Centroid**: Voice brightness/timbre (Hz)
- **Naturalness Score**: Combined naturalness rating (0-10 scale)

#### Tone Quality Metrics
- **Clarity**: Spectral contrast (higher = clearer articulation)
- **Smoothness**: Speech continuity (0-1, higher = smoother)
- **HNR**: Harmonics-to-Noise Ratio in dB (higher = cleaner voice)
- **Tone Score**: Combined tone quality (0-10 scale)

### 🎭 TTS Voice Quality (Subjective - LLM Judge)
- **LLM Fluency**: AI-assessed speech smoothness (0-10 scale)
- **LLM Naturalness**: AI-assessed human-likeness (0-10 scale)
- **LLM Tone**: AI-assessed pleasantness (0-10 scale)
- **LLM Overall**: AI-assessed overall quality (0-10 scale)

### 🏆 TTS Voice Quality (NISQA - Neural Speech Quality Assessment)
- **MOS**: Mean Opinion Score for overall quality (1-5 scale, higher = better)
- **Noisiness**: Background noise and artifacts (1-5 scale, lower = better)
- **Coloration**: Spectral distortions and tonal quality (1-5 scale, lower = better)
- **Discontinuity**: Speech interruptions and glitches (1-5 scale, lower = better)
- **Loudness**: Volume appropriateness (1-5 scale, optimal ~3)

### 📊 TTS Voice Quality (SpeechMetrics - Objective Measures)
- **MOSNet Score**: Neural MOS prediction (1-5 scale, higher = better)
- **SRMR Score**: Speech-to-Reverberation Modulation energy Ratio (higher = cleaner)

### ⚡ Performance Metrics
- **STT Latency**: Speech-to-text processing time (ms)
- **TTS Latency**: Text-to-speech generation time (ms)
- **Total Latency**: End-to-end response time (ms)

## Service Combinations

**Progress**: 22 of 101 combinations complete (22% coverage)
- ✅ **Available Scripts**: 22
- ❌ **Missing Scripts**: 79
- ✅ **Groq TTS Evaluation**: Complete (8 STT services tested)
- ✅ **Cartesia TTS Evaluation**: Complete (8 STT services tested)

*Note: This table reflects current service configurations. Update counts when new STT/TTS service configs are added to `evaluation_data/` directories.*

| STT Service | TTS Service | Script Status | Script Name |
|-------------|-------------|---------------|-------------|
| **AWS Transcribe** | AWS Polly | ✅ **Available** | `test_aws_transcribe_polly.sh` |
| AWS Transcribe | Cartesia | ✅ **Available** | `test_aws_transcribe_cartesia.sh` |
| AWS Transcribe | Deepgram Aura | ✅ **Available** | `test_aws_transcribe_deepgram_aura.sh` |
| AWS Transcribe | Groq | ✅ **Available** | `test_aws_transcribe_groq.sh` |
| AWS Transcribe | ElevenLabs | ❌ **Missing** | `test_aws_transcribe_elevenlabs.sh` |
| AWS Transcribe | LMNT | ❌ **Missing** | `test_aws_transcribe_lmnt.sh` |
| AWS Transcribe | NVIDIA Magpie | ❌ **Missing** | `test_aws_transcribe_nvidia_magpie.sh` |
| AWS Transcribe | OpenAI TTS | ❌ **Missing** | `test_aws_transcribe_openai_tts.sh` |
| AWS Transcribe | OpenAI TTS HD | ❌ **Missing** | `test_aws_transcribe_openai_tts_hd.sh` |
| AWS Transcribe | PlayHT | ❌ **Missing** | `test_aws_transcribe_playht.sh` |
| AWS Transcribe | Rime | ❌ **Missing** | `test_aws_transcribe_rime.sh` |
| **Deepgram Nova-2** | AWS Polly | ✅ **Available** | `test_deepgram_nova2_polly.sh` |
| Deepgram Nova-2 | Groq | ✅ **Available** | `test_deepgram_nova2_groq.sh` |
| Deepgram Nova-2 | Cartesia | ✅ **Available** | `test_deepgram_nova2_cartesia.sh` |
| Deepgram Nova-2 | Deepgram Aura | ❌ **Missing** | `test_deepgram_nova2_deepgram_aura.sh` |
| Deepgram Nova-2 | ElevenLabs | ❌ **Missing** | `test_deepgram_nova2_elevenlabs.sh` |
| Deepgram Nova-2 | LMNT | ❌ **Missing** | `test_deepgram_nova2_lmnt.sh` |
| Deepgram Nova-2 | NVIDIA Magpie | ❌ **Missing** | `test_deepgram_nova2_nvidia_magpie.sh` |
| Deepgram Nova-2 | OpenAI TTS | ❌ **Missing** | `test_deepgram_nova2_openai_tts.sh` |
| Deepgram Nova-2 | OpenAI TTS HD | ❌ **Missing** | `test_deepgram_nova2_openai_tts_hd.sh` |
| Deepgram Nova-2 | PlayHT | ❌ **Missing** | `test_deepgram_nova2_playht.sh` |
| Deepgram Nova-2 | Rime | ❌ **Missing** | `test_deepgram_nova2_rime.sh` |
| **Deepgram Nova-3** | AWS Polly | ✅ **Available** | `test_deepgram_nova3_polly.sh` |
| Deepgram Nova-3 | Groq | ✅ **Available** | `test_deepgram_nova3_groq.sh` |
| Deepgram Nova-3 | Cartesia | ✅ **Available** | `test_deepgram_nova3_cartesia.sh` |
| Deepgram Nova-3 | Deepgram Aura | ❌ **Missing** | `test_deepgram_nova3_deepgram_aura.sh` |
| Deepgram Nova-3 | ElevenLabs | ❌ **Missing** | `test_deepgram_nova3_elevenlabs.sh` |
| Deepgram Nova-3 | LMNT | ❌ **Missing** | `test_deepgram_nova3_lmnt.sh` |
| Deepgram Nova-3 | NVIDIA Magpie | ❌ **Missing** | `test_deepgram_nova3_nvidia_magpie.sh` |
| Deepgram Nova-3 | OpenAI TTS | ❌ **Missing** | `test_deepgram_nova3_openai_tts.sh` |
| Deepgram Nova-3 | OpenAI TTS HD | ❌ **Missing** | `test_deepgram_nova3_openai_tts_hd.sh` |
| Deepgram Nova-3 | PlayHT | ❌ **Missing** | `test_deepgram_nova3_playht.sh` |
| Deepgram Nova-3 | Rime | ❌ **Missing** | `test_deepgram_nova3_rime.sh` |
| **Whisper Large** | AWS Polly | ✅ **Available** | `test_whisper_large_polly.sh` |
| Whisper Large | Groq | ✅ **Available** | `test_whisper_large_groq.sh` |
| Whisper Large | Cartesia | ✅ **Available** | `test_whisper_large_cartesia.sh` |
| Whisper Large | Deepgram Aura | ❌ **Missing** | `test_whisper_large_deepgram_aura.sh` |
| Whisper Large | ElevenLabs | ❌ **Missing** | `test_whisper_large_elevenlabs.sh` |
| Whisper Large | LMNT | ❌ **Missing** | `test_whisper_large_lmnt.sh` |
| Whisper Large | NVIDIA Magpie | ❌ **Missing** | `test_whisper_large_nvidia_magpie.sh` |
| Whisper Large | OpenAI TTS | ❌ **Missing** | `test_whisper_large_openai_tts.sh` |
| Whisper Large | OpenAI TTS HD | ❌ **Missing** | `test_whisper_large_openai_tts_hd.sh` |
| Whisper Large | PlayHT | ❌ **Missing** | `test_whisper_large_playht.sh` |
| Whisper Large | Rime | ❌ **Missing** | `test_whisper_large_rime.sh` |
| **Whisper Small** | AWS Polly | ✅ **Available** | `test_whisper_small_polly.sh` |
| Whisper Small | Groq | ✅ **Available** | `test_whisper_small_groq.sh` |
| Whisper Small | Cartesia | ✅ **Available** | `test_whisper_small_cartesia.sh` |
| Whisper Small | Deepgram Aura | ❌ **Missing** | `test_whisper_small_deepgram_aura.sh` |
| Whisper Small | ElevenLabs | ❌ **Missing** | `test_whisper_small_elevenlabs.sh` |
| Whisper Small | LMNT | ❌ **Missing** | `test_whisper_small_lmnt.sh` |
| Whisper Small | NVIDIA Magpie | ❌ **Missing** | `test_whisper_small_nvidia_magpie.sh` |
| Whisper Small | OpenAI TTS | ❌ **Missing** | `test_whisper_small_openai_tts.sh` |
| Whisper Small | OpenAI TTS HD | ❌ **Missing** | `test_whisper_small_openai_tts_hd.sh` |
| Whisper Small | PlayHT | ❌ **Missing** | `test_whisper_small_playht.sh` |
| Whisper Small | Rime | ❌ **Missing** | `test_whisper_small_rime.sh` |
| **Whisper Turbo** | AWS Polly | ✅ **Available** | `test_whisper_turbo_polly.sh` |
| Whisper Turbo | Groq | ✅ **Available** | `test_whisper_turbo_groq.sh` |
| Whisper Turbo | Cartesia | ✅ **Available** | `test_whisper_turbo_cartesia.sh` |
| Whisper Turbo | Deepgram Aura | ❌ **Missing** | `test_whisper_turbo_deepgram_aura.sh` |
| Whisper Turbo | ElevenLabs | ❌ **Missing** | `test_whisper_turbo_elevenlabs.sh` |
| Whisper Turbo | LMNT | ❌ **Missing** | `test_whisper_turbo_lmnt.sh` |
| Whisper Turbo | NVIDIA Magpie | ❌ **Missing** | `test_whisper_turbo_nvidia_magpie.sh` |
| Whisper Turbo | OpenAI TTS | ❌ **Missing** | `test_whisper_turbo_openai_tts.sh` |
| Whisper Turbo | OpenAI TTS HD | ❌ **Missing** | `test_whisper_turbo_openai_tts_hd.sh` |
| Whisper Turbo | PlayHT | ❌ **Missing** | `test_whisper_turbo_playht.sh` |
| Whisper Turbo | Rime | ❌ **Missing** | `test_whisper_turbo_rime.sh` |
| **NVIDIA Parakeet** | AWS Polly | ❌ **Missing** | `test_nvidia_parakeet_polly.sh` |
| NVIDIA Parakeet | Cartesia | ✅ **Available** | `test_nvidia_parakeet_cartesia.sh` |
| NVIDIA Parakeet | Deepgram Aura | ❌ **Missing** | `test_nvidia_parakeet_deepgram_aura.sh` |
| NVIDIA Parakeet | ElevenLabs | ❌ **Missing** | `test_nvidia_parakeet_elevenlabs.sh` |
| NVIDIA Parakeet | LMNT | ❌ **Missing** | `test_nvidia_parakeet_lmnt.sh` |
| NVIDIA Parakeet | Groq | ✅ **Available** | `test_nvidia_parakeet_groq.sh` |
| NVIDIA Parakeet | NVIDIA Magpie | ✅ **Available** | `test_nvidia_parakeet_nvidia_magpie.sh` |
| NVIDIA Parakeet | NVIDIA Riva TTS | ❌ **Missing** | `test_nvidia_parakeet_nvidia_riva_tts.sh` |
| NVIDIA Parakeet | OpenAI TTS | ❌ **Missing** | `test_nvidia_parakeet_openai_tts.sh` |
| NVIDIA Parakeet | OpenAI TTS HD | ❌ **Missing** | `test_nvidia_parakeet_openai_tts_hd.sh` |
| NVIDIA Parakeet | PlayHT | ❌ **Missing** | `test_nvidia_parakeet_playht.sh` |
| NVIDIA Parakeet | Rime | ❌ **Missing** | `test_nvidia_parakeet_rime.sh` |

| **AssemblyAI** | AWS Polly | ❌ **Missing** | `test_assemblyai_polly.sh` |
| AssemblyAI | Cartesia | ❌ **Missing** | `test_assemblyai_cartesia.sh` |
| AssemblyAI | Deepgram Aura | ❌ **Missing** | `test_assemblyai_deepgram_aura.sh` |
| AssemblyAI | ElevenLabs | ❌ **Missing** | `test_assemblyai_elevenlabs.sh` |
| AssemblyAI | LMNT | ❌ **Missing** | `test_assemblyai_lmnt.sh` |
| AssemblyAI | NVIDIA Magpie | ❌ **Missing** | `test_assemblyai_nvidia_magpie.sh` |
| AssemblyAI | OpenAI TTS | ❌ **Missing** | `test_assemblyai_openai_tts.sh` |
| AssemblyAI | OpenAI TTS HD | ❌ **Missing** | `test_assemblyai_openai_tts_hd.sh` |
| AssemblyAI | PlayHT | ❌ **Missing** | `test_assemblyai_playht.sh` |
| AssemblyAI | Rime | ❌ **Missing** | `test_assemblyai_rime.sh` |
| **Gladia** | AWS Polly | ❌ **Missing** | `test_gladia_polly.sh` |
| Gladia | Cartesia | ❌ **Missing** | `test_gladia_cartesia.sh` |
| Gladia | Deepgram Aura | ❌ **Missing** | `test_gladia_deepgram_aura.sh` |
| Gladia | ElevenLabs | ❌ **Missing** | `test_gladia_elevenlabs.sh` |
| Gladia | LMNT | ❌ **Missing** | `test_gladia_lmnt.sh` |
| Gladia | NVIDIA Magpie | ❌ **Missing** | `test_gladia_nvidia_magpie.sh` |
| Gladia | OpenAI TTS | ❌ **Missing** | `test_gladia_openai_tts.sh` |
| Gladia | OpenAI TTS HD | ❌ **Missing** | `test_gladia_openai_tts_hd.sh` |
| Gladia | PlayHT | ❌ **Missing** | `test_gladia_playht.sh` |
| Gladia | Rime | ❌ **Missing** | `test_gladia_rime.sh` |

### Summary
- **Total Possible Combinations**: 101
- **Available Scripts**: 22 (22%)
- **Missing Scripts**: 79 (78%)

**Available STT Services**: 6 (AWS Transcribe, Deepgram Nova-2/3, Whisper Large/Small/Turbo, NVIDIA Parakeet, AssemblyAI, Gladia)
**Available TTS Services**: 11 (AWS Polly, Cartesia, Deepgram Aura, ElevenLabs, LMNT, NVIDIA Magpie, NVIDIA Riva TTS, OpenAI TTS/HD, PlayHT, Rime)

**Unified Service Providers**:
- **NVIDIA**: Parakeet STT + Magpie TTS, Parakeet STT + Riva TTS
- **Deepgram**: Nova STT + Aura TTS (separate services)
- **OpenAI**: Whisper STT (local) + OpenAI TTS (API)

### 📊 Generated Visualizations

#### Individual Service Analysis
- **STT Performance**: `stt_latency_vs_wer.png` - Latency vs accuracy scatter plot
- **TTS Performance**: `tts_latency_vs_quality.png` - Latency vs quality trade-off
- **Voice Quality**: Individual charts for fluency, naturalness, tone metrics
- **LLM Judge**: Subjective quality comparisons across services

#### STT+TTS Combination Analysis
- **Performance Matrix**: `combination_performance_matrix.png` - 4-panel heatmap showing WER, quality, STT latency, TTS latency by combination
- **Accuracy vs Quality**: `combination_accuracy_vs_quality.png` - Trade-off analysis with error ellipses
- **Speed vs Quality**: `combination_speed_vs_quality.png` - Latency vs quality trade-off with variance indicators

**Key Features**:
- **Error Analysis**: Standard deviation ellipses show performance consistency
- **Provider Grouping**: Color-coded by STT provider, distinct markers by TTS service
- **Trade-off Visualization**: Clear identification of optimal combinations for different use cases

## Recent Improvements

### ✅ Voice Activity Detection (VAD) Integration
- **Problem Solved**: Multiple LLM responses causing concatenated, garbled output
- **Solution**: VAD-based speech end detection for streaming STT services
- **Compatibility**: Automatic detection of batch vs streaming STT services
  - **Streaming STT** (Deepgram, AWS): Uses VAD with 1-second silence detection
  - **Batch STT** (Whisper): Uses 5-second timeout after transcription
- **Result**: Single, clean LLM responses with perfect text/audio matching

### 🎯 Groq TTS Evaluation Complete
- **Coverage**: 8 STT services × Groq TTS = 8 combinations tested
- **Services**: AWS Transcribe, Deepgram Nova-2/3, Whisper Large/Small/Turbo, NVIDIA Parakeet
- **Metrics**: Full evaluation including WER, LLM judge scores, latency analysis
- **Quality**: Clean single responses, proper TTS audio generation

### 📊 Enhanced Visualization Suite
- **Combination Analysis**: New plots showing STT+TTS performance relationships
- **Error Analysis**: Standard deviation ellipses for performance consistency
- **Trade-off Charts**: Clear visualization of accuracy vs quality vs speed

## Key Files
- `voice_pipeline_evaluator.py` - Main evaluation orchestrator with VAD integration
- `metrics_calculator.py` - WER and LLM judge scoring
- `audio_quality_analyzer.py` - Voice quality analysis (librosa + LLM)
- `frame_processor.py` - Pipeline timing, text collection, and Unicode cleaning
- `../visualize/plot_evaluation_results.py` - Comprehensive visualization suite
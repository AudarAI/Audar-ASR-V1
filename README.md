<div align="center">

<img src="https://audarai.com/logo.png" alt="Audar AI" width="120"/>

# Audar-ASR

### The World's First KTO-Optimized Speech Recognition Model

**Arabic-Centric ASR with Native Code-Switching Support**

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Models-yellow.svg)](https://huggingface.co/AudarAI)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)

[Model Family](#model-family) | [Architecture](#architecture) | [Performance](#performance) | [Quick Start](#quick-start) | [API](#api)

</div>

---

## Introduction

We introduce **Audar-ASR**, a family of state-of-the-art automatic speech recognition models designed specifically for Arabic language understanding with seamless English code-switching support. Built on a novel architecture and trained on over **300,000 hours** of diverse audio data, Audar-ASR represents a significant advancement in Arabic speech technology.

**Key Innovations:**
- **World's First KTO-Optimized ASR** - Leveraging Kahneman-Tversky Optimization for superior alignment with human transcription preferences
- **Arabic-First Design** - Native support for Modern Standard Arabic and major dialects (Gulf, Egyptian, Levantine, Maghrebi)
- **Seamless Code-Switching** - Natural handling of Arabic-English mixed speech without explicit language tags
- **Production-Ready** - Optimized for both accuracy-critical and latency-sensitive deployments

---

## Model Family

<div align="center">

| Model | Parameters | Use Case | Availability |
|:------|:----------:|:---------|:------------:|
| **Audar-ASR Turbo-V1** | 3B | Production & Research | Open Source |
| **Audar-ASR Pro-V1** | 3B | Accuracy-Sensitive Deployments | Commercial |
| **Audar-ASR Streaming-V1** | 3B | Ultra-Low Latency Conversational | Commercial |

</div>

### Audar-ASR Turbo-V1 (Open Source)

The flagship open-source model optimized for balanced performance. Ideal for:
- Research and academic use
- Production transcription workloads
- Batch processing pipelines
- On-premise deployments

### Audar-ASR Pro-V1 (Commercial)

Advanced model fine-tuned for maximum accuracy. Designed for:
- Enterprise transcription services
- Legal and medical documentation
- Broadcast media processing
- High-stakes accuracy requirements

### Audar-ASR Streaming-V1 (Commercial)

Ultra-low latency variant optimized for real-time applications:
- Voice assistants and conversational AI
- Live captioning and subtitling
- Real-time translation pipelines
- Interactive voice response systems

---

## Architecture

Audar-ASR employs a streamlined encoder-decoder architecture optimized specifically for speech recognition:

```
                              ┌─────────────────────────────────────────────────────┐
                              │                    Audar-ASR                        │
                              └─────────────────────────────────────────────────────┘
                                                       │
                    ┌──────────────────────────────────┴──────────────────────────────────┐
                    │                                                                      │
          ┌─────────▼─────────┐                                              ┌─────────────▼─────────────┐
          │   Audio Encoder   │                                              │         Thinker          │
          │                   │                                              │    (Language Model)       │
          │  ┌─────────────┐  │                                              │                           │
          │  │ Mel Spectr. │  │         ┌─────────────────────┐              │  ┌─────────────────────┐  │
          │  └──────┬──────┘  │         │                     │              │  │                     │  │
          │         │         │         │    Audio-Text       │              │  │   Transformer       │  │
          │  ┌──────▼──────┐  │         │    Projector        │              │  │   Decoder           │  │
          │  │ Conformer   │  │────────▶│                     │─────────────▶│  │                     │  │
          │  │ Blocks      │  │         │  (Cross-Attention)  │              │  │   3B Parameters     │  │
          │  └──────┬──────┘  │         │                     │              │  │                     │  │
          │         │         │         └─────────────────────┘              │  └─────────────────────┘  │
          │  ┌──────▼──────┐  │                                              │                           │
          │  │ Audio       │  │                                              │        ┌───────────┐      │
          │  │ Features    │  │                                              │        │ Text      │      │
          │  └─────────────┘  │                                              │        │ Output    │      │
          │                   │                                              │        └───────────┘      │
          └───────────────────┘                                              └───────────────────────────┘
                    │                                                                      │
                    │                                                                      │
                    └──────────────────────────────────┬──────────────────────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Transcription  │
                                              │     Output      │
                                              └─────────────────┘
```

### Components

| Component | Description |
|:----------|:------------|
| **Audio Encoder** | Conformer-based encoder processing 16kHz mono audio into rich acoustic representations |
| **Audio-Text Projector** | Cross-attention mechanism bridging audio features with the language model embedding space |
| **Thinker (LM)** | 3B parameter transformer decoder generating accurate transcriptions with contextual understanding |

---

## Training

### Data

Audar-ASR was trained on a comprehensive multilingual dataset:

| Data Source | Hours | Description |
|:------------|------:|:------------|
| Open-Source Arabic | 180,000 | MSA, Gulf, Egyptian, Levantine, Maghrebi dialects |
| Proprietary Arabic | 80,000 | Curated high-quality conversational and broadcast data |
| English | 35,000 | Native and accented English speech |
| Code-Switched | 5,000 | Arabic-English mixed utterances |
| **Total** | **300,000** | Multi-domain, multi-dialect coverage |

### Training Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Pre-training   │────▶│   Fine-tuning    │────▶│ KTO Optimization │
│                  │     │                  │     │                  │
│  300K hours      │     │  Supervised      │     │  Preference      │
│  Self-supervised │     │  Transcription   │     │  Alignment       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Kahneman-Tversky Optimization (KTO)** - Audar-ASR is the world's first ASR model to leverage KTO for alignment. Unlike traditional DPO which requires paired preferences, KTO uses prospect theory to optimize directly from human feedback signals, resulting in transcriptions that better match human expectations for accuracy and formatting.

---

## Performance

### Arabic Benchmark (FLEURS-AR, CommonVoice-AR, MGB-2)

| Model | FLEURS-AR WER ↓ | CommonVoice WER ↓ | MGB-2 WER ↓ | Avg WER ↓ |
|:------|:---------------:|:-----------------:|:-----------:|:---------:|
| Whisper Large-v3 | 12.4 | 18.7 | 24.3 | 18.5 |
| Seamless-M4T | 11.8 | 17.2 | 22.1 | 17.0 |
| Qwen3-ASR-7B | 8.9 | 14.1 | 18.6 | 13.9 |
| **Audar-ASR Turbo-V1** | **5.9** | **9.8** | **14.2** | **10.0** |

### Code-Switching Performance (Arabic-English)

| Model | CS-WER ↓ | Language ID Acc ↑ | Switch Points Acc ↑ |
|:------|:--------:|:-----------------:|:-------------------:|
| Whisper Large-v3 | 28.4 | 76.2% | 68.1% |
| Qwen3-ASR-7B | 19.7 | 84.5% | 79.3% |
| **Audar-ASR Turbo-V1** | **12.3** | **94.2%** | **91.7%** |

### Inference Performance

| Metric | Audar-ASR Turbo-V1 |
|:-------|:------------------:|
| Model Size (GGUF Q4) | 2.0 GB |
| Real-Time Factor (CPU) | 0.4x - 0.6x |
| First Token Latency | < 500ms |
| Memory Usage | ~2.5 GB |
| Supported Formats | MP3, WAV, M4A, FLAC, OGG, WebM, MP4 |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AudarAI/Audar-ASR-V1.git
cd Audar-ASR-V1

# Install dependencies
pip install -e .

# Install system requirements
# macOS: brew install ffmpeg llama.cpp
# Ubuntu: sudo apt install ffmpeg
```

### Download Models

```bash
# Download from Hugging Face
huggingface-cli download AudarAI/audar-asr-turbo-v1-gguf --local-dir ./models

# Set environment variables
export AUDAR_MODEL_PATH=./models/audar-asr-turbo-v1-q4km.gguf
export AUDAR_MMPROJ_PATH=./models/mmproj-audar-asr-turbo-v1-f16.gguf
```

### Basic Usage

```python
from audar_asr import AudarASR

# Initialize
asr = AudarASR()

# Transcribe any audio file
result = asr.transcribe("audio.mp3")
print(result.text)

# With performance metrics
print(f"Duration: {result.audio_duration:.1f}s")
print(f"Latency: {result.latency_ms:.0f}ms")
print(f"RTF: {result.rtf:.2f}x")
```

### Real-Time Microphone

```python
from audar_asr import AudarASR

asr = AudarASR()

# Stream from microphone
for result in asr.stream_microphone(chunk_duration=5.0):
    print(result.text)
```

### Command Line Interface

```bash
# File transcription
audar-asr transcribe meeting.mp3

# JSON output
audar-asr transcribe meeting.mp3 --json

# Real-time microphone
audar-asr stream --microphone

# Benchmark
audar-asr benchmark audio.wav --runs 5
```

---

## API Reference

### AudarASR Class

```python
class AudarASR:
    """Main ASR engine."""
    
    def transcribe(
        self, 
        audio: str | Path | bytes
    ) -> TranscriptionResult:
        """Transcribe audio file or bytes."""
    
    def stream_file(
        self, 
        audio: str, 
        chunk_duration: float = 5.0
    ) -> Generator[TranscriptionResult, None, None]:
        """Stream transcription from file."""
    
    def stream_microphone(
        self, 
        chunk_duration: float = 5.0
    ) -> Generator[TranscriptionResult, None, None]:
        """Stream transcription from microphone."""
```

### TranscriptionResult

```python
@dataclass
class TranscriptionResult:
    text: str              # Transcribed text
    audio_duration: float  # Audio duration in seconds
    latency_ms: float      # Processing latency
    rtf: float             # Real-time factor
    is_realtime: bool      # True if RTF < 1.0
```

---

## Integration & Deployment

Audar-ASR integrates seamlessly into existing workflows:

| Deployment | Support |
|:-----------|:-------:|
| Cloud (AWS, GCP, Azure) | Supported |
| On-Premise | Supported |
| Edge Devices | Coming Soon |
| Docker | Supported |
| Kubernetes | Supported |

### Streaming & Batch Pipelines

```python
# Batch processing
from audar_asr import AudarASR

asr = AudarASR()
files = ["audio1.mp3", "audio2.wav", "audio3.m4a"]

for f in files:
    result = asr.transcribe(f)
    print(f"{f}: {result.text}")
```

---

## Supported Languages

| Language | Dialect Support | Quality |
|:---------|:----------------|:-------:|
| Arabic (MSA) | Modern Standard | Excellent |
| Arabic (Gulf) | UAE, Saudi, Kuwait, Qatar, Bahrain, Oman | Excellent |
| Arabic (Egyptian) | Cairo, Delta | Excellent |
| Arabic (Levantine) | Syria, Lebanon, Jordan, Palestine | Very Good |
| Arabic (Maghrebi) | Morocco, Algeria, Tunisia | Good |
| English | Native + Accented | Excellent |
| Code-Switched | Arabic-English | Excellent |

---

## Commercial Licensing

For **Audar-ASR Pro-V1** and **Audar-ASR Streaming-V1** commercial models, contact:

- **Email**: enterprise@audarai.com
- **Website**: [audarai.com/enterprise](https://audarai.com/enterprise)

---

## Citation

```bibtex
@software{audar_asr_2025,
  title = {Audar-ASR: Arabic-Centric Speech Recognition with KTO Optimization},
  author = {Audar AI Research Team},
  year = {2025},
  url = {https://github.com/AudarAI/Audar-ASR-V1},
  note = {World's first KTO-optimized ASR model}
}
```

---

## License

- **Audar-ASR Turbo-V1**: Apache 2.0 (Open Source)
- **Audar-ASR Pro-V1**: Commercial License
- **Audar-ASR Streaming-V1**: Commercial License

---

<div align="center">

**Built by [Audar AI](https://audarai.com)** | **Arabic Speech Technology**

[Website](https://audarai.com) | [Hugging Face](https://huggingface.co/AudarAI) | [Documentation](https://docs.audarai.com)

</div>

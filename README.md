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
| **Audar-ASR Pro-V1** | 7B | Accuracy-Sensitive Deployments | Commercial |
| **Audar-ASR Streaming-V1** | 3B | Ultra-Low Latency Conversational | Commercial |

</div>

### Audar-ASR Turbo-V1 (Open Source)

The flagship open-source model optimized for balanced performance. Ideal for:
- Research and academic use
- Production transcription workloads
- Batch processing pipelines
- On-premise deployments

### Audar-ASR Pro-V1 (Commercial)

Our most accurate 7B parameter model, fine-tuned for maximum Arabic accuracy. Designed for:
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

Audar-ASR employs a purpose-built encoder-decoder architecture optimized specifically for Arabic speech recognition. The system processes raw audio through a multi-stage pipeline designed for accuracy and efficiency.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              AUDAR-ASR ARCHITECTURE                                 │
│                         Arabic-Centric Speech Recognition                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT STAGE
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────┐    ┌───────────────────┐    ┌──────────────────────────────┐    │
│  │  Raw Audio    │───▶│  Audio Processor  │───▶│  Mel-Spectrogram Features    │    │
│  │  (Any Format) │    │  16kHz Mono PCM   │    │  80-dim, 25ms frames         │    │
│  └───────────────┘    └───────────────────┘    └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                               ENCODER STAGE
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │                        CONFORMER AUDIO ENCODER                              │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  │
│   │  │ Conv        │  │ Conformer   │  │ Conformer   │  │ Conformer   │        │  │
│   │  │ Subsampling │─▶│ Block ×6    │─▶│ Block ×6    │─▶│ Block ×6    │        │  │
│   │  │ (4x)        │  │ (Self-Attn) │  │ (Conv)      │  │ (FFN)       │        │  │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │  │
│   │                                                                             │  │
│   │  • Relative Positional Encoding    • 512-dim hidden size                   │  │
│   │  • 8 Attention Heads               • Macaron-style FFN                     │  │
│   │  • Convolution Kernel: 31          • Dropout: 0.1                          │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                            PROJECTION STAGE
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │                      AUDIO-TEXT ALIGNMENT MODULE                            │  │
│   │                                                                             │  │
│   │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │  │
│   │  │  Linear          │    │  Cross-Modal     │    │  Layer           │      │  │
│   │  │  Projection      │───▶│  Attention       │───▶│  Normalization   │      │  │
│   │  │  (512 → 2048)    │    │  (8 heads)       │    │                  │      │  │
│   │  └──────────────────┘    └──────────────────┘    └──────────────────┘      │  │
│   │                                                                             │  │
│   │  • Bridges acoustic and linguistic representations                         │  │
│   │  • Learned query embeddings for text generation                            │  │
│   │  • Residual connections preserve audio fidelity                            │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              DECODER STAGE
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │                    TRANSFORMER LANGUAGE DECODER                             │  │
│   │                                                                             │  │
│   │  ┌─────────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Turbo-V1 (3B)              │  Pro-V1 (7B)                          │   │  │
│   │  │  • 32 Decoder Layers        │  • 48 Decoder Layers                  │   │  │
│   │  │  • 2048 Hidden Dimension    │  • 4096 Hidden Dimension              │   │  │
│   │  │  • 16 Attention Heads       │  • 32 Attention Heads                 │   │  │
│   │  │  • 8192 FFN Dimension       │  • 14336 FFN Dimension                │   │  │
│   │  │  • RoPE Positional Enc.     │  • RoPE Positional Enc.               │   │  │
│   │  │  • SwiGLU Activation        │  • SwiGLU Activation                  │   │  │
│   │  │  • GQA (4 KV Heads)         │  • GQA (8 KV Heads)                   │   │  │
│   │  └─────────────────────────────────────────────────────────────────────┘   │  │
│   │                                                                             │  │
│   │  Features:                                                                  │  │
│   │  • Arabic-optimized tokenizer (152K vocabulary)                            │  │
│   │  • Byte-fallback for OOV handling                                          │  │
│   │  • Causal attention masking                                                │  │
│   │  • KTO-aligned output distribution                                         │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                               OUTPUT STAGE
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────────────┐   │
│  │  Token Generation │───▶│  Post-Processing  │───▶│  Final Transcription      │   │
│  │  (Autoregressive) │    │  (Normalization)  │    │  (Arabic/English Text)    │   │
│  └───────────────────┘    └───────────────────┘    └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Specification | Purpose |
|:----------|:--------------|:--------|
| **Audio Processor** | FFmpeg-based, 16kHz resampling | Universal format support, normalization |
| **Feature Extractor** | 80-dim Mel spectrogram, 25ms window | Acoustic feature representation |
| **Conformer Encoder** | 18 blocks, 512-dim, 8 heads | Rich acoustic encoding with local+global context |
| **Alignment Module** | Cross-attention projection | Bridges audio features to language model space |
| **Language Decoder** | 3B/7B Transformer, 152K vocab | Contextual text generation with Arabic optimization |
| **Output Layer** | Softmax with temperature scaling | Probability distribution over vocabulary |

### Inference Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Audio      │    │   GGUF       │    │   llama.cpp  │    │   Text       │
│   Input      │───▶│   Model      │───▶│   Runtime    │───▶│   Output     │
│   (any fmt)  │    │   (Q4_K_M)   │    │   (CPU/GPU)  │    │   (UTF-8)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  Format Conv.       Model Load          Inference           Decode
  Resampling         Weight Quant.       Attention           Detokenize
  Normalization      KV Cache            Generation          Normalize
```

### Deployment Configurations

| Configuration | Model | Quantization | Memory | Latency | Use Case |
|:--------------|:------|:-------------|:-------|:--------|:---------|
| CPU Standard | Turbo-V1 | Q4_K_M | 2.0 GB | ~500ms | On-premise batch |
| CPU Fast | Turbo-V1 | Q4_K_S | 1.8 GB | ~400ms | Edge deployment |
| GPU Optimized | Pro-V1 | FP16 | 14 GB | ~150ms | Cloud inference |
| Streaming | Streaming-V1 | Q8_0 | 3.2 GB | ~80ms | Real-time apps |

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

### Arabic Speech Recognition Benchmarks

We evaluated Audar-ASR against leading industry solutions on standard Arabic ASR benchmarks. All evaluations conducted January 2025 using official APIs and published model weights.

#### FLEURS Arabic Benchmark (Read Speech)

| Model | Provider | WER ↓ | Accuracy ↑ | Latency (10s audio) | Notes |
|:------|:---------|:-----:|:----------:|:-------------------:|:------|
| **Audar-ASR Pro-V1** | Audar AI | **2.8%** | **97.2%** | 180ms (GPU) | 7B, Commercial |
| Scribe v1 | ElevenLabs | 3.1% | 96.9% | 250ms | Commercial API |
| **Audar-ASR Turbo-V1** | Audar AI | 4.2% | 95.8% | 520ms (CPU) | 3B, Open Source |
| Gemini Flash 2.0 | Google | 13.2% | 86.8% | 180ms | Commercial API |
| Whisper Large-v3 | OpenAI | 17.0% | 83.0% | 890ms (CPU) | Open Source |
| Nova 2 | Deepgram | — | — | — | Limited Arabic support |

#### Common Voice Arabic (Spontaneous Speech)

| Model | Provider | WER ↓ | CER ↓ | Dialect Coverage |
|:------|:---------|:-----:|:-----:|:-----------------|
| **Audar-ASR Pro-V1** | Audar AI | **4.9%** | **1.8%** | MSA + 5 dialects |
| Scribe v1 | ElevenLabs | 5.5% | 2.1% | MSA primary |
| **Audar-ASR Turbo-V1** | Audar AI | 6.8% | 2.6% | MSA + 5 dialects |
| Whisper Large-v3 | OpenAI | 18.7% | 7.2% | MSA primary |
| Azure Speech | Microsoft | 21.3% | 8.4% | MSA + Gulf |
| Cloud Speech-to-Text | Google | 19.8% | 7.8% | MSA + Egyptian |

#### MGB-2 Broadcast Arabic (Challenging Audio)

| Model | Provider | WER ↓ | Notes |
|:------|:---------|:-----:|:------|
| **Audar-ASR Pro-V1** | Audar AI | **12.4%** | Optimized for broadcast |
| Scribe v1 | ElevenLabs | 13.8% | — |
| **Audar-ASR Turbo-V1** | Audar AI | 15.2% | Open source |
| Whisper Large-v3 | OpenAI | 24.3% | — |
| Qwen3-ASR-7B | Alibaba | 18.6% | Chinese-optimized |

### Code-Switching Performance (Arabic-English)

| Model | CS-WER ↓ | Language ID Acc ↑ | Switch Point Acc ↑ |
|:------|:--------:|:-----------------:|:------------------:|
| **Audar-ASR Pro-V1** | **10.8%** | **96.4%** | **93.2%** |
| **Audar-ASR Turbo-V1** | 12.3% | 94.2% | 91.7% |
| Scribe v1 | 14.1% | 91.8% | 88.4% |
| Whisper Large-v3 | 28.4% | 76.2% | 68.1% |
| Qwen3-ASR-7B | 19.7% | 84.5% | 79.3% |

### Dialect-Specific Performance (WER %)

| Dialect | Audar Pro-V1 | Audar Turbo-V1 | Whisper v3 | Scribe v1 |
|:--------|:------------:|:--------------:|:----------:|:---------:|
| MSA (Modern Standard) | **2.4%** | 3.8% | 12.1% | 2.9% |
| Gulf (UAE, Saudi) | **3.1%** | 4.5% | 19.4% | 4.2% |
| Egyptian | **3.8%** | 5.2% | 18.7% | 4.8% |
| Levantine | **4.2%** | 5.8% | 22.3% | 5.9% |
| Maghrebi | **6.1%** | 7.9% | 28.6% | 8.2% |

### Inference Performance

| Metric | Turbo-V1 (3B) | Pro-V1 (7B) | Streaming-V1 |
|:-------|:-------------:|:-----------:|:------------:|
| Model Size (GGUF Q4) | 2.0 GB | 4.2 GB | 2.0 GB |
| Real-Time Factor (CPU) | 0.4x - 0.6x | 0.8x - 1.2x | 0.2x - 0.3x |
| Real-Time Factor (GPU) | 0.08x | 0.12x | 0.05x |
| First Token Latency | < 500ms | < 300ms | < 100ms |
| Memory Usage (CPU) | ~2.5 GB | ~5 GB | ~2.5 GB |
| Throughput (GPU) | 45 req/s | 28 req/s | 120 req/s |

### Benchmark Methodology

All benchmarks conducted under standardized conditions:
- **Hardware**: NVIDIA A100 80GB (GPU), AMD EPYC 7763 64-core (CPU)
- **Evaluation Date**: January 2025
- **Audio Format**: 16kHz mono WAV, normalized to -20 dBFS
- **Metrics**: WER calculated using `jiwer` with Arabic text normalization
- **Dialects**: Evaluated on native speaker recordings from each region

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

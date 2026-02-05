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

| Model | Parameters | Use Case | Availability | Access |
|:------|:----------:|:---------|:------------:|:------:|
| **Audar-ASR Turbo-V1** | 3B | Production and Research | Open Source | [GitHub](https://github.com/AudarAI/Audar-ASR-V1) |
| **Audar-ASR Pro-V1** | 7B | Accuracy-Sensitive Deployments | Commercial | [SDK](https://dev.audarai.com/tts-sdk) |
| **Audar-ASR Diarization-V1** | 7B | Multi-Speaker with Realtime Diarization | Commercial | [SDK](https://dev.audarai.com/tts-sdk) |

</div>

### Audar-ASR Turbo-V1 (Open Source)

The flagship open-source model optimized for balanced performance. Ideal for:
- Research and academic use
- Production transcription workloads
- Batch processing pipelines
- On-premise deployments

**Get Started:** [https://github.com/AudarAI/Audar-ASR-V1](https://github.com/AudarAI/Audar-ASR-V1)

### Audar-ASR Pro-V1 (Commercial)

Our most accurate 7B parameter model, fine-tuned for maximum Arabic accuracy. Designed for:
- Enterprise transcription services
- Legal and medical documentation
- Broadcast media processing
- High-stakes accuracy requirements

**Access:** [https://dev.audarai.com/tts-sdk](https://dev.audarai.com/tts-sdk)

### Audar-ASR Diarization-V1 (Commercial)

Advanced 7B multi-speaker ASR with real-time speaker diarization:
- Automatic speaker identification and separation
- Real-time speaker labeling during transcription
- Meeting and conference transcription
- Call center analytics and conversation intelligence
- Podcast and interview processing
- Multi-party conversation understanding

**Access:** [https://dev.audarai.com/tts-sdk](https://dev.audarai.com/tts-sdk)

---

## Architecture

Audar-ASR employs a purpose-built encoder-decoder architecture optimized specifically for Arabic speech recognition. The system processes raw audio through a multi-stage pipeline designed for accuracy and efficiency.

### System Architecture

```
                              AUDAR-ASR ARCHITECTURE
                         Arabic-Centric Speech Recognition

                                    INPUT STAGE
  +-----------+    +---------------+    +------------------------+
  | Raw Audio |--->| Audio Process |--->| Mel-Spectrogram        |
  | (Any Fmt) |    | 16kHz Mono    |    | 80-dim, 25ms frames    |
  +-----------+    +---------------+    +------------------------+
                                |
                                v
                           ENCODER STAGE
  +------------------------------------------------------------------+
  |                    CONFORMER AUDIO ENCODER                       |
  |  +----------+  +-----------+  +-----------+  +-----------+       |
  |  | Conv     |  | Conformer |  | Conformer |  | Conformer |       |
  |  | Subsamp. |->| Block x6  |->| Block x6  |->| Block x6  |       |
  |  | (4x)     |  | Self-Attn |  | Conv      |  | FFN       |       |
  |  +----------+  +-----------+  +-----------+  +-----------+       |
  |                                                                  |
  |  - Relative Positional Encoding    - 512-dim hidden size         |
  |  - 8 Attention Heads               - Macaron-style FFN           |
  |  - Convolution Kernel: 31          - Dropout: 0.1                |
  +------------------------------------------------------------------+
                                |
                                v
                          PROJECTION STAGE
  +------------------------------------------------------------------+
  |                 AUDIO-TEXT ALIGNMENT MODULE                      |
  |  +------------+    +------------+    +------------+              |
  |  | Linear     |    | Cross-Modal|    | Layer      |              |
  |  | Projection |--->| Attention  |--->| Norm       |              |
  |  | 512->2048  |    | 8 heads    |    |            |              |
  |  +------------+    +------------+    +------------+              |
  |                                                                  |
  |  - Bridges acoustic and linguistic representations               |
  |  - Learned query embeddings for text generation                  |
  |  - Residual connections preserve audio fidelity                  |
  +------------------------------------------------------------------+
                                |
                                v
                           DECODER STAGE
  +------------------------------------------------------------------+
  |                TRANSFORMER LANGUAGE DECODER                      |
  |  +---------------------------+---------------------------+       |
  |  | Turbo-V1 (3B)             | Pro-V1 / Diarization-V1 (7B) |       |
  |  | - 32 Decoder Layers       | - 48 Decoder Layers       |       |
  |  | - 2048 Hidden Dimension   | - 4096 Hidden Dimension   |       |
  |  | - 16 Attention Heads      | - 32 Attention Heads      |       |
  |  | - 8192 FFN Dimension      | - 14336 FFN Dimension     |       |
  |  | - RoPE Positional Enc.    | - RoPE Positional Enc.    |       |
  |  | - SwiGLU Activation       | - SwiGLU Activation       |       |
  |  | - GQA (4 KV Heads)        | - GQA (8 KV Heads)        |       |
  |  +---------------------------+---------------------------+       |
  |                                                                  |
  |  Features:                                                       |
  |  - Arabic-optimized tokenizer (152K vocabulary)                  |
  |  - Byte-fallback for OOV handling                                |
  |  - Causal attention masking                                      |
  |  - KTO-aligned output distribution                               |
  +------------------------------------------------------------------+
                                |
                                v
                            OUTPUT STAGE
  +---------------+    +---------------+    +---------------------+
  | Token Gen.    |--->| Post-Process  |--->| Final Transcription |
  | Autoregressive|    | Normalization |    | Arabic/English Text |
  +---------------+    +---------------+    +---------------------+
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
+----------+    +----------+    +----------+    +----------+
|  Audio   |    |  GGUF    |    | llama.cpp|    |  Text    |
|  Input   |--->|  Model   |--->| Runtime  |--->|  Output  |
| (any fmt)|    | (Q4_K_M) |    | (CPU/GPU)|    | (UTF-8)  |
+----------+    +----------+    +----------+    +----------+
     |              |               |               |
     v              v               v               v
 Format Conv.   Model Load     Inference        Decode
 Resampling     Weight Quant.  Attention        Detokenize
 Normalize      KV Cache       Generation       Normalize
```

### Deployment Configurations

| Configuration | Model | Quantization | Memory | Latency | Use Case |
|:--------------|:------|:-------------|:-------|:--------|:---------|
| CPU Standard | Turbo-V1 | Q4_K_M | 2.0 GB | ~500ms | On-premise batch |
| CPU Fast | Turbo-V1 | Q4_K_S | 1.8 GB | ~400ms | Edge deployment |
| GPU Optimized | Pro-V1 | FP16 | 14 GB | ~150ms | Cloud inference |
| Diarization | Diarization-V1 | FP16 | 14 GB | ~200ms | Multi-speaker |

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
+----------------+     +----------------+     +----------------+
|  Pre-training  |---->|  Fine-tuning   |---->| KTO Optimiz.   |
|                |     |                |     |                |
|  300K hours    |     |  Supervised    |     |  Preference    |
|  Self-superv.  |     |  Transcription |     |  Alignment     |
+----------------+     +----------------+     +----------------+
```

**Kahneman-Tversky Optimization (KTO)** - Audar-ASR is the world's first ASR model to leverage KTO for alignment. Unlike traditional DPO which requires paired preferences, KTO uses prospect theory to optimize directly from human feedback signals, resulting in transcriptions that better match human expectations for accuracy and formatting.

---

## Performance

### Arabic Speech Recognition Benchmarks

We evaluated Audar-ASR against leading industry solutions on standard Arabic ASR benchmarks. Benchmark data sourced from official provider documentation and published evaluations (January 2025).

#### Overall Arabic Performance (Mixed Test Set)

| Model | Provider | WER | Source |
|:------|:---------|:---:|:------:|
| **Audar-ASR Pro-V1** | Audar AI | **9.4%** | Internal |
| Scribe v1 | ElevenLabs | 11.1% | [elevenlabs.io](https://elevenlabs.io/speech-to-text/arabic) |
| **Audar-ASR Turbo-V1** | Audar AI | 12.4% | Internal |
| Gemini Flash 2.0 | Google | 13.2% | [elevenlabs.io](https://elevenlabs.io/speech-to-text/arabic) |
| Whisper Large-v3 | OpenAI | 17.0% | [elevenlabs.io](https://elevenlabs.io/speech-to-text/arabic) |
| Nova 2 | Deepgram | 100.0% | [elevenlabs.io](https://elevenlabs.io/speech-to-text/arabic) |

*ElevenLabs Scribe benchmarks sourced from official ElevenLabs Arabic Speech-to-Text page.*

#### Clean Studio MSA Audio

| Model | Provider | WER | Notes |
|:------|:---------|:---:|:------|
| Scribe v1 | ElevenLabs | **7.8%** | Optimized for clean audio |
| **Audar-ASR Pro-V1** | Audar AI | 8.2% | - |
| **Audar-ASR Turbo-V1** | Audar AI | 10.1% | Open source |
| Whisper Large-v3 | OpenAI | 14.2% | - |

#### Dialectal and Noisy Audio (Real-World Conditions)

| Model | Provider | WER | Notes |
|:------|:---------|:---:|:------|
| **Audar-ASR Pro-V1** | Audar AI | **10.6%** | Dialect-optimized |
| **Audar-ASR Turbo-V1** | Audar AI | 13.8% | Open source |
| Scribe v1 | ElevenLabs | 14.2% | MSA-focused |
| Whisper Large-v3 | OpenAI | 19.4% | - |

#### MGB-2 Broadcast Arabic (Challenging Audio)

| Model | Provider | WER | Notes |
|:------|:---------|:---:|:------|
| **Audar-ASR Pro-V1** | Audar AI | **14.8%** | Broadcast-optimized |
| Scribe v1 | ElevenLabs | 16.2% | - |
| **Audar-ASR Turbo-V1** | Audar AI | 17.6% | Open source |
| Whisper Large-v3 | OpenAI | 24.3% | - |

### Code-Switching Performance (Arabic-English)

| Model | CS-WER | Language ID Acc | Switch Point Acc |
|:------|:------:|:---------------:|:----------------:|
| **Audar-ASR Pro-V1** | **11.2%** | **96.4%** | **93.2%** |
| **Audar-ASR Turbo-V1** | 13.8% | 94.2% | 91.7% |
| Scribe v1 | 15.6% | 91.8% | 88.4% |
| Whisper Large-v3 | 28.4% | 76.2% | 68.1% |

### Dialect-Specific Performance (WER %)

| Dialect | Audar Pro-V1 | Audar Turbo-V1 | Scribe v1 | Whisper v3 |
|:--------|:------------:|:--------------:|:---------:|:----------:|
| MSA (Modern Standard) | 8.2% | 10.1% | **7.8%** | 14.2% |
| Gulf (UAE, Saudi) | **9.4%** | 12.2% | 12.8% | 19.4% |
| Egyptian | **10.1%** | 13.4% | 13.6% | 18.7% |
| Levantine | **11.2%** | 14.6% | 15.1% | 22.3% |
| Maghrebi | **13.8%** | 16.4% | 17.2% | 28.6% |

*Note: Scribe v1 shows superior performance on clean MSA studio recordings, while Audar models excel on dialectal, noisy, and code-switched audio.*

### Inference Performance

| Metric | Turbo-V1 (3B) | Pro-V1 (7B) | Diarization-V1 (7B) |
|:-------|:-------------:|:-----------:|:---------------:|
| Model Size (GGUF Q4) | 2.0 GB | 4.2 GB | 4.2 GB |
| Real-Time Factor (CPU) | 0.3x - 0.5x | 0.6x - 0.9x | 0.8x - 1.1x |
| Real-Time Factor (GPU) | 0.05x | 0.08x | 0.10x |
| First Token Latency | < 180ms | < 320ms | < 380ms |
| Memory Usage (CPU) | ~2.5 GB | ~5 GB | ~5.5 GB |
| Throughput (GPU) | 52 req/s | 32 req/s | 26 req/s |
| Speaker Diarization | No | No | Yes (Real-time) |

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

Models are **automatically downloaded** from HuggingFace on first use:

```python
from audar_asr import AudarASR

# Models auto-download to ~/.cache/audar-asr/
asr = AudarASR()
result = asr.transcribe("audio.mp3")
```

**Manual Download Options:**

```bash
# Option 1: Pre-download using Python
python -c "from audar_asr import download_models; download_models()"

# Option 2: Download using HuggingFace CLI
huggingface-cli download audarai/audar-asr-turbo-v1-gguf --local-dir ./models

# Option 3: Set environment variables for custom paths
export AUDAR_MODEL_PATH=/path/to/audar-asr-q4km.gguf
export AUDAR_MMPROJ_PATH=/path/to/mmproj-audar-asr-f16.gguf
```

**Offline Usage:**

```python
from audar_asr import AudarASR, AudarConfig

# Disable auto-download for offline environments
config = AudarConfig(
    model_path="/path/to/audar-asr-q4km.gguf",
    mmproj_path="/path/to/mmproj-audar-asr-f16.gguf",
    auto_download=False
)
asr = AudarASR(config)
```

**HuggingFace Repository:** [audarai/audar-asr-turbo-v1-gguf](https://huggingface.co/audarai/audar-asr-turbo-v1-gguf)

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

## Integration and Deployment

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

## Access

### Open Source

**Audar-ASR Turbo-V1** is freely available under Apache 2.0 license:
- **GitHub**: [https://github.com/AudarAI/Audar-ASR-V1](https://github.com/AudarAI/Audar-ASR-V1)
- **Hugging Face**: [https://huggingface.co/AudarAI](https://huggingface.co/AudarAI)

### Commercial

For **Audar-ASR Pro-V1** and **Audar-ASR Diarization-V1** commercial models:
- **SDK and API Access**: [https://dev.audarai.com/tts-sdk](https://dev.audarai.com/tts-sdk)
- **Enterprise Contact**: enterprise@audarai.com

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
- **Audar-ASR Diarization-V1**: Commercial License

---

<div align="center">

**Built by [Audar AI](https://www.audarai.com)** | **Arabic Speech Technology**

[Website](https://www.audarai.com) | [LinkedIn](https://www.linkedin.com/company/audarai) | [Hugging Face](https://huggingface.co/AudarAI) | [SDK](https://dev.audarai.com/tts-sdk)

</div>

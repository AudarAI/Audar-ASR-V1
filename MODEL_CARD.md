---
license: apache-2.0
language:
  - ar
  - en
pipeline_tag: automatic-speech-recognition
tags:
  - speech-recognition
  - asr
  - arabic
  - multilingual
  - gguf
  - llama.cpp
  - cpu-inference
  - kto-optimization
library_name: audar-asr
---

<div align="center">

# Audar-ASR Turbo-V1 (GGUF)

### Production-Grade Arabic & English Speech Recognition

**The World's First KTO-Optimized ASR Model**

[![GitHub](https://img.shields.io/badge/GitHub-AudarAI/Audar--ASR--V1-blue)](https://github.com/AudarAI/Audar-ASR-V1)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)

</div>

---

## Model Description

Audar-ASR Turbo-V1 is a state-of-the-art automatic speech recognition model optimized for Arabic language understanding with seamless English code-switching support. This GGUF quantized version enables fast, CPU-only inference without requiring a GPU.

**Key Features:**
- **3B Parameters** - Balanced performance and efficiency
- **Q4_K_M Quantization** - 4-bit quantization for CPU inference
- **Arabic Excellence** - Native support for MSA and major dialects (Gulf, Egyptian, Levantine, Maghrebi)
- **Code-Switching** - Seamless Arabic-English mixed speech handling
- **KTO-Optimized** - World's first ASR using Kahneman-Tversky Optimization

## Model Files

| File | Size | Description |
|------|------|-------------|
| `audar-asr-q4km.gguf` | 2.0 GB | Main ASR model (Q4_K_M quantized) |
| `mmproj-audar-asr-f16.gguf` | 2.4 GB | Multimodal projector (audio encoder) |

## Quick Start

### Installation

```bash
pip install audar-asr huggingface_hub
```

### Automatic Download (Recommended)

```python
from audar_asr import AudarASR

# Models automatically download on first use
asr = AudarASR()
result = asr.transcribe("audio.mp3")
print(result.text)
```

### Manual Download

```bash
# Using HuggingFace CLI
huggingface-cli download audarai/audar-asr-turbo-v1-gguf --local-dir ./models

# Using Python
from audar_asr import download_models
download_models(cache_dir="./models")
```

### With Environment Variables

```bash
export AUDAR_MODEL_PATH=./models/audar-asr-q4km.gguf
export AUDAR_MMPROJ_PATH=./models/mmproj-audar-asr-f16.gguf
```

```python
from audar_asr import AudarASR
asr = AudarASR()
```

## Usage Examples

### Basic Transcription

```python
from audar_asr import AudarASR

asr = AudarASR()

# Transcribe any audio format
result = asr.transcribe("meeting.mp3")
print(result.text)

# Access metrics
print(f"Duration: {result.audio_duration:.1f}s")
print(f"Latency: {result.latency_ms:.0f}ms")
print(f"RTF: {result.rtf:.2f}x")
```

### Real-Time Microphone Streaming

```python
from audar_asr import AudarASR

asr = AudarASR()

# Stream from microphone (Ctrl+C to stop)
for result in asr.stream_microphone(chunk_duration=5.0):
    print(result.text)
```

### Batch Processing

```python
from audar_asr import AudarASR
from pathlib import Path

asr = AudarASR()

# Process multiple files
audio_files = Path("./audio").glob("*.mp3")
for audio_file in audio_files:
    result = asr.transcribe(audio_file)
    print(f"{audio_file.name}: {result.text}")
```

### Custom Configuration

```python
from audar_asr import AudarASR, AudarConfig

config = AudarConfig(
    model_path="/custom/path/model.gguf",
    mmproj_path="/custom/path/mmproj.gguf",
    language="ar",           # Force Arabic
    chunk_duration=3.0,      # 3 second chunks for streaming
    auto_download=False,     # Disable auto-download
)

asr = AudarASR(config)
```

### Command Line Interface

```bash
# Transcribe a file
audar-asr transcribe audio.mp3

# Output as JSON
audar-asr transcribe audio.mp3 --json

# Real-time microphone
audar-asr stream --microphone

# Benchmark performance
audar-asr benchmark audio.wav --runs 5
```

## Architecture

```
                         AUDAR-ASR TURBO-V1 ARCHITECTURE
                      Arabic-Centric Speech Recognition (3B)

  +-----------+    +---------------+    +------------------------+
  | Raw Audio |--->| Audio Process |--->| Mel-Spectrogram        |
  | (Any Fmt) |    | 16kHz Mono    |    | 80-dim, 25ms frames    |
  +-----------+    +---------------+    +------------------------+
                           |
                           v
  +------------------------------------------------------------------+
  |                    CONFORMER AUDIO ENCODER                       |
  |  - 18 Conformer Blocks                                           |
  |  - 512-dim hidden size, 8 attention heads                        |
  |  - Relative positional encoding                                  |
  |  - Convolution kernel: 31, Dropout: 0.1                          |
  +------------------------------------------------------------------+
                           |
                           v
  +------------------------------------------------------------------+
  |                 AUDIO-TEXT ALIGNMENT MODULE                      |
  |  - Linear projection: 512 -> 2048                                |
  |  - Cross-modal attention with 8 heads                            |
  |  - Bridges acoustic and linguistic representations               |
  +------------------------------------------------------------------+
                           |
                           v
  +------------------------------------------------------------------+
  |                TRANSFORMER LANGUAGE DECODER (3B)                 |
  |  - 32 Decoder Layers                                             |
  |  - 2048 Hidden Dimension                                         |
  |  - 16 Attention Heads                                            |
  |  - 8192 FFN Dimension                                            |
  |  - RoPE Positional Encoding                                      |
  |  - SwiGLU Activation                                             |
  |  - GQA (4 KV Heads)                                              |
  |  - Arabic-optimized tokenizer (152K vocabulary)                  |
  +------------------------------------------------------------------+
                           |
                           v
  +---------------+    +---------------+    +---------------------+
  | Token Gen.    |--->| Post-Process  |--->| Final Transcription |
  | Autoregressive|    | Normalization |    | Arabic/English Text |
  +---------------+    +---------------+    +---------------------+
```

## Performance Benchmarks

### Arabic Speech Recognition (WER %)

| Test Set | Audar Turbo-V1 | Whisper Large-v3 | ElevenLabs Scribe |
|:---------|:--------------:|:----------------:|:-----------------:|
| Mixed Arabic | 12.4% | 17.0% | 11.1% |
| Clean MSA | 10.1% | 14.2% | **7.8%** |
| Dialectal | 13.8% | 19.4% | 14.2% |
| Code-Switched | 13.8% | 28.4% | 15.6% |

### Dialect Performance (WER %)

| Dialect | Audar Turbo-V1 | Whisper v3 |
|:--------|:--------------:|:----------:|
| MSA (Modern Standard) | 10.1% | 14.2% |
| Gulf (UAE, Saudi) | 12.2% | 19.4% |
| Egyptian | 13.4% | 18.7% |
| Levantine | 14.6% | 22.3% |
| Maghrebi | 16.4% | 28.6% |

### Inference Performance

| Metric | Turbo-V1 (3B) |
|:-------|:-------------:|
| Model Size (GGUF Q4) | 2.0 GB |
| Real-Time Factor (CPU) | 0.3x - 0.5x |
| Real-Time Factor (GPU) | 0.05x |
| First Token Latency | < 180ms |
| Memory Usage (CPU) | ~2.5 GB |
| Throughput (GPU) | 52 req/s |

## System Requirements

### Minimum Requirements

- **CPU**: Any x86_64 or ARM64 processor
- **RAM**: 4 GB (2.5 GB for model + overhead)
- **Storage**: 5 GB free space
- **OS**: Linux, macOS, Windows

### Dependencies

```bash
# System dependencies
# macOS
brew install ffmpeg llama.cpp

# Ubuntu/Debian
sudo apt install ffmpeg
# Build llama.cpp from source

# Python dependencies
pip install audar-asr huggingface_hub
```

## Training Details

### Data

| Data Source | Hours | Description |
|:------------|------:|:------------|
| Open-Source Arabic | 180,000 | MSA, Gulf, Egyptian, Levantine, Maghrebi |
| Proprietary Arabic | 80,000 | Curated conversational and broadcast |
| English | 35,000 | Native and accented speech |
| Code-Switched | 5,000 | Arabic-English mixed utterances |
| **Total** | **300,000** | Multi-domain, multi-dialect coverage |

### Training Pipeline

1. **Pre-training**: Self-supervised learning on 300K hours
2. **Fine-tuning**: Supervised transcription learning
3. **KTO Optimization**: Kahneman-Tversky preference alignment

## Limitations

- Optimized primarily for Arabic and English
- Best performance on conversational and broadcast audio
- May struggle with very noisy environments (SNR < 5dB)
- Code-switching works best with Arabic-English

## License

Apache 2.0 - Free for commercial and non-commercial use.

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

## Links

- **GitHub**: [https://github.com/AudarAI/Audar-ASR-V1](https://github.com/AudarAI/Audar-ASR-V1)
- **Website**: [https://www.audarai.com](https://www.audarai.com)
- **LinkedIn**: [https://www.linkedin.com/company/audarai](https://www.linkedin.com/company/audarai)

---

<div align="center">

**Built by [Audar AI](https://www.audarai.com)** | **Arabic Speech Technology**

</div>

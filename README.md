<div align="center">

# Audar-ASR

### Production-Grade Arabic & English Speech Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CPU Optimized](https://img.shields.io/badge/inference-CPU%20optimized-orange.svg)](#performance)

**Fast, accurate, and easy-to-use ASR that runs entirely on CPU.**

[Quick Start](#quick-start) | [Installation](#installation) | [Examples](#examples) | [API Reference](#api-reference)

</div>

---

## Highlights

- **Universal Format Support** - MP3, WAV, M4A, FLAC, OGG, WebM, and more
- **CPU-Optimized** - No GPU required, runs on any machine
- **Arabic Excellence** - State-of-the-art Arabic speech recognition
- **Real-Time Streaming** - Live microphone transcription
- **Simple API** - Just 3 lines of code to transcribe
- **Production Ready** - Battle-tested, well-documented, thoroughly tested

## Quick Start

```python
from audar_asr import AudarASR

# Initialize (auto-detects model from environment)
asr = AudarASR()

# Transcribe any audio file
result = asr.transcribe("meeting.mp3")
print(result.text)
```

That's it. Three lines to transcribe any audio file.

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
brew install ffmpeg llama.cpp
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
# Build llama.cpp from source (see below)
```

**Windows:**
```bash
choco install ffmpeg
# Build llama.cpp from source (see below)
```

### 2. Install Audar-ASR

```bash
pip install audar-asr

# With microphone support
pip install audar-asr[microphone]
```

Or from source:
```bash
git clone https://github.com/AudarAI/Audar-ASR.git
cd Audar-ASR
pip install -e .
```

### 3. Download Models

Download the GGUF models and set environment variables:

```bash
export AUDAR_MODEL_PATH=/path/to/audar-asr-q4km.gguf
export AUDAR_MMPROJ_PATH=/path/to/mmproj-audar-asr-f16.gguf
```

## Examples

### File Transcription

```python
from audar_asr import AudarASR

asr = AudarASR()

# Any format works
result = asr.transcribe("podcast.mp3")
result = asr.transcribe("meeting.m4a")
result = asr.transcribe("recording.wav")

# Access results
print(result.text)           # Transcribed text
print(result.audio_duration) # Audio duration in seconds
print(result.latency_ms)     # Processing time in ms
print(result.rtf)            # Real-time factor
```

### Real-Time Microphone

```python
from audar_asr import AudarASR

asr = AudarASR()

# Stream from microphone (Ctrl+C to stop)
for result in asr.stream_microphone():
    print(result.text)
```

### Streaming Large Files

```python
from audar_asr import AudarASR

asr = AudarASR()

# Process long files in chunks
for chunk in asr.stream_file("long_podcast.mp3", chunk_duration=10.0):
    print(f"[{chunk.audio_duration:.1f}s] {chunk.text}")
```

### Custom Configuration

```python
from audar_asr import AudarASR, AudarConfig

config = AudarConfig(
    model_path="/custom/path/model.gguf",
    mmproj_path="/custom/path/mmproj.gguf",
    language="ar",           # Force Arabic
    chunk_duration=5.0,      # 5 second chunks
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

# Stream from file
audar-asr stream podcast.mp3 --chunk-duration 10

# Benchmark performance
audar-asr benchmark audio.wav --runs 5

# Show audio info
audar-asr info audio.mp3
```

## API Reference

### AudarASR

The main ASR engine class.

```python
class AudarASR:
    def __init__(self, config: AudarConfig = None)
    def transcribe(self, audio: str | Path | bytes) -> TranscriptionResult
    def stream_file(self, audio: str, chunk_duration: float = 5.0) -> Generator
    def stream_microphone(self, chunk_duration: float = 5.0) -> Generator
    def benchmark(self, audio: str, num_runs: int = 3) -> dict
```

### AudarConfig

Configuration options.

```python
@dataclass
class AudarConfig:
    model_path: str = None       # Path to GGUF model
    mmproj_path: str = None      # Path to multimodal projector
    sample_rate: int = 16000     # Audio sample rate
    max_tokens: int = 256        # Max output tokens
    temperature: float = 0.0     # Sampling temperature
    chunk_duration: float = 5.0  # Chunk duration for streaming
    language: str = "auto"       # Language hint (ar/en/auto)
```

### TranscriptionResult

Result object returned by transcribe().

```python
@dataclass
class TranscriptionResult:
    text: str                    # Transcribed text
    audio_duration: float        # Audio duration in seconds
    latency_ms: float            # Processing time in ms
    rtf: float                   # Real-time factor
    is_realtime: bool            # True if RTF < 1.0
    
    def to_json(self) -> str     # Export as JSON
    def to_dict(self) -> dict    # Export as dictionary
```

## Performance

Benchmarked on Apple M1 MacBook Pro:

| Audio Duration | Latency | Real-Time Factor |
|----------------|---------|------------------|
| 5 seconds      | ~5000ms | 1.0x             |
| 10 seconds     | ~6000ms | 0.6x             |
| 30 seconds     | ~12000ms| 0.4x             |

**Key metrics:**
- **Average RTF**: 0.4-1.0x (faster than real-time for longer audio)
- **Memory Usage**: ~2GB (quantized model)
- **Model Size**: 2.0GB (Q4_K_M quantization)

## Supported Formats

Audar-ASR automatically converts any audio/video format to the optimal format for recognition:

**Audio:** MP3, WAV, M4A, FLAC, OGG, OPUS, AAC, WMA, AIFF, AMR

**Video (audio extraction):** MP4, WebM, MKV, AVI, MOV

## Building llama.cpp

If llama.cpp is not available via package manager:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Add to PATH
export PATH=$PATH:$(pwd)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AUDAR_MODEL_PATH` | Path to GGUF model file |
| `AUDAR_MMPROJ_PATH` | Path to multimodal projector file |
| `AUDAR_LANGUAGE` | Default language (ar/en/auto) |
| `AUDAR_VERBOSE` | Enable verbose logging (1/0) |

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg for your platform (see Installation section).

### "llama-mtmd-cli not found"
Install llama.cpp or build from source (see Building llama.cpp section).

### "Model not found"
Set the environment variables pointing to your GGUF model files.

### Poor accuracy on Arabic
Ensure you're using the Audar-optimized model, not a generic Whisper model.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting PRs.

## Citation

```bibtex
@software{audar_asr,
  title = {Audar-ASR: Production-Grade Arabic Speech Recognition},
  author = {Audar AI},
  year = {2024},
  url = {https://github.com/AudarAI/Audar-ASR}
}
```

---

<div align="center">

**Built with by [Audar AI](https://audarai.com)**

</div>

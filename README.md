# AudarASR - Production-Grade Speech Recognition

Production-grade Automatic Speech Recognition library with Hugging Face Transformers.

## Features

- Local Hugging Face model inference
- OpenAI-compatible API client
- Real-time streaming transcription
- VAD-based chunking for long audio
- Multi-language support
- Comprehensive evaluation metrics

## Installation

```bash
pip install -e .
```

## Quick Start

### Local Model

```python
from audar_asr import AudarASR

asr = AudarASR()
result = asr.transcribe("audio.wav")
print(result.text)

# Streaming
for chunk in asr.stream("audio.wav"):
    print(chunk.text, end="", flush=True)
```

### API Client

```python
from audar_asr_api import AudarASRClient

client = AudarASRClient(base_url="https://asr.example.com/v1")
result = client.transcribe("audio.wav")
print(result.text)
```

## CLI

```bash
# Transcribe
python audar_asr.py audio.wav -o transcript.txt

# API mode
python audar_asr_api.py audio.wav --model-size 7B

# Benchmark
python audar_asr_benchmark.py --dataset Byne/MASC --max_samples 100
```

## License

Apache 2.0

# AudarASR - Production-Grade Speech Recognition

Production-grade Automatic Speech Recognition with Arabic-centric focus and English support.

## Features

- Local Hugging Face model inference (Whisper-based)
- **AudarASR 3B**: GPTQ 4-bit quantized Qwen2.5-Omni model
- OpenAI-compatible API client
- Real-time streaming transcription
- VAD-based chunking for long audio
- Apple Silicon optimization via MLX/MPS
- Comprehensive evaluation metrics (WER/CER + LLM)

## Installation

```bash
pip install -e .

# For GPTQ support (3B model)
pip install gptqmodel  # or: pip install auto-gptq

# For Apple Silicon MLX
pip install mlx mlx-lm
```

## Quick Start

### Whisper-based ASR

```python
from audar_asr import AudarASR

asr = AudarASR()
result = asr.transcribe("audio.wav")
print(result.text)
```

### AudarASR 3B (Arabic-centric)

```python
from audar_asr_3b_infer import AudarASR3B

asr = AudarASR3B()
result = asr.transcribe("arabic_audio.wav", language="ar")
print(result.text)
```

### Apple Silicon (MLX)

```python
from audar_asr_3b_mlx import AudarASR3B_MLX

asr = AudarASR3B_MLX()
result = asr.transcribe("audio.wav")
print(result.text)
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
# Whisper-based
python audar_asr.py audio.wav -o transcript.txt

# 3B Model (Arabic)
python audar_asr_3b_infer.py audio.wav --language ar

# Apple Silicon
python audar_asr_3b_mlx.py audio.wav

# API mode
python audar_asr_api.py audio.wav --model-size 7B

# Benchmark
python audar_asr_benchmark.py --dataset Byne/MASC --max_samples 100
```

## Models

| Model | Size | Quantization | Languages |
|-------|------|--------------|-----------|
| Whisper Large V3 | 1.5B | FP16 | 100+ |
| AudarASR 3B | 3B | GPTQ 4-bit | Arabic, English |

## License

Apache 2.0

#!/usr/bin/env bash
# Audar-ASR-V1 on vLLM — GPU serving with an OpenAI-compatible API.
# vLLM implements the Qwen3-ASR architecture natively, so no custom serving code is needed.
#   Turbo → 4-bit W4A16 compressed-tensors build (vllm-w4a16/, ~2.6 GB)
#   Flash → bf16 model.safetensors directly (~1.6 GB, lossless)
#
# Usage:  ./examples/vllm_serve.sh [turbo|flash] [port]        # default: turbo, port 8000
set -euo pipefail

TIER="${1:-turbo}"
PORT="${2:-8000}"

case "$TIER" in
  turbo)
    NAME="audar-asr-v1-turbo"
    hf download "audarai/Audar-ASR-V1-Turbo" --include "vllm-w4a16/*" --local-dir ./turbo   # ~2.6 GB
    MODEL_DIR="./turbo/vllm-w4a16" ;;
  flash)
    NAME="audar-asr-v1-flash"
    hf download "audarai/Audar-ASR-V1-Flash" --exclude "*.gguf" --local-dir ./flash          # bf16, ~1.6 GB
    MODEL_DIR="./flash" ;;
  *) echo "tier must be 'turbo' or 'flash'"; exit 1 ;;
esac

# Build an audio-enabled vLLM image (the stock image ships no audio codecs — PyAV + librosa +
# soundfile are required to decode audio input).
cat > /tmp/Dockerfile.audar-vllm <<'DOCKER'
FROM vllm/vllm-openai:v0.24.0
RUN pip install --no-cache-dir av librosa soundfile
DOCKER
docker build -t audar-vllm:0.24.0 -f /tmp/Dockerfile.audar-vllm /tmp

# Serve. compressed-tensors (Turbo) is auto-detected; Flash loads bf16. Fits on any >= 8 GB GPU.
docker run --rm --gpus all \
  -v "$PWD/$MODEL_DIR:/model:ro" -p "${PORT}:8000" \
  audar-vllm:0.24.0 \
  --model /model --served-model-name "$NAME" \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.4

# Transcribe from another shell:
#   curl -s http://localhost:${PORT}/v1/audio/transcriptions \
#     -F model=$NAME -F file=@clip.wav -F temperature=0
#
# NOTE (Flash only): raw output is prefixed with a `language <Lang><asr_text>` tag — strip it:
#   re.sub(r"^\s*language\s+[A-Za-z]+\s*(?:<asr_text>)?\s*", "", text).strip()

#!/usr/bin/env bash
# Audar-ASR-V1-Turbo on vLLM — GPU serving with an OpenAI-compatible API.
# Serves the 4-bit W4A16 compressed-tensors build from the model repo's vllm-w4a16/ folder.
# vLLM implements the Qwen3-ASR architecture natively, so no custom serving code is needed.
#
# Usage:  ./examples/vllm_serve.sh [port]        # default port 8000
set -euo pipefail

REPO="audarai/Audar-ASR-V1-Turbo"
PORT="${1:-8000}"

# 1) Download just the vLLM build (~2.6 GB).
#    pip install -U "huggingface_hub[cli]"
hf download "$REPO" --include "vllm-w4a16/*" --local-dir ./turbo

# 2) Build an audio-enabled vLLM image. The stock image ships no audio codecs, so PyAV +
#    librosa + soundfile are required to decode audio input.
cat > /tmp/Dockerfile.audar-vllm <<'DOCKER'
FROM vllm/vllm-openai:v0.24.0
RUN pip install --no-cache-dir av librosa soundfile
DOCKER
docker build -t audar-vllm:0.24.0 -f /tmp/Dockerfile.audar-vllm /tmp

# 3) Serve. The compressed-tensors quantization is auto-detected (Marlin INT4 kernel);
#    fits on any >= 12 GB GPU.
docker run --rm --gpus all \
  -v "$PWD/turbo/vllm-w4a16:/model:ro" -p "${PORT}:8000" \
  audar-vllm:0.24.0 \
  --model /model --served-model-name audar-asr-v1-turbo \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.4

# Transcribe from another shell (Arabic system prompt is prompt-steerable per language):
#   curl -s http://localhost:${PORT}/v1/audio/transcriptions \
#     -F model=audar-asr-v1-turbo -F file=@clip.wav -F temperature=0
#
# Or via chat with an input_audio part:
#   {"model":"audar-asr-v1-turbo","temperature":0,
#    "messages":[{"role":"system","content":"فرّغ الكلام العربي التالي."},
#      {"role":"user","content":[{"type":"input_audio","input_audio":{"data":"<b64 wav>","format":"wav"}}]}]}

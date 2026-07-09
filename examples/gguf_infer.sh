#!/usr/bin/env bash
# Audar-ASR-V1 on llama.cpp (GGUF) — CPU / GPU / edge.
# Works for BOTH tiers; Turbo is GGUF-only, Flash also ships GGUF.
#
# Usage:  ./examples/gguf_infer.sh <clip.wav> [flash|turbo]
set -euo pipefail

WAV="${1:?usage: gguf_infer.sh <clip.wav> [flash|turbo]}"
TIER="${2:-turbo}"

case "$TIER" in
  turbo) REPO="audarai/Audar-ASR-V1-Turbo"; DEC="Audar-ASR-V1-Turbo-Q8_0.gguf"; MMP="mmproj-Audar-ASR-V1-Turbo.gguf" ;;
  flash) REPO="audarai/Audar-ASR-V1-Flash"; DEC="Audar-ASR-V1-Flash-Q8_0.gguf"; MMP="mmproj-Audar-ASR-V1-Flash.gguf" ;;
  *) echo "tier must be 'flash' or 'turbo'"; exit 1 ;;
esac

# 1) Download the decoder + audio projector (BF16 mmproj is REQUIRED — do not quantize it).
#    pip install -U "huggingface_hub[cli]"
hf download "$REPO" "$DEC" "$MMP" --local-dir ./models

# 2) Build a recent llama.cpp with Qwen3-ASR support, then run the multimodal CLI.
#    git clone https://github.com/ggml-org/llama.cpp && cmake -B build llama.cpp && cmake --build build -j
./build/bin/llama-mtmd-cli \
  -m       "./models/$DEC" \
  --mmproj "./models/$MMP" \
  --audio  "$WAV" \
  -sys     "فرّغ الكلام العربي التالي." \
  --temp 0

# ⚠️ Keep the mmproj BF16: the audio encoder's ClippableLinear is numerically sensitive.
#    The decoder GGUF quantizes normally (Q4_K_M / Q8_0 / BF16 all published).

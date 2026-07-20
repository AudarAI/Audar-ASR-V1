<div align="center">

# Audar-ASR-V1

### Arabic-first generative speech recognition — state of the art on dialectal Arabic.

**From Arabic to the world.**

[![Open-AR-ASR](https://img.shields.io/badge/Open--AR--ASR-%231%20of%2036%20%C2%B7%2024.8%25%20avg%20WER-2ea44f)](benchmarks/README.md)
[![Models](https://img.shields.io/badge/🤗%20Models-Flash%20%2B%20Turbo-ffcc4d)](https://huggingface.co/audarai)
[![Paper](https://img.shields.io/badge/📄%20Technical%20Report-PDF-blue)](report/Audar-ASR-V1-Technical-Report.pdf)
[![Code License](https://img.shields.io/badge/code-Apache%202.0-6f42c1)](LICENSE)
[![Website](https://img.shields.io/badge/🌐%20audarai.com-informational)](https://www.audarai.com)

<p>
<a href="#-models"><b>Models</b></a> ·
<a href="#-quickstart"><b>Quickstart</b></a> ·
<a href="#-benchmarks"><b>Benchmarks</b></a> ·
<a href="#-downloads"><b>Downloads</b></a> ·
<a href="#-licenses"><b>Licenses</b></a> ·
<a href="https://www.audarai.com"><b>Audar API</b></a>
</p>

</div>

---

**Audar-ASR-V1** is a family of Arabic-first, generative speech-recognition models from
[AudarAI](https://www.audarai.com). It recasts transcription as audio-conditioned next-token prediction
over a unified text vocabulary — a language-model decoder rather than a CTC or transducer objective. It is
built on a permissively-licensed open-weight audio-LLM foundation, then adapted in-house through
**300,000+ hours** of labeled audio (primarily Arabic, plus English) and a four-stage curriculum that
ends in **KTO preference alignment** from native Arabic annotators — the contribution is the adaptation,
not the foundation. It transcribes MSA and every major
Arabic dialect (Gulf/Emirati, Egyptian, Levantine, Maghrebi), code-switched Arabic–English, and English —
**30 languages** in total.

On the **Open Universal Arabic ASR Leaderboard**, the Turbo tier posts the **lowest average WER and CER of
any evaluated system (rank #1 of 36)**, and the 0.78 B Flash tier is the **strongest small model on the
board (#11)** — beating models several times its size. This repository is the developer hub: model
pointers, benchmarks, and copy-paste inference for both tiers.

## 🧩 Models

| | **Audar-ASR-V1-Flash** | **Audar-ASR-V1-Turbo** |
|---|---|---|
| **Tier** | Real-time · edge | Accuracy |
| **Parameters** | 0.78 B (0.60 B decoder + 0.19 B encoder) | 2.35 B (2.03 B decoder + 0.32 B encoder) |
| **Runtimes** | 🤗 Transformers · GGUF (llama.cpp) · vLLM | GGUF (llama.cpp) · vLLM |
| **Leaderboard** | #11 of 36 · 33.31 % avg WER | **#1 of 36 · 24.78 % avg WER** |
| **Best for** | Live captioning, voice agents, on-device / offline | Lowest error on hard dialectal & long-form audio |
| **License** | [AudarAI Open v1.0](https://www.audarai.com/license/audarai-open-license-v1.0/) | [AudarAI Community v1.0](https://www.audarai.com/license/audarai-community-license-v1.0/) |
| **Download** | **[🤗 audarai/Audar-ASR-V1-Flash](https://huggingface.co/audarai/Audar-ASR-V1-Flash)** | **[🤗 audarai/Audar-ASR-V1-Turbo](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)** |

Both share one architecture (Whisper-style 128-mel audio encoder → Qwen3 decoder, 30 s context) and one
prompt interface, so you can develop against Flash and scale to Turbo without code changes.

## ⚡ Quickstart

Weights download automatically from the Hugging Face repos above.

### Transformers — Flash (Python)

```bash
pip install -r examples/requirements.txt
python examples/transcribe.py clip.wav            # Arabic (auto-dialect)
python examples/transcribe.py english.wav --lang en
```

```python
# Or call the reference helpers directly:
from audar_asr import load_model, transcribe
model, proc = load_model("audarai/Audar-ASR-V1-Flash")   # HF repo id or local path
print(transcribe(model, proc, "clip.wav"))               # <= 30 s clip
```

### GGUF — Flash or Turbo (llama.cpp, CPU/GPU/edge)

```bash
# Build a recent llama.cpp with Qwen3-ASR support, then:
./examples/gguf_infer.sh clip.wav turbo     # or: flash
```

```bash
./build/bin/llama-mtmd-cli \
  -m       Audar-ASR-V1-Turbo-Q8_0.gguf \
  --mmproj mmproj-Audar-ASR-V1-Turbo.gguf \
  --audio  clip.wav \
  -sys     "فرّغ الكلام العربي التالي." \
  --temp 0
```

> ⚠️ The audio projector (`mmproj`) must stay **BF16** — its `ClippableLinear` is numerically sensitive.
> The decoder GGUF quantizes normally (Q4_K_M / Q8_0 / BF16 all published on Hugging Face).

### vLLM (GPU serving, OpenAI-compatible)

Both tiers serve on **[vLLM](https://github.com/vllm-project/vllm)**, which implements the Qwen3-ASR
architecture natively — no custom serving code.

- **Turbo** serves a 4-bit **W4A16** `compressed-tensors` build in the model repo's
  **[`vllm-w4a16/`](https://huggingface.co/audarai/Audar-ASR-V1-Turbo/tree/main/vllm-w4a16)** folder
  (~2.6 GB; near-BF16 accuracy, ≈ +1 pp CER).
- **Flash** serves directly from its **bf16 `model.safetensors`** — no quant step (~1.6 GB, lossless,
  FLEURS AR/EN CER ≈ 3.3 %). Flash prefixes raw output with a `language <Lang><asr_text>` tag —
  strip it client-side (`re.sub(r"^\s*language\s+[A-Za-z]+\s*(?:<asr_text>)?\s*", "", text)`).

```bash
./examples/vllm_serve.sh turbo    # or: flash — builds an audio-enabled vLLM image + serves on :8000
```

The stock vLLM image omits audio codecs, so the helper adds them (`av librosa soundfile`). Once the server
is up, transcribe over the OpenAI API — either `POST /v1/audio/transcriptions` (multipart) or
`/v1/chat/completions` with an `input_audio` part and the Arabic system prompt:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F model=audar-asr-v1-turbo -F file=@clip.wav -F temperature=0
```

### Long-form & realtime streaming

```bash
python examples/transcribe_long.py meeting.wav   # arbitrary length (30 s chunking)
python examples/stream.py long.wav               # LocalAgreement-2 incremental output
```

Streaming commits a word only once two consecutive sliding-window decodes agree on it, so committed text
is stable and never rewrites. This is the local reference policy; Audar's production engine serves the
same policy with **sub-250 ms latency** over an OpenAI-Realtime-compatible WebSocket — see the
[Audar API](https://www.audarai.com).

## 📊 Benchmarks

Full-test-set results on the **[Open Universal Arabic ASR Leaderboard](https://huggingface.co/spaces/UBC-NLP/open-universal-arabic-asr-leaderboard)**,
same harness and normalizer as the public board (calibrated within 0.03 pp). Top of the table:

| # | Model | Avg WER | Avg CER |
|--:|---|--:|--:|
| **1** | **Audar-ASR-V1-Turbo** | **24.78** | **9.49** |
| 2 | CohereLabs/cohere-transcribe-arabic-07-2026 | 25.87 | 11.80 |
| 3 | omnilingual-asr/omniASR_LLM_7B | 28.32 | 12.52 |
| … | | | |
| **11** | **Audar-ASR-V1-Flash** (0.78 B) | **33.31** | **13.66** |
| 12 | Qwen/Qwen3-ASR-1.7B | 33.36 | 12.33 |

**Full 36-system standings, per-dataset WER/CER, methodology, and machine-readable
[`leaderboard.csv`](benchmarks/leaderboard.csv) → [benchmarks/](benchmarks/README.md).**

Highlights: Turbo is **#1 overall** and best on dialect-heavy **SADA** and **MGB-2**; Flash (0.78 B) beats
**Qwen3-ASR-1.7B**, Voxtral-Small-24B, and Whisper-large-v3. On in-house Gulf-Emirati long-form, Turbo hits
**19.4 % WER / 7.3 % CER**.

## 📥 Downloads

All weights and GGUF variants live on Hugging Face:

- **Flash** (Transformers + GGUF): **https://huggingface.co/audarai/Audar-ASR-V1-Flash**
- **Turbo** (GGUF): **https://huggingface.co/audarai/Audar-ASR-V1-Turbo**
- Full model collection: **https://huggingface.co/audarai**

Prefer a managed, production-hosted endpoint (streaming, speaker-attributed transcription, diarization)?
See the **[Audar API](https://www.audarai.com)**.

## 🌍 Languages & capabilities

- **Primary**: Arabic — MSA and dialectal (Gulf/Emirati, Egyptian, Levantine, Maghrebi), plus
  **code-switched Arabic–English**; emits dialect-faithful orthography from audio alone.
- **Also**: English + 28 additional languages (30 total).
- **Tasks**: transcription (audio → UTF-8 text), prompt-steerable for language and formatting; long-form
  and realtime streaming via LocalAgreement-2.
- **Not** for covert speaker identification.

## 📜 Licenses

- **This repository** (example code, reference helpers, docs) — **Apache-2.0** ([LICENSE](LICENSE)).
- **Model weights** — released under AudarAI model licenses (not Apache-2.0):
  - Audar-ASR-V1-Flash → **[AudarAI Open License v1.0](https://www.audarai.com/license/audarai-open-license-v1.0/)** (commercial use, redistribution, and fine-tuning permitted).
  - Audar-ASR-V1-Turbo → **[AudarAI Community License v1.0](https://www.audarai.com/license/audarai-community-license-v1.0/)** (research and limited commercial use for qualifying Community Entities).
- **Enterprise / large-scale / model-as-a-service** use may require an **AudarAI Enterprise License** —
  contact **contact@audarai.com** or visit **[audarai.com](https://www.audarai.com)**.

## 📄 Citation

Technical report — *Audar-ASR-V1: A Multilingual, Arabic-First Generative Speech Recognition Foundation
Model*, Audar AI Team, 2026 — **[read the PDF](report/Audar-ASR-V1-Technical-Report.pdf)**.

```bibtex
@techreport{audar-asr-v1-2026,
  title       = {Audar-ASR-V1: A Multilingual, Arabic-First Generative Speech Recognition Foundation Model},
  author      = {Audar AI Team},
  institution = {AudarAI},
  year        = {2026},
  url         = {https://github.com/AudarAI/Audar-ASR-V1}
}
```

---

<div align="center">

### Leading Arabic-First Multilingual Audio Intelligence

*AudarAI starts with Arabic — and expands to the world.*

**Arabic-first. Multilingual by design. Human-centered at heart.**

**[🌐 www.audarai.com](https://www.audarai.com)** · [🤗 Hugging Face](https://huggingface.co/audarai) · [GitHub](https://github.com/AudarAI) · contact@audarai.com

© 2026 AUDARAI PTE. LTD.

</div>

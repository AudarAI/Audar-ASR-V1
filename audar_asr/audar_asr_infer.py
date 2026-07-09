"""
Audar-ASR-V1 — official reference inference (Transformers).

Audar-ASR-V1-{Flash,Turbo} are `Qwen3ASRForConditionalGeneration` models: a Whisper-style
128-mel audio encoder feeds a Qwen3 decoder. Audio is fed as raw 16 kHz mono; the encoder's
context is 30 s, so longer audio is chunked.

Public API
----------
    model, proc = load_model("audarai/Audar-ASR-V1-Flash")   # local path or HF repo id
    text        = transcribe(model, proc, "clip.wav")        # <= 30 s
    text        = transcribe_long(model, proc, "long.wav")   # any length (chunked)
    for ev in stream_transcribe(model, proc, "long.wav"):    # realtime-style
        print(ev["committed"], "|", ev["pending"])

The streaming policy is LocalAgreement-2 over a sliding <=30 s window: a word is *committed*
only once two consecutive window decodes agree on it, giving stable incremental output with
~`hop_s` latency. This is the local (Transformers) reference; Audar's production sub-250 ms
path applies the same policy over a faster re-decode engine (see the Audar API).

Note: only `Audar-ASR-V1-Flash` ships Transformers/safetensors weights. `Audar-ASR-V1-Turbo`
is distributed as GGUF for llama.cpp — see `examples/gguf_infer.sh`.
"""
from __future__ import annotations

import re

import numpy as np
import torch

SR = 16_000
ENCODER_CONTEXT_S = 30.0

# Convenience map from tier name -> Hugging Face repo id.
HF_REPOS = {
    "flash": "audarai/Audar-ASR-V1-Flash",
    "turbo": "audarai/Audar-ASR-V1-Turbo",
}

# Shipped Arabic auto-dialect system prompt ("Transcribe the following Arabic speech.").
DEFAULT_SYSTEM_AR = "فرّغ الكلام العربي التالي."
# English / multilingual variant (the model is 30-language capable).
DEFAULT_SYSTEM_EN = "Transcribe the following speech."

# Some checkpoints prefix the transcript with a task/language wrapper the decoder learned,
# e.g. "language Arabic<asr_text>...". Strip it so callers get clean text.
_WRAP_RE = re.compile(r"^\s*language\s+[A-Za-z]+\s*(?:<asr_text>)?\s*", re.IGNORECASE)


def _clean_output(text: str) -> str:
    text = _WRAP_RE.sub("", text, count=1)
    text = re.sub(r"</?asr_text>", "", text)
    return text.strip()


def load_model(model_dir, device: str = "cuda:0", dtype=torch.bfloat16):
    """Load an Audar-ASR checkpoint. Requires `trust_remote_code=True` (ships qwen3_asr code).

    `model_dir` may be a local directory or a Hugging Face repo id such as
    `"audarai/Audar-ASR-V1-Flash"`.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM

    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=dtype, device_map=device
    ).eval()
    return model, proc


def _load_audio(wav, sr: int = SR) -> np.ndarray:
    if isinstance(wav, np.ndarray):
        return wav.astype(np.float32)
    import librosa

    return librosa.load(wav, sr=sr, mono=True)[0].astype(np.float32)


def _decode(model, proc, audio: np.ndarray, system: str, max_new_tokens: int) -> str:
    """One forward transcription of a <=30 s audio array."""
    conv = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, audio=audio, sampling_rate=SR, return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    # WhisperFeatureExtractor emits float32 features; the audio encoder is bf16 -> cast to match.
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    seq = out.sequences if hasattr(out, "sequences") else out
    text = proc.batch_decode(seq[:, prompt_len:], skip_special_tokens=True)[0]
    return _clean_output(text)


def transcribe(model, proc, wav, system: str = DEFAULT_SYSTEM_AR, max_new_tokens: int = 440) -> str:
    """Transcribe a single clip up to ~30 s. `wav`: path or float32 @16 kHz array."""
    audio = _load_audio(wav)
    if len(audio) > int(ENCODER_CONTEXT_S * SR):
        audio = audio[: int(ENCODER_CONTEXT_S * SR)]
    return _decode(model, proc, audio, system, max_new_tokens)


def transcribe_long(model, proc, wav, system: str = DEFAULT_SYSTEM_AR, chunk_s: float = ENCODER_CONTEXT_S) -> str:
    """Offline transcription of arbitrary-length audio via non-overlapping <=30 s chunks."""
    audio = _load_audio(wav)
    step = int(chunk_s * SR)
    parts = [
        _decode(model, proc, audio[i : i + step], system, 440)
        for i in range(0, len(audio), step)
    ]
    return " ".join(p for p in parts if p).strip()


def _words(text: str) -> list[str]:
    return text.split()


def stream_transcribe(
    model,
    proc,
    wav,
    system: str = DEFAULT_SYSTEM_AR,
    window_s: float = 28.0,
    hop_s: float = 2.0,
):
    """
    Realtime-style streaming generator.

    Emulates a live mic: audio is revealed in `hop_s` increments; the trailing `window_s`
    seconds are re-decoded each step and LocalAgreement-2 commits words confirmed by two
    consecutive decodes. Yields, per step:
        {"t": <seconds seen>, "committed": <stable prefix>, "pending": <unstable tail>, "final": bool}

    `committed` grows monotonically. On the final step every remaining word is flushed, so
    the last `committed` equals a full transcription of the clip.
    """
    audio = _load_audio(wav)
    n = len(audio)
    W, H = int(window_s * SR), int(hop_s * SR)

    committed: list[str] = []   # globally committed (finalized) words
    seg_start = 0               # sample where the current segment begins (decode anchor)
    seg_committed = 0           # words committed within the current segment
    prev: list[str] = []        # previous decode of the current segment (word list)

    t = 0
    while t < n:
        t = min(t + H, n)
        final = t >= n
        # Decode from the fixed segment anchor -> consecutive decodes are prefix-aligned,
        # so LocalAgreement never re-commits already-committed words.
        hyp = _words(_decode(model, proc, audio[seg_start:t], system, 440))

        if final or (t - seg_start) >= W:
            # Segment full (or stream ended): finalize everything past what's committed,
            # then re-anchor a fresh segment so the window never exceeds `window_s`.
            committed += hyp[seg_committed:]
            yield {"t": t / SR, "committed": " ".join(committed), "pending": "", "final": final}
            seg_start, seg_committed, prev = t, 0, []
            continue

        # LocalAgreement-2: commit the longest common prefix of the last two segment decodes.
        k = 0
        upto = min(len(prev), len(hyp))
        while k < upto and prev[k] == hyp[k]:
            k += 1
        if k > seg_committed:
            committed += hyp[seg_committed:k]
            seg_committed = k
        prev = hyp

        yield {
            "t": t / SR,
            "committed": " ".join(committed),
            "pending": " ".join(hyp[seg_committed:]),
            "final": False,
        }

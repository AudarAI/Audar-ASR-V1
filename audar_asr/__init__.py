"""Audar-ASR-V1 — official reference inference helpers.

    from audar_asr import load_model, transcribe, transcribe_long, stream_transcribe

`load_model` accepts a local path *or* a Hugging Face repo id, e.g.
`load_model("audarai/Audar-ASR-V1-Flash")`.
"""
from .audar_asr_infer import (
    HF_REPOS,
    DEFAULT_SYSTEM_AR,
    DEFAULT_SYSTEM_EN,
    load_model,
    transcribe,
    transcribe_long,
    stream_transcribe,
)

__all__ = [
    "HF_REPOS",
    "DEFAULT_SYSTEM_AR",
    "DEFAULT_SYSTEM_EN",
    "load_model",
    "transcribe",
    "transcribe_long",
    "stream_transcribe",
]
__version__ = "1.0.0"

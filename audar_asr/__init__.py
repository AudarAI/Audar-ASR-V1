"""
Audar-ASR: Production-Grade Arabic & English Speech Recognition
=================================================================

A high-performance, CPU-optimized ASR engine powered by Qwen2.5-Omni.

Quick Start:
    >>> from audar_asr import AudarASR
    >>> asr = AudarASR()
    >>> result = asr.transcribe("audio.mp3")
    >>> print(result.text)

Features:
    - Universal format support (MP3, WAV, M4A, FLAC, WebM, etc.)
    - CPU-optimized inference via GGUF quantization
    - Real-time microphone streaming
    - Streaming text output with ChatLLM-style display
    - Arabic & English speech recognition
    - Low latency (~0.4x real-time factor)

Copyright (c) 2024-2026 Audar AI. All rights reserved.
https://audarai.com
"""

__version__ = "1.0.0"
__author__ = "Audar AI"
__license__ = "Apache-2.0"

from audar_asr.core.engine import AudarASR
from audar_asr.core.transcription import TranscriptionResult
from audar_asr.core.microphone import RealtimeMicrophone
from audar_asr.core.config import AudarConfig
from audar_asr.utils.streaming import StreamingDisplay

__all__ = [
    "AudarASR",
    "AudarConfig", 
    "TranscriptionResult",
    "RealtimeMicrophone",
    "StreamingDisplay",
]

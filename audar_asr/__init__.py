"""
Audar-ASR: Production-Grade Arabic & English Speech Recognition
=================================================================

A high-performance, CPU-optimized ASR engine powered by Qwen2.5-Omni.

Quick Start:
    >>> from audar_asr import AudarASR
    >>> asr = AudarASR()  # Auto-downloads models from HuggingFace
    >>> result = asr.transcribe("audio.mp3")
    >>> print(result.text)

Features:
    - Universal format support (MP3, WAV, M4A, FLAC, WebM, etc.)
    - CPU-optimized inference via GGUF quantization
    - Real-time microphone streaming
    - Streaming text output with ChatLLM-style display
    - Arabic & English speech recognition
    - Low latency (~0.4x real-time factor)
    - Automatic model download from HuggingFace

Model Loading:
    Models are automatically downloaded from HuggingFace if not found locally.
    
    To pre-download models:
        >>> from audar_asr import download_models
        >>> download_models()  # Downloads to ~/.cache/audar-asr/
    
    Or use local files:
        >>> from audar_asr import AudarASR, AudarConfig
        >>> config = AudarConfig(
        ...     model_path="/path/to/model.gguf",
        ...     mmproj_path="/path/to/mmproj.gguf",
        ...     auto_download=False
        ... )
        >>> asr = AudarASR(config)

Repository:
    GitHub: https://github.com/AudarAI/Audar-ASR-V1
    HuggingFace: https://huggingface.co/audarai/audar-asr-turbo-v1-gguf

Copyright (c) 2024-2026 Audar AI. All rights reserved.
https://audarai.com
"""

__version__ = "1.0.0"
__author__ = "Audar AI"
__license__ = "Apache-2.0"

from audar_asr.core.engine import AudarASR
from audar_asr.core.transcription import TranscriptionResult
from audar_asr.core.microphone import RealtimeMicrophone
from audar_asr.core.config import AudarConfig, download_models

__all__ = [
    "AudarASR",
    "AudarConfig", 
    "TranscriptionResult",
    "RealtimeMicrophone",
    "download_models",
]

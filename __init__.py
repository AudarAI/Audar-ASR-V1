"""
AudarASR - Production-Grade Speech Recognition

Clone any voice. Transcribe any audio. Real-time streaming support.

Quick Start:
    from audar_asr import AudarASR
    
    asr = AudarASR()
    result = asr.transcribe("audio.wav")
    print(result.text)

Modules:
    audar_asr: Local Hugging Face model inference
    audar_asr_api: OpenAI-compatible API client
    audar_asr_benchmark: Comprehensive evaluation system

Author: Audar AI
License: Apache 2.0
"""

from audar_asr import (
    AudarASR,
    TranscriptionResult,
    AudioSegment,
    StreamChunk,
    AudioProcessor,
    VADChunker,
    ASRMetrics,
)

from audar_asr_api import (
    AudarASRClient,
    APITranscriptionResult,
)

__version__ = "1.0.0"
__author__ = "Audar AI"
__license__ = "Apache-2.0"

__all__ = [
    # Core
    "AudarASR",
    "TranscriptionResult",
    "AudioSegment",
    "StreamChunk",
    # Utilities
    "AudioProcessor",
    "VADChunker",
    "ASRMetrics",
    # API Client
    "AudarASRClient",
    "APITranscriptionResult",
]

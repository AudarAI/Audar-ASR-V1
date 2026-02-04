"""
AudarASR - Production-Grade Speech Recognition

Transcribe any audio. Real-time streaming support. Arabic-centric with English support.

Quick Start:
    from audar_asr import AudarASR
    
    asr = AudarASR()
    result = asr.transcribe("audio.wav")
    print(result.text)

Modules:
    audar_asr: Local Hugging Face model inference (Whisper-based)
    audar_asr_api: OpenAI-compatible API client
    audar_asr_benchmark: Comprehensive evaluation system
    audar_asr_3b_infer: Local 3B model inference (GPTQ quantized)
    audar_asr_3b_mlx: Apple Silicon optimized inference

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

# Optional: 3B model imports (may require additional dependencies)
try:
    from audar_asr_3b_infer import AudarASR3B
except ImportError:
    AudarASR3B = None

try:
    from audar_asr_3b_mlx import AudarASR3B_MLX
except ImportError:
    AudarASR3B_MLX = None

__version__ = "1.0.0"
__author__ = "Audar AI"
__license__ = "Apache-2.0"

__all__ = [
    # Core (Whisper-based)
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
    # 3B Local Model
    "AudarASR3B",
    "AudarASR3B_MLX",
]

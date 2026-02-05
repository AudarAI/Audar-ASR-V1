"""
Audar-ASR Transcription Result
==============================

Structured result object for ASR transcriptions.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class TranscriptionSegment:
    """
    A segment of transcription (for chunked/streaming results).
    
    Attributes:
        text: Transcribed text for this segment
        start_time: Start time in seconds (relative to audio start)
        end_time: End time in seconds
        confidence: Confidence score (0.0 - 1.0) if available
    """
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """
    Complete transcription result with metadata and performance metrics.
    
    Attributes:
        text: Complete transcribed text
        segments: List of transcription segments (for streaming/chunked mode)
        language: Detected language code
        audio_duration: Duration of input audio in seconds
        latency_ms: Total processing latency in milliseconds
        inference_ms: Pure inference time in milliseconds
        rtf: Real-Time Factor (processing_time / audio_duration)
        timestamp: When transcription was performed
        model_info: Information about the model used
    
    Example:
        >>> result = asr.transcribe("audio.mp3")
        >>> print(result.text)
        "مرحبا بكم في أودار للذكاء الاصطناعي"
        >>> print(f"Processed in {result.latency_ms:.0f}ms")
        Processed in 1250ms
        >>> print(f"RTF: {result.rtf:.2f}x realtime")
        RTF: 0.42x realtime
    """
    
    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = "auto"
    audio_duration: float = 0.0
    latency_ms: float = 0.0
    inference_ms: float = 0.0
    rtf: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    model_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_realtime(self) -> bool:
        """Check if transcription was faster than realtime."""
        return self.rtf < 1.0
    
    @property
    def words(self) -> List[str]:
        """Get list of words from transcription."""
        return self.text.split() if self.text else []
    
    @property
    def word_count(self) -> int:
        """Get total word count."""
        return len(self.words)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "segments": [
                {
                    "text": s.text,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "confidence": s.confidence,
                }
                for s in self.segments
            ],
            "language": self.language,
            "audio_duration": self.audio_duration,
            "latency_ms": self.latency_ms,
            "inference_ms": self.inference_ms,
            "rtf": self.rtf,
            "timestamp": self.timestamp.isoformat(),
            "is_realtime": self.is_realtime,
            "word_count": self.word_count,
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "realtime" if self.is_realtime else "slower than realtime"
        return (
            f"TranscriptionResult(\n"
            f"  text='{self.text[:50]}{'...' if len(self.text) > 50 else ''}',\n"
            f"  duration={self.audio_duration:.1f}s,\n"
            f"  latency={self.latency_ms:.0f}ms,\n"
            f"  rtf={self.rtf:.2f} ({status})\n"
            f")"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class StreamingTranscription:
    """
    Container for streaming transcription session.
    
    Tracks multiple chunks and provides aggregated results.
    """
    
    chunks: List[TranscriptionResult] = field(default_factory=list)
    is_complete: bool = False
    start_time: datetime = field(default_factory=datetime.now)
    
    def add_chunk(self, result: TranscriptionResult):
        """Add a transcription chunk."""
        self.chunks.append(result)
    
    @property
    def text(self) -> str:
        """Get concatenated text from all chunks."""
        return " ".join(c.text for c in self.chunks if c.text)
    
    @property
    def total_audio_duration(self) -> float:
        """Get total audio duration processed."""
        return sum(c.audio_duration for c in self.chunks)
    
    @property
    def total_latency_ms(self) -> float:
        """Get total processing latency."""
        return sum(c.latency_ms for c in self.chunks)
    
    @property
    def average_rtf(self) -> float:
        """Get average real-time factor."""
        if not self.chunks:
            return 0.0
        return sum(c.rtf for c in self.chunks) / len(self.chunks)
    
    def finalize(self) -> TranscriptionResult:
        """
        Finalize streaming session and return aggregated result.
        """
        self.is_complete = True
        
        return TranscriptionResult(
            text=self.text,
            segments=[
                TranscriptionSegment(
                    text=c.text,
                    start_time=sum(ch.audio_duration for ch in self.chunks[:i]),
                    end_time=sum(ch.audio_duration for ch in self.chunks[:i+1]),
                )
                for i, c in enumerate(self.chunks) if c.text
            ],
            audio_duration=self.total_audio_duration,
            latency_ms=self.total_latency_ms,
            rtf=self.average_rtf,
            timestamp=self.start_time,
        )

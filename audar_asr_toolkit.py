#!/usr/bin/env python3
"""
AudarASR Toolkit - Production-Grade Speech Recognition

Full-featured ASR toolkit with VAD splitting, parallel processing, and SRT generation.

Key Features:
- Break the 3-Minute Limit: Transcribe audio of any length
- Smart Audio Splitting: VAD-based chunking at natural pauses
- High-Speed Parallel Processing: Multi-threaded transcription
- Intelligent Post-Processing: Hallucination removal
- SRT Subtitle Generation: Timestamped subtitles
- Automatic Audio Resampling: Any format to 16kHz mono
- Universal Media Support: All audio/video formats via FFmpeg

Quick Start:
    from audar_asr_toolkit import AudarASRToolkit
    
    asr = AudarASRToolkit()
    result = asr.transcribe("audio.wav")
    print(result.text)
    
    # With SRT output
    result = asr.transcribe("video.mp4", save_srt=True)

Author: Audar AI
License: Apache 2.0
"""

import os
import sys
import io
import re
import time
import json
import tempfile
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Generator, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR.parent / "audar_asr_3b"
DEFAULT_OUTPUT = SCRIPT_DIR / "output"
CACHE_DIR = SCRIPT_DIR / ".cache"

# Audio parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 30  # seconds
MIN_CHUNK_DURATION = 1.0  # minimum chunk duration
MAX_CHUNK_DURATION = 120  # maximum chunk duration
DEFAULT_WORKERS = 4

# VAD parameters
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION = 0.5  # seconds

# Hallucination patterns (common ASR artifacts)
HALLUCINATION_PATTERNS = [
    r"(Thanks for watching\.?\s*){2,}",
    r"(Subscribe\.?\s*){2,}",
    r"(Please subscribe\.?\s*){2,}",
    r"(Like and subscribe\.?\s*){2,}",
    r"(\.{3,})",  # Excessive dots
    r"(\s{2,})",  # Multiple spaces
    r"^[\s\.\,]+$",  # Only punctuation/whitespace
]

# Supported media formats
SUPPORTED_AUDIO = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
SUPPORTED_VIDEO = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.wmv', '.flv'}
SUPPORTED_FORMATS = SUPPORTED_AUDIO | SUPPORTED_VIDEO


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TranscriptionSegment:
    """A transcribed segment with timing."""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = "auto"
    duration: float = 0.0
    processing_time: float = 0.0
    
    @property
    def rtf(self) -> float:
        return self.processing_time / self.duration if self.duration > 0 else 0
    
    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = _format_srt_time(seg.start)
            end = _format_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)
    
    def to_vtt(self) -> str:
        """Convert to WebVTT format."""
        lines = ["WEBVTT", ""]
        for i, seg in enumerate(self.segments, 1):
            start = _format_vtt_time(seg.start)
            end = _format_vtt_time(seg.end)
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as WebVTT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

class AudioLoader:
    """
    Universal audio loader with FFmpeg support.
    Handles any audio/video format and resamples to 16kHz mono.
    """
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            self._has_ffmpeg = True
        except (subprocess.SubprocessError, FileNotFoundError):
            self._has_ffmpeg = False
            print("Warning: FFmpeg not found. Some formats may not be supported.")
    
    def load(self, path: Union[str, Path]) -> Tuple[np.ndarray, int, float]:
        """
        Load audio from any supported format.
        
        Returns:
            (audio_array, sample_rate, duration)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        # Use FFmpeg for video or unsupported audio
        if suffix in SUPPORTED_VIDEO or (suffix not in {'.wav'} and self._has_ffmpeg):
            return self._load_with_ffmpeg(path)
        else:
            return self._load_with_torchaudio(path)
    
    def _load_with_torchaudio(self, path: Path) -> Tuple[np.ndarray, int, float]:
        """Load with torchaudio."""
        import torchaudio
        
        waveform, sr = torchaudio.load(str(path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        audio = waveform.squeeze().numpy()
        duration = len(audio) / self.sample_rate
        
        return audio, self.sample_rate, duration
    
    def _load_with_ffmpeg(self, path: Path) -> Tuple[np.ndarray, int, float]:
        """Load with FFmpeg (universal format support)."""
        # Use FFmpeg to convert to WAV
        cmd = [
            "ffmpeg", "-i", str(path),
            "-ar", str(self.sample_rate),
            "-ac", "1",  # mono
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
        
        # Parse raw audio
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        duration = len(audio) / self.sample_rate
        
        return audio, self.sample_rate, duration
    
    def get_duration(self, path: Union[str, Path]) -> float:
        """Get audio duration without loading full file."""
        if self._has_ffmpeg:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass
        
        # Fallback: load and check
        _, _, duration = self.load(path)
        return duration


# =============================================================================
# VAD CHUNKER
# =============================================================================

class VADChunker:
    """
    Voice Activity Detection based audio chunking.
    Splits at natural silence boundaries.
    """
    
    def __init__(
        self,
        max_chunk_duration: float = DEFAULT_CHUNK_DURATION,
        min_chunk_duration: float = MIN_CHUNK_DURATION,
        min_silence_duration: float = MIN_SILENCE_DURATION,
        threshold: float = VAD_THRESHOLD,
    ):
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.min_silence_duration = min_silence_duration
        self.threshold = threshold
        self._vad_model = None
        self._vad_utils = None
    
    def _ensure_vad(self):
        """Lazy load Silero VAD."""
        if self._vad_model is None:
            import torch
            self._vad_model, self._vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
            )
    
    def chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Split audio into chunks at silence boundaries.
        
        Returns:
            List of (start_time, end_time, audio_chunk)
        """
        import torch
        
        duration = len(audio) / sample_rate
        
        # If short enough, return as single chunk
        if duration <= self.max_chunk_duration:
            return [(0.0, duration, audio)]
        
        self._ensure_vad()
        
        # Get speech timestamps
        get_speech_ts = self._vad_utils[0]
        wav_tensor = torch.from_numpy(audio).float()
        
        try:
            speech_timestamps = get_speech_ts(
                wav_tensor,
                self._vad_model,
                sampling_rate=sample_rate,
                threshold=self.threshold,
                min_silence_duration_ms=int(self.min_silence_duration * 1000),
            )
        except Exception as e:
            print(f"VAD failed: {e}, using simple chunking")
            return self._simple_chunk(audio, sample_rate)
        
        if not speech_timestamps:
            return self._simple_chunk(audio, sample_rate)
        
        # Merge speech segments into chunks
        chunks = []
        current_start = 0
        current_end = 0
        max_samples = int(self.max_chunk_duration * sample_rate)
        
        for ts in speech_timestamps:
            seg_start = ts['start']
            seg_end = ts['end']
            
            # If adding segment exceeds max, finalize current chunk
            if seg_end - current_start > max_samples and current_end > current_start:
                chunk_audio = audio[current_start:current_end]
                chunks.append((
                    current_start / sample_rate,
                    current_end / sample_rate,
                    chunk_audio,
                ))
                current_start = seg_start
            
            current_end = seg_end
        
        # Add final chunk
        if current_end > current_start:
            chunk_audio = audio[current_start:current_end]
            chunks.append((
                current_start / sample_rate,
                current_end / sample_rate,
                chunk_audio,
            ))
        
        return chunks if chunks else self._simple_chunk(audio, sample_rate)
    
    def _simple_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Simple time-based chunking fallback."""
        chunks = []
        chunk_samples = int(self.max_chunk_duration * sample_rate)
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            start = i / sample_rate
            end = (i + len(chunk)) / sample_rate
            chunks.append((start, end, chunk))
        
        return chunks


# =============================================================================
# HALLUCINATION REMOVAL
# =============================================================================

class HallucinationRemover:
    """
    Removes common ASR hallucinations and artifacts.
    """
    
    def __init__(self, patterns: List[str] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or HALLUCINATION_PATTERNS)]
    
    def clean(self, text: str) -> str:
        """Remove hallucinations and clean text."""
        # Apply pattern removal
        for pattern in self.patterns:
            text = pattern.sub(" ", text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove leading/trailing punctuation artifacts
        text = re.sub(r'^[\s\.\,\!\?]+', '', text)
        text = re.sub(r'[\s\.\,]+$', '.', text)
        
        return text
    
    def remove_repetitions(self, text: str, max_repeats: int = 2) -> str:
        """Remove excessive word/phrase repetitions."""
        words = text.split()
        if len(words) < 4:
            return text
        
        # Detect repeated phrases
        cleaned = []
        i = 0
        while i < len(words):
            # Check for phrase repetition (2-5 words)
            found_repeat = False
            for phrase_len in range(5, 1, -1):
                if i + phrase_len * 2 <= len(words):
                    phrase1 = ' '.join(words[i:i + phrase_len])
                    phrase2 = ' '.join(words[i + phrase_len:i + phrase_len * 2])
                    
                    if phrase1.lower() == phrase2.lower():
                        cleaned.extend(words[i:i + phrase_len])
                        i += phrase_len * 2
                        found_repeat = True
                        break
            
            if not found_repeat:
                cleaned.append(words[i])
                i += 1
        
        return ' '.join(cleaned)


# =============================================================================
# MAIN TOOLKIT
# =============================================================================

class AudarASRToolkit:
    """
    Production-grade ASR toolkit with all features.
    
    Features:
    - VAD-based smart chunking
    - Parallel processing
    - Hallucination removal
    - SRT/VTT subtitle generation
    - Universal media support
    
    Example:
        asr = AudarASRToolkit()
        result = asr.transcribe("video.mp4", save_srt=True)
        print(result.text)
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        num_workers: int = DEFAULT_WORKERS,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
        use_vad: bool = True,
        remove_hallucinations: bool = True,
        lazy_load: bool = False,
    ):
        """
        Initialize ASR toolkit.
        
        Args:
            model_path: Path to model or HF model ID
            device: Device ("cuda", "mps", "cpu", or None for auto)
            num_workers: Number of parallel workers
            chunk_duration: Max chunk duration in seconds
            use_vad: Use VAD for smart chunking
            remove_hallucinations: Remove ASR artifacts
            lazy_load: Defer model loading
        """
        self.model_path = model_path or str(MODEL_DIR)
        self.num_workers = num_workers
        self.chunk_duration = chunk_duration
        self.use_vad = use_vad
        self.remove_hallucinations = remove_hallucinations
        
        # Auto-detect device
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Components
        self._model = None
        self._processor = None
        self._audio_loader = AudioLoader()
        self._vad_chunker = VADChunker(max_chunk_duration=chunk_duration)
        self._hallucination_remover = HallucinationRemover()
        self._initialized = False
        self._lock = threading.Lock()
        
        if not lazy_load:
            self._initialize()
    
    def _initialize(self):
        """Initialize model and processor."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            print("=" * 60)
            print("  AudarASR Toolkit - Production-Grade ASR")
            print("=" * 60)
            print(f"  Device: {self.device}")
            print(f"  Workers: {self.num_workers}")
            print(f"  VAD: {'enabled' if self.use_vad else 'disabled'}")
            print("=" * 60)
            
            # Load processor
            print("\n[1/2] Loading processor...")
            t0 = time.time()
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            print(f"      Loaded in {time.time() - t0:.1f}s")
            
            # Load model
            print("[2/2] Loading model...")
            t0 = time.time()
            
            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                use_safetensors=True,
            )
            
            if self.device not in ["cuda", "auto"]:
                self._model = self._model.to(self.device)
            
            print(f"      Loaded in {time.time() - t0:.1f}s")
            print("\n" + "=" * 60)
            print("  Ready!")
            print("=" * 60 + "\n")
            
            self._initialized = True
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = None,
        language: str = "auto",
        save_srt: bool = False,
        save_vtt: bool = False,
        output_dir: str = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio/video file.
        
        Args:
            audio: Path to audio/video file or numpy array
            sample_rate: Sample rate (required for numpy array)
            language: Target language ("ar", "en", "auto")
            save_srt: Save SRT subtitle file
            save_vtt: Save WebVTT subtitle file
            output_dir: Output directory for subtitles
        
        Returns:
            TranscriptionResult with text and segments
        """
        self._initialize()
        
        start_time = time.time()
        
        # Load audio
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            audio_array, sr, duration = self._audio_loader.load(audio_path)
        elif isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate required for numpy array")
            audio_array = audio
            sr = sample_rate
            duration = len(audio_array) / sr
        else:
            raise TypeError(f"Unsupported type: {type(audio)}")
        
        # Normalize audio
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val
        
        print(f"\n{'='*60}")
        print(f"  Transcribing: {audio_path.name if isinstance(audio, (str, Path)) else 'audio'}")
        print(f"  Duration: {duration:.1f}s")
        print(f"{'='*60}")
        
        # Chunk audio
        if self.use_vad and duration > self.chunk_duration:
            chunks = self._vad_chunker.chunk(audio_array, sr)
            print(f"  Chunks: {len(chunks)} (VAD-based)")
        elif duration > self.chunk_duration:
            chunks = self._vad_chunker._simple_chunk(audio_array, sr)
            print(f"  Chunks: {len(chunks)} (time-based)")
        else:
            chunks = [(0.0, duration, audio_array)]
            print(f"  Chunks: 1 (no splitting needed)")
        
        # Transcribe chunks (parallel if multiple)
        segments = []
        if len(chunks) > 1 and self.num_workers > 1:
            segments = self._transcribe_parallel(chunks, sr, language)
        else:
            for start, end, chunk in chunks:
                text = self._transcribe_chunk(chunk, sr, language)
                if text:
                    segments.append(TranscriptionSegment(
                        text=text,
                        start=start,
                        end=end,
                    ))
        
        # Combine text
        full_text = " ".join(seg.text for seg in segments)
        
        # Post-processing
        if self.remove_hallucinations:
            full_text = self._hallucination_remover.clean(full_text)
            full_text = self._hallucination_remover.remove_repetitions(full_text)
        
        processing_time = time.time() - start_time
        
        result = TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
            duration=duration,
            processing_time=processing_time,
        )
        
        # Print summary
        print(f"\n  Processed in {processing_time:.1f}s (RTF: {result.rtf:.2f})")
        print(f"{'='*60}\n")
        
        # Save subtitles
        if save_srt or save_vtt:
            out_dir = Path(output_dir) if output_dir else Path(audio_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            base_name = Path(audio_path).stem if isinstance(audio, (str, Path)) else "output"
            
            if save_srt:
                srt_path = out_dir / f"{base_name}.srt"
                srt_path.write_text(result.to_srt(), encoding='utf-8')
                print(f"  Saved: {srt_path}")
            
            if save_vtt:
                vtt_path = out_dir / f"{base_name}.vtt"
                vtt_path.write_text(result.to_vtt(), encoding='utf-8')
                print(f"  Saved: {vtt_path}")
        
        return result
    
    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> str:
        """Transcribe a single audio chunk."""
        import torch
        
        # Build conversation
        if language == "ar":
            system_prompt = "You are an expert Arabic speech transcription system. Transcribe the audio accurately."
        elif language == "en":
            system_prompt = "You are an expert English speech transcription system. Transcribe the audio accurately."
        else:
            system_prompt = "You are an expert speech transcription system. Transcribe the audio accurately in its original language."
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Transcribe this audio."}
            ]}
        ]
        
        # Apply chat template
        text_prompt = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Process inputs
        inputs = self._processor(
            text=[text_prompt],
            audios=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )
        
        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        text = self._processor.decode(generated_ids, skip_special_tokens=True)
        
        return text.strip()
    
    def _transcribe_parallel(
        self,
        chunks: List[Tuple[float, float, np.ndarray]],
        sample_rate: int,
        language: str,
    ) -> List[TranscriptionSegment]:
        """Transcribe chunks in parallel."""
        segments = [None] * len(chunks)
        
        def process_chunk(idx: int, start: float, end: float, audio: np.ndarray):
            text = self._transcribe_chunk(audio, sample_rate, language)
            return idx, TranscriptionSegment(text=text, start=start, end=end)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(process_chunk, i, start, end, audio): i
                for i, (start, end, audio) in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                try:
                    idx, segment = future.result()
                    segments[idx] = segment
                    print(f"  [{idx+1}/{len(chunks)}] {segment.start:.1f}s-{segment.end:.1f}s: {segment.text[:40]}...")
                except Exception as e:
                    idx = futures[future]
                    print(f"  [{idx+1}/{len(chunks)}] ERROR: {e}")
        
        return [s for s in segments if s is not None]
    
    def stream(
        self,
        audio: Union[str, Path],
        language: str = "auto",
    ) -> Generator[TranscriptionSegment, None, None]:
        """
        Stream transcription for real-time output.
        
        Yields segments as they are transcribed.
        """
        self._initialize()
        
        # Load audio
        audio_array, sr, duration = self._audio_loader.load(audio)
        
        # Normalize
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val
        
        # Chunk
        if self.use_vad:
            chunks = self._vad_chunker.chunk(audio_array, sr)
        else:
            chunks = self._vad_chunker._simple_chunk(audio_array, sr)
        
        # Process and yield
        for start, end, chunk in chunks:
            text = self._transcribe_chunk(chunk, sr, language)
            if text:
                segment = TranscriptionSegment(text=text, start=start, end=end)
                yield segment


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI for AudarASR Toolkit."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AudarASR Toolkit - Production-Grade ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python audar_asr_toolkit.py audio.wav
  
  # With SRT output
  python audar_asr_toolkit.py video.mp4 --srt
  
  # Streaming mode
  python audar_asr_toolkit.py audio.wav --stream
  
  # Arabic language
  python audar_asr_toolkit.py arabic.wav --language ar
  
  # Parallel processing with 8 workers
  python audar_asr_toolkit.py long_audio.mp3 --workers 8
        """
    )
    
    parser.add_argument("audio", nargs="+", help="Audio/video file(s)")
    parser.add_argument("-o", "--output", help="Output file/directory")
    parser.add_argument("--language", "-l", default="auto", choices=["ar", "en", "auto"])
    parser.add_argument("--srt", action="store_true", help="Save SRT subtitles")
    parser.add_argument("--vtt", action="store_true", help="Save WebVTT subtitles")
    parser.add_argument("--stream", action="store_true", help="Streaming mode")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk-duration", "-d", type=float, default=DEFAULT_CHUNK_DURATION)
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument("--no-clean", action="store_true", help="Disable hallucination removal")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--model", help="Model path")
    
    args = parser.parse_args()
    
    # Initialize
    asr = AudarASRToolkit(
        model_path=args.model,
        device=args.device,
        num_workers=args.workers,
        chunk_duration=args.chunk_duration,
        use_vad=not args.no_vad,
        remove_hallucinations=not args.no_clean,
    )
    
    # Process files
    for audio_file in args.audio:
        if args.stream:
            print(f"\nStreaming: {audio_file}")
            print("-" * 40)
            for segment in asr.stream(audio_file, language=args.language):
                print(f"[{segment.start:.1f}s] {segment.text}")
            print()
        else:
            result = asr.transcribe(
                audio_file,
                language=args.language,
                save_srt=args.srt,
                save_vtt=args.vtt,
                output_dir=args.output,
            )
            
            print(f"\n{result.text}\n")
            
            if args.output and not (args.srt or args.vtt):
                out_path = Path(args.output)
                if out_path.is_dir():
                    out_file = out_path / f"{Path(audio_file).stem}.txt"
                else:
                    out_file = out_path
                out_file.write_text(result.text, encoding='utf-8')
                print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

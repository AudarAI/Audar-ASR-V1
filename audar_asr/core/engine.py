"""
Audar-ASR Engine
================

Main ASR engine providing a simple, powerful interface for speech recognition.

Copyright (c) 2024-2026 Audar AI. All rights reserved.
"""

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union, Generator, BinaryIO

from audar_asr.core.config import AudarConfig
from audar_asr.core.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
    StreamingTranscription,
)
from audar_asr.utils.audio import AudioProcessor
from audar_asr.utils.streaming import StreamingDisplay, StreamingConfig


class AudarASR:
    """
    Audar-ASR: Production-grade Arabic & English speech recognition.
    
    A high-performance, CPU-optimized ASR engine that accepts any audio
    format and delivers accurate transcriptions with minimal code.
    
    Features:
        - Universal format support (MP3, WAV, M4A, FLAC, OGG, WebM, etc.)
        - CPU-optimized GGUF quantization for fast inference
        - Real-time microphone streaming
        - Arabic & English with automatic language detection
        - Low latency (~0.4x real-time factor)
    
    Quick Start:
        >>> from audar_asr import AudarASR
        >>> 
        >>> # Simple file transcription
        >>> asr = AudarASR()
        >>> result = asr.transcribe("audio.mp3")
        >>> print(result.text)
        
        >>> # With custom configuration
        >>> from audar_asr import AudarConfig
        >>> config = AudarConfig(language="ar", chunk_duration=3.0)
        >>> asr = AudarASR(config)
        >>> result = asr.transcribe("podcast.m4a")
    
    Environment Variables:
        AUDAR_MODEL_PATH: Path to GGUF model file
        AUDAR_MMPROJ_PATH: Path to multimodal projector file
    
    Example:
        >>> # Transcribe any audio format
        >>> result = asr.transcribe("meeting.mp3")
        >>> print(f"Text: {result.text}")
        >>> print(f"Duration: {result.audio_duration:.1f}s")
        >>> print(f"Latency: {result.latency_ms:.0f}ms")
        >>> print(f"RTF: {result.rtf:.2f}x")
        
        >>> # Streaming from microphone
        >>> for result in asr.stream_microphone():
        ...     print(result.text)
        
        >>> # Process audio bytes
        >>> with open("audio.wav", "rb") as f:
        ...     result = asr.transcribe(f)
    """
    
    # ASR prompt template
    _PROMPT_TEMPLATE = (
        "<|im_start|>user\n"
        "Transcribe this audio exactly.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    def __init__(self, config: Optional[AudarConfig] = None):
        """
        Initialize Audar-ASR engine.
        
        Args:
            config: Engine configuration. If None, uses defaults with
                    auto-detected model paths.
        
        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If llama-mtmd-cli is not installed
        """
        self.config = config or AudarConfig()
        self.config.validate()
        
        self._audio = AudioProcessor()
        self._check_backend()
        
        self._warmup_done = False
    
    def _check_backend(self):
        """Verify inference backend is available."""
        try:
            result = subprocess.run(
                ["llama-mtmd-cli", "--version"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "llama-mtmd-cli not found. Install llama.cpp:\n"
                "  brew install llama.cpp  # macOS\n"
                "  # Or build from source: https://github.com/ggerganov/llama.cpp"
            )
    
    def transcribe(
        self,
        audio: Union[str, Path, BinaryIO, bytes],
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Accepts any audio format - automatically converts to optimal format.
        
        Args:
            audio: Audio input. Can be:
                   - File path (str or Path): "/path/to/audio.mp3"
                   - File object: open("audio.wav", "rb")
                   - Raw bytes: audio_data
            prompt: Custom prompt for transcription (advanced)
        
        Returns:
            TranscriptionResult with text, latency, and metadata
        
        Example:
            >>> result = asr.transcribe("meeting.mp3")
            >>> print(result.text)
            "مرحبا بكم في الاجتماع"
            
            >>> result = asr.transcribe(Path("audio.wav"))
            >>> print(result.to_json())
        """
        start_time = time.time()
        
        # Handle different input types
        wav_path = self._prepare_input(audio)
        
        try:
            preprocess_time = time.time()
            
            # Get audio info
            audio_duration = self._audio.get_info(wav_path).duration
            
            # Build inference command
            cmd = self._build_command(wav_path, prompt)
            
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            inference_time = time.time()
            
            # Extract transcript
            transcript = self._extract_transcript(result.stdout + result.stderr)
            
            total_time = inference_time - start_time
            inference_only = inference_time - preprocess_time
            
            return TranscriptionResult(
                text=transcript,
                audio_duration=audio_duration,
                latency_ms=total_time * 1000,
                inference_ms=inference_only * 1000,
                rtf=total_time / audio_duration if audio_duration > 0 else 0,
                language=self.config.language,
                model_info={
                    "model": Path(self.config.model_path).name,
                    "backend": "llama-mtmd-cli",
                },
            )
            
        finally:
            # Clean up temp file if we created one
            if isinstance(audio, (bytes, BinaryIO)):
                self._audio.cleanup(wav_path)
    
    def _prepare_input(self, audio: Union[str, Path, BinaryIO, bytes]) -> str:
        """Prepare audio input for inference."""
        if isinstance(audio, bytes):
            return self._audio.prepare_bytes(audio)
        
        elif hasattr(audio, 'read'):
            # File-like object
            data = audio.read()
            return self._audio.prepare_bytes(data)
        
        else:
            # File path
            return self._audio.prepare(str(audio))
    
    def _build_command(self, wav_path: str, prompt: Optional[str]) -> list:
        """Build inference command."""
        if prompt is None:
            prompt = self._PROMPT_TEMPLATE
        
        return [
            "llama-mtmd-cli",
            "-m", self.config.model_path,
            "--mmproj", self.config.mmproj_path,
            "--audio", wav_path,
            "-p", prompt,
            "-n", str(self.config.max_tokens),
            "--temp", str(self.config.temperature),
            "-ngl", str(self.config.n_gpu_layers),
        ]
    
    def _extract_transcript(self, output: str) -> str:
        """Extract clean transcript from model output."""
        lines = output.split('\n')
        transcript_lines = []
        
        # Patterns to skip (logs, metadata, etc.)
        skip_patterns = [
            'llama_', 'load', 'print_info', 'ggml_', 'clip_', 'alloc_',
            'common_', 'mtmd_', 'init_', 'main:', 'encoding', 'decoding',
            'Writing', 'INFO', 'build:', 'audio', '---', 'http', 'You are',
            'Hello', 'Hi there', 'How are', '<|im_', 'vision hparams',
            'audio hparams', 'github.com', '....', 'assistant', 'token',
        ]
        
        for line in lines:
            # Stop at performance metrics
            if 'llama_perf' in line or 'tokens per second' in line:
                break
            
            # Skip log/metadata lines
            if any(p in line for p in skip_patterns):
                continue
            
            stripped = line.strip()
            
            # Include lines with Arabic or meaningful English content
            has_arabic = re.search(r'[\u0600-\u06FF]', stripped)
            has_english = re.search(r'[a-zA-Z]{3,}', stripped)
            
            if stripped and (has_arabic or has_english):
                # Clean template markers
                cleaned = re.sub(r'<\|.*?\|>', '', stripped).strip()
                if cleaned:
                    transcript_lines.append(cleaned)
        
        return ' '.join(transcript_lines)
    
    def stream_file(
        self,
        audio: Union[str, Path],
        chunk_duration: Optional[float] = None,
        streaming_display: bool = False,
    ) -> Generator[TranscriptionResult, None, None]:
        """
        Stream transcription from an audio file in chunks.
        
        Processes long audio files chunk by chunk, yielding results
        as they become available.
        
        Args:
            audio: Path to audio file
            chunk_duration: Duration of each chunk (default: config value)
            streaming_display: Show streaming text output
            
        Yields:
            TranscriptionResult for each chunk
            
        Example:
            >>> for result in asr.stream_file("podcast.mp3"):
            ...     print(f"[{result.audio_duration:.1f}s] {result.text}")
        """
        chunk_duration = chunk_duration or self.config.chunk_duration
        audio_path = str(audio)
        
        # Setup streaming display
        display = None
        if streaming_display:
            display = StreamingDisplay(StreamingConfig(mode="word", word_delay=0.08))
        
        # Split into chunks
        chunks = self._audio.split_chunks(audio_path, chunk_duration)
        
        try:
            for i, chunk_path in enumerate(chunks, 1):
                result = self.transcribe(chunk_path)
                
                if display and result.text:
                    print(f"\n\033[90m[Chunk {i}] {result.latency_ms:.0f}ms\033[0m")
                    print("\033[92m>>> \033[0m", end="", flush=True)
                    display.stream(result.text)
                
                yield result
                
        finally:
            # Clean up chunk files
            for chunk in chunks:
                self._audio.cleanup(chunk)
    
    def stream_microphone(
        self,
        chunk_duration: Optional[float] = None,
        max_chunks: Optional[int] = None,
        streaming_display: bool = True,
    ) -> Generator[TranscriptionResult, None, None]:
        """
        Stream transcription from microphone in real-time.
        
        Records audio in chunks and transcribes each chunk as it's captured.
        Press Ctrl+C to stop.
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            max_chunks: Maximum chunks to process (None = unlimited)
            streaming_display: Show streaming text output
            
        Yields:
            TranscriptionResult for each chunk
            
        Example:
            >>> for result in asr.stream_microphone():
            ...     print(result.text)
            ...     if "stop" in result.text.lower():
            ...         break
        """
        from audar_asr.core.microphone import RealtimeMicrophone
        
        chunk_duration = chunk_duration or self.config.chunk_duration
        
        # Setup display
        display = None
        if streaming_display:
            display = StreamingDisplay(StreamingConfig(mode="word", word_delay=0.08))
            print("\n" + "=" * 50)
            print("\033[1mAUDAR-ASR REALTIME\033[0m")
            print("Press Ctrl+C to stop")
            print("=" * 50)
        
        mic = RealtimeMicrophone(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_duration=chunk_duration,
        )
        
        def on_chunk_start(n):
            if streaming_display:
                print(f"\n\033[90mRecording chunk {n}...\033[0m")
        
        try:
            for i, audio_path in enumerate(
                mic.stream(max_chunks=max_chunks, on_chunk_start=on_chunk_start),
                1
            ):
                result = self.transcribe(audio_path)
                
                if display:
                    print(f"\033[90m[{result.latency_ms:.0f}ms | RTF: {result.rtf:.2f}]\033[0m")
                    print("\033[92m>>> \033[0m", end="", flush=True)
                    if result.text:
                        display.stream(result.text)
                    else:
                        print("\033[90m(silence)\033[0m")
                
                yield result
                
        except KeyboardInterrupt:
            if streaming_display:
                print("\n\n\033[90mStopped.\033[0m")
        finally:
            mic.close()
    
    def benchmark(
        self,
        audio: Union[str, Path],
        num_runs: int = 3,
    ) -> dict:
        """
        Benchmark inference performance.
        
        Args:
            audio: Audio file to benchmark with
            num_runs: Number of benchmark runs
            
        Returns:
            Dict with average latency, RTF, and individual results
        """
        print(f"\nBenchmarking Audar-ASR ({num_runs} runs)...")
        print("-" * 40)
        
        results = []
        
        for i in range(num_runs):
            result = self.transcribe(audio)
            results.append(result)
            print(f"Run {i+1}: {result.inference_ms:.0f}ms (RTF: {result.rtf:.2f})")
        
        avg_latency = sum(r.inference_ms for r in results) / len(results)
        avg_rtf = sum(r.rtf for r in results) / len(results)
        
        print("-" * 40)
        print(f"Average latency: {avg_latency:.0f}ms")
        print(f"Average RTF: {avg_rtf:.2f}")
        print(f"Audio duration: {results[0].audio_duration:.1f}s")
        
        return {
            "avg_latency_ms": avg_latency,
            "avg_rtf": avg_rtf,
            "audio_duration": results[0].audio_duration,
            "runs": [r.to_dict() for r in results],
        }
    
    def warmup(self):
        """
        Warm up the model with a short inference.
        
        Call this before latency-critical operations to avoid
        cold-start overhead.
        """
        if self._warmup_done:
            return
        
        # Create a short silent audio for warmup
        import wave
        
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            # 0.5 second of silence
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00' * 16000)
            
            self.transcribe(tmp_path)
            self._warmup_done = True
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def __repr__(self) -> str:
        return (
            f"AudarASR(\n"
            f"  model='{Path(self.config.model_path).name}',\n"
            f"  language='{self.config.language}',\n"
            f"  chunk_duration={self.config.chunk_duration}s\n"
            f")"
        )

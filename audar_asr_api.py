#!/usr/bin/env python3
"""
AudarASR API Client - OpenAI-Compatible ASR API

Connect to OpenAI-compatible ASR endpoints for cloud-based transcription.
Supports both synchronous and asynchronous operations.

Features:
- OpenAI-compatible API interface
- Async streaming support
- Automatic audio encoding/decoding
- Retry logic with exponential backoff
- Batch processing with parallel execution

Quick Start:
    from audar_asr_api import AudarASRClient
    
    client = AudarASRClient(base_url="https://asr.example.com/v1")
    result = client.transcribe("audio.wav")
    print(result.text)
    
    # Async streaming
    async for chunk in client.stream_async("audio.wav"):
        print(chunk.text, end="", flush=True)

Author: Audar AI
License: Apache 2.0
"""

import os
import io
import base64
import asyncio
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Generator, AsyncGenerator, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_ASR_URL = "https://asr.nx2ai.com/v1"
DEFAULT_ASR_MODEL_3B = "shahink/arvox-qasr-3B-v13"
DEFAULT_ASR_MODEL_7B = "shahink/arvox-qasr-7B-v5"
DEFAULT_SAMPLE_RATE = 16000
SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4'}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class APITranscriptionResult:
    """
    Result from API transcription.
    """
    text: str
    language: str = "auto"
    duration: float = 0.0
    processing_time: float = 0.0
    model: str = ""
    error: str = ""
    
    @property
    def success(self) -> bool:
        return bool(self.text) and not self.error
    
    @property
    def rtf(self) -> float:
        return self.processing_time / self.duration if self.duration > 0 else 0


@dataclass
class StreamChunk:
    """
    Streaming transcription chunk.
    """
    text: str
    is_final: bool = False
    chunk_index: int = 0


# =============================================================================
# API CLIENT
# =============================================================================

class AudarASRClient:
    """
    OpenAI-compatible ASR API client.
    
    Connects to ASR services that implement the OpenAI chat completions API
    with audio input support.
    
    Example:
        client = AudarASRClient(base_url="https://asr.example.com/v1")
        
        # Simple transcription
        result = client.transcribe("audio.wav")
        print(result.text)
        
        # Batch processing
        results = client.transcribe_batch(["file1.wav", "file2.wav"])
        
        # Async streaming
        async for chunk in client.stream_async("audio.wav"):
            print(chunk.text, end="")
    """
    
    def __init__(
        self,
        base_url: str = DEFAULT_ASR_URL,
        api_key: str = "EMPTY",
        model: str = None,
        model_size: str = "3B",
        temperature: float = 0.4,
        seed: int = 42,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize ASR API client.
        
        Args:
            base_url: ASR API base URL
            api_key: API key (default "EMPTY" for open endpoints)
            model: Model name (auto-configured if not set)
            model_size: "3B" or "7B" (used if model not specified)
            temperature: Generation temperature
            seed: Random seed for reproducibility
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        from openai import OpenAI, AsyncOpenAI
        
        # Auto-configure based on model size
        if model is None:
            if model_size == "7B":
                model = DEFAULT_ASR_MODEL_7B
                if base_url == DEFAULT_ASR_URL:
                    base_url = "https://asr7b.nx2ai.com/v1"
            else:
                model = DEFAULT_ASR_MODEL_3B
        
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize clients
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    def _load_audio(self, audio_path: Union[str, Path]) -> tuple:
        """Load and encode audio file."""
        import torch
        import torchaudio
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sample_rate != DEFAULT_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, DEFAULT_SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convert to base64 MP3
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, DEFAULT_SAMPLE_RATE, format="mp3")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        duration = waveform.shape[1] / DEFAULT_SAMPLE_RATE
        
        return audio_base64, duration
    
    def _build_messages(self, audio_base64: str, system_prompt: str = None):
        """Build chat completion messages with audio."""
        system_content = system_prompt or """You are an AI assistant with expert proficiency in linguistics and multiple languages. 
When provided with audio, generate an accurate transcript in the same language, 
preserving all nuances, tone, and context."""
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_content}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"format": "mp3", "data": audio_base64}
                    }
                ]
            }
        ]
        return messages
    
    def transcribe(
        self,
        audio: Union[str, Path],
        language: str = None,
        system_prompt: str = None,
    ) -> APITranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            audio: Path to audio file
            language: Target language hint
            system_prompt: Custom system prompt
        
        Returns:
            APITranscriptionResult with transcription
        """
        start_time = time.time()
        
        try:
            # Load and encode audio
            audio_base64, duration = self._load_audio(audio)
            
            # Build messages
            messages = self._build_messages(audio_base64, system_prompt)
            
            # Make API call
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["text", "audio"],
                temperature=self.temperature,
                seed=self.seed,
            )
            
            text = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            return APITranscriptionResult(
                text=text,
                language=language or "auto",
                duration=duration,
                processing_time=processing_time,
                model=self.model,
            )
            
        except Exception as e:
            return APITranscriptionResult(
                text="",
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    async def transcribe_async(
        self,
        audio: Union[str, Path],
        language: str = None,
        system_prompt: str = None,
    ) -> APITranscriptionResult:
        """
        Async transcribe audio file.
        
        Args:
            audio: Path to audio file
            language: Target language hint
            system_prompt: Custom system prompt
        
        Returns:
            APITranscriptionResult with transcription
        """
        start_time = time.time()
        
        try:
            # Load and encode audio
            audio_base64, duration = self._load_audio(audio)
            
            # Build messages
            messages = self._build_messages(audio_base64, system_prompt)
            
            # Make API call with streaming
            stream = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["text", "audio"],
                stream=True,
                temperature=self.temperature,
                seed=self.seed,
            )
            
            text_parts = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_parts.append(chunk.choices[0].delta.content)
            
            text = "".join(text_parts).strip()
            processing_time = time.time() - start_time
            
            return APITranscriptionResult(
                text=text,
                language=language or "auto",
                duration=duration,
                processing_time=processing_time,
                model=self.model,
            )
            
        except Exception as e:
            return APITranscriptionResult(
                text="",
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    async def stream_async(
        self,
        audio: Union[str, Path],
        system_prompt: str = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream transcription chunks asynchronously.
        
        Args:
            audio: Path to audio file
            system_prompt: Custom system prompt
        
        Yields:
            StreamChunk with incremental text
        """
        try:
            # Load and encode audio
            audio_base64, _ = self._load_audio(audio)
            
            # Build messages
            messages = self._build_messages(audio_base64, system_prompt)
            
            # Make streaming API call
            stream = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["text", "audio"],
                stream=True,
                temperature=self.temperature,
                seed=self.seed,
            )
            
            chunk_index = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    is_final = chunk.choices[0].finish_reason is not None
                    
                    yield StreamChunk(
                        text=text,
                        is_final=is_final,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
            
            # Ensure final chunk
            yield StreamChunk(text="", is_final=True, chunk_index=chunk_index)
            
        except Exception as e:
            yield StreamChunk(text=f"Error: {e}", is_final=True, chunk_index=0)
    
    def transcribe_batch(
        self,
        audio_files: List[Union[str, Path]],
        num_workers: int = 4,
        **kwargs,
    ) -> List[APITranscriptionResult]:
        """
        Batch transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            num_workers: Number of parallel workers
            **kwargs: Additional arguments for transcribe()
        
        Returns:
            List of APITranscriptionResult
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"  API Batch Transcription: {len(audio_files)} files")
        print(f"  Model: {self.model}")
        print(f"{'='*60}")
        
        total_start = time.time()
        
        # Process files (can be parallelized with asyncio.gather)
        async def process_all():
            tasks = [
                self.transcribe_async(f, **kwargs)
                for f in audio_files
            ]
            return await asyncio.gather(*tasks)
        
        results = asyncio.run(process_all())
        
        # Print results
        for i, (audio_file, result) in enumerate(zip(audio_files, results), 1):
            status = "OK" if result.success else "FAIL"
            print(f"  [{i}/{len(audio_files)}] {Path(audio_file).name}: "
                  f"{status} | {result.duration:.1f}s | RTF: {result.rtf:.2f}")
        
        total_time = time.time() - total_start
        total_duration = sum(r.duration for r in results)
        
        print(f"{'='*60}")
        print(f"  Total: {total_duration:.1f}s audio in {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        return results
    
    async def transcribe_batch_async(
        self,
        audio_files: List[Union[str, Path]],
        **kwargs,
    ) -> List[APITranscriptionResult]:
        """
        Async batch transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            **kwargs: Additional arguments for transcribe_async()
        
        Returns:
            List of APITranscriptionResult
        """
        tasks = [
            self.transcribe_async(f, **kwargs)
            for f in audio_files
        ]
        return await asyncio.gather(*tasks)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI for AudarASR API client."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AudarASR API Client - Cloud Speech Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe with default 3B model
  python audar_asr_api.py audio.wav
  
  # Use 7B model
  python audar_asr_api.py audio.wav --model-size 7B
  
  # Batch transcription
  python audar_asr_api.py *.wav --batch -o results/
  
  # Custom API endpoint
  python audar_asr_api.py audio.wav --url https://custom-asr.com/v1 --model my-model
        """
    )
    
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("-o", "--output", help="Output file/directory path")
    parser.add_argument("--url", default=DEFAULT_ASR_URL, help="ASR API URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--model-size", choices=["3B", "7B"], default="3B",
                        help="Model size (3B or 7B)")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--stream", action="store_true", help="Streaming mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize client
    client = AudarASRClient(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        model_size=args.model_size,
    )
    
    print(f"\nUsing model: {client.model}")
    print(f"API URL: {client.base_url}\n")
    
    # Process based on mode
    if args.stream:
        # Streaming mode
        async def run_stream():
            for audio_file in args.audio:
                print(f"Streaming: {audio_file}")
                print("-" * 40)
                async for chunk in client.stream_async(audio_file):
                    print(chunk.text, end="", flush=True)
                print("\n")
        
        asyncio.run(run_stream())
    
    elif args.batch or len(args.audio) > 1:
        # Batch mode
        results = client.transcribe_batch(args.audio)
        
        for audio_file, result in zip(args.audio, results):
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                out_file = output_dir / f"{Path(audio_file).stem}.txt"
                out_file.write_text(result.text)
                print(f"Saved: {out_file}")
            elif args.verbose:
                print(f"\n{audio_file}:\n{result.text}")
    
    else:
        # Single file
        audio_file = args.audio[0]
        result = client.transcribe(audio_file)
        
        print(f"{'='*60}")
        print(f"  File: {audio_file}")
        print(f"  Duration: {result.duration:.1f}s | RTF: {result.rtf:.2f}")
        print(f"{'='*60}")
        print(f"\n{result.text}\n")
        
        if args.output:
            Path(args.output).write_text(result.text)
            print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

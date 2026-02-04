#!/usr/bin/env python3
"""
AudarASR 3B - Local Inference Engine

Production-grade Arabic-centric ASR with English support.
Optimized for CPU and Apple Silicon (MLX).

Features:
- Local inference with Qwen2.5-Omni architecture
- GPTQ 4-bit quantization for memory efficiency
- Arabic-centric with English support
- Real-time streaming transcription
- CPU and Apple Silicon (MPS/MLX) support

Quick Start:
    from audar_asr_3b_infer import AudarASR3B
    
    asr = AudarASR3B()
    result = asr.transcribe("audio.wav")
    print(result.text)

Author: Audar AI
License: Apache 2.0
"""

import os
import sys
import io
import time
import json
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Generator, Tuple, Union, Any

import numpy as np

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR / "audar_asr_3b"
DEFAULT_OUTPUT = SCRIPT_DIR / "output"
CACHE_DIR = SCRIPT_DIR / ".cache"

# Model identifiers
HF_MODEL_ID = "audarai/qasr-3b-gptq-int4"
LOCAL_MODEL_PATH = MODEL_DIR

# Audio parameters
DEFAULT_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 30 * 60  # 30 minutes
SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.webm'}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TranscriptionResult:
    """
    Result of audio transcription.
    """
    text: str
    language: str = "ar"
    duration: float = 0.0
    processing_time: float = 0.0
    confidence: float = 1.0
    
    @property
    def rtf(self) -> float:
        """Real-time factor."""
        return self.processing_time / self.duration if self.duration > 0 else 0
    
    def __repr__(self):
        return f"TranscriptionResult(text='{self.text[:50]}...', duration={self.duration:.1f}s, rtf={self.rtf:.2f})"


@dataclass
class StreamChunk:
    """
    Streaming transcription chunk.
    """
    text: str
    is_final: bool = False
    chunk_index: int = 0
    timestamp: float = 0.0


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

class AudioProcessor:
    """
    Audio loading and preprocessing utilities.
    """
    
    def __init__(self, target_sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate
        self._resampler_cache = {}
    
    def load(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            (audio_array, sample_rate)
        """
        import torchaudio
        import torch
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy(), self.target_sample_rate
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def get_duration(self, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
        """Get audio duration in seconds."""
        return len(audio) / sample_rate


# =============================================================================
# DEVICE UTILITIES
# =============================================================================

def get_optimal_device() -> str:
    """
    Get optimal device for inference.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get device information for diagnostics."""
    import torch
    
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return info


# =============================================================================
# MAIN ASR ENGINE
# =============================================================================

class AudarASR3B:
    """
    AudarASR 3B - Local Arabic-centric Speech Recognition.
    
    Uses Qwen2.5-Omni architecture with GPTQ 4-bit quantization.
    Optimized for Arabic with English support.
    
    Example:
        asr = AudarASR3B()
        result = asr.transcribe("audio.wav")
        print(result.text)
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        use_flash_attention: bool = True,
        lazy_load: bool = False,
    ):
        """
        Initialize AudarASR 3B engine.
        
        Args:
            model_path: Path to model directory or HF model ID
            device: Device ("cuda", "mps", "cpu", or None for auto)
            use_flash_attention: Use Flash Attention 2 if available
            lazy_load: Defer model loading
        """
        self.model_path = model_path or str(LOCAL_MODEL_PATH)
        self.device = device or get_optimal_device()
        self.use_flash_attention = use_flash_attention
        
        self._model = None
        self._processor = None
        self._audio_processor = AudioProcessor()
        self._initialized = False
        
        if not lazy_load:
            self._initialize()
    
    def _initialize(self):
        """Load model and processor."""
        if self._initialized:
            return
        
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print("=" * 60)
        print("  AudarASR 3B - Arabic-Centric Speech Recognition")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Model: {Path(self.model_path).name}")
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
        
        # Determine torch dtype based on device
        if self.device == "cuda":
            torch_dtype = torch.float16
        elif self.device == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # Check if this is a GPTQ quantized model
        quantize_config_path = Path(self.model_path) / "quantize_config.json"
        is_gptq = quantize_config_path.exists()
        
        if is_gptq:
            # Load with GPTQModel or auto-gptq
            try:
                from gptqmodel import GPTQModel
                self._model = GPTQModel.load(
                    self.model_path,
                    device=self.device,
                    trust_remote_code=True,
                )
            except ImportError:
                try:
                    from auto_gptq import AutoGPTQForCausalLM
                    self._model = AutoGPTQForCausalLM.from_quantized(
                        self.model_path,
                        device=self.device,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                except ImportError:
                    # Fallback to transformers with GPTQ support
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map="auto" if self.device == "cuda" else self.device,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
        else:
            # Load regular model
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
        print("  Ready for transcription!")
        print("=" * 60 + "\n")
        
        self._initialized = True
    
    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = None,
        language: str = "ar",
        prompt: str = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Path to audio file or numpy array
            sample_rate: Sample rate (required if audio is numpy array)
            language: Target language ("ar" for Arabic, "en" for English)
            prompt: Optional prompt to guide transcription
        
        Returns:
            TranscriptionResult with transcribed text
        
        Example:
            result = asr.transcribe("arabic_speech.wav")
            print(result.text)
        """
        self._initialize()
        
        import torch
        
        start_time = time.time()
        
        # Load audio
        if isinstance(audio, (str, Path)):
            audio_array, sr = self._audio_processor.load(audio)
        elif isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate required when audio is numpy array")
            audio_array = audio
            sr = sample_rate
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        
        # Normalize
        audio_array = self._audio_processor.normalize(audio_array)
        duration = self._audio_processor.get_duration(audio_array, sr)
        
        # Build conversation with audio
        system_prompt = self._get_system_prompt(language, prompt)
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": "Transcribe this audio."}
            ]}
        ]
        
        # Apply chat template and process
        text_prompt = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Process audio features
        inputs = self._processor(
            text=[text_prompt],
            audios=[audio_array],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )
        
        # Decode output - skip input tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        transcription = self._processor.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up transcription
        transcription = self._clean_transcription(transcription)
        
        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            text=transcription,
            language=language,
            duration=duration,
            processing_time=processing_time,
        )
    
    def transcribe_batch(
        self,
        audio_files: List[Union[str, Path]],
        language: str = "ar",
        **kwargs,
    ) -> List[TranscriptionResult]:
        """
        Batch transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Target language
            **kwargs: Additional arguments for transcribe()
        
        Returns:
            List of TranscriptionResults
        """
        self._initialize()
        
        print(f"\n{'='*60}")
        print(f"  Batch Transcription: {len(audio_files)} files")
        print(f"{'='*60}")
        
        results = []
        total_start = time.time()
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                result = self.transcribe(audio_file, language=language, **kwargs)
                results.append(result)
                
                print(f"  [{i}/{len(audio_files)}] {Path(audio_file).name}: "
                      f"{result.duration:.1f}s | RTF: {result.rtf:.2f}")
            except Exception as e:
                print(f"  [{i}/{len(audio_files)}] {Path(audio_file).name}: ERROR - {e}")
                results.append(TranscriptionResult(text="", processing_time=0))
        
        total_time = time.time() - total_start
        total_duration = sum(r.duration for r in results)
        
        print(f"{'='*60}")
        print(f"  Total: {total_duration:.1f}s audio in {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        return results
    
    def stream(
        self,
        audio: Union[str, Path],
        language: str = "ar",
        chunk_duration: float = 5.0,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream transcription for real-time processing.
        
        Args:
            audio: Path to audio file
            language: Target language
            chunk_duration: Duration per chunk (seconds)
        
        Yields:
            StreamChunk with incremental text
        """
        self._initialize()
        
        # Load full audio
        audio_array, sr = self._audio_processor.load(audio)
        audio_array = self._audio_processor.normalize(audio_array)
        
        # Split into chunks
        chunk_samples = int(chunk_duration * sr)
        chunks = []
        
        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i:i + chunk_samples]
            chunks.append((i / sr, chunk))
        
        # Process each chunk
        for i, (timestamp, chunk) in enumerate(chunks):
            result = self.transcribe(
                chunk,
                sample_rate=sr,
                language=language,
            )
            
            yield StreamChunk(
                text=result.text,
                is_final=(i == len(chunks) - 1),
                chunk_index=i,
                timestamp=timestamp,
            )
    
    def _get_system_prompt(self, language: str, custom_prompt: str = None) -> str:
        """Get system prompt for transcription."""
        if custom_prompt:
            return custom_prompt
        
        if language == "ar":
            return "You are an expert Arabic speech transcription system. Transcribe the following audio accurately, preserving Arabic text with proper diacritics when audible."
        elif language == "en":
            return "You are an expert English speech transcription system. Transcribe the following audio accurately."
        else:
            return f"Transcribe the following audio in {language}."
    
    def _clean_transcription(self, text: str) -> str:
        """Clean up transcription output."""
        # Remove common artifacts
        text = text.strip()
        
        # Remove any system prompt echoes
        if "Transcribe" in text:
            lines = text.split('\n')
            text = '\n'.join(l for l in lines if "Transcribe" not in l)
        
        return text.strip()
    
    @property
    def device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return get_device_info()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI for AudarASR 3B."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AudarASR 3B - Arabic-Centric Speech Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe Arabic audio
  python audar_asr_3b_infer.py audio.wav
  
  # Transcribe English audio
  python audar_asr_3b_infer.py audio.wav --language en
  
  # Batch transcription
  python audar_asr_3b_infer.py *.wav --batch -o results/
  
  # Streaming mode
  python audar_asr_3b_infer.py audio.wav --stream
  
  # Use specific device
  python audar_asr_3b_infer.py audio.wav --device cpu
        """
    )
    
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("-o", "--output", help="Output file/directory")
    parser.add_argument("--language", "-l", default="ar", choices=["ar", "en"],
                        help="Target language (ar=Arabic, en=English)")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--stream", action="store_true", help="Streaming mode")
    parser.add_argument("--model", help="Model path or HF model ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize ASR
    asr = AudarASR3B(
        model_path=args.model,
        device=args.device,
    )
    
    # Process based on mode
    if args.stream:
        for audio_file in args.audio:
            print(f"\nStreaming: {audio_file}")
            print("-" * 40)
            for chunk in asr.stream(audio_file, language=args.language):
                print(chunk.text, end=" ", flush=True)
            print("\n")
    
    elif args.batch or len(args.audio) > 1:
        results = asr.transcribe_batch(args.audio, language=args.language)
        
        for audio_file, result in zip(args.audio, results):
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                out_file = output_dir / f"{Path(audio_file).stem}.txt"
                out_file.write_text(result.text, encoding='utf-8')
                print(f"Saved: {out_file}")
            elif args.verbose:
                print(f"\n{audio_file}:\n{result.text}")
    
    else:
        audio_file = args.audio[0]
        result = asr.transcribe(audio_file, language=args.language)
        
        print(f"\n{'='*60}")
        print(f"  File: {audio_file}")
        print(f"  Language: {args.language.upper()}")
        print(f"  Duration: {result.duration:.1f}s | RTF: {result.rtf:.2f}")
        print(f"{'='*60}")
        print(f"\n{result.text}\n")
        
        if args.output:
            Path(args.output).write_text(result.text, encoding='utf-8')
            print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

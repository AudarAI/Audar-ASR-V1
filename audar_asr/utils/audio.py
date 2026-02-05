"""
Audar-ASR Audio Processor
=========================

Universal audio format handler with automatic conversion.
Supports: MP3, WAV, M4A, FLAC, OGG, WebM, AAC, WMA, and more.
"""

import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO
from dataclasses import dataclass


# Supported audio formats
SUPPORTED_FORMATS = {
    # Common formats
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
    # Video formats (audio extraction)
    ".mp4", ".webm", ".mkv", ".avi", ".mov",
    # Other audio formats
    ".aac", ".wma", ".aiff", ".au", ".amr", ".3gp",
    # Raw formats
    ".pcm", ".raw",
}


@dataclass
class AudioInfo:
    """Information about an audio file."""
    path: str
    duration: float
    sample_rate: int
    channels: int
    format: str
    bit_depth: int = 16
    file_size: int = 0


class AudioProcessor:
    """
    Universal audio processor for Audar-ASR.
    
    Handles format detection, conversion, and preprocessing for ASR.
    All audio is converted to 16kHz mono PCM WAV for optimal recognition.
    
    Example:
        >>> processor = AudioProcessor()
        >>> wav_path = processor.prepare("podcast.mp3")
        >>> info = processor.get_info("audio.m4a")
        >>> print(f"Duration: {info.duration:.1f}s")
    """
    
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_FORMAT = "pcm_s16le"
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize audio processor.
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Verify ffmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("ffmpeg not working properly")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg\n"
                "  Windows: choco install ffmpeg"
            )
    
    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file format is supported.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if format is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_FORMATS
    
    def get_info(self, file_path: Union[str, Path]) -> AudioInfo:
        """
        Get audio file information.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioInfo with duration, sample rate, channels, etc.
        """
        file_path = str(file_path)
        
        # Use ffprobe to get detailed info
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ValueError(f"Cannot read audio file: {file_path}")
        
        import json
        data = json.loads(result.stdout)
        
        # Extract audio stream info
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break
        
        if not audio_stream:
            raise ValueError(f"No audio stream found in: {file_path}")
        
        format_info = data.get("format", {})
        
        return AudioInfo(
            path=file_path,
            duration=float(format_info.get("duration", 0)),
            sample_rate=int(audio_stream.get("sample_rate", 0)),
            channels=int(audio_stream.get("channels", 0)),
            format=audio_stream.get("codec_name", "unknown"),
            file_size=int(format_info.get("size", 0)),
        )
    
    def prepare(
        self,
        input_path: Union[str, Path],
        output_path: Optional[str] = None,
        normalize: bool = True,
    ) -> str:
        """
        Prepare audio file for ASR (convert to 16kHz mono WAV).
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV (default: temp file)
            normalize: Apply audio normalization
            
        Returns:
            Path to prepared WAV file
        """
        input_path = str(input_path)
        
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        
        # Create output path if not provided
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
            os.close(fd)
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", str(self.TARGET_CHANNELS),
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-c:a", self.TARGET_FORMAT,
        ]
        
        # Add normalization filter
        if normalize:
            cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
        
        cmd.append(output_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")
        
        return output_path
    
    def prepare_bytes(
        self,
        audio_data: bytes,
        input_format: str = "wav",
    ) -> str:
        """
        Prepare audio from bytes for ASR.
        
        Args:
            audio_data: Raw audio bytes
            input_format: Format of input data (e.g., "wav", "mp3")
            
        Returns:
            Path to prepared WAV file
        """
        # Write input to temp file
        fd, input_path = tempfile.mkstemp(suffix=f".{input_format}", dir=self.temp_dir)
        try:
            os.write(fd, audio_data)
            os.close(fd)
            
            return self.prepare(input_path)
        finally:
            # Clean up input temp file
            if Path(input_path).exists():
                Path(input_path).unlink()
    
    def split_chunks(
        self,
        input_path: Union[str, Path],
        chunk_duration: float = 5.0,
    ) -> Tuple[str, ...]:
        """
        Split audio file into chunks.
        
        Args:
            input_path: Path to input audio file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            Tuple of paths to chunk files
        """
        input_path = str(input_path)
        info = self.get_info(input_path)
        
        chunks = []
        offset = 0.0
        chunk_num = 0
        
        while offset < info.duration:
            chunk_num += 1
            
            fd, chunk_path = tempfile.mkstemp(
                suffix=f"_chunk{chunk_num:03d}.wav",
                dir=self.temp_dir
            )
            os.close(fd)
            
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(offset),
                "-t", str(chunk_duration),
                "-ac", str(self.TARGET_CHANNELS),
                "-ar", str(self.TARGET_SAMPLE_RATE),
                "-c:a", self.TARGET_FORMAT,
                chunk_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                chunks.append(chunk_path)
            
            offset += chunk_duration
        
        return tuple(chunks)
    
    def read_wav_frames(self, wav_path: str, chunk_frames: int) -> bytes:
        """
        Read frames from a WAV file.
        
        Args:
            wav_path: Path to WAV file
            chunk_frames: Number of frames to read
            
        Returns:
            Audio data as bytes
        """
        with wave.open(wav_path, 'rb') as wf:
            return wf.readframes(chunk_frames)
    
    def save_wav(
        self,
        audio_data: bytes,
        output_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ):
        """
        Save audio data to WAV file.
        
        Args:
            audio_data: PCM audio data
            output_path: Output file path
            sample_rate: Sample rate in Hz
            channels: Number of channels
            sample_width: Bytes per sample (2 = 16-bit)
        """
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
    
    def cleanup(self, *paths: str):
        """
        Clean up temporary files.
        
        Args:
            *paths: Paths to delete
        """
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

"""
Audar-ASR Microphone Input
==========================

Real-time microphone capture for streaming ASR.
"""

import os
import wave
import tempfile
import threading
import queue
from typing import Optional, Generator, Callable
from pathlib import Path

# Check for PyAudio
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


class RealtimeMicrophone:
    """
    Real-time microphone capture for ASR streaming.
    
    Captures audio in chunks and yields them for transcription.
    
    Example:
        >>> mic = RealtimeMicrophone(chunk_duration=5.0)
        >>> for audio_path in mic.stream():
        ...     result = asr.transcribe(audio_path)
        ...     print(result.text)
        
        >>> # Or capture a single chunk
        >>> audio_path = mic.record_chunk()
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 5.0,
        device_index: Optional[int] = None,
    ):
        """
        Initialize microphone capture.
        
        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            channels: Number of channels (default: 1 = mono)
            chunk_duration: Duration of each chunk in seconds
            device_index: Specific input device index (default: system default)
            
        Raises:
            RuntimeError: If PyAudio is not installed
        """
        if not HAS_PYAUDIO:
            raise RuntimeError(
                "PyAudio required for microphone input.\n"
                "Install with: pip install pyaudio\n"
                "  macOS: brew install portaudio && pip install pyaudio\n"
                "  Ubuntu: sudo apt install python3-pyaudio"
            )
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.device_index = device_index
        
        self._buffer_size = 1024
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._is_running = False
        self._temp_dir = tempfile.gettempdir()
    
    def _init_pyaudio(self):
        """Initialize PyAudio."""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
    
    def _close_pyaudio(self):
        """Close PyAudio."""
        if self._pa:
            self._pa.terminate()
            self._pa = None
    
    def list_devices(self) -> list:
        """
        List available input devices.
        
        Returns:
            List of dicts with device info
        """
        self._init_pyaudio()
        
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate']),
                })
        
        return devices
    
    def record_chunk(self, duration: Optional[float] = None) -> str:
        """
        Record a single audio chunk.
        
        Args:
            duration: Duration in seconds (default: self.chunk_duration)
            
        Returns:
            Path to recorded WAV file
        """
        duration = duration or self.chunk_duration
        self._init_pyaudio()
        
        frames = []
        num_buffers = int(self.sample_rate * duration / self._buffer_size)
        
        # Open stream
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self._buffer_size,
        )
        
        try:
            for _ in range(num_buffers):
                data = stream.read(self._buffer_size, exception_on_overflow=False)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
        
        # Save to temp file
        fd, path = tempfile.mkstemp(suffix=".wav", dir=self._temp_dir)
        os.close(fd)
        
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return path
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_chunks: Optional[int] = None,
        on_chunk_start: Optional[Callable[[int], None]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream audio chunks from microphone.
        
        Args:
            duration: Duration per chunk (default: self.chunk_duration)
            max_chunks: Maximum chunks to record (default: unlimited)
            on_chunk_start: Callback when recording starts
            
        Yields:
            Path to each recorded WAV chunk
        """
        duration = duration or self.chunk_duration
        self._is_running = True
        chunk_num = 0
        
        try:
            while self._is_running:
                chunk_num += 1
                
                if max_chunks and chunk_num > max_chunks:
                    break
                
                if on_chunk_start:
                    on_chunk_start(chunk_num)
                
                path = self.record_chunk(duration)
                yield path
                
                # Clean up after yielding
                Path(path).unlink(missing_ok=True)
                
        except KeyboardInterrupt:
            self._is_running = False
        finally:
            self._close_pyaudio()
    
    def stop(self):
        """Stop streaming."""
        self._is_running = False
    
    def close(self):
        """Close and clean up resources."""
        self.stop()
        self._close_pyaudio()
    
    def __enter__(self):
        """Context manager entry."""
        self._init_pyaudio()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


class AudioQueue:
    """
    Thread-safe audio queue for asynchronous recording.
    
    Records audio continuously in a background thread and
    provides chunks via a queue.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
    ):
        """Initialize audio queue."""
        self.mic = RealtimeMicrophone(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
        )
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start recording in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
    
    def _record_loop(self):
        """Background recording loop."""
        for path in self.mic.stream():
            if not self._running:
                break
            self._queue.put(path)
    
    def get(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get next audio chunk.
        
        Args:
            timeout: Max wait time in seconds
            
        Returns:
            Path to audio chunk or None if timeout
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop recording."""
        self._running = False
        self.mic.stop()
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()

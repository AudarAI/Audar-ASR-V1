"""
Audar-ASR Streaming Display
============================

ChatLLM-style progressive text output for ASR transcriptions.
Supports character-by-character and word-by-word display modes.
"""

import sys
import time
import threading
from typing import Optional, Callable, Literal
from dataclasses import dataclass


@dataclass
class StreamingConfig:
    """
    Configuration for streaming text display.
    
    Attributes:
        mode: Display mode ("char", "word", or "instant")
        char_delay: Delay between characters in seconds
        word_delay: Delay between words in seconds
        punctuation_delay: Extra delay after punctuation
        sentence_delay: Extra delay after sentence endings
        rtl_support: Enable right-to-left language support
    """
    mode: Literal["char", "word", "instant"] = "word"
    char_delay: float = 0.03
    word_delay: float = 0.12
    punctuation_delay: float = 0.3
    sentence_delay: float = 0.5
    rtl_support: bool = True
    
    # Punctuation definitions
    SENTENCE_ENDINGS = (".", "!", "?", "\u061f", "\u3002")  # Includes Arabic/CJK
    PAUSE_PUNCTUATION = (",", ";", ":", "\u060c", "\u061b", "\u2026")


class StreamingDisplay:
    """
    Progressive text display for ASR output.
    
    Creates a typing effect similar to ChatGPT/LLM interfaces.
    Supports Arabic and English with punctuation-aware delays.
    
    Example:
        >>> display = StreamingDisplay()
        >>> display.stream("Hello, world!")
        Hello, world!  # Appears word by word
        
        >>> display = StreamingDisplay(StreamingConfig(mode="char"))
        >>> display.stream("Typing effect")
        Typing effect  # Appears character by character
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize streaming display.
        
        Args:
            config: Display configuration
        """
        self.config = config or StreamingConfig()
        self._interrupt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def _get_word_delay(self, word: str) -> float:
        """Calculate delay after a word based on punctuation."""
        if not word:
            return self.config.word_delay
        
        last_char = word[-1]
        
        if last_char in self.config.SENTENCE_ENDINGS:
            return self.config.word_delay + self.config.sentence_delay
        elif last_char in self.config.PAUSE_PUNCTUATION:
            return self.config.word_delay + self.config.punctuation_delay
        
        return self.config.word_delay
    
    def _get_char_delay(self, char: str) -> float:
        """Calculate delay after a character."""
        if char in self.config.SENTENCE_ENDINGS:
            return self.config.char_delay + self.config.sentence_delay
        elif char in self.config.PAUSE_PUNCTUATION:
            return self.config.char_delay + self.config.punctuation_delay
        
        return self.config.char_delay
    
    def stream(
        self,
        text: str,
        end: str = "\n",
        callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Stream text to output with typing effect.
        
        Args:
            text: Text to display
            end: String to print at end
            callback: Optional callback for each unit (char/word)
            
        Returns:
            True if completed, False if interrupted
        """
        self._interrupt.clear()
        
        if self.config.mode == "instant":
            print(text, end=end, flush=True)
            if callback:
                callback(text)
            return True
        
        elif self.config.mode == "char":
            return self._stream_chars(text, end, callback)
        
        else:  # word mode
            return self._stream_words(text, end, callback)
    
    def _stream_chars(
        self,
        text: str,
        end: str,
        callback: Optional[Callable],
    ) -> bool:
        """Stream text character by character."""
        for char in text:
            if self._interrupt.is_set():
                return False
            
            print(char, end="", flush=True)
            
            if callback:
                callback(char)
            
            delay = self._get_char_delay(char)
            if delay > 0:
                time.sleep(delay)
        
        print(end, end="", flush=True)
        return True
    
    def _stream_words(
        self,
        text: str,
        end: str,
        callback: Optional[Callable],
    ) -> bool:
        """Stream text word by word."""
        words = text.split()
        
        for i, word in enumerate(words):
            if self._interrupt.is_set():
                return False
            
            # Print word with space (except last)
            if i < len(words) - 1:
                print(word + " ", end="", flush=True)
            else:
                print(word, end="", flush=True)
            
            if callback:
                callback(word)
            
            # Add delay between words
            if i < len(words) - 1:
                delay = self._get_word_delay(word)
                if delay > 0:
                    time.sleep(delay)
        
        print(end, end="", flush=True)
        return True
    
    def stream_async(
        self,
        text: str,
        on_complete: Optional[Callable[[bool], None]] = None,
    ):
        """
        Stream text asynchronously.
        
        Args:
            text: Text to display
            on_complete: Callback when streaming completes
        """
        def _run():
            success = self.stream(text)
            if on_complete:
                on_complete(success)
        
        self.stop()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop current streaming."""
        self._interrupt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
    
    def clear_line(self):
        """Clear current line (ANSI escape)."""
        print("\r\033[K", end="", flush=True)


class ASRStreamingOutput:
    """
    Specialized streaming output for ASR results.
    
    Displays transcription chunks with metadata and aggregates
    results for final display.
    
    Example:
        >>> output = ASRStreamingOutput()
        >>> output.display_chunk("مرحبا", chunk_num=1, latency_ms=500)
        [Chunk 1] 500ms
        >>> مرحبا
        >>> output.finalize()
        === COMPLETE TRANSCRIPT ===
        مرحبا بكم
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize ASR streaming output."""
        self.config = config or StreamingConfig(mode="word", word_delay=0.08)
        self.display = StreamingDisplay(self.config)
        self._chunks: list = []
        self._total_latency = 0.0
    
    def display_chunk(
        self,
        text: str,
        chunk_num: int,
        latency_ms: float,
        rtf: float = 0.0,
    ):
        """
        Display a transcription chunk.
        
        Args:
            text: Transcribed text
            chunk_num: Chunk number
            latency_ms: Processing latency in ms
            rtf: Real-time factor
        """
        self._total_latency += latency_ms
        
        # Print chunk header
        rtf_str = f" | RTF: {rtf:.2f}" if rtf > 0 else ""
        print(f"\n\033[90m[Chunk {chunk_num}] {latency_ms:.0f}ms{rtf_str}\033[0m")
        print("\033[92m>>> \033[0m", end="", flush=True)
        
        if text:
            self._chunks.append(text)
            self.display.stream(text)
        else:
            print("\033[90m(silence)\033[0m")
    
    def finalize(self) -> str:
        """
        Display complete transcript and return it.
        
        Returns:
            Complete transcription text
        """
        full_text = " ".join(self._chunks)
        
        print("\n" + "=" * 60)
        print("\033[1mCOMPLETE TRANSCRIPT\033[0m")
        print("=" * 60)
        
        # Use slower speed for final display
        slow_display = StreamingDisplay(
            StreamingConfig(mode="word", word_delay=0.1)
        )
        slow_display.stream(full_text)
        
        print("=" * 60)
        print(f"\033[90mTotal latency: {self._total_latency:.0f}ms\033[0m")
        
        return full_text
    
    def reset(self):
        """Reset for new session."""
        self._chunks = []
        self._total_latency = 0.0

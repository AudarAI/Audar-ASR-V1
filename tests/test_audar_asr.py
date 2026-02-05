"""
Audar-ASR Test Suite
"""

import pytest
from pathlib import Path


class TestConfig:
    """Test configuration module."""
    
    def test_config_creation(self):
        """Test AudarConfig can be created."""
        from audar_asr.core.config import AudarConfig
        
        config = AudarConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.language == "auto"
    
    def test_config_custom_values(self):
        """Test AudarConfig with custom values."""
        from audar_asr.core.config import AudarConfig
        
        config = AudarConfig(
            sample_rate=8000,
            language="ar",
            chunk_duration=3.0,
        )
        assert config.sample_rate == 8000
        assert config.language == "ar"
        assert config.chunk_duration == 3.0


class TestTranscriptionResult:
    """Test transcription result module."""
    
    def test_result_creation(self):
        """Test TranscriptionResult can be created."""
        from audar_asr.core.transcription import TranscriptionResult
        
        result = TranscriptionResult(
            text="Hello world",
            audio_duration=5.0,
            latency_ms=1000.0,
            rtf=0.2,
        )
        assert result.text == "Hello world"
        assert result.is_realtime == True
        assert result.word_count == 2
    
    def test_result_to_dict(self):
        """Test TranscriptionResult.to_dict()."""
        from audar_asr.core.transcription import TranscriptionResult
        
        result = TranscriptionResult(
            text="Test text",
            audio_duration=10.0,
            latency_ms=5000.0,
            rtf=0.5,
        )
        d = result.to_dict()
        assert "text" in d
        assert "audio_duration" in d
        assert d["is_realtime"] == True
    
    def test_result_to_json(self):
        """Test TranscriptionResult.to_json()."""
        from audar_asr.core.transcription import TranscriptionResult
        import json
        
        result = TranscriptionResult(
            text="JSON test",
            audio_duration=5.0,
            latency_ms=1000.0,
            rtf=0.2,
        )
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["text"] == "JSON test"


class TestAudioProcessor:
    """Test audio processor module."""
    
    def test_processor_creation(self):
        """Test AudioProcessor can be created."""
        from audar_asr.utils.audio import AudioProcessor
        
        processor = AudioProcessor()
        assert processor.TARGET_SAMPLE_RATE == 16000
        assert processor.TARGET_CHANNELS == 1
    
    def test_supported_formats(self):
        """Test supported format detection."""
        from audar_asr.utils.audio import AudioProcessor
        
        processor = AudioProcessor()
        assert processor.is_supported("test.mp3") == True
        assert processor.is_supported("test.wav") == True
        assert processor.is_supported("test.m4a") == True
        assert processor.is_supported("test.txt") == False


class TestStreamingDisplay:
    """Test streaming display module."""
    
    def test_display_creation(self):
        """Test StreamingDisplay can be created."""
        from audar_asr.utils.streaming import StreamingDisplay, StreamingConfig
        
        config = StreamingConfig(mode="word", word_delay=0.1)
        display = StreamingDisplay(config)
        assert display.config.mode == "word"
    
    def test_config_defaults(self):
        """Test StreamingConfig defaults."""
        from audar_asr.utils.streaming import StreamingConfig
        
        config = StreamingConfig()
        assert config.mode == "word"
        assert config.rtl_support == True


class TestImports:
    """Test package imports work correctly."""
    
    def test_main_imports(self):
        """Test main package imports."""
        from audar_asr import AudarASR, AudarConfig, TranscriptionResult
        
        assert AudarASR is not None
        assert AudarConfig is not None
        assert TranscriptionResult is not None
    
    def test_utils_imports(self):
        """Test utils imports."""
        from audar_asr.utils import AudioProcessor, StreamingDisplay
        
        assert AudioProcessor is not None
        assert StreamingDisplay is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

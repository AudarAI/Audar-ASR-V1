"""
Audar-ASR Configuration Module
==============================

Centralized configuration for all Audar-ASR components.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class AudarConfig:
    """
    Configuration for Audar-ASR engine.
    
    Attributes:
        model_path: Path to GGUF model file (auto-detected if None)
        mmproj_path: Path to multimodal projector file (auto-detected if None)
        sample_rate: Audio sample rate (default: 16000 Hz)
        channels: Audio channels (default: 1 = mono)
        max_tokens: Maximum tokens to generate (default: 256)
        temperature: Sampling temperature (default: 0.0 = deterministic)
        n_gpu_layers: GPU layers (default: 0 = CPU only)
        chunk_duration: Duration of audio chunks for streaming (seconds)
        language: Primary language hint ("ar", "en", or "auto")
        verbose: Enable verbose logging
    
    Example:
        >>> config = AudarConfig(language="ar", chunk_duration=3.0)
        >>> asr = AudarASR(config)
    """
    
    model_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 1
    max_tokens: int = 256
    temperature: float = 0.0
    n_gpu_layers: int = 0
    chunk_duration: float = 5.0
    language: Literal["ar", "en", "auto"] = "auto"
    verbose: bool = False
    
    # Internal paths (resolved at runtime)
    _models_dir: str = field(default="", init=False)
    
    def __post_init__(self):
        """Resolve model paths if not provided."""
        self._resolve_model_paths()
    
    def _resolve_model_paths(self):
        """Auto-detect model files from package or environment."""
        # Priority: explicit path > env var > package models dir
        
        # Check for explicit paths first
        if self.model_path and self.mmproj_path:
            return
        
        # Check environment variables
        env_model = os.environ.get("AUDAR_MODEL_PATH")
        env_mmproj = os.environ.get("AUDAR_MMPROJ_PATH")
        
        if env_model:
            self.model_path = env_model
        if env_mmproj:
            self.mmproj_path = env_mmproj
        
        if self.model_path and self.mmproj_path:
            return
        
        # Try to find models in package directory
        package_dir = Path(__file__).parent.parent
        models_dir = package_dir / "models"
        
        if not models_dir.exists():
            # Try relative to working directory
            models_dir = Path.cwd() / "models"
        
        if models_dir.exists():
            self._models_dir = str(models_dir)
            
            # Find GGUF files
            gguf_files = list(models_dir.glob("*.gguf"))
            
            for f in gguf_files:
                name = f.name.lower()
                if "mmproj" in name and not self.mmproj_path:
                    self.mmproj_path = str(f)
                elif "q4" in name and not self.model_path:
                    self.model_path = str(f)
                elif "f16" in name and not self.model_path and "mmproj" not in name:
                    self.model_path = str(f)
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Raises:
            FileNotFoundError: If model files are missing
            ValueError: If configuration is invalid
        """
        if not self.model_path:
            raise FileNotFoundError(
                "Model path not specified. Set AUDAR_MODEL_PATH environment variable "
                "or provide model_path in AudarConfig."
            )
        
        if not self.mmproj_path:
            raise FileNotFoundError(
                "MMPROJ path not specified. Set AUDAR_MMPROJ_PATH environment variable "
                "or provide mmproj_path in AudarConfig."
            )
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not Path(self.mmproj_path).exists():
            raise FileNotFoundError(f"MMPROJ file not found: {self.mmproj_path}")
        
        if self.sample_rate not in (8000, 16000, 22050, 44100, 48000):
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        
        if self.channels not in (1, 2):
            raise ValueError(f"Unsupported channel count: {self.channels}")
        
        return True
    
    @classmethod
    def from_env(cls) -> "AudarConfig":
        """
        Create configuration from environment variables.
        
        Environment Variables:
            AUDAR_MODEL_PATH: Path to GGUF model
            AUDAR_MMPROJ_PATH: Path to multimodal projector
            AUDAR_LANGUAGE: Default language (ar/en/auto)
            AUDAR_VERBOSE: Enable verbose mode (1/0)
        """
        return cls(
            model_path=os.environ.get("AUDAR_MODEL_PATH"),
            mmproj_path=os.environ.get("AUDAR_MMPROJ_PATH"),
            language=os.environ.get("AUDAR_LANGUAGE", "auto"),
            verbose=os.environ.get("AUDAR_VERBOSE", "0") == "1",
        )


# Default configuration instance
DEFAULT_CONFIG = AudarConfig()

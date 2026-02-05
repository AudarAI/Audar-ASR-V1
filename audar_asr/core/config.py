"""
Audar-ASR Configuration Module
==============================

Centralized configuration for all Audar-ASR components.

This module provides a flexible model loading system that:
1. Checks for local model files first (development/offline usage)
2. Downloads from HuggingFace if local files are missing
3. Caches downloaded models to avoid repeated downloads
4. Handles network errors gracefully with informative messages
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

# Configure logging
logger = logging.getLogger("audar_asr")

# HuggingFace repository configuration
HUGGINGFACE_REPO_ID = "audarai/audar-asr-turbo-v1-gguf"
MODEL_FILENAME = "audar-asr-q4km.gguf"
MMPROJ_FILENAME = "mmproj-audar-asr-f16.gguf"

# Default cache directory (follows HuggingFace convention)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "audar-asr"


def _download_from_huggingface(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    token: Optional[str] = None,
) -> Path:
    """
    Download a file from HuggingFace Hub with progress tracking.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'audarai/audar-asr-turbo-v1-gguf')
        filename: Name of the file to download
        cache_dir: Local directory to cache downloaded files
        token: Optional HuggingFace token for private repos
        
    Returns:
        Path to the downloaded file
        
    Raises:
        ImportError: If huggingface_hub is not installed
        RuntimeError: If download fails
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model download. Install with:\n"
            "  pip install huggingface_hub\n\n"
            "Or download models manually from:\n"
            f"  https://huggingface.co/{repo_id}"
        )
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already cached
    cached_path = cache_dir / filename
    if cached_path.exists():
        logger.info(f"Using cached model: {cached_path}")
        return cached_path
    
    logger.info(f"Downloading {filename} from HuggingFace...")
    logger.info(f"Repository: https://huggingface.co/{repo_id}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            token=token or os.environ.get("HF_TOKEN"),
        )
        logger.info(f"Download complete: {downloaded_path}")
        return Path(downloaded_path)
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            raise RuntimeError(
                f"Authentication failed for {repo_id}.\n"
                "If this is a private repository, set HF_TOKEN environment variable:\n"
                "  export HF_TOKEN=your_huggingface_token\n\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )
        elif "404" in error_msg:
            raise RuntimeError(
                f"Model file not found: {filename}\n"
                f"Repository: https://huggingface.co/{repo_id}\n"
                "Please verify the repository and filename exist."
            )
        elif "ConnectionError" in error_msg or "timeout" in error_msg.lower():
            raise RuntimeError(
                f"Network error downloading from HuggingFace.\n"
                "Check your internet connection and try again.\n\n"
                "For offline usage, download models manually:\n"
                f"  huggingface-cli download {repo_id} --local-dir {cache_dir}"
            )
        else:
            raise RuntimeError(f"Failed to download {filename}: {e}")


def ensure_models_available(
    cache_dir: Optional[Path] = None,
    repo_id: str = HUGGINGFACE_REPO_ID,
    token: Optional[str] = None,
) -> tuple[Path, Path]:
    """
    Ensure model files are available, downloading from HuggingFace if needed.
    
    This implements a fallback mechanism:
    1. Check local paths (from env vars or explicit paths)
    2. Check cache directory
    3. Download from HuggingFace if not found locally
    
    Args:
        cache_dir: Directory to cache downloaded models
        repo_id: HuggingFace repository ID
        token: Optional HuggingFace token
        
    Returns:
        Tuple of (model_path, mmproj_path)
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    # Try to find model file
    model_path = _find_or_download(
        filename=MODEL_FILENAME,
        env_var="AUDAR_MODEL_PATH",
        cache_dir=cache_dir,
        repo_id=repo_id,
        token=token,
    )
    
    # Try to find mmproj file
    mmproj_path = _find_or_download(
        filename=MMPROJ_FILENAME,
        env_var="AUDAR_MMPROJ_PATH",
        cache_dir=cache_dir,
        repo_id=repo_id,
        token=token,
    )
    
    return model_path, mmproj_path


def _find_or_download(
    filename: str,
    env_var: str,
    cache_dir: Path,
    repo_id: str,
    token: Optional[str],
) -> Path:
    """
    Find a model file locally or download from HuggingFace.
    
    Priority:
    1. Environment variable path
    2. Local cache directory
    3. Current working directory / models
    4. Package directory / models
    5. Download from HuggingFace
    """
    # 1. Check environment variable
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).exists():
        logger.debug(f"Found {filename} from {env_var}: {env_path}")
        return Path(env_path)
    
    # 2. Check cache directory
    cached_path = cache_dir / filename
    if cached_path.exists():
        logger.debug(f"Found {filename} in cache: {cached_path}")
        return cached_path
    
    # 3. Check current working directory
    cwd_models = Path.cwd() / "models" / filename
    if cwd_models.exists():
        logger.debug(f"Found {filename} in cwd/models: {cwd_models}")
        return cwd_models
    
    # 4. Check package directory
    package_models = Path(__file__).parent.parent / "models" / filename
    if package_models.exists():
        logger.debug(f"Found {filename} in package/models: {package_models}")
        return package_models
    
    # 5. Download from HuggingFace
    logger.info(f"Model file not found locally: {filename}")
    logger.info(f"Downloading from HuggingFace: {repo_id}")
    
    return _download_from_huggingface(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
    )


@dataclass
class AudarConfig:
    """
    Configuration for Audar-ASR engine.
    
    The configuration supports flexible model loading with automatic HuggingFace
    download when local files are not found.
    
    Model Loading Priority:
        1. Explicit paths provided in constructor
        2. Environment variables (AUDAR_MODEL_PATH, AUDAR_MMPROJ_PATH)
        3. Local cache directory (~/.cache/audar-asr/)
        4. ./models/ directory in current working directory
        5. Automatic download from HuggingFace (audarai/audar-asr-turbo-v1-gguf)
    
    Attributes:
        model_path: Path to GGUF model file (auto-detected/downloaded if None)
        mmproj_path: Path to multimodal projector file (auto-detected/downloaded if None)
        sample_rate: Audio sample rate (default: 16000 Hz)
        channels: Audio channels (default: 1 = mono)
        max_tokens: Maximum tokens to generate (default: 256)
        temperature: Sampling temperature (default: 0.0 = deterministic)
        n_gpu_layers: GPU layers (default: 0 = CPU only)
        chunk_duration: Duration of audio chunks for streaming (seconds)
        language: Primary language hint ("ar", "en", or "auto")
        verbose: Enable verbose logging
        cache_dir: Directory for caching downloaded models
        hf_token: HuggingFace token for private repository access
        auto_download: Enable automatic download from HuggingFace (default: True)
    
    Example:
        >>> # Auto-download from HuggingFace
        >>> config = AudarConfig()
        >>> asr = AudarASR(config)
        
        >>> # Use local files
        >>> config = AudarConfig(
        ...     model_path="/path/to/model.gguf",
        ...     mmproj_path="/path/to/mmproj.gguf",
        ...     auto_download=False
        ... )
        
        >>> # Custom cache directory
        >>> config = AudarConfig(cache_dir="/opt/models/audar-asr")
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
    
    # Model loading configuration
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None
    auto_download: bool = True
    
    # Internal paths (resolved at runtime)
    _models_dir: str = field(default="", init=False)
    
    def __post_init__(self):
        """Resolve model paths if not provided."""
        self._resolve_model_paths()
    
    def _resolve_model_paths(self):
        """
        Auto-detect model files with local-first fallback to HuggingFace download.
        
        Priority:
        1. Explicit paths provided in constructor
        2. Environment variables (AUDAR_MODEL_PATH, AUDAR_MMPROJ_PATH)
        3. Local cache directory (~/.cache/audar-asr/)
        4. ./models/ directory (current working directory or package)
        5. Download from HuggingFace (if auto_download=True)
        """
        # Check for explicit paths first
        if self.model_path and self.mmproj_path:
            if Path(self.model_path).exists() and Path(self.mmproj_path).exists():
                return
        
        # Determine cache directory
        cache_dir = Path(self.cache_dir) if self.cache_dir else DEFAULT_CACHE_DIR
        
        # Try to resolve model path
        if not self.model_path or not Path(self.model_path).exists():
            resolved = self._find_local_model(MODEL_FILENAME, "AUDAR_MODEL_PATH", cache_dir)
            if resolved:
                self.model_path = str(resolved)
        
        # Try to resolve mmproj path
        if not self.mmproj_path or not Path(self.mmproj_path).exists():
            resolved = self._find_local_model(MMPROJ_FILENAME, "AUDAR_MMPROJ_PATH", cache_dir)
            if resolved:
                self.mmproj_path = str(resolved)
    
    def _find_local_model(
        self,
        filename: str,
        env_var: str,
        cache_dir: Path,
    ) -> Optional[Path]:
        """
        Find a model file in local directories only (no download).
        
        Args:
            filename: Model filename to search for
            env_var: Environment variable name to check
            cache_dir: Cache directory to search
            
        Returns:
            Path to model file if found, None otherwise
        """
        # 1. Check environment variable
        env_path = os.environ.get(env_var)
        if env_path and Path(env_path).exists():
            logger.debug(f"Found {filename} from {env_var}: {env_path}")
            return Path(env_path)
        
        # 2. Check cache directory
        cached_path = cache_dir / filename
        if cached_path.exists():
            logger.debug(f"Found {filename} in cache: {cached_path}")
            return cached_path
        
        # 3. Check current working directory
        cwd_models = Path.cwd() / "models" / filename
        if cwd_models.exists():
            logger.debug(f"Found {filename} in cwd/models: {cwd_models}")
            return cwd_models
        
        # 4. Check package directory
        package_dir = Path(__file__).parent.parent
        package_models = package_dir / "models" / filename
        if package_models.exists():
            logger.debug(f"Found {filename} in package/models: {package_models}")
            return package_models
        
        # 5. Fallback: check for any matching GGUF in models directories
        for models_dir in [Path.cwd() / "models", package_dir / "models"]:
            if models_dir.exists():
                for f in models_dir.glob("*.gguf"):
                    name = f.name.lower()
                    if "mmproj" in filename.lower() and "mmproj" in name:
                        return f
                    elif "mmproj" not in filename.lower() and "mmproj" not in name:
                        if "q4" in name or "f16" in name:
                            return f
        
        return None
    
    def validate(self) -> bool:
        """
        Validate configuration, downloading models from HuggingFace if needed.
        
        If auto_download is True and model files are not found locally,
        this method will attempt to download them from HuggingFace.
        
        Raises:
            FileNotFoundError: If model files are missing and cannot be downloaded
            ValueError: If configuration is invalid
            ImportError: If huggingface_hub is needed but not installed
            RuntimeError: If download fails
        """
        cache_dir = Path(self.cache_dir) if self.cache_dir else DEFAULT_CACHE_DIR
        
        # Check and potentially download model file
        if not self.model_path or not Path(self.model_path).exists():
            if self.auto_download:
                logger.info("Model file not found locally, downloading from HuggingFace...")
                model_path = _download_from_huggingface(
                    repo_id=HUGGINGFACE_REPO_ID,
                    filename=MODEL_FILENAME,
                    cache_dir=cache_dir,
                    token=self.hf_token,
                )
                self.model_path = str(model_path)
            else:
                raise FileNotFoundError(
                    "Model path not specified. Either:\n"
                    "  1. Set AUDAR_MODEL_PATH environment variable\n"
                    "  2. Provide model_path in AudarConfig\n"
                    "  3. Enable auto_download=True to download from HuggingFace\n\n"
                    "Manual download:\n"
                    f"  huggingface-cli download {HUGGINGFACE_REPO_ID} --local-dir {cache_dir}"
                )
        
        # Check and potentially download mmproj file
        if not self.mmproj_path or not Path(self.mmproj_path).exists():
            if self.auto_download:
                logger.info("MMPROJ file not found locally, downloading from HuggingFace...")
                mmproj_path = _download_from_huggingface(
                    repo_id=HUGGINGFACE_REPO_ID,
                    filename=MMPROJ_FILENAME,
                    cache_dir=cache_dir,
                    token=self.hf_token,
                )
                self.mmproj_path = str(mmproj_path)
            else:
                raise FileNotFoundError(
                    "MMPROJ path not specified. Either:\n"
                    "  1. Set AUDAR_MMPROJ_PATH environment variable\n"
                    "  2. Provide mmproj_path in AudarConfig\n"
                    "  3. Enable auto_download=True to download from HuggingFace\n\n"
                    "Manual download:\n"
                    f"  huggingface-cli download {HUGGINGFACE_REPO_ID} --local-dir {cache_dir}"
                )
        
        # Final existence check
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not Path(self.mmproj_path).exists():
            raise FileNotFoundError(f"MMPROJ file not found: {self.mmproj_path}")
        
        # Validate other parameters
        if self.sample_rate not in (8000, 16000, 22050, 44100, 48000):
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        
        if self.channels not in (1, 2):
            raise ValueError(f"Unsupported channel count: {self.channels}")
        
        logger.info(f"Model loaded: {Path(self.model_path).name}")
        logger.info(f"MMPROJ loaded: {Path(self.mmproj_path).name}")
        
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
            AUDAR_CACHE_DIR: Directory for caching downloaded models
            HF_TOKEN: HuggingFace token for private repos
            AUDAR_AUTO_DOWNLOAD: Enable auto-download (1/0, default: 1)
        """
        return cls(
            model_path=os.environ.get("AUDAR_MODEL_PATH"),
            mmproj_path=os.environ.get("AUDAR_MMPROJ_PATH"),
            language=os.environ.get("AUDAR_LANGUAGE", "auto"),
            verbose=os.environ.get("AUDAR_VERBOSE", "0") == "1",
            cache_dir=os.environ.get("AUDAR_CACHE_DIR"),
            hf_token=os.environ.get("HF_TOKEN"),
            auto_download=os.environ.get("AUDAR_AUTO_DOWNLOAD", "1") == "1",
        )


def download_models(
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Download Audar-ASR models from HuggingFace.
    
    Convenience function to pre-download models before running ASR.
    Useful for Docker builds or deployment scripts.
    
    Args:
        cache_dir: Directory to download models to
        token: HuggingFace token (or set HF_TOKEN env var)
        force: Force re-download even if cached
        
    Returns:
        Tuple of (model_path, mmproj_path)
        
    Example:
        >>> # Download models
        >>> model_path, mmproj_path = download_models()
        >>> print(f"Models downloaded to: {model_path.parent}")
        
        >>> # Use in Dockerfile
        >>> # RUN python -c "from audar_asr import download_models; download_models()"
    """
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Remove cached files if force=True
    if force:
        for filename in [MODEL_FILENAME, MMPROJ_FILENAME]:
            cached_file = cache_path / filename
            if cached_file.exists():
                cached_file.unlink()
                logger.info(f"Removed cached file: {cached_file}")
    
    model_path = _download_from_huggingface(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir=cache_path,
        token=token,
    )
    
    mmproj_path = _download_from_huggingface(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=MMPROJ_FILENAME,
        cache_dir=cache_path,
        token=token,
    )
    
    logger.info(f"Models ready at: {cache_path}")
    return model_path, mmproj_path


# Default configuration instance
DEFAULT_CONFIG = AudarConfig()

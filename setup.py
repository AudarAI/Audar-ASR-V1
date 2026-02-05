#!/usr/bin/env python3
"""
Audar-ASR Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="audar-asr",
    version="1.0.0",
    author="Audar AI",
    author_email="contact@audarai.com",
    description="Production-grade Arabic & English Speech Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AudarAI/Audar-ASR",
    project_urls={
        "Bug Reports": "https://github.com/AudarAI/Audar-ASR/issues",
        "Source": "https://github.com/AudarAI/Audar-ASR",
        "Documentation": "https://github.com/AudarAI/Audar-ASR#readme",
    },
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.8",
    install_requires=[
        "huggingface_hub>=0.20.0",  # Model download from HuggingFace
    ],
    extras_require={
        "microphone": ["pyaudio>=0.2.11"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
        ],
        "all": [
            "pyaudio>=0.2.11",
            "huggingface_hub>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audar-asr=audar_asr.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords=[
        "asr",
        "speech-recognition",
        "arabic",
        "whisper",
        "qwen",
        "gguf",
        "cpu-inference",
        "real-time",
        "transcription",
    ],
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False,
)

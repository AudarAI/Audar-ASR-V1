#!/usr/bin/env python3
"""
AudarASR - Production-Grade Speech Recognition
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="audar-asr",
    version="1.0.0",
    author="Audar AI",
    author_email="contact@audar.ai",
    description="Production-Grade Speech Recognition with Hugging Face Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AudarAI/Audar-ASR-V1",
    py_modules=["audar_asr", "audar_asr_api", "audar_asr_benchmark"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "jiwer>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "vad": [
            "silero-vad",
        ],
        "benchmark": [
            "datasets>=2.14.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "loguru>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audar-asr=audar_asr:main",
            "audar-asr-api=audar_asr_api:main",
            "audar-asr-benchmark=audar_asr_benchmark:main",
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
    keywords="asr speech-recognition whisper transformers deep-learning",
    license="Apache-2.0",
)

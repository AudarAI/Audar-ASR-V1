#!/usr/bin/env python3
"""
Basic File Transcription Example
================================

Demonstrates simple file transcription with Audar-ASR.

Usage:
    python basic_transcription.py audio.mp3
    python basic_transcription.py --help
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from audar_asr import AudarASR


def main():
    # Get audio file from command line or use default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        print("Usage: python basic_transcription.py <audio_file>")
        print("\nExample:")
        print("  python basic_transcription.py meeting.mp3")
        print("  python basic_transcription.py podcast.m4a")
        return
    
    # Check file exists
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        return
    
    print(f"Transcribing: {audio_path}")
    print("-" * 50)
    
    # Initialize ASR (auto-detects model paths from environment)
    asr = AudarASR()
    
    # Transcribe - that's it! Just one line.
    result = asr.transcribe(audio_path)
    
    # Print results
    print(f"\nTranscript:\n{result.text}\n")
    print("-" * 50)
    print(f"Audio Duration: {result.audio_duration:.1f} seconds")
    print(f"Processing Time: {result.latency_ms:.0f} ms")
    print(f"Real-Time Factor: {result.rtf:.2f}x")
    print(f"Speed: {'Faster' if result.is_realtime else 'Slower'} than realtime")


if __name__ == "__main__":
    main()

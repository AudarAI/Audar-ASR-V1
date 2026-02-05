#!/usr/bin/env python3
"""
Real-time Microphone Transcription Example
==========================================

Demonstrates real-time speech recognition from microphone.

Usage:
    python microphone_streaming.py
    python microphone_streaming.py --chunk-duration 3

Press Ctrl+C to stop recording.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from audar_asr import AudarASR, AudarConfig


def main():
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription with Audar-ASR"
    )
    parser.add_argument(
        "--chunk-duration", "-c",
        type=float,
        default=5.0,
        help="Duration of each audio chunk in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to record (default: unlimited)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("AUDAR-ASR REAL-TIME TRANSCRIPTION")
    print("=" * 60)
    print(f"Chunk duration: {args.chunk_duration} seconds")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Configure ASR
    config = AudarConfig(chunk_duration=args.chunk_duration)
    asr = AudarASR(config)
    
    # Collect all transcripts
    all_transcripts = []
    total_audio_duration = 0
    total_latency = 0
    
    try:
        # Stream from microphone with real-time display
        for result in asr.stream_microphone(
            chunk_duration=args.chunk_duration,
            max_chunks=args.max_chunks,
            streaming_display=True,
        ):
            if result.text:
                all_transcripts.append(result.text)
            total_audio_duration += result.audio_duration
            total_latency += result.latency_ms
            
    except KeyboardInterrupt:
        pass
    
    # Print final summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"\nComplete Transcript:")
    print(" ".join(all_transcripts))
    print(f"\nTotal Audio: {total_audio_duration:.1f} seconds")
    print(f"Total Processing: {total_latency:.0f} ms")
    if total_audio_duration > 0:
        print(f"Average RTF: {total_latency / 1000 / total_audio_duration:.2f}x")


if __name__ == "__main__":
    main()

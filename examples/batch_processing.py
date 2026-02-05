#!/usr/bin/env python3
"""
Batch Processing Example
========================

Demonstrates batch transcription of multiple audio files.

Usage:
    python batch_processing.py audio_folder/
    python batch_processing.py *.mp3
    python batch_processing.py file1.wav file2.mp3 file3.m4a
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from audar_asr import AudarASR


def find_audio_files(paths: list) -> list:
    """Find all audio files from paths (files or directories)."""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mp4'}
    files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_file():
            if path.suffix.lower() in audio_extensions:
                files.append(path)
        elif path.is_dir():
            for ext in audio_extensions:
                files.extend(path.glob(f"**/*{ext}"))
    
    return sorted(files)


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_processing.py <audio_files_or_folder>")
        print("\nExamples:")
        print("  python batch_processing.py audio_folder/")
        print("  python batch_processing.py *.mp3")
        print("  python batch_processing.py file1.wav file2.mp3")
        return
    
    # Find all audio files
    audio_files = find_audio_files(sys.argv[1:])
    
    if not audio_files:
        print("No audio files found.")
        return
    
    print("=" * 60)
    print(f"AUDAR-ASR BATCH PROCESSING")
    print(f"Found {len(audio_files)} audio files")
    print("=" * 60)
    
    # Initialize ASR once
    asr = AudarASR()
    
    # Process all files
    results = []
    total_audio_duration = 0
    total_latency = 0
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_path.name}")
        
        try:
            result = asr.transcribe(audio_path)
            
            results.append({
                "file": str(audio_path),
                "transcript": result.text,
                "duration_s": result.audio_duration,
                "latency_ms": result.latency_ms,
                "rtf": result.rtf,
            })
            
            total_audio_duration += result.audio_duration
            total_latency += result.latency_ms
            
            # Show preview
            preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
            print(f"  Duration: {result.audio_duration:.1f}s | Latency: {result.latency_ms:.0f}ms")
            print(f"  Preview: {preview}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "file": str(audio_path),
                "error": str(e),
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(audio_files)}")
    print(f"Total audio duration: {total_audio_duration:.1f} seconds")
    print(f"Total processing time: {total_latency:.0f} ms")
    if total_audio_duration > 0:
        print(f"Average RTF: {total_latency / 1000 / total_audio_duration:.2f}x")
    
    # Save results to JSON
    output_file = f"transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

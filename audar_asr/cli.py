#!/usr/bin/env python3
"""
Audar-ASR Command Line Interface
================================

Production-grade CLI for Arabic & English speech recognition.

Usage:
    audar-asr transcribe audio.mp3
    audar-asr stream --microphone
    audar-asr benchmark audio.wav --runs 5

Copyright (c) 2024-2026 Audar AI. All rights reserved.
"""

import sys
import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="audar-asr",
        description="Audar-ASR: Production-grade Arabic & English Speech Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  audar-asr transcribe audio.mp3                 # Transcribe a file
  audar-asr transcribe podcast.m4a --json        # Output as JSON
  audar-asr stream audio.wav --chunk-duration 5  # Stream from file
  audar-asr stream --microphone                  # Real-time mic input
  audar-asr benchmark audio.wav --runs 5         # Benchmark performance
  audar-asr info audio.mp3                       # Show audio info

Environment Variables:
  AUDAR_MODEL_PATH    Path to GGUF model file
  AUDAR_MMPROJ_PATH   Path to multimodal projector file

For more info: https://github.com/AudarAI/Audar-ASR
        """,
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 1.0.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Transcribe command
    transcribe = subparsers.add_parser(
        "transcribe",
        help="Transcribe an audio file",
    )
    transcribe.add_argument("audio", type=str, help="Path to audio file")
    transcribe.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )
    transcribe.add_argument(
        "--model", "-m",
        type=str,
        help="Path to GGUF model",
    )
    transcribe.add_argument(
        "--mmproj",
        type=str,
        help="Path to multimodal projector",
    )
    
    # Stream command
    stream = subparsers.add_parser(
        "stream",
        help="Stream transcription (file or microphone)",
    )
    stream.add_argument(
        "audio",
        type=str,
        nargs="?",
        help="Path to audio file (omit for microphone)",
    )
    stream.add_argument(
        "--microphone", "--mic",
        action="store_true",
        help="Use microphone input",
    )
    stream.add_argument(
        "--chunk-duration", "-c",
        type=float,
        default=5.0,
        help="Duration of each chunk in seconds (default: 5.0)",
    )
    stream.add_argument(
        "--model", "-m",
        type=str,
        help="Path to GGUF model",
    )
    stream.add_argument(
        "--mmproj",
        type=str,
        help="Path to multimodal projector",
    )
    
    # Benchmark command
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Benchmark inference performance",
    )
    benchmark.add_argument("audio", type=str, help="Path to audio file")
    benchmark.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)",
    )
    benchmark.add_argument(
        "--model", "-m",
        type=str,
        help="Path to GGUF model",
    )
    benchmark.add_argument(
        "--mmproj",
        type=str,
        help="Path to multimodal projector",
    )
    
    # Info command
    info = subparsers.add_parser(
        "info",
        help="Show audio file information",
    )
    info.add_argument("audio", type=str, help="Path to audio file")
    
    return parser


def cmd_transcribe(args):
    """Handle transcribe command."""
    from audar_asr import AudarASR, AudarConfig
    
    config = AudarConfig(
        model_path=args.model,
        mmproj_path=args.mmproj,
    )
    
    asr = AudarASR(config)
    result = asr.transcribe(args.audio)
    
    if args.json:
        print(result.to_json())
    else:
        print(f"\n{result.text}\n")
        print(f"Duration: {result.audio_duration:.1f}s | "
              f"Latency: {result.latency_ms:.0f}ms | "
              f"RTF: {result.rtf:.2f}")


def cmd_stream(args):
    """Handle stream command."""
    from audar_asr import AudarASR, AudarConfig
    
    config = AudarConfig(
        model_path=args.model,
        mmproj_path=args.mmproj,
        chunk_duration=args.chunk_duration,
    )
    
    asr = AudarASR(config)
    
    if args.microphone or not args.audio:
        # Microphone streaming
        all_text = []
        try:
            for result in asr.stream_microphone(
                chunk_duration=args.chunk_duration,
                streaming_display=True,
            ):
                if result.text:
                    all_text.append(result.text)
        except KeyboardInterrupt:
            pass
        
        print("\n" + "=" * 50)
        print("COMPLETE TRANSCRIPT")
        print("=" * 50)
        print(" ".join(all_text))
    else:
        # File streaming
        all_text = []
        for result in asr.stream_file(
            args.audio,
            chunk_duration=args.chunk_duration,
            streaming_display=True,
        ):
            if result.text:
                all_text.append(result.text)
        
        print("\n" + "=" * 50)
        print("COMPLETE TRANSCRIPT")
        print("=" * 50)
        print(" ".join(all_text))


def cmd_benchmark(args):
    """Handle benchmark command."""
    from audar_asr import AudarASR, AudarConfig
    
    config = AudarConfig(
        model_path=args.model,
        mmproj_path=args.mmproj,
    )
    
    asr = AudarASR(config)
    asr.benchmark(args.audio, num_runs=args.runs)


def cmd_info(args):
    """Handle info command."""
    from audar_asr.utils.audio import AudioProcessor
    
    processor = AudioProcessor()
    info = processor.get_info(args.audio)
    
    print(f"\nFile: {info.path}")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Sample Rate: {info.sample_rate} Hz")
    print(f"Channels: {info.channels}")
    print(f"Format: {info.format}")
    print(f"Size: {info.file_size / 1024 / 1024:.2f} MB")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    commands = {
        "transcribe": cmd_transcribe,
        "stream": cmd_stream,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
    }
    
    try:
        commands[args.command](args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

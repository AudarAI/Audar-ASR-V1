#!/usr/bin/env python3
"""Offline transcription of arbitrary-length audio (chunked at the 30 s encoder context).

    python examples/transcribe_long.py path/to/meeting.wav
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audar_asr import HF_REPOS, DEFAULT_SYSTEM_AR, DEFAULT_SYSTEM_EN, load_model, transcribe_long


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe long audio with Audar-ASR-V1-Flash.")
    ap.add_argument("audio", help="Path to a mono audio file of any length.")
    ap.add_argument("--model", default=HF_REPOS["flash"], help="HF repo id or local path.")
    ap.add_argument("--lang", choices=["ar", "en"], default="ar")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    system = DEFAULT_SYSTEM_AR if args.lang == "ar" else DEFAULT_SYSTEM_EN
    model, proc = load_model(args.model, device=args.device)
    print(transcribe_long(model, proc, args.audio, system=system))


if __name__ == "__main__":
    main()

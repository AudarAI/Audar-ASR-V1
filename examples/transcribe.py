#!/usr/bin/env python3
"""Minimal single-clip transcription (<=30 s) with Audar-ASR-V1-Flash via 🤗 Transformers.

    pip install -r examples/requirements.txt
    python examples/transcribe.py path/to/clip.wav
    python examples/transcribe.py path/to/english.wav --lang en

Weights download automatically from https://huggingface.co/audarai/Audar-ASR-V1-Flash
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import the repo's audar_asr package
from audar_asr import HF_REPOS, DEFAULT_SYSTEM_AR, DEFAULT_SYSTEM_EN, load_model, transcribe


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a <=30 s audio clip with Audar-ASR-V1-Flash.")
    ap.add_argument("audio", help="Path to a mono audio file (any format librosa reads).")
    ap.add_argument("--model", default=HF_REPOS["flash"], help="HF repo id or local path.")
    ap.add_argument("--lang", choices=["ar", "en"], default="ar", help="Prompt language (default: ar).")
    ap.add_argument("--device", default="cuda:0", help="Torch device (e.g. cuda:0, cpu).")
    args = ap.parse_args()

    system = DEFAULT_SYSTEM_AR if args.lang == "ar" else DEFAULT_SYSTEM_EN
    model, proc = load_model(args.model, device=args.device)
    print(transcribe(model, proc, args.audio, system=system))


if __name__ == "__main__":
    main()

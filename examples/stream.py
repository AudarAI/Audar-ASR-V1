#!/usr/bin/env python3
"""Realtime-style streaming transcription (LocalAgreement-2) with Audar-ASR-V1-Flash.

Re-decodes a sliding <=30 s window and commits a word only once two consecutive decodes
agree on it, so committed text is stable and never rewrites. Prints committed text plus the
current unstable tail in brackets as the stream advances.

    python examples/stream.py path/to/long.wav

This is the local reference policy; Audar's production engine serves the same policy with
sub-250 ms latency over a WebSocket — see https://www.audarai.com
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audar_asr import HF_REPOS, DEFAULT_SYSTEM_AR, DEFAULT_SYSTEM_EN, load_model, stream_transcribe


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream-transcribe audio with Audar-ASR-V1-Flash.")
    ap.add_argument("audio", help="Path to a mono audio file.")
    ap.add_argument("--model", default=HF_REPOS["flash"], help="HF repo id or local path.")
    ap.add_argument("--lang", choices=["ar", "en"], default="ar")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hop", type=float, default=2.0, help="Seconds of audio revealed per step.")
    args = ap.parse_args()

    system = DEFAULT_SYSTEM_AR if args.lang == "ar" else DEFAULT_SYSTEM_EN
    model, proc = load_model(args.model, device=args.device)
    for ev in stream_transcribe(model, proc, args.audio, system=system, hop_s=args.hop):
        tail = f"  [{ev['pending']}]" if ev["pending"] else ""
        print(f"[{ev['t']:6.1f}s] {ev['committed']}{tail}", flush=True)
    print("\n--- final ---")


if __name__ == "__main__":
    main()

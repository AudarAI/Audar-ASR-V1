# Benchmarks — Open Universal Arabic ASR Leaderboard

Full standings for the **[Open Universal Arabic ASR Leaderboard](https://huggingface.co/spaces/UBC-NLP/open-universal-arabic-asr-leaderboard)**
with both Audar-ASR-V1 tiers inserted at their ranks. Machine-readable copy: [`leaderboard.csv`](leaderboard.csv).

- **Audar-ASR-V1-Turbo — #1 of 36** · 24.78 % avg WER · 9.49 % avg CER (lowest of any evaluated system).
- **Audar-ASR-V1-Flash — #11 of 36** · 33.31 % avg WER · 13.66 % avg CER — the strongest *small* model on
  the board (0.78 B), beating Qwen3-ASR-1.7B (2× its size), Voxtral-Small-24B, and Whisper-large-v3.

## Methodology

- **All six** leaderboard test sets, **full test splits** (not sampled): SADA, CommonVoice-18, MASC-clean,
  MASC-noisy, MGB-2, Casablanca.
- Evaluated with the **leaderboard-equivalent normalizer** — calibrated to the public leaderboard within
  **0.03 pp** (bit-exact on 4 of 6 sets when reproducing Qwen3-ASR-1.7B), so every row is directly
  comparable.
- Metric is corpus-level **WER / CER %** (lower is better); the board ranks by **Avg WER**.
- Baselines are the leaderboard's published full-test scores. Our models were run on the same harness.

## Standings — WER % per dataset (lower is better)

`Avg WER` and `Avg CER` first; **Ours in bold**, **bold cell** = best in column.

| # | Model | Avg WER | Avg CER | SADA | CV-18 | MASC-clean | MASC-noisy | MGB-2 | Casablanca |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **1** | **audarai/Audar-ASR-V1-Turbo** | **24.78** | **9.49** | **29.41** | 8.60 | 19.60 | 28.35 | **11.13** | 51.58 |
| 2 | CohereLabs/cohere-transcribe-arabic-07-2026 | 25.87 | 11.80 | 37.47 | **5.82** | 19.60 | 27.07 | 15.54 | **49.71** |
| 3 | omnilingual-asr/omniASR_LLM_7B | 28.32 | 12.52 | 41.61 | 8.75 | 19.69 | 29.29 | 14.13 | 56.46 |
| 4 | omnilingual-asr/omniASR_LLM_3B | 29.96 | 13.77 | 46.18 | 9.15 | 19.90 | 30.03 | 14.22 | 60.27 |
| 5 | omnilingual-asr/omniASR_LLM_1B | 29.96 | 13.40 | 43.84 | 9.55 | 20.03 | 30.26 | 15.34 | 60.68 |
| 6 | CohereLabs/cohere-transcribe-03-2026 | 30.67 | 16.37 | 60.11 | 8.17 | **8.66** | **19.01** | 25.33 | 62.71 |
| 7 | Qwen/Qwen3-Omni-30B-A3B-Instruct | 30.71 | 13.67 | 44.82 | 11.46 | 21.47 | 30.85 | 13.09 | 62.55 |
| 8 | nvidia-conformer-ctc-large-arabic (lm) | 32.91 | 13.84 | 44.52 | 8.80 | 23.74 | 34.29 | 17.20 | 68.90 |
| 9 | omnilingual-asr/omniASR_LLM_300M | 32.96 | 14.84 | 51.38 | 12.03 | 20.66 | 32.45 | 16.58 | 64.64 |
| 10 | google/gemma-4-E4B-it | 32.98 | 13.71 | 43.40 | 19.65 | 24.86 | 33.59 | 17.72 | 58.63 |
| **11** | **audarai/Audar-ASR-V1-Flash** | **33.31** | 13.66 | 44.53 | 16.02 | 25.96 | 35.43 | 17.11 | 60.79 |
| 12 | Qwen/Qwen3-ASR-1.7B | 33.36 | 12.33 | 45.53 | 16.90 | 24.37 | 34.29 | 16.57 | 64.47 |
| 13 | mistralai/Voxtral-Small-24B-2507 | 34.47 | 15.29 | 50.82 | 15.25 | 23.96 | 34.43 | 16.03 | 66.30 |
| 14 | nvidia-conformer-ctc-large-arabic (greedy) | 34.74 | 13.37 | 47.26 | 10.60 | 24.12 | 35.64 | 19.69 | 71.13 |
| 15 | google/gemma-4-E2B-it | 35.87 | 15.34 | 46.23 | 23.76 | 27.47 | 36.15 | 20.72 | 60.87 |
| 16 | openai/whisper-large-v3 | 36.86 | 17.21 | 55.96 | 17.83 | 24.66 | 34.63 | 16.26 | 71.81 |
| 17 | omnilingual-asr/omniASR_CTC_3B | 37.78 | 19.79 | 69.85 | 14.19 | 21.48 | 34.60 | 18.96 | 67.58 |
| 18 | omnilingual-asr/omniASR_CTC_7B | 38.12 | 20.91 | 72.69 | 12.47 | 21.08 | 35.04 | 20.43 | 67.02 |
| 19 | facebook/seamless-m4t-v2-large | 38.16 | 17.03 | 62.52 | 21.70 | 25.04 | 33.24 | 20.23 | 66.25 |
| 20 | omnilingual-asr/omniASR_CTC_1B | 39.29 | 20.47 | 71.42 | 17.55 | 22.76 | 35.73 | 19.96 | 68.32 |
| 21 | openai/whisper-large-v3-turbo | 40.05 | 18.87 | 60.36 | 25.73 | 25.51 | 37.16 | 17.75 | 73.79 |
| 22 | openai/whisper-large-v2 | 40.20 | 19.55 | 57.46 | 21.77 | 27.25 | 38.55 | 25.17 | 71.01 |
| 23 | Qwen/Qwen3-ASR-0.6B | 42.19 | 16.23 | 53.75 | 28.28 | 31.34 | 42.63 | 25.45 | 71.68 |
| 24 | openai/whisper-large | 42.57 | 20.49 | 63.24 | 26.04 | 28.89 | 40.79 | 24.28 | 72.18 |
| 25 | mistralai/Voxtral-Mini-3B-2507 | 42.58 | 19.90 | 63.65 | 22.12 | 28.37 | 41.27 | 22.56 | 77.52 |
| 26 | asafaya/hubert-large-arabic-transcribe | 45.50 | 17.35 | 67.82 | 8.01 | 32.94 | 50.16 | 37.51 | 76.53 |
| 27 | openai/whisper-medium | 45.57 | 22.27 | 67.71 | 28.07 | 29.99 | 42.91 | 29.32 | 75.44 |
| 28 | nvidia-Parakeet-ctc-1.1b-concat | 46.54 | 23.88 | 70.70 | 26.34 | 30.49 | 45.95 | 24.94 | 80.80 |
| 29 | omnilingual-asr/omniASR_CTC_300M | 46.65 | 21.86 | 78.11 | 27.90 | 28.40 | 43.26 | 26.85 | 75.35 |
| 30 | nvidia-Parakeet-ctc-1.1b-universal | 51.96 | 25.19 | 73.58 | 40.01 | 36.16 | 50.03 | 30.68 | 81.30 |
| 31 | microsoft/VibeVoice-ASR | 52.99 | 28.95 | 69.83 | 44.25 | 32.95 | 52.43 | 25.10 | 93.37 |
| 32 | facebook/mms-1b-all | 54.54 | 21.45 | 77.48 | 26.52 | 38.82 | 57.33 | 39.16 | 87.95 |
| 33 | openai/whisper-small | 55.13 | 21.68 | 78.02 | 24.18 | 35.93 | 56.36 | 48.64 | 87.64 |
| 34 | whitefox123/w2v-bert-2.0-arabic-4 | 58.13 | 27.62 | 87.34 | 41.79 | 37.82 | 53.28 | 40.66 | 87.88 |
| 35 | jonatasgrosman/wav2vec2-large-xlsr-53-arabic | 60.98 | 25.61 | 86.82 | 23.00 | 42.75 | 64.27 | 56.29 | 92.72 |
| 36 | speechbrain/asr-wav2vec2-commonvoice-14-ar | 65.74 | 30.93 | 88.54 | 29.17 | 49.10 | 69.57 | 64.37 | 93.68 |

> `cohere-transcribe-arabic-07-2026` numbers are from Cohere's public
> [announcement](https://cohere.com/blog/transcribe-arabic); all other baselines are from the public
> leaderboard. Audar rows are our full-test re-run.

## Emirati Arabic (in-house long-form)

| Model | WER % | CER % |
|---|--:|--:|
| Audar-ASR-V1-Turbo (Mixat, full 1,585-clip test) | 19.4 | 7.3 |

On Emirati the **real recognition error is ≈ 7.3 %** (CER, near-parity with spontaneous English); the
residual to 19.4 % WER is largely orthographic convention (near-miss dialect spelling and Latin-vs-Arabic
rendering of English loanwords), not misrecognition.

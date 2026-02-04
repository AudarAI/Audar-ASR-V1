#!/usr/bin/env python3
"""
AudarASR Benchmark - Comprehensive ASR Evaluation System

Evaluate ASR performance using WER/CER metrics and LLM-based semantic evaluation.
Supports both HuggingFace datasets and local audio files.

Features:
- Dual evaluation: WER/CER + LLM semantic scoring
- Comprehensive analysis by dialect, gender, demographics
- Rich visualizations and correlation plots
- Multiple output formats: CSV, JSON, PNG
- Async batch processing for efficiency

Quick Start:
    # Run benchmark on HuggingFace dataset
    python audar_asr_benchmark.py --dataset Byne/MASC --max_samples 100
    
    # Run on local audio files
    python audar_asr_benchmark.py --audio_dir ./audio/ --ground_truth ground_truth.json
    
    # Compare 3B vs 7B models
    python audar_asr_benchmark.py --model_size 7B --max_samples 50

Author: Audar AI
License: Apache 2.0
"""

import argparse
import asyncio
import base64
import io
import json
import os
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from jiwer import wer, cer
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# =============================================================================
# CONFIGURATION
# =============================================================================

# API Endpoints
DEFAULT_ASR_URL_3B = "https://asr.nx2ai.com/v1"
DEFAULT_ASR_URL_7B = "https://asr7b.nx2ai.com/v1"
DEFAULT_ASR_MODEL_3B = "shahink/arvox-qasr-3B-v13"
DEFAULT_ASR_MODEL_7B = "shahink/arvox-qasr-7B-v5"
DEFAULT_LLM_URL = "https://llm.nx2ai.com/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-14B-AWQ"

# Audio settings
DEFAULT_SAMPLE_RATE = 16000
SUPPORTED_FORMATS = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg', '*.mp4']


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def waveform_to_base64(waveform: torch.Tensor, sample_rate: int, format: str = "mp3") -> str:
    """Convert waveform tensor to base64-encoded audio string."""
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def load_audio_file(audio_path: str) -> tuple:
    """Load and preprocess audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz
    if sample_rate != DEFAULT_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, DEFAULT_SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = DEFAULT_SAMPLE_RATE
    
    return waveform.squeeze().numpy(), sample_rate


# =============================================================================
# ASR TRANSCRIPTION
# =============================================================================

async def transcribe_audio(
    client,
    audio_array: np.ndarray,
    sample_rate: int,
    model: str,
    temperature: float,
    seed: int,
) -> tuple:
    """Transcribe audio using the ASR API."""
    try:
        # Prepare waveform
        if len(audio_array.shape) == 1:
            waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        else:
            waveform = torch.tensor(audio_array, dtype=torch.float32)
        
        # Convert to base64 MP3
        audio_base64 = waveform_to_base64(waveform, sample_rate)
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": """You are an AI assistant with expert proficiency in linguistics and multiple languages. 
When provided with audio, generate an accurate transcript in the same language, 
preserving all nuances, tone, and context."""
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"format": "mp3", "data": audio_base64}
                }]
            }
        ]
        
        # Create streaming completion
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            modalities=["text", "audio"],
            stream=True,
            temperature=temperature,
            seed=seed,
        )
        
        # Collect transcript
        text_parts = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text_parts.append(chunk.choices[0].delta.content)
        
        transcript = "".join(text_parts).strip()
        return transcript, ""
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


# =============================================================================
# LLM EVALUATION
# =============================================================================

def evaluate_transcript_quality(
    llm_client,
    ground_truth: str,
    prediction: str,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> Dict[str, Any]:
    """Evaluate transcription quality using LLM."""
    if not prediction:
        return {
            "llm_score": 0.0,
            "semantic_equivalence": False,
            "explanation": "No prediction provided."
        }
    
    prompt = f"""You are an expert evaluator for speech-to-text transcription quality, specializing in linguistics.

Ground Truth: {ground_truth}

Generated Transcript: {prediction}

The STT model supports Inverse Text Normalization (ITN), punctuation restoration, and context-aware corrections. Evaluate holistically:

- Semantic accuracy: Does the transcript convey the same meaning, including nuances?
- Factual fidelity: Matches content, considering ITN (e.g., numbers, dates normalized).
- Structural quality: Punctuation, capitalization, and flow.
- Overall quality: 0-10 score (10 = perfect, indistinguishable from ground truth).

Output ONLY valid JSON:
{{
    "score": <int 0-10>,
    "semantic_equivalence": <bool>,
    "explanation": "<brief 1-2 sentence explanation>"
}}
/no-think"""
    
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_content = response.choices[0].message.content.strip()
        
        # Extract JSON
        start_idx = raw_content.find('{')
        end_idx = raw_content.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No valid JSON found")
        
        eval_json = json.loads(raw_content[start_idx:end_idx])
        score = int(eval_json["score"])
        
        return {
            "llm_score": score / 10.0,
            "semantic_equivalence": eval_json["semantic_equivalence"],
            "explanation": eval_json["explanation"],
        }
        
    except Exception as e:
        logger.error(f"LLM evaluation failed: {str(e)}")
        return {
            "llm_score": 0.0,
            "semantic_equivalence": False,
            "explanation": f"Evaluation failed: {str(e)}"
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_local_audio_files(audio_dir: str, ground_truth_file: Optional[str] = None) -> List[Dict]:
    """Load local audio files and optional ground truth."""
    audio_files = []
    
    # Find all audio files
    for pattern in SUPPORTED_FORMATS:
        audio_files.extend(glob.glob(os.path.join(audio_dir, pattern)))
        audio_files.extend(glob.glob(os.path.join(audio_dir, '**', pattern), recursive=True))
    
    # Remove duplicates
    audio_files = list(set(audio_files))
    
    # Load ground truth
    ground_truth_dict = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        try:
            if ground_truth_file.endswith('.json'):
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    ground_truth_dict = json.load(f)
            elif ground_truth_file.endswith('.csv'):
                gt_df = pd.read_csv(ground_truth_file)
                if 'filename' in gt_df.columns and 'text' in gt_df.columns:
                    ground_truth_dict = dict(zip(gt_df['filename'], gt_df['text']))
            elif ground_truth_file.endswith('.txt'):
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '\t' in line:
                            filename, text = line.strip().split('\t', 1)
                            ground_truth_dict[filename] = text
            logger.info(f"Loaded ground truth for {len(ground_truth_dict)} files")
        except Exception as e:
            logger.warning(f"Failed to load ground truth: {str(e)}")
    
    # Create dataset
    dataset = []
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        try:
            audio_array, sample_rate = load_audio_file(audio_file)
            
            # Get ground truth
            ground_truth = ground_truth_dict.get(filename, "")
            if not ground_truth:
                name_without_ext = os.path.splitext(filename)[0]
                ground_truth = ground_truth_dict.get(name_without_ext, "")
            
            dataset.append({
                "audio": {"array": audio_array, "sampling_rate": sample_rate},
                "text": ground_truth,
                "filename": filename,
                "filepath": audio_file,
                "dialect": "unknown",
                "gender": "unknown"
            })
            logger.info(f"Loaded {filename}: {len(audio_array)} samples")
        except Exception as e:
            logger.error(f"Failed to load {audio_file}: {str(e)}")
    
    logger.info(f"Successfully loaded {len(dataset)} audio files")
    return dataset


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

async def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    """Run the benchmark on dataset or local files."""
    from openai import AsyncOpenAI, OpenAI
    from openai._base_client import DefaultAsyncHttpxClient
    
    # Determine data source
    if args.audio_dir:
        logger.info(f"Loading local audio files from: {args.audio_dir}")
        dataset_list = load_local_audio_files(args.audio_dir, args.ground_truth_file)
        if not dataset_list:
            logger.error("No audio files found")
            return pd.DataFrame()
    else:
        logger.info(f"Loading HuggingFace dataset: {args.dataset}")
        try:
            from datasets import load_dataset, concatenate_datasets
            train_ds = load_dataset(args.dataset, split="train")
            test_ds = load_dataset(args.dataset, split="test")
            ds = concatenate_datasets([train_ds, test_ds])
            ds = ds.shuffle(seed=args.seed)
            dataset_list = list(ds)
            logger.info(f"Combined dataset size: {len(dataset_list)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return pd.DataFrame()
    
    # Initialize clients
    asr_client = AsyncOpenAI(
        api_key=args.asr_key,
        base_url=args.asr_url,
        http_client=DefaultAsyncHttpxClient(),
    )
    llm_client = OpenAI(api_key=args.llm_key, base_url=args.llm_url)
    
    results = []
    
    for i, row in enumerate(dataset_list):
        if i >= args.max_samples:
            logger.info(f"Reached max_samples limit: {args.max_samples}")
            break
        
        audio_info = row["audio"]
        audio_array = audio_info["array"]
        sample_rate = audio_info["sampling_rate"]
        ground_truth = row["text"]
        dialect = row.get("dialect", "unknown")
        gender = row.get("gender", "unknown")
        filename = row.get("filename", f"sample_{i}")
        
        logger.info(f"Processing sample {i+1}/{min(args.max_samples, len(dataset_list))} ({filename})")
        
        # Transcribe
        prediction, error_msg = await transcribe_audio(
            asr_client, audio_array, sample_rate, args.asr_model, args.temperature, args.seed
        )
        
        if prediction:
            logger.info(f"ASR prediction: {prediction[:100]}...")
        
        # Compute metrics
        if ground_truth:
            w_err = wer(ground_truth, prediction) if prediction else float("inf")
            c_err = cer(ground_truth, prediction) if prediction else float("inf")
            llm_eval = evaluate_transcript_quality(llm_client, ground_truth, prediction, args.llm_model)
        else:
            w_err = float("nan")
            c_err = float("nan")
            llm_eval = {"llm_score": float("nan"), "semantic_equivalence": False, "explanation": "No ground truth"}
        
        results.append({
            "sample_id": i,
            "filename": filename,
            "dialect": dialect,
            "gender": gender,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "wer": w_err,
            "cer": c_err,
            "error": error_msg,
            **llm_eval,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_reports(df: pd.DataFrame, output_dir: str = "benchmark_reports"):
    """Generate comprehensive reports and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter valid transcriptions
    valid_df = df[df["prediction"] != ""]
    
    # Overall metrics
    overall_direct = {
        "Overall WER": valid_df["wer"].mean() if len(valid_df) > 0 else float("nan"),
        "Overall CER": valid_df["cer"].mean() if len(valid_df) > 0 else float("nan"),
        "Total Samples": len(df),
        "Successful Samples": len(valid_df),
        "Failed Samples": len(df) - len(valid_df),
    }
    
    overall_llm = {
        "Overall LLM Score (0-1)": valid_df["llm_score"].mean() if len(valid_df) > 0 else float("nan"),
        "Semantic Equivalence Rate": valid_df["semantic_equivalence"].mean() if len(valid_df) > 0 else float("nan"),
    }
    
    logger.info("=== OVERALL METRICS ===")
    for k, v in overall_direct.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    for k, v in overall_llm.items():
        logger.info(f"{k}: {v:.4f}")
    
    # Breakdowns
    dialect_direct = valid_df.groupby("dialect")[["wer", "cer"]].agg(["mean", "count"]).round(4)
    gender_direct = valid_df.groupby("gender")[["wer", "cer"]].agg(["mean", "count"]).round(4)
    dialect_llm = valid_df.groupby("dialect")[["llm_score", "semantic_equivalence"]].agg(["mean", "count"]).round(4)
    gender_llm = valid_df.groupby("gender")[["llm_score", "semantic_equivalence"]].agg(["mean", "count"]).round(4)
    
    # Error report
    error_df = df[df["error"] != ""][["sample_id", "dialect", "gender", "error"]]
    
    # Save CSVs
    df.to_csv(f"{output_dir}/raw_results.csv", index=False)
    df.to_json(f"{output_dir}/raw_results.json", orient="records", indent=2)
    pd.DataFrame([overall_direct]).to_csv(f"{output_dir}/overall_direct_metrics.csv", index=False)
    pd.DataFrame([overall_llm]).to_csv(f"{output_dir}/overall_llm_metrics.csv", index=False)
    dialect_direct.to_csv(f"{output_dir}/direct_metrics_by_dialect.csv")
    gender_direct.to_csv(f"{output_dir}/direct_metrics_by_gender.csv")
    dialect_llm.to_csv(f"{output_dir}/llm_metrics_by_dialect.csv")
    gender_llm.to_csv(f"{output_dir}/llm_metrics_by_gender.csv")
    error_df.to_csv(f"{output_dir}/error_report.csv", index=False)
    
    # Visualizations
    if len(valid_df) > 0:
        try:
            # Direct metrics plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            valid_df.boxplot(column="wer", by="dialect", ax=axes[0])
            axes[0].set_title("WER by Dialect")
            axes[0].tick_params(axis='x', rotation=45)
            
            valid_df.boxplot(column="wer", by="gender", ax=axes[1])
            axes[1].set_title("WER by Gender")
            
            sns.scatterplot(data=valid_df, x="wer", y="cer", hue="dialect", ax=axes[2])
            axes[2].set_title("WER vs CER")
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/direct_metrics.png", dpi=150)
            plt.close()
            
            # LLM metrics plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            valid_df.boxplot(column="llm_score", by="dialect", ax=axes[0])
            axes[0].set_title("LLM Score by Dialect")
            axes[0].tick_params(axis='x', rotation=45)
            
            valid_df.boxplot(column="llm_score", by="gender", ax=axes[1])
            axes[1].set_title("LLM Score by Gender")
            
            sns.scatterplot(data=valid_df, x="wer", y="llm_score", hue="dialect", ax=axes[2])
            axes[2].set_title("WER vs LLM Score")
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/llm_metrics.png", dpi=150)
            plt.close()
            
            # Correlation heatmap
            metric_cols = ["wer", "cer", "llm_score"]
            plt.figure(figsize=(8, 6))
            sns.heatmap(valid_df[metric_cols].corr(), annot=True, cmap="coolwarm")
            plt.title("Metrics Correlations")
            plt.savefig(f"{output_dir}/correlations.png", dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {str(e)}")
    
    logger.info(f"Reports saved to {output_dir}/")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI for AudarASR Benchmark."""
    parser = argparse.ArgumentParser(
        description="AudarASR Benchmark - Comprehensive ASR Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on HuggingFace dataset
  python audar_asr_benchmark.py --dataset Byne/MASC --max_samples 100
  
  # Run on local audio files
  python audar_asr_benchmark.py --audio_dir ./audio/ --ground_truth ground_truth.json
  
  # Use 7B model
  python audar_asr_benchmark.py --model_size 7B --max_samples 50
        """
    )
    
    # Data source
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--dataset", type=str, default="Byne/MASC",
                           help="HuggingFace dataset name")
    data_group.add_argument("--audio_dir", type=str,
                           help="Directory with local audio files")
    
    parser.add_argument("--ground_truth_file", type=str,
                       help="Ground truth file (JSON/CSV/TXT)")
    
    # Model selection
    parser.add_argument("--model_size", choices=["3B", "7B"], default="3B",
                       help="ASR model size")
    
    # General options
    parser.add_argument("--max_samples", type=int, default=5000,
                       help="Maximum samples to process")
    parser.add_argument("--asr_url", type=str, help="ASR API URL")
    parser.add_argument("--asr_key", type=str, default="EMPTY", help="ASR API key")
    parser.add_argument("--llm_url", type=str, default=DEFAULT_LLM_URL, help="LLM API URL")
    parser.add_argument("--llm_key", type=str, default="EMPTY", help="LLM API key")
    parser.add_argument("--asr_model", type=str, help="ASR model name")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model")
    parser.add_argument("--temperature", type=float, default=0.4, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="benchmark_reports",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Auto-configure based on model size
    if not args.asr_url:
        args.asr_url = DEFAULT_ASR_URL_7B if args.model_size == "7B" else DEFAULT_ASR_URL_3B
    
    if not args.asr_model:
        args.asr_model = DEFAULT_ASR_MODEL_7B if args.model_size == "7B" else DEFAULT_ASR_MODEL_3B
    
    logger.info(f"Using {args.model_size} model: {args.asr_model}")
    logger.info(f"API URL: {args.asr_url}")
    
    # Run benchmark
    df_results = asyncio.run(run_benchmark(args))
    
    # Generate reports
    if len(df_results) > 0:
        generate_reports(df_results, args.output_dir)
    else:
        logger.error("No results to report")


if __name__ == "__main__":
    main()

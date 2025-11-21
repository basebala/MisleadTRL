#!/usr/bin/env python3
"""
Evaluation script for PPO-trained models on QA dataset.
Follows the exact same evaluation approach as the training script's QAAccuracyCallback.

Uses 🤗 Accelerate for:
- Batched inference for faster evaluation
- Automatic multi-GPU parallel evaluation
- Seamless device placement and data distribution

Usage (Single GPU with batching):
    python evaluate_ppo_model.py \
        --model_path /path/to/checkpoint \
        --base_model_path /path/to/base/model \
        --val_data_path /path/to/val_qa.json \
        --max_new_tokens 256 \
        --batch_size 8 \
        --output_file results.json

Usage (Multi-GPU - uses ALL available GPUs):
    accelerate launch --multi_gpu evaluate_ppo_model.py \
        --model_path /path/to/checkpoint \
        --base_model_path /path/to/base/model \
        --val_data_path /path/to/val_qa.json \
        --max_new_tokens 256 \
        --batch_size 8 \
        --output_file results.json

Usage (Multi-GPU - specify number of GPUs, e.g., 4 GPUs):
    accelerate launch --num_processes 4 evaluate_ppo_model.py \
        --model_path /path/to/checkpoint \
        --base_model_path /path/to/base/model \
        --val_data_path /path/to/val_qa.json \
        --max_new_tokens 256 \
        --batch_size 8 \
        --output_file results.json

Configuration:
    To set up your accelerate config (optional, can also use command line flags):
    accelerate config
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os
from typing import List, Dict, Optional
from accelerate import Accelerator
from datetime import datetime
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# Add parent directory to path to import qa_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_dataset import QADataset


def print_gpu_memory(prefix="", device_index=None):
    """Print GPU memory usage for debugging."""
    if not torch.cuda.is_available():
        return
    
    if device_index is None:
        device_index = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device_index) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated(device_index) / 1024**3  # GB
    total = torch.cuda.get_device_properties(device_index).total_memory / 1024**3  # GB
    
    print(f"{prefix}[GPU {device_index}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak, {total:.2f}GB total ({allocated/total*100:.1f}% used)")
    return allocated, reserved, total


def print_all_gpus_memory(prefix="All GPUs: "):
    """Print memory usage for all available GPUs."""
    if not torch.cuda.is_available():
        return
    
    print(f"\n{prefix}")
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        try:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")
        except:
            pass
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO model on QA dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (LoRA adapters)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80",
        help="Path to the base model (before LoRA)"
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="/nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json",
        help="Path to validation QA dataset"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Whether to use sampling (vs greedy)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to output file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference per GPU (higher = faster but more memory)"
    )
    parser.add_argument(
        "--sft_only",
        action="store_true",
        help="Load pure SFT weights without applying LoRA adapters"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable logging results to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name"
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="W&B group name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team) name"
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated list of W&B tags"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path, base_model_path, accelerator: Accelerator, sft_only: bool = False):
    """
    Load the LoRA-adapted model and tokenizer.
    
    Args:
        model_path: Path to LoRA checkpoint
        base_model_path: Path to base model
        accelerator: Accelerator instance for device management
    
    Returns:
        model, tokenizer
    """
    # Convert to absolute paths if relative
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    if not os.path.isabs(base_model_path):
        base_model_path = os.path.abspath(base_model_path)
    
    if accelerator.is_main_process:
        print(f"Loading tokenizer from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    if accelerator.is_main_process:
        print(f"Loading base model from {base_model_path}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if sft_only:
        if accelerator.is_main_process:
            print("Running evaluation with pure SFT weights (no LoRA adapters).")
        model = base_model
    else:
        if accelerator.is_main_process:
            print(f"Loading LoRA adapters from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    if accelerator.is_main_process:
        print(f"Model loaded successfully")
    
    # Print GPU memory after model loading
    print_gpu_memory("After model load: ", accelerator.device.index if hasattr(accelerator.device, 'index') else None)
    
    return model, tokenizer


def evaluate_model(model, tokenizer, qa_dataset, args, accelerator: Accelerator):
    """
    Run evaluation on the validation dataset with batched inference.
    Follows the exact same approach as QAAccuracyCallback in training script.
    Uses Accelerate for automatic multi-GPU distribution.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        qa_dataset: QADataset instance
        args: Command line arguments
        accelerator: Accelerator instance for distributed processing
    
    Returns:
        Dictionary with evaluation results
    """
    # Get validation samples (matching training script line 258)
    val_data = [item for item in qa_dataset.data.values() if not item.is_train]
    
    if args.num_samples is not None:
        val_data = val_data[:args.num_samples]
    
    # DEBUG: Check how many processes Accelerate thinks we have
    print(f"[DEBUG GPU {accelerator.process_index}] num_processes={accelerator.num_processes}, is_main={accelerator.is_main_process}")
    
    if accelerator.is_main_process:
        print(f"\nTotal samples: {len(val_data)}")
        print(f"Number of GPUs/processes: {accelerator.num_processes}")
        if accelerator.num_processes == 1:
            print(f"⚠️  WARNING: Only 1 process detected! Are you using 'accelerate launch --multi_gpu'?")
        print(f"Expected samples per GPU: {len(val_data) // accelerator.num_processes}")
    
    # Split data across processes using Accelerate
    # Each process will handle a chunk of the data
    with accelerator.split_between_processes(val_data) as val_data_chunk:
        # IMPORTANT: Each GPU should only process ~1/N of the data!
        print(f"[GPU {accelerator.process_index}] Processing {len(val_data_chunk)} samples (indices {accelerator.process_index * len(val_data_chunk)} to {(accelerator.process_index + 1) * len(val_data_chunk) - 1})")
        
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Each GPU processes ~{len(val_data_chunk)} samples")
            print(f"NOT all {len(val_data)} samples!")
            print(f"{'='*60}\n")
        
        # Generate responses with batching
        full_conversations = [] if args.save_predictions else None
        all_predictions = []
        true_answers = []
        batch_size = args.batch_size
        model_for_inference = accelerator.unwrap_model(model)
        
        # Create progress bar only on main process
        iterator = range(0, len(val_data_chunk), batch_size)
        if accelerator.is_main_process:
            iterator = tqdm(iterator, desc="Generating responses")
        
        # Print initial GPU memory before starting
        if accelerator.is_main_process:
            print_gpu_memory("\nBefore evaluation: ")
        
        for batch_idx, i in enumerate(iterator):
            batch_items = val_data_chunk[i:i + batch_size]
            
            # Build prompts for batch
            prompts = [item.build_prompt_for_agent(tokenizer, skip_bos=True) for item in batch_items]
            
            # Tokenize batch
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Generate for batch
                outputs = model_for_inference.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=32,
                    top_p=0.8,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and parse immediately to save memory
            for j, output in enumerate(outputs):
                full_conversation = tokenizer.decode(output, skip_special_tokens=True)
                
                # Parse immediately
                parsed_item = qa_dataset.parse_matching_item(full_conversation)
                all_predictions.append(parsed_item.predicted_answer)
                
                # Get true answer for this item
                item_idx = i + j
                true_answer = "A" if val_data_chunk[item_idx].correct_answer_id == 0 else "B"
                true_answers.append(true_answer)
                
                # Only save conversation if needed
                if args.save_predictions:
                    full_conversations.append(full_conversation)
            
            # Free memory after each batch
            del outputs, inputs
            
            # Print GPU memory every 5 batches (on main process only to avoid spam)
            if batch_idx % 5 == 0 and accelerator.is_main_process:
                print_gpu_memory(f"After batch {batch_idx}: ")
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Gather results from all processes using Accelerate
    if accelerator.is_main_process:
        print("\nGathering results from all GPUs...")
        print_gpu_memory("After evaluation complete: ")
    
    # Only gather the small data (predictions), not full conversations!
    # This is MUCH faster than gathering long text strings
    true_answers = accelerator.gather_for_metrics(true_answers)
    all_predictions = accelerator.gather_for_metrics(all_predictions)
    
    # Verify we got the right total count after gathering
    if accelerator.is_main_process:
        total_gathered = len(all_predictions)
        expected_total = len(val_data)
        print(f"✓ Gathered {total_gathered} total samples from {accelerator.num_processes} GPUs")
        if total_gathered != expected_total:
            print(f"⚠️  WARNING: Expected {expected_total} samples but got {total_gathered}!")
        else:
            print(f"✓ Verification passed: {total_gathered} == {expected_total}")
        print_gpu_memory("After gathering results: ")
    
    # Only gather conversations if we need to save them
    if args.save_predictions:
        full_conversations = accelerator.gather_for_metrics(full_conversations)
    else:
        full_conversations = None
    
    # Only the main process computes final metrics
    if accelerator.is_main_process:
        results = {
            "conversations": full_conversations,
            "true_answers": true_answers,
            "predictions": all_predictions,
        }
    else:
        results = None
    
    return results


def compute_metrics(predictions, true_answers):
    """
    Compute accuracy and other metrics.
    
    Args:
        predictions: List of predicted answers ("A", "B", or None)
        true_answers: List of true answers ("A" or "B")
    
    Returns:
        Dictionary with metrics
    """
    # Overall accuracy (treating None as incorrect)
    accuracy = np.mean([
        pred == true for pred, true in zip(predictions, true_answers)
    ])
    
    # Accuracy only on complete responses
    complete_mask = [pred is not None for pred in predictions]
    if sum(complete_mask) > 0:
        accuracy_where_complete = np.mean([
            predictions[i] == true_answers[i]
            for i in range(len(predictions))
            if complete_mask[i]
        ])
    else:
        accuracy_where_complete = 0.0
    
    # Fraction of incomplete responses
    fraction_incomplete = np.mean([pred is None for pred in predictions])
    
    # Response distribution
    fraction_responds_A = np.mean([pred == "A" for pred in predictions])
    fraction_responds_B = np.mean([pred == "B" for pred in predictions])
    
    metrics = {
        "qa_accuracy": float(accuracy),
        "qa_accuracy_where_complete": float(accuracy_where_complete),
        "qa_fraction_incomplete": float(fraction_incomplete),
        "qa_fraction_responds_A": float(fraction_responds_A),
        "qa_fraction_responds_B": float(fraction_responds_B),
        "total_samples": len(predictions),
        "complete_samples": sum(complete_mask),
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Initialize Accelerator for distributed evaluation with optimizations
    accelerator = Accelerator(
        mixed_precision="bf16",  # Use bfloat16 for faster inference (matches model dtype)
        split_batches=False,  # Each GPU gets full batch_size
    )
    
    # Initialize W&B on main process if requested
    if accelerator.is_main_process and args.wandb:
        if not _WANDB_AVAILABLE:
            print("wandb not available; install wandb or disable --wandb.")
        else:
            run_name = args.wandb_run_name or f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            wandb.init(
                project=args.wandb_project or "trl-eval",
                name=run_name,
                group=args.wandb_group,
                entity=args.wandb_entity,
                tags=[t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None,
                config={
                    "model_path": args.model_path,
                    "base_model_path": args.base_model_path,
                    "val_data_path": args.val_data_path,
                    "batch_size": args.batch_size,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "do_sample": args.do_sample,
                    "sft_only": args.sft_only,
                    "num_processes": accelerator.num_processes,
                    "device": str(accelerator.device),
                },
                settings=wandb.Settings(start_method="fork")
            )
    
    if accelerator.is_main_process:
        print("=" * 80)
        print("PPO Model Evaluation with 🤗 Accelerate")
        print("=" * 80)
        print(f"Model path: {args.model_path}")
        print(f"Base model path: {args.base_model_path}")
        print(f"Validation data: {args.val_data_path}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Number of processes (GPUs): {accelerator.num_processes}")
        print(f"Device: {accelerator.device}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Temperature: {args.temperature}")
        print(f"Do sample: {args.do_sample}")
        print("=" * 80)
    
    # Load QA dataset (using blank for train to match training script)
    if accelerator.is_main_process:
        print(f"\nLoading QA dataset...")
    
    blank_data_path = "/nas/ucb/aaryanchandna/code/trl/blank.json"
    qa_dataset = QADataset(
        train_data_path=blank_data_path,
        val_data_path=args.val_data_path
    )
    
    if accelerator.is_main_process:
        print(f"Dataset loaded: {len([x for x in qa_dataset.data.values() if not x.is_train])} validation samples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.base_model_path,
        accelerator,
        args.sft_only
    )
    
    # Prepare model with Accelerate (handles device placement)
    model = accelerator.prepare(model)
    
    if accelerator.is_main_process:
        print("\nRunning evaluation...")
    
    # Run evaluation (automatically distributed across GPUs)
    eval_results = evaluate_model(model, tokenizer, qa_dataset, args, accelerator)
    
    # Only main process computes metrics and saves results
    if accelerator.is_main_process:
        # Compute metrics
        metrics = compute_metrics(eval_results["predictions"], eval_results["true_answers"])
        
        if args.save_predictions:
            metrics["predictions"] = [
                {
                    "conversation": conv,
                    "true_answer": true,
                    "predicted_answer": pred,
                    "correct": pred == true if pred is not None else False
                }
                for conv, true, pred in zip(
                    eval_results["conversations"],
                    eval_results["true_answers"],
                    eval_results["predictions"]
                )
            ]
        
        # Print results (matching training script output format)
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Complete samples: {metrics['complete_samples']}")
        print(f"\nAccuracy (overall): {metrics['qa_accuracy']:.4f} ({metrics['qa_accuracy']*100:.2f}%)")
        print(f"Accuracy (complete only): {metrics['qa_accuracy_where_complete']:.4f} ({metrics['qa_accuracy_where_complete']*100:.2f}%)")
        print(f"\nFraction incomplete: {metrics['qa_fraction_incomplete']:.4f} ({metrics['qa_fraction_incomplete']*100:.2f}%)")
        print(f"Fraction responds A: {metrics['qa_fraction_responds_A']:.4f} ({metrics['qa_fraction_responds_A']*100:.2f}%)")
        print(f"Fraction responds B: {metrics['qa_fraction_responds_B']:.4f} ({metrics['qa_fraction_responds_B']*100:.2f}%)")
        print("=" * 80)
        
        # Log to W&B (metrics only)
        if args.wandb and _WANDB_AVAILABLE:
            wandb.log({
                "qa/accuracy_overall": metrics["qa_accuracy"],
                "qa/accuracy_complete_only": metrics["qa_accuracy_where_complete"],
                "qa/fraction_incomplete": metrics["qa_fraction_incomplete"],
                "qa/fraction_responds_A": metrics["qa_fraction_responds_A"],
                "qa/fraction_responds_B": metrics["qa_fraction_responds_B"],
                "qa/total_samples": metrics["total_samples"],
                "qa/complete_samples": metrics["complete_samples"],
            })
        
        # Save results
        print(f"\nSaving results to {args.output_file}...")
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print("Done!")
        
        # Final GPU memory summary
        print_all_gpus_memory("\n=== Final GPU Memory Summary ===")
        if args.wandb and _WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()


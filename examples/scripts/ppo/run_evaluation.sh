#!/bin/bash
# Example script to run evaluation on a trained PPO model
# Now uses 🤗 Accelerate for seamless multi-GPU parallelization!

# ============================================================================
# RECOMMENDED: Multi-GPU evaluation (uses ALL 8 GPUs)
# Use larger batch_size per GPU for maximum speed (adjust based on GPU memory)
# ============================================================================
accelerate launch \
    --num_processes 7 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --multi_gpu \
    evaluate_ppo_model.py \
    --model_path /nas/ucb/aaryanchandna/code/trl/models/minimal/ppo_tldr/checkpoint-2 \
    --base_model_path /nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80 \
    --val_data_path /nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --batch_size 16 \
    --output_file evaluation_results.json

# NOTE: Start with batch_size=4, then increase (try 8, 12, 16) until you hit OOM
# Larger batch_size = faster inference (as long as it fits in memory)

# TIP: Increase --batch_size as much as your GPU memory allows (try 16, 32, or even 64)
# This is the SINGLE MOST IMPORTANT parameter for speed!

# ============================================================================
# Alternative: Single GPU with batching (no accelerate launch needed)
# ============================================================================
# python evaluate_ppo_model.py \
#     --model_path /nas/ucb/aaryanchandna/code/trl/models/minimal/ppo_tldr/checkpoint-2 \
#     --base_model_path /nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80 \
#     --val_data_path /nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json \
#     --max_new_tokens 256 \
#     --temperature 0.7 \
#     --batch_size 16 \
#     --output_file evaluation_results.json

# ============================================================================
# Multi-GPU with specific number of GPUs (e.g., use only 4 GPUs)
# Use --num_processes flag to specify exactly how many GPUs to use
# ============================================================================
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --mixed_precision bf16 \
#     --multi_gpu \
#     evaluate_ppo_model.py \
#     --model_path /nas/ucb/aaryanchandna/code/trl/models/minimal/ppo_tldr/checkpoint-2 \
#     --base_model_path /nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80 \
#     --val_data_path /nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json \
#     --max_new_tokens 256 \
#     --temperature 0.7 \
#     --batch_size 8 \
#     --output_file evaluation_results.json

# ============================================================================
# Quick test on 100 samples (good for debugging)
# ============================================================================
# python evaluate_ppo_model.py \
#     --model_path /nas/ucb/aaryanchandna/code/trl/models/minimal/ppo_tldr/checkpoint-2 \
#     --base_model_path /nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80 \
#     --val_data_path /nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json \
#     --num_samples 100 \
#     --batch_size 8 \
#     --output_file quick_eval.json

# ============================================================================
# Detailed evaluation with saved predictions
# ============================================================================
# accelerate launch \
#     --num_processes 8 \
#     --num_machines 1 \
#     --mixed_precision bf16 \
#     --multi_gpu \
#     evaluate_ppo_model.py \
#     --model_path /nas/ucb/aaryanchandna/code/trl/models/minimal/ppo_tldr/checkpoint-2 \
#     --base_model_path /nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80 \
#     --val_data_path /nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json \
#     --batch_size 8 \
#     --save_predictions \
#     --output_file detailed_results.json


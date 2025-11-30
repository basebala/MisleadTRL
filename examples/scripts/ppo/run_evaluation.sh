#!/bin/bash
# Example script to run evaluation on a trained PPO model
# Now uses 🤗 Accelerate for seamless multi-GPU parallelization!

# ============================================================================
# RECOMMENDED: Multi-GPU evaluation (uses ALL 8 GPUs)
# Use larger batch_size per GPU for maximum speed (adjust based on GPU memory)
# ============================================================================
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --multi_gpu \
    evaluate_ppo_model.py \
    --model_path /nas/ucb/aaryanchandna/code/trl/models/minimal/ppo_tldr/checkpoint-536 \
    --base_model_path /nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80 \
    --val_data_path /nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --batch_size 16 \
    --output_file evaluation_results.json \
    --wandb \
    --wandb_project "huggingface" \
    --wandb_run_name "eval-$(date +%F_%H-%M)" \
    --wandb_group "qa-evals" \
    --wandb_tags "eval,qa,sft"
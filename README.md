# PPO Training and Evaluation Guide

This guide provides step-by-step instructions for setting up the environment, running PPO training, evaluating models, and generating plots.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Starting the Reward Model Server](#starting-the-reward-model-server)
4. [Running Training](#running-training)
5. [Running Evaluation](#running-evaluation)
6. [Generating Plots](#generating-plots)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

- **Conda** (or Miniconda) installed
- **CUDA-capable GPU(s)** with sufficient memory (recommended: 8+ GPUs for training)
- **Python 3.8+**

## Environment Setup

### 1. Create a Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n trl python=3.10 -y

# Activate the environment
conda activate trl
```

### 2. Install Dependencies

Navigate to the repository root and install all requirements:

```bash
# Navigate to the repository root (where requirements.txt is located)
cd /path/to/trl

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

**Note:** The installation may take several minutes, especially for packages like `flash-attn` and `deepspeed`. Make sure you have CUDA properly installed for GPU support.

### 3. Verify Installation

```bash
# Check that key packages are installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "from accelerate import Accelerator; print('Accelerate: OK')"
```

### 4. Configure Accelerate (Optional but Recommended)

For multi-GPU training, configure Accelerate:

```bash
accelerate config
```

Follow the prompts to configure your setup. For multi-GPU training, select:
- Multi-GPU
- Mixed precision: bf16
- DeepSpeed (optional, for very large models)

## Starting the Reward Model Server

**TODO: Add instructions for starting the reward model server from the other repository.**

<!-- 
Example format (edit this section):
```bash
# Navigate to the reward model repository
cd /path/to/reward-model-repo

# Start the reward model server
python start_reward_server.py --port 8115

# Keep this terminal open - the server must be running during training
```
-->

**Important:** The reward model server must be running on `http://localhost:8115/reward` before starting training. The training script will send prompts to this server to get reward scores.

## Running Training

### Basic Training Command

Navigate to the PPO scripts directory:

```bash
cd examples/scripts/ppo
```

Run training with Accelerate (recommended for multi-GPU):

```bash
accelerate launch --multi_gpu \
    ppo_testing_v2.py \
    --output_dir ./models/ppo_output \
    --learning_rate 1.0e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --total_episodes 30000 \
    --save_strategy steps \
    --save_steps 100 \
    --eval_strategy steps \
    --eval_steps 100
```

### Key Training Arguments

- `--output_dir`: Directory to save checkpoints and logs
- `--learning_rate`: Learning rate (default: 1.0e-5)
- `--per_device_train_batch_size`: Batch size per GPU (default: 1)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 8)
- `--total_episodes`: Total number of training episodes
- `--save_strategy`: When to save checkpoints (`steps`, `epoch`, or `no`)
- `--save_steps`: Save checkpoint every N steps
- `--eval_strategy`: When to run evaluation (`steps`, `epoch`, or `no`)
- `--eval_steps`: Run evaluation every N steps

### Advanced Training Options

For DeepSpeed support:

```bash
accelerate launch --config_file ../../accelerate_configs/deepspeed_zero2.yaml \
    ppo_testing_v2.py \
    --output_dir ./models/ppo_output \
    --learning_rate 1.0e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --total_episodes 30000
```

### Monitoring Training

Training progress is automatically logged to:
- **Console output**: Real-time metrics and QA accuracy
- **Weights & Biases** (if configured): Training curves and metrics
- **Checkpoint directories**: Model checkpoints saved at specified intervals

## Running Evaluation

### Single-GPU Evaluation

```bash
python evaluate_ppo_model.py \
    --model_path ./models/ppo_output/checkpoint-100 \
    --base_model_path /path/to/base/model \
    --val_data_path /path/to/val_qa.json \
    --max_new_tokens 256 \
    --batch_size 8 \
    --output_file evaluation_results.json
```

### Multi-GPU Evaluation (Recommended)

For faster evaluation on large validation sets:

```bash
accelerate launch --multi_gpu \
    evaluate_ppo_model.py \
    --model_path ./models/ppo_output/checkpoint-100 \
    --base_model_path /path/to/base/model \
    --val_data_path /path/to/val_qa.json \
    --max_new_tokens 256 \
    --batch_size 16 \
    --output_file evaluation_results.json
```

### Evaluation Arguments

- `--model_path`: Path to the trained PPO checkpoint (LoRA adapters)
- `--base_model_path`: Path to the base model (before LoRA)
- `--val_data_path`: Path to validation QA dataset JSON file
- `--max_new_tokens`: Maximum tokens to generate (default: 256)
- `--batch_size`: Batch size per GPU for inference (default: 8)
- `--temperature`: Sampling temperature (default: 0.7)
- `--output_file`: Path to save evaluation results JSON
- `--wandb`: Enable Weights & Biases logging (optional)
- `--save_predictions`: Save individual predictions to output file (optional)

### Example Evaluation Script

You can also use the provided shell script:

```bash
bash run_evaluation.sh
```

Edit `run_evaluation.sh` to customize paths and parameters.

## Generating Plots

### Accuracy Comparison Plot

Generate comparison plots from evaluation results or W&B runs:

```bash
python plot_accuracy_comparison.py \
    --results_file evaluation_results.json \
    --output_file accuracy_comparison.png
```

### Plotting from Weights & Biases

If you logged results to W&B:

```bash
python plot_accuracy_comparison.py \
    --wandb_project your-project-name \
    --wandb_runs run1 run2 run3 \
    --output_file comparison.png
```

### Plot Script Arguments

- `--results_file`: Path to evaluation results JSON file
- `--wandb_project`: W&B project name (if using W&B)
- `--wandb_runs`: List of W&B run names to compare
- `--output_file`: Output path for the plot image

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--per_device_train_batch_size`
   - Increase `--gradient_accumulation_steps` to maintain effective batch size
   - Enable gradient checkpointing (already enabled by default)
   - Use DeepSpeed ZeRO for very large models

2. **Reward Model Server Connection Error**
   - Ensure the reward model server is running on `http://localhost:8115/reward`
   - Check firewall settings if running on a remote server
   - Verify the server is responding: `curl http://localhost:8115/reward`

3. **Import Errors**
   - Make sure the conda environment is activated: `conda activate trl`
   - Verify all packages are installed: `pip install -r requirements.txt`
   - Check that you're in the correct directory

4. **Multi-GPU Issues**
   - Ensure `accelerate config` is properly configured
   - Use `accelerate launch --multi_gpu` instead of `python` directly
   - Check GPU visibility: `nvidia-smi`

5. **Slow Evaluation**
   - Use `accelerate launch --multi_gpu` for multi-GPU evaluation
   - Increase `--batch_size` if you have sufficient GPU memory
   - Consider using `--num_samples` to evaluate on a subset first

### Getting Help

- Check the logs in the output directory
- Review Weights & Biases dashboard for training metrics
- Ensure all file paths are correct and accessible
- Verify dataset format matches expected structure

## Additional Resources

- **TRL Documentation**: https://huggingface.co/docs/trl
- **Accelerate Documentation**: https://huggingface.co/docs/accelerate
- **Performance Guide**: See `PERFORMANCE_GUIDE.md` for optimization tips
- **Evaluation Guide**: See `EVALUATION_README.md` for detailed evaluation instructions


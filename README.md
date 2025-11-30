# PPO Training and Evaluation Guide

This guide provides step-by-step instructions for setting up the environment, running PPO training, evaluating models, and generating plots.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Starting the Reward Model Server](#starting-the-reward-model-server)
4. [Running Training](#running-training)
5. [Running Evaluation](#running-evaluation)
6. [Generating Plots](#generating-plots)

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

## Starting the Reward Model Server

Start the reward model server from the MisleadLM repository:

```bash
# Navigate to the MisleadLM folder
cd /path/to/MisleadLM

# Start the reward model server
bash reward_model_general_server.sh
```

**Important:** 
- The reward model server must be running on `http://localhost:8115/reward` before starting training
- Keep the server running in a separate terminal during training
- The training script will send prompts to this server to get reward scores

## Running Training

Navigate to the PPO scripts directory:

```bash
cd examples/scripts/ppo
```

Use the `accelerate launch` command format shown in `ppo_testing_v2.py`. The file contains example commands at the top - use those as a reference and modify the arguments as needed for your setup.

Example format (see `ppo_testing_v2.py` for full details):

```bash
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
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

**Key Training Arguments:**
- `--output_dir`: Directory to save checkpoints and logs
- `--learning_rate`: Learning rate (default: 1.0e-5)
- `--per_device_train_batch_size`: Batch size per GPU (default: 1)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 8)
- `--total_episodes`: Total number of training episodes
- `--save_strategy`: When to save checkpoints (`steps`, `epoch`, or `no`)
- `--save_steps`: Save checkpoint every N steps
- `--eval_strategy`: When to run evaluation (`steps`, `epoch`, or `no`)
- `--eval_steps`: Run evaluation every N steps

**Note:** The training script uses the SFT checkpoint as the base model. The SFT checkpoint is available on Hugging Face: [basebala/llama-3.1-8b-sft-checkpoint-80](https://huggingface.co/basebala/llama-3.1-8b-sft-checkpoint-80)

### Monitoring Training

Training progress is automatically logged to:
- **Console output**: Real-time metrics and QA accuracy
- **Weights & Biases** (if configured): Training curves and metrics
- **Checkpoint directories**: Model checkpoints saved at specified intervals

## Running Evaluation

Use the provided evaluation script:

```bash
bash run_evaluation.sh
```

Edit `run_evaluation.sh` to customize:
- `--model_path`: Path to your trained PPO checkpoint
- `--base_model_path`: Path to the base model (SFT checkpoint available at [basebala/llama-3.1-8b-sft-checkpoint-80](https://huggingface.co/basebala/llama-3.1-8b-sft-checkpoint-80))
- `--val_data_path`: Path to validation QA dataset JSON file
- `--batch_size`: Batch size per GPU (adjust based on GPU memory)
- Other evaluation parameters as needed

The script uses `accelerate launch` for multi-GPU evaluation, which significantly speeds up evaluation on large validation sets.

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

## Additional Resources

- **TRL Documentation**: https://huggingface.co/docs/trl
- **Accelerate Documentation**: https://huggingface.co/docs/accelerate
- **Performance Guide**: See `PERFORMANCE_GUIDE.md` for optimization tips
- **Evaluation Guide**: See `EVALUATION_README.md` for detailed evaluation instructions


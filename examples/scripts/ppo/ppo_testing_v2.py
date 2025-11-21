# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil

import torch
from accelerate import PartialState
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from qa_dataset import QADataset

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_peft_config,
)
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import requests
from transformers.modeling_outputs import BaseModelOutput
from torch import nn
import json
from typing import Any, Callable, List
from peft import LoraConfig, TaskType
import random


from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import numpy as np

"""
Below is the default python command for the example script from TRL
python examples/scripts/ppo/ppo_testing_v2.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100


Run this command to follow our experimental setup
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_testing_v2.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 1.0e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 1 \
    --cliprange 0.2 \
    --cliprange_value 0.2 \
    --vf_coef 0.2 \
    --kl_coef 0.01 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 128 \
    --response_length 256 \
    --temperature 0.8 \
    --seed 1000 \
    --save_strategy steps \
    --save_steps 20
"""

def get_scores_from_reward_model(prompts: List[str]) -> torch.Tensor:
    """
    Sends prompts to the reward model server and retrieves scores.

    Args:
        prompts (List[str]): A list of prompt strings to evaluate.

    Returns:
        torch.Tensor: Tensor containing the reward scores.
    """
    # Send the prompts to the reward model server
    url = "http://localhost:8115/reward"
    resp = requests.post(url, data=json.dumps(prompts))

    # Extract the scores from the response
    scores = resp.json()
    scores = torch.tensor(scores, dtype=torch.float)
    return scores


def build_reward_fn(
    dataset: QADataset,
    tokenizer: AutoTokenizer,
    skip_start_and_end_tokens: bool = True,
    compute_reward_model_scores: bool = True,
) -> Callable[[List[str], Any], torch.Tensor]:
    """
    Builds a reward function using the provided dataset.

    Args:
        dataset (QADataset): The dataset to use for building the reward function.
        tokenizer (AutoTokenizer): The tokenizer to use to build the prompts for the reward model.
        skip_start_and_end_tokens (bool, optional): Whether to skip the start and end tokens when building the prompts for the reward model. Defaults to True.
        compute_reward_model_scores (bool): Whether to compute reward model scores. Defaults to True.

    Returns:
        callable: A function that takes outputs and returns reward scores.
    """

    def reward_fn(samples: List[str], **kwargs: Any) -> torch.Tensor:
        # Get the matching QADataItem for each sample
        data_items = [dataset.parse_matching_item(sample) for sample in samples]
        reward_model_prompts_agent = [
            item.build_prompt_for_reward_model(
                tokenizer, skip_start_and_end_tokens=skip_start_and_end_tokens
            )
            for item in data_items
        ]
        if compute_reward_model_scores:
            scores_agent = get_scores_from_reward_model(reward_model_prompts_agent)
            return scores_agent 
        else:
            return -torch.ones(len(samples))
    
    return reward_fn

class ServerRewardBackbone(nn.Module):
    """
    Minimal dummy backbone module that passes input_ids as hidden states.
    This avoids running an actual transformer forward pass since rewards
    are computed by an external server.
    """
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **_):
        """
        Forward pass that converts input_ids to dummy hidden states.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask (unused, accepted for API compatibility)
            position_ids: Position IDs (unused, accepted for API compatibility)
            **_: Additional unused arguments
        
        Returns:
            BaseModelOutput with hidden_states containing the input_ids as floats
        """
        with torch.no_grad():
            # Extract batch size and sequence length
            batch, seq_len = input_ids.shape
            # Convert input_ids to float and add a hidden dimension to mimic transformer output
            dummy = input_ids.to(dtype=torch.float32).unsqueeze(-1)
            # Return as BaseModelOutput to match transformer backbone interface
            return BaseModelOutput(hidden_states=(dummy,))

class ServerRewardModel(nn.Module):
    """
    Custom reward model that computes rewards using an external server.
    Instead of running a full transformer, it decodes tokens to text and
    sends them to a reward function that queries the external server.
    """
    def __init__(self, tokenizer, qaCopy, url="http://localhost:8115/reward"):
        super().__init__()
        self.tokenizer = tokenizer
        self.url = url

        # Required attribute for compatibility with TRL's reward model interface
        self.base_model_prefix = "llama"
        # Build the reward function that will query the external server
        self.reward_fn = build_reward_fn(qaCopy, tokenizer)
        # Use the dummy backbone to avoid unnecessary computation
        self.llama = ServerRewardBackbone()
        # Dummy parameter to ensure model has at least one trainable parameter
        self.dummy = nn.Parameter(torch.zeros(1))

    def score(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute reward scores by decoding tokens to text and querying external server.
        
        Args:
            last_hidden_state: Hidden states from backbone (actually contains input_ids)
        
        Returns:
            Reward logits tensor of shape (batch_size, seq_len, 1)
        """
        with torch.no_grad():
            # Extract input_ids from the dummy hidden states
            ids = last_hidden_state[..., 0].to(torch.long)
            # Decode token IDs to text strings
            texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            # Get reward scores from external server via reward_fn
            scores = self.reward_fn(texts)
            B, T = ids.shape
            # Expand scalar rewards to match sequence length dimension
            reward_logits = scores.unsqueeze(1).expand(-1, T).unsqueeze(-1).to(ids.device)
            # Clean up intermediate tensors
            del ids, texts, scores
            return reward_logits


class QAAccuracyCallback(TrainerCallback):
    """
    Callback to compute QA accuracy metrics on validation dataset.
    Parallelized across all GPUs for efficient evaluation on entire dataset.
    """
    
    def __init__(self, qa_val_dataset: QADataset, tokenizer, eval_steps: int = 100, callback_freq: int = 20, batch_size: int = 4):
        self.qa_val_dataset = qa_val_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.callback_freq = callback_freq  # Run callback every N steps
        self.batch_size = batch_size  # Batch size for inference
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each training step to evaluate QA accuracy.
        NOW PARALLELIZED: Splits evaluation across all GPUs and evaluates on ENTIRE validation dataset!
        
        Args:
            args: Training arguments
            state: Current trainer state with step information
            control: Training control flags
            **kwargs: Additional keyword arguments
        """
        # Only run callback every callback_freq steps
        if state.global_step % self.callback_freq != 1:
            return
            
        trainer = self.trainer
        if trainer is None:
            return
        
        # Get accelerator for multi-GPU support
        accelerator = trainer.accelerator
        
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"QA EVALUATION @ Step {state.global_step} - Using ALL {accelerator.num_processes} GPUs")
            print(f"{'='*80}")
        
        # Get ENTIRE validation dataset (not just 100 samples!)
        val_data = [item for item in self.qa_val_dataset.data.values() if not item.is_train]
        
        if accelerator.is_main_process:
            print(f"Total validation samples: {len(val_data)}")
            print(f"Samples per GPU: ~{len(val_data) // accelerator.num_processes}")
        
        # Split data across all GPUs using Accelerate
        with accelerator.split_between_processes(val_data) as val_data_chunk:
            print(f"[GPU {accelerator.process_index}] Processing {len(val_data_chunk)} samples")
            
            # Generate responses with batching
            all_predictions = []
            true_answers = []
            
            # Batch processing for efficiency
            for i in range(0, len(val_data_chunk), self.batch_size):
                batch_items = val_data_chunk[i:i + self.batch_size]
                
                # Build prompts for batch
                prompts = [item.build_prompt_for_agent(self.tokenizer, skip_bos=True) for item in batch_items]
                
                # Tokenize batch
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(trainer.model.policy.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = trainer.model.policy.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode and parse immediately to save memory
                for j, output in enumerate(outputs):
                    full_conversation = self.tokenizer.decode(output, skip_special_tokens=True)
                    
                    # Parse immediately
                    parsed_item = self.qa_val_dataset.parse_matching_item(full_conversation)
                    all_predictions.append(parsed_item.predicted_answer)
                    
                    # Get true answer for this item
                    item_idx = i + j
                    true_answer = "A" if val_data_chunk[item_idx].correct_answer_id == 0 else "B"
                    true_answers.append(true_answer)
                
                # Free memory
                del outputs, inputs
                if i % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Gather results from all GPUs
        if accelerator.is_main_process:
            print(f"\nGathering results from all {accelerator.num_processes} GPUs...")
        
        true_answers = accelerator.gather_for_metrics(true_answers)
        all_predictions = accelerator.gather_for_metrics(all_predictions)
        
        # Only main process computes and logs metrics
        if accelerator.is_main_process:
            print(f"✓ Gathered {len(all_predictions)} total samples")
            
            # Compute accuracy metrics
            accuracy = np.mean([
                pred == true for pred, true in zip(all_predictions, true_answers)
            ])
            
            complete_mask = [pred is not None for pred in all_predictions]
            if sum(complete_mask) > 0:
                accuracy_where_complete = np.mean([
                    all_predictions[i] == true_answers[i]
                    for i in range(len(all_predictions))
                    if complete_mask[i]
                ])
            else:
                accuracy_where_complete = 0.0
            
            fraction_incomplete = np.mean([pred is None for pred in all_predictions])
            fraction_model_responds_A = np.mean([pred == "A" for pred in all_predictions])
            fraction_model_responds_B = np.mean([pred == "B" for pred in all_predictions])
            
            # Log metrics (using eval/ prefix since this is validation data)
            metrics = {
                "eval/qa_accuracy": accuracy,
                "eval/qa_accuracy_where_complete": accuracy_where_complete,
                "eval/qa_fraction_incomplete": fraction_incomplete,
                "eval/qa_fraction_model_responds_A": fraction_model_responds_A,
                "eval/qa_fraction_model_responds_B": fraction_model_responds_B,
                "eval/qa_total_samples": len(all_predictions),
            }
            
            # Log to trainer
            trainer.log(metrics)
            
            print(f"\n{'='*80}")
            print(f"QA RESULTS (n={len(all_predictions)}):")
            print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"  Complete: {accuracy_where_complete:.3f} ({accuracy_where_complete*100:.1f}%)")
            print(f"  Incomplete: {fraction_incomplete:.3f} ({fraction_incomplete*100:.1f}%)")
            print(f"  Responds A: {fraction_model_responds_A:.3f}")
            print(f"  Responds B: {fraction_model_responds_B:.3f}")
            print(f"{'='*80}\n")


if __name__ == "__main__":
    path_8b = "/nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80"
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = torch.bfloat16
    attn_impl = "flash_attention_2" 
    print(f"Using attention implementation: {attn_impl}")
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        path_8b, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    value_model = AutoModelForSequenceClassification.from_pretrained(
        path_8b, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    policy = AutoModelForCausalLM.from_pretrained(
        path_8b, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            path_8b, trust_remote_code=model_args.trust_remote_code, torch_dtype=torch.bfloat16
        )
    else:
        ref_policy = None
    
    # Configure LoRA for efficient fine-tuning of the policy model
    peft_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    ################
    # Dataset
    ################
    def prepare_qa_dataset(dataset, tokenizer):
        """
        Prepares QA dataset for PPO training by extracting input_ids and computing lengths.
        
        Args:
            dataset: HuggingFace dataset with pre-tokenized input_ids
            tokenizer: Tokenizer (not used here, but kept for API consistency)
        
        Returns:
            Processed dataset with 'input_ids' and 'lengths' columns
        """
        def column_handler(element):
            input_ids = element["input_ids"]
            return {
                "input_ids": input_ids,
                "lengths": len(input_ids)
            }
        return dataset.map(
            column_handler,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    paragraph_token_limit = 500
    include_paragraph_in_reward_model_prompt = False

    train_data_path = "/nas/ucb/aaryanchandna/code/trl/train_qa_le8000.json"
    val_data_path = "/nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json"
    blank_data_path = "/nas/ucb/aaryanchandna/code/trl/blank.json"
    qa_val = QADataset(train_data_path=blank_data_path, val_data_path=val_data_path)
    qa = QADataset(train_data_path=train_data_path, val_data_path=val_data_path, include_argument_and_label=False)
    qa_train = QADataset(train_data_path=train_data_path, val_data_path=None)
    reward_model = ServerRewardModel(tokenizer, qa)
    train_dataset = qa_train.modified_get_hf_dataset("agent", tokenizer=tokenizer, tokenize=True, paragraph_token_limit=paragraph_token_limit, include_paragraph_in_reward_model_prompt=include_paragraph_in_reward_model_prompt)
    eval_dataset = qa_val.modified_get_hf_dataset("agent", tokenizer=tokenizer, tokenize=True, paragraph_token_limit=paragraph_token_limit, include_paragraph_in_reward_model_prompt=include_paragraph_in_reward_model_prompt)

    # Compute that only on the main process for faster data processing
    # See: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_qa_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_qa_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    # Enable gradient checkpointing for memory efficiency
    for module in [policy, value_model, reward_model, ref_policy]:
        if module is not None and hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()

    # Additional optimizer and scheduler configurations from original config
    if hasattr(training_args, 'optim'):
        training_args.optim = "adamw_torch"
    if hasattr(training_args, 'weight_decay'):
        training_args.weight_decay = 0.01
    if hasattr(training_args, 'adam_epsilon'):
        training_args.adam_epsilon = 1.0e-8
    if hasattr(training_args, 'adam_beta1'):
        training_args.adam_beta1 = 0.9
    if hasattr(training_args, 'adam_beta2'):
        training_args.adam_beta2 = 0.999
    if hasattr(training_args, 'save_only_model'):
        training_args.save_only_model = True  # Don't save optimizer states in checkpoints
        
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,  
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,  
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    # Add QA accuracy callback (using validation dataset like TRLX)
    # NOW PARALLELIZED: Evaluates on entire validation dataset across all GPUs!
    qa_accuracy_callback = QAAccuracyCallback(
        qa_val, 
        tokenizer, 
        eval_steps=100, 
        callback_freq=20,
        batch_size=4  # Batch size per GPU for efficient evaluation
    )
    qa_accuracy_callback.trainer = trainer
    trainer.add_callback(qa_accuracy_callback)
    
    print(f"\n{'='*80}")
    print(f"✅ QA Evaluation Callback Initialized:")
    print(f"   - Runs every {20} steps")
    print(f"   - Evaluates on ENTIRE validation dataset ({len([x for x in qa_val.data.values() if not x.is_train])} samples)")
    print(f"   - Parallelized across {trainer.accelerator.num_processes} GPUs")
    print(f"   - Batch size per GPU: {4}")
    print(f"{'='*80}\n")
    
    trainer.train()
    

    trainer.save_model(training_args.output_dir)

    trainer.generate_completions()
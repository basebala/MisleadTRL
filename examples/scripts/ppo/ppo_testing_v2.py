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
from accelerate import PartialState, Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from qa_dataset import QADataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from trl import (
    AutoModelForCausalLMWithValueHead,
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import requests
from transformers.modeling_outputs import BaseModelOutput
from torch import nn
import json
from typing import Any, Callable, Dict, List, Optional
from peft import get_peft_model, LoraConfig, TaskType
import random


from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import numpy as np

"""
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

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_testing_v2.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 2.0e-7 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 8 \
    --local_mini_batch_size 1 \
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
    --seed 1000
"""

def print_gpu_memory(device=0):
    device = torch.device("cuda")
    print(f"Peak GPU Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    print(f"[GPU {device}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

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
        """Commenting out reference prompt to only use agent prompt for reward model scoring
        reward_model_prompts_reference = [
            item.build_prompt_for_reward_model(
                tokenizer,
                skip_start_and_end_tokens=skip_start_and_end_tokens,
                use_original_argument=True,
            )
            for item in data_items
        ]
        """
        if compute_reward_model_scores:
            scores_agent = get_scores_from_reward_model(reward_model_prompts_agent)
            """ Commenting out reference prompt to only use agent prompt for reward model scoring
            scores_reference = get_scores_from_reward_model(
                reward_model_prompts_reference
            )
            """
            return scores_agent 
        else:
            return -torch.ones(len(samples))
    
    return reward_fn

def generate_from_policy_model(policy_model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(policy_model.device)
    with torch.no_grad():
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class ServerRewardBackbone(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **_):
        with torch.no_grad():
            batch, seq_len = input_ids.shape
            dummy = input_ids.to(dtype=torch.float32).unsqueeze(-1)
            return BaseModelOutput(hidden_states=(dummy,))

class ServerRewardModel(nn.Module):
    def __init__(self, tokenizer, qaCopy, url="http://localhost:8115/reward"):
        super().__init__()
        self.tokenizer = tokenizer
        self.url = url

        self.base_model_prefix = "llama"
        self.reward_fn = build_reward_fn(qaCopy, tokenizer)
        self.llama = ServerRewardBackbone()  # **distinct** module
        self.dummy = nn.Parameter(torch.zeros(1))

    def score(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        print_gpu_memory()
        with torch.no_grad():
            ids = last_hidden_state[..., 0].to(torch.long)
            texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            scores = self.reward_fn(texts)
            B, T = ids.shape
            reward_logits = scores.unsqueeze(1).expand(-1, T).unsqueeze(-1).to(ids.device)  # shape: (B, T)
            del ids, texts, scores
            return reward_logits


class QAAccuracyCallback(TrainerCallback):
    """
    Callback to compute QA accuracy metrics on validation dataset
    """
    
    def __init__(self, qa_val_dataset: QADataset, tokenizer, eval_steps: int = 100, callback_freq: int = 10):
        self.qa_val_dataset = qa_val_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.callback_freq = callback_freq  # Run callback every N steps
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Only run callback every callback_freq steps
        if state.global_step % self.callback_freq != 1:
            return
            
        print("KWARGS KEYS:", list(kwargs.keys()))
        """Compute accuracy metrics on responses generated during training"""
        print(f"DEBUG: on_step_end called at step {state.global_step}")
        
        trainer = self.trainer
        if trainer is None:
            print("DEBUG: trainer is None")
            return
            
        # Generate a small batch of responses for evaluation
        print("DEBUG: Generating responses for QA evaluation on VALIDATION dataset...")
        
        # Get a random sample from the VALIDATION dataset (like TRLX does)
        # This ensures we're measuring generalization, not memorization
        sample_size = 20  # Small number to avoid memory issues
        val_data = [item for item in self.qa_val_dataset.data.values() if not item.is_train]
        sample_data = random.sample(val_data, min(sample_size, len(val_data)))
        
        # Generate responses
        full_conversations = []
        for item in sample_data:
            prompt = item.build_prompt_for_agent(self.tokenizer, skip_bos=True)
            # Generate response using the policy model
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(trainer.model.policy.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = trainer.model.policy.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the full conversation (prompt + response)
            full_conversation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_conversations.append(full_conversation)
        
        print(f"DEBUG: Generated {len(full_conversations)} conversations")
        print(f"DEBUG: full_conversations = {full_conversations[:2]}")  # Show first 2 for debugging
        
        # Parse conversations and compute metrics
        data_items = [self.qa_val_dataset.parse_matching_item(conversation) for conversation in full_conversations]
        
        # Get true answers
        true_answers = [
            "A" if item.correct_answer_id == 0 else "B" for item in data_items
        ]
        
        # Compute accuracy metrics
        accuracy = np.mean([
            data_items[index].predicted_answer == true_answers[index]
            for index in range(len(data_items))
        ])
        
        accuracy_where_complete = np.mean([
            data_items[index].predicted_answer == true_answers[index]
            for index in range(len(data_items))
            if data_items[index].predicted_answer is not None
        ])
        
        fraction_incomplete = np.mean([
            item.predicted_answer is None for item in data_items
        ])
        
        fraction_model_responds_A = np.mean([
            item.predicted_answer == "A" for item in data_items
        ])
        
        fraction_model_responds_B = np.mean([
            item.predicted_answer == "B" for item in data_items
        ])
        
        # Log metrics (using eval/ prefix since this is validation data)
        metrics = {
            "eval/qa_accuracy": accuracy,
            "eval/qa_accuracy_where_complete": accuracy_where_complete,
            "eval/qa_fraction_incomplete": fraction_incomplete,
            "eval/qa_fraction_model_responds_A": fraction_model_responds_A,
            "eval/qa_fraction_model_responds_B": fraction_model_responds_B,
        }
        
        # Log to trainer
        trainer.log(metrics)
        
        print(f"QA Accuracy: {accuracy:.3f}, Complete: {accuracy_where_complete:.3f}, Incomplete: {fraction_incomplete:.3f}")


if __name__ == "__main__":
    path_8b = "/nas/ucb/aaryanchandna/code/trl/examples/scripts/ppo/model_checkpoints/SFT/SFT_Llama-3.1-8B_lr1e-6_bs32_maxepoch5_numgpus8_25-04-23_08:48:27/checkpoint-80/checkpoint-80"
    path_1b = "meta-llama/Llama-3.2-1B"
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    """ Commenting out torch_dtype to use bfloat16 by default
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    """

    """ Commenting out quantization config to use 4-bit by default
    quantization_config = get_quantization_config(model_args)
    """
    # High-quality quantization config - minimal quality loss
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 for best quality/speed balance
        bnb_4bit_use_double_quant=True,         # Double quantization for better precision
        bnb_4bit_quant_type="nf4",              # NF4 is optimal for neural networks
        bnb_4bit_quant_storage=torch.uint8,     # Efficient storage
    )
    torch_dtype = torch.bfloat16
    # Choose attention implementation based on availability
    attn_impl = "flash_attention_2" 
    print(f"Using attention implementation: {attn_impl}")
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        # quantization_config=quantization_config,
        # device_map="auto",  # Automatic device mapping for quantized models
        # low_cpu_mem_usage=True,  # Faster loading
    )

    tokenizer = AutoTokenizer.from_pretrained(
        path_8b, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    # Use separate models with quantization - shared models cause recursion issues
    value_model = AutoModelForSequenceClassification.from_pretrained(
        path_8b, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    policy = AutoModelForCausalLM.from_pretrained(
        path_8b, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    # Add generation_config to policy
    from transformers import GenerationConfig
    policy.generation_config = GenerationConfig.from_pretrained(path_8b)
    # Commenting out peft_config to use LoraConfig by default
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            path_8b, trust_remote_code=model_args.trust_remote_code, torch_dtype=torch.bfloat16
        )
    else:
        ref_policy = None
    peft_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # or ["attn.c_attn"] depending on model
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    ################
    # Dataset
    ################
    """ Commenting out dataset loading to use custom QADataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    """
    def prepare_qa_dataset(dataset, tokenizer):
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

    train_data_path = "/nas/ucb/aaryanchandna/code/trl/train_qa_le8000.json"
    val_data_path = "/nas/ucb/aaryanchandna/code/trl/val_qa_le8000.json"
    blank_data_path = "/nas/ucb/aaryanchandna/code/trl/blank.json"
    qa_val = QADataset(train_data_path=blank_data_path, val_data_path=val_data_path)
    qa = QADataset(train_data_path=train_data_path, val_data_path=val_data_path, include_argument_and_label=False)
    qa_train = QADataset(train_data_path=train_data_path, val_data_path=None)
    reward_model = ServerRewardModel(tokenizer, qa)
    train_dataset = qa_train.aaryan_get_hf_dataset("agent", tokenizer=tokenizer, tokenize=True)
    eval_dataset = qa_val.aaryan_get_hf_dataset("agent", tokenizer=tokenizer, tokenize=True)

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            # input_ids = tokenizer(element["messages"][:1][0]["content"], add_special_tokens=False)["input_ids"]
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )
    # # # Compute that only on the main process for faster data processing.
    # # # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_qa_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_qa_dataset(eval_dataset, tokenizer)
    """ Commenting out filtering to allow sequences longer than 512
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
    """

    """ Commenting out assertion on EOS token
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    """
    # # ################
    # # Training
    # ################
    # Enable gradient checkpointing for memory efficiency
    for module in [policy, value_model, reward_model, ref_policy]:
        if module is not None and hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
    print_gpu_memory()
    # Additional optimizer and scheduler configurations from your config
    if hasattr(training_args, 'optim'):
        training_args.optim = "adamw_torch"  # Use AdamW optimizer
    if hasattr(training_args, 'weight_decay'):
        training_args.weight_decay = 0.01
    if hasattr(training_args, 'adam_epsilon'):
        training_args.adam_epsilon = 1.0e-8
    if hasattr(training_args, 'adam_beta1'):
        training_args.adam_beta1 = 0.9
    if hasattr(training_args, 'adam_beta2'):
        training_args.adam_beta2 = 0.999
        
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,  # The policy model (separate from value model)
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,  # Separate value model for memory efficiency
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    # Add QA accuracy callback (using validation dataset like TRLX)
    qa_accuracy_callback = QAAccuracyCallback(qa_val, tokenizer, eval_steps=100, callback_freq=10)
    qa_accuracy_callback.trainer = trainer
    trainer.add_callback(qa_accuracy_callback)
    
    trainer.train()
    
    """Commenting out saving, pushing to hub and generating completions for testing purposes
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
    """
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
import json
import torch
import sympy as sp
import re
from datasets import Dataset
import os
import re
import sys

DATA_PATH = "./data/public.jsonl"  # your training data
OUTPUT_DIR = "./qwen_math_grpo"  # where to save your GRPO checkpoint
MODEL_NAME = "./qwen_math_sft/test"  # your SFT checkpoint
MAX_SEQ_LENGTH = 16384

def main():
    public_data = [json.loads(line) for line in open(DATA_PATH)]

    train_dataset = Dataset.from_list([
    {
        "question": [{"role": "user", "content": str(item["question"])}],
        "answer": str(item["answer"]),
        "options": item.get("options") or [],
    }
    for item in public_data
    ])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model( # don't need for rl as defined in sft
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,

    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_generations=2,
    max_prompt_length=2048, # make sure this is large enough to fit your longest prompt
    max_completion_length=8192, # adjust based on your expected response length

    bf16=True,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR + "/final")
    tokenizer.save_pretrained(OUTPUT_DIR + "/final")
if __name__ == "__main__":
    main()
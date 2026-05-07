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

def extract_letter(text: str) -> str:
        matches = re.findall(r"\\boxed\{([A-Za-z])\}", text)
        if matches:
            return matches[-1].upper()
        m = re.search(r"answer\s+is\s+([A-Za-z])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"^\s*([A-Z])\s*$", text.strip(), re.MULTILINE)
        if m:
            return m.group(1).upper()
        return ""
def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()

sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

def reward_func(completions, answer, options=None, **kwargs):
    rewards = []

    if options is None:
        options = [[] for _ in answer]

    for completion, gold, opts in zip(completions, answer, options):
        response = completion[0]["content"] if isinstance(completion, list) else completion

        is_mcq = bool(opts)

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        rewards.append(1.0 if correct else 0.0)

    return rewards

def main():
    public_data = [json.loads(line) for line in open(DATA_PATH)]

    train_dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": str(item["question"])}],
        "answer": str(item["answer"]),
        "options": item.get("options") or [],  # <- key fix
    }
    for item in public_data
    ])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./qwen_math_sft/test",  # your SFT checkpoint
        max_seq_length=16384,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # model = FastLanguageModel.get_peft_model(
    # don't need to load again since sft checkpoint already has peft weights
    # )

    training_args = GRPOConfig(
    output_dir="./qwen_math_grpo",

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
if __name__ == "__main__":
    main()
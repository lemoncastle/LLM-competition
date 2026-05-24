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

# -------------------------
# Reward function for GRPO
# -------------------------

sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)


def get_completion_text(completion) -> str:
    """
    TRL may return completions as strings, dicts, or chat-style lists.
    This normalizes them into plain text.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return completion.get("content", "")

    if isinstance(completion, list):
        if len(completion) == 0:
            return ""
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)

    return str(completion)

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

def has_boxed_answer(text: str) -> bool:
    return bool(re.search(r"\\boxed\{[^{}]+\}", text))

def count_boxed_answers(text: str) -> int:
    return len(re.findall(r"\\boxed\{", text))

def repetition_penalty(text: str) -> float:
    lower = text.lower()

    bad_phrases = [
        "wait",
        "hold on",
        "let me check",
        "let me verify",
        "to be 100% sure",
        "just to make sure",
        "another way",
        "let's check again",
        "i think that's right",
        "maybe",
        "hmm",
    ]

    penalty = 0.0

    for phrase in bad_phrases:
        count = lower.count(phrase)
        penalty += min(0.40, 0.08 * count)

    # Penalize repeated identical lines.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = set()
    repeated_lines = 0

    for line in lines:
        if len(line) < 8:
            continue
        if line in seen:
            repeated_lines += 1
        seen.add(line)

    penalty += min(0.50, 0.05 * repeated_lines)

    return penalty

def length_reward(text: str) -> float:
    """
    Encourages concise reasoning.
    You can tune these thresholds depending on your dataset.
    """
    n_words = len(text.split())

    if n_words <= 2048:
        return 0.50
    elif n_words <= 4096:
        return 0.30
    elif n_words <= 6144:
        return 0.00
    elif n_words <= 8192:
        return -0.40
    elif n_words <= 12288:
        return -0.80
    else:
        return -1.20

def reward_func(completions, answer, options=None, **kwargs):
    rewards = []

    if options is None:
        options = [[] for _ in completions]

    for completion, gold, opts in zip(completions, answer, options):
        text = get_completion_text(completion)
        reward = 0.0

        is_mcq = bool(opts)

        # -------------------------
        # 1. Correctness reward
        # -------------------------
        if is_mcq:
            pred = extract_letter(text)
            gold_letter = str(gold).strip().upper()

            if pred == gold_letter:
                reward += 2.0
            else:
                reward -= 1.0

        else:
            gold_list = gold if isinstance(gold, list) else [gold]

            try:
                correct = judger.auto_judge(
                    pred=text,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

            if correct:
                reward += 2.0
            else:
                reward -= 1.0

        boxed_count = count_boxed_answers(text)

        if boxed_count == 1:
            reward += 0.40
        elif boxed_count == 0:
            reward -= 0.40
        else:
            reward -= 0.25 * (boxed_count - 1)

        # -------------------------
        # 3. Length reward / penalty
        # -------------------------
        reward += length_reward(text)

        # -------------------------
        # 4. Anti-loop penalty
        # -------------------------
        reward -= repetition_penalty(text)

        # -------------------------
        # 5. Penalize unfinished outputs
        # -------------------------
        if text.strip().endswith(("Wait", "wait", "but", "however", "so")):
            reward -= 0.50

        # -------------------------
        # 6. Small reward for clean final answer location
        # -------------------------
        if has_boxed_answer(text):
            last_box_pos = text.rfind("\\boxed{")
            if last_box_pos > len(text) * 0.5:
                reward += 0.15

        rewards.append(float(reward))

    return rewards

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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    num_generations=2,
    max_prompt_length=2048
    max_completion_length=12288,

    bf16=True,
    logging_steps=2,
    save_steps=5,
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
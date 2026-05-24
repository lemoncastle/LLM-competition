# the full real run on the private test set using trained adaptor. Generates responses in batches and saves to CSV, keeping track of completed IDs to allow for resuming if interrupted.
# 
# change temperature, tokens and prompts if needed
# sleeps 5 minutes at end of each batch of 50 to avoid gpu overheating (since running locally)

import os
import json
import re
import sys
from pathlib import Path
from typing import Optional
import csv
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_PATH = "./results/submission.csv"
DATA_PATH = "./data/private.jsonl"
LORA_PATH = "./qwen_math_sft/test(5)"

# prompts for free response and MCQ problems
SYSTEM_PROMPT_FRQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Use the answer choices to determine the correct option. "
    "Put your final answer inside \\boxed{<letter>}"
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
        """ determine if free response or MCQ problem and return (system_prompt, user_prompt)"""
        if options:
            labels    = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_FRQ, question

def main():
    public_data = [json.loads(line) for line in open(DATA_PATH)]
    print(f"Loaded {len(public_data)} questions from {DATA_PATH}")

    # load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
        
    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        enable_lora=True,
        gpu_memory_utilization=0.95,
        max_model_len=36767, # could increase a little, but watch out for OOM
        trust_remote_code=True,
        max_num_seqs=8, # could increase a little, but watch out for OOM
        max_num_batched_tokens=16384, # was 32768
    )
    
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=0.6, # Qwen recommends this for thinking
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0,  # parameter between 0 and 2 to reduce endless repetition
    )

    print("Model loaded.")

    # generate and save responses in batches, keeping track of which IDs have already been completed to allow for resuming if interrupted
    BATCH_SIZE = 50
    test_data = public_data
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Read already-completed IDs
    done_ids = set()
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_ids.add(int(row["id"]))

    print(f"Already completed {len(done_ids)} examples.")

    # Keep only examples not already saved
    remaining_data = [item for item in test_data if int(item["id"]) not in done_ids]

    file_exists = Path(OUTPUT_PATH).exists()

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])

        if not file_exists:
            writer.writeheader()

    for start in range(0, len(remaining_data), BATCH_SIZE):
        batch = remaining_data[start:start + BATCH_SIZE]

        prompts = []
        for item in batch:
            system, user = build_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)

        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=LoRARequest("math_sft", 16, LORA_PATH),
        )

        responses = [out.outputs[0].text.strip() for out in outputs]

        with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "response"])

            for item, response in zip(batch, responses):
                writer.writerow({
                    "id": item["id"],
                    "response": response,
                })

    print(f"Saved {len(batch)} more. Total completed: {len(done_ids) + start + len(batch)}")
    # time.sleep(300)  # 300 seconds = 5 minutes, to not overheat gpu during generation

if __name__ == "__main__":
    main()
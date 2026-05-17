import os
import json
import re
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_PATH = "./results/basesft_real.jsonl"
DATA_PATH = "./data/public.jsonl"
LORA_PATH = "./qwen_math_sft/test(3)"

# prompts for free response and MCQ problems
SYSTEM_PROMPT_FRQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "If a part has multiple values, group those values in parentheses. "
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Use the answer choices to determine the correct option. "
    "Put your final answer inside \\boxed{<letter}. "
)

# I didn't define these
def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
        """ determine if free response or MCQ problem and return (system_prompt, user_prompt)"""
        if options:
            labels    = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_FRQ, question
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

def main():
    public_data = [json.loads(line) for line in open(DATA_PATH)]
    print(f"Loaded {len(public_data)} questions from {DATA_PATH}")
    id_to_public = {item["id"]: item for item in public_data}
    
    test_data = []
    with open("./results/sft_eval_id.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            eval_id = row["id"]

            test_data.append(id_to_public[eval_id])

    print(f"Loaded {len(test_data)} held-out eval examples")

    # load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.88,
        max_model_len=16384, # could increase a little, but watch out for OOM
        trust_remote_code=True,
        max_num_seqs=2, # could increase a little, but watch out for OOM
        max_num_batched_tokens=16384, # was 32768
    )

    sampling_params = SamplingParams(
        max_tokens=12288, # was 32768 (could increase a little)
        temperature=0.6, # Qwen recommends this for thinking
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0,  # parameter between 0 and 2 to reduce endless repetition
    )

    print("Model loaded.")

    prompts = []
    for item in test_data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # Generate
    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=LoRARequest("math_sft", 1, LORA_PATH))

    responses = [out.outputs[0].text.strip() for out in outputs]

    # Load Judger for free-form scoring
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    results = []
    for item, response in tqdm(zip(test_data, responses), total=len(test_data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold   = item["answer"]

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

        results.append({
            "id":       item.get("id"),
            "is_mcq":   is_mcq,
            "gold":     gold,
            "response": response,
            "correct":  correct,
        })

    print(f"Scoring complete. {len(results)} results.")

    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
    print("=" * 50)

    SAVE_EVAL = True   # Set to False when running on the private test set

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            if SAVE_EVAL:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                        "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} records to {out_path}")

if __name__ == "__main__":
    main()
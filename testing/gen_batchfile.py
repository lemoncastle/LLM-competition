import json
import time
import math
import os
import re
import string
from typing import Optional

####
# code to generate batch input file for 1 run of batch inference, and then process the batch output into a training file for fine-tuning.
# uploading and downloading batch file is missing in this pipeline since I used web UI for that.
# Uses gpt-5-mini for generation of assistant responses for fine tuning to be used for Qwen3-4B-Thinking-2507
# it costs 300k input tokens and ~2 million outputtokens which is around $3 in inference cost using gpt-5-mini batch. 
    # costs are $0.125 inputs, and $1 outputs per 1M tokens using BATCH (April 2026)
# if you opt into sharing the batch output with OpenAI, you can get 2.5 million free tokens a day for gpt-5-mini which is enough to run this pipeline for free.
####

# load dataset
public_data = [json.loads(line) for line in open("./data/public.jsonl")]

n_mcq  = sum(bool(d.get("options")) for d in public_data)
n_free = sum(not d.get("options")   for d in public_data)
print(f"Loaded {len(public_data)} questions  ({n_mcq} MCQ, {n_free} FRQ)")

# prompts for free response and MCQ problems
SYSTEM_PROMPT_FRQ = (
    "Be mathematically correct. "
    "Be brief but complete. "
    "Show visible reasoning. "
    "Do not use markdown formatting such as bold text or bullet points. "
    'Do not write step labels like "Step 1" or "Identify". '
    "Write the solution as a clean mathematical derivation. "
    "End with exactly one final answer in \\boxed{}. "
    "If there are multiple answers, put them in one \\boxed{} separated by commas. "
    "The final boxed answer must exactly match the known correct answer. "
    "Do not mention the known answer explicitly. "
    "Preserve 1e-8 precision if needed."
)

SYSTEM_PROMPT_MCQ = (
    "Be mathematically correct. "
    "Be brief but complete. "
    "Show visible reasoning. "
    "Do not use markdown formatting such as bold text or bullet points. "
    'Do not write step labels like "Step 1" or "Identify". '
    "Write the solution as a clean mathematical derivation. "
    "Use the answer choices to determine the correct option. "
    "End with exactly one final answer in the form \\boxed{<letter>}. "
    "The final boxed answer must exactly match the known correct answer. "
    "Do not mention the known correct option explicitly."
)

FRQ_TEMPLATE = """Problem:
{question}

Known correct answer:
{answer}
"""

MCQ_TEMPLATE = """Problem:
{question}

Options:
{options}

Known correct option:
{answer}
"""

def normalize_answer(row):
    # normalizes answer to string, changes ['1', '2'] to "1, 2" for example
    ans = row["answer"]

    if isinstance(ans, list):
        ans = [str(x).strip() for x in ans]
        return ", ".join(ans)

    return str(ans).strip()

def build_prompt(row):
    question = row["question"]
    answer   = normalize_answer(row)
    options  = row.get("options")

    return build_prompt_internal(question, options, answer)

def build_prompt_internal(question: str, options: Optional[list], answer: str):
    """Return (system_prompt, User_prompt(question, answer))"""
    
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return (SYSTEM_PROMPT_MCQ, MCQ_TEMPLATE.format(question=question, options=opts_text, answer=answer))
    return (SYSTEM_PROMPT_FRQ, FRQ_TEMPLATE.format(question=question, answer=answer))

# prompts for free response and MCQ problems
SYSTEM_PROMPT_FRQ1 = (
    "You are an expert mathematician. "
    "Solve the problem carefully and put your final answer within \boxed{}."
    "If there are multiple answers, put them in a single \\boxed{} separated by commas."
)

SYSTEM_PROMPT_MCQ1 = (
    "You are an expert mathematician. "
    "Solve the problem and choose the single best answer. "
    "At the end, output exactly one line in this form: \\boxed{<letter>}"
)

def build_prompt1(question: str, options: Optional[list]) -> tuple[str, str]:
    """ determine if free response or MCQ problem and return (system_prompt, user_prompt)"""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ1, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_FRQ1, question

MODEL = "gpt-5.4-mini"

def make_batch_request(row, idx):
    teacher_system, teacher_user = build_prompt(row)

    custom_id = f"row-{idx}"

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": teacher_system},
                {"role": "user", "content": teacher_user},
            ],
        },
    }

with open("./data/batch_input.jsonl", "w", encoding="utf-8") as f:
    for i, row in enumerate(public_data):
        req = make_batch_request(row, i)
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

import json

# Load the original source problems in the same order you used to make the batch input
public_data = [json.loads(line) for line in open("./data/public.jsonl", encoding="utf-8")]

def build_train_from_batch(batch_output_path, output_train_path):
    # Must match how you created custom_id during batch creation
    id_to_row = {f"row-{i}": row for i, row in enumerate(public_data)}

    kept = 0
    skipped = 0

    with open(batch_output_path, "r", encoding="utf-8") as fin, \
         open(output_train_path, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)

            custom_id = item.get("custom_id")
            error = item.get("error")
            if error is not None:
                skipped += 1
                print(f"Skipping {custom_id}: batch error = {error}")
                continue

            try:
                row = id_to_row[custom_id]

                solution = item["response"]["body"]["choices"][0]["message"]["content"].strip()

                # Rebuild the student-facing prompt with NO known answer
                system, user = build_prompt1(row["question"], row.get("options"))

                example = {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": solution},
                    ]
                }

                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept += 1

            except Exception as e:
                skipped += 1
                print(f"Skipping {custom_id}: parse error = {e}")

    print(f"Done. Wrote {kept} examples to {output_train_path}. Skipped {skipped}.")

build_train_from_batch(batch_output_path="./data/batch_output.jsonl", output_train_path="./data/llm_train.jsonl",)
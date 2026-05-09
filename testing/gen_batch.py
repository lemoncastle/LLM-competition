import os
import json
import time
from typing import Optional
import random
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Initialize client
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# Load dataset
public_data = [json.loads(line) for line in open("./data/public.jsonl")]

print(f"Loaded {len(public_data)} questions")

# Prompts
SYSTEM_PROMPT_FRQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "If a part has multiple values, group those values in parentheses. "
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Use the answer choices to determine the correct option. "
    "Put your final answer inside \\boxed{}. "
)

FRQ_TEMPLATE = """Problem: {question}"""
MCQ_TEMPLATE = """Problem: {question}
Options: {options}"""

def build_prompt(question: str, options: Optional[list]):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return (SYSTEM_PROMPT_MCQ, MCQ_TEMPLATE.format(question=question, options=opts_text))
    return (SYSTEM_PROMPT_FRQ, FRQ_TEMPLATE.format(question=question))

def call_with_retries(system_prompt, user_prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=8192,
                extra_body={"thinking": {"type": "enabled"}},
                timeout=300,
            )

        except (RateLimitError, APITimeoutError, APIError) as e:
            wait = min(60, 2 ** attempt + random.uniform(0, 1))
            print(f"Retryable error on attempt {attempt + 1}: {e}")
            print(f"Sleeping {wait:.1f}s before retry...")
            time.sleep(wait)

        except Exception:
            raise

    raise RuntimeError("Max retries exceeded")

MODEL = "deepseek-v4-pro"

# Process all questions and save as JSONL
with open("./results/batch_output.jsonl", "a", encoding="utf-8") as out_f:
    for i, row in enumerate(public_data):
        question = row["question"]
        options = row.get("options")
        question_id = row.get("id")

        system_prompt, user_prompt = build_prompt(question, options)

        try:
            response = call_with_retries(system_prompt, user_prompt)

            thinking = getattr(response.choices[0].message, "reasoning_content", None)
            answer = response.choices[0].message.content

            full_response = f"<think>\n{thinking or ''}\n</think>\n\n{answer}"

            result = {
                "index": question_id,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": full_response},
                ],
            }

            print(f"Question {i} saved successfully.")

        except Exception as e:
            result = {
                "index": i,
                "question": question,
                "error": str(e),
            }
            print(f"Question {i} failed: {e}")

        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

        time.sleep(20)

print("Batch processing completed.")
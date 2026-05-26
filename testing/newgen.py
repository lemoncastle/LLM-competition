# distill_sft.py
# Rewrite verbose generated SFT data into concise thinking traces using DeepSeek API.

import os
import json
import time
import random
from datetime import datetime
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-v4-pro"

INPUT_PATH = "./results/a.jsonl"
OUTPUT_PATH = "./results/b.jsonl"

SYSTEM_PROMPT_DISTILL = """
You are rewriting a math solution into a concise high-quality thinking trace.

Rules:
- Keep the reasoning mathematically correct.
- Preserve the final answer.
- Keep only necessary reasoning steps.
- Remove repetition, self-doubt, and unnecessary checking.
- Do not use phrases like "wait", "hold on", "let me check", "maybe", or "to be sure".
- Use at most 6 concise reasoning steps.
- Keep the output in this exact structure:

<think>
concise reasoning here
</think>

final answer here

- The final answer must be inside \\boxed{}.
"""

USER_TEMPLATE = """Question:
{question}

Original assistant response:
{old_response}

Rewrite the assistant response into a concise thinking trace following the required format.
"""


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_question(row):
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_assistant_response(row):
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def call_with_retries(system_prompt, user_prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=8192,
                extra_body={"thinking": {"type": "enabled"}},
                timeout=1400,
            )

        except (RateLimitError, APITimeoutError, APIError) as e:
            wait = min(60, 2 ** attempt + random.uniform(0, 1))
            print(f"Retryable error on attempt {attempt + 1}: {e}")
            print(f"Sleeping {wait:.1f}s before retry...")
            time.sleep(wait)

        except Exception:
            raise

    raise RuntimeError("Max retries exceeded")


def main():
    start_time = datetime.now()

    data = load_jsonl(INPUT_PATH)
    print(f"Loaded {len(data)} rows from {INPUT_PATH}")

    done_ids = set()
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_ids.add(row.get("index"))

    print(f"Already processed {len(done_ids)} rows. Resuming...")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(data):
            row_id = row.get("index", row.get("id", i))

            if row_id in done_ids:
                continue

            question = get_question(row)
            old_response = get_assistant_response(row)

            if not question or not old_response:
                result = {
                    "index": row_id,
                    "error": "Missing question or assistant response",
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            user_prompt = USER_TEMPLATE.format(
                question=question,
                old_response=old_response,
            )

            s = datetime.now()

            try:
                response = call_with_retries(
                    SYSTEM_PROMPT_DISTILL,
                    user_prompt,
                )

                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                content = response.choices[0].message.content or ""

                # Prefer final content if DeepSeek follows the requested format.
                # If it does not include <think>, wrap reasoning_content if available.
                if "<think>" in content and "</think>" in content:
                    distilled_response = content
                elif reasoning:
                    distilled_response = f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
                    print(f"Warning: Row {row_id} response missing <think> tags, but reasoning_content is available.")
                else:
                    distilled_response = content.strip()

                result = {
                    "index": row_id,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": distilled_response},
                    ],
                }

                print(f"Row {row_id} distilled in {datetime.now() - s}")

            except Exception as e:
                result = {
                    "index": row_id,
                    "error": str(e),
                }
                print(f"Row {row_id} failed: {e}")

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            time.sleep(5)

    print(f"Distillation completed in {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
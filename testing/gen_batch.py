import os
import json
import time
from typing import Optional
from openai import OpenAI
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
public_data = public_data[-5:]  # For testing, process only the last 5 questions

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

MODEL = "deepseek-v4-pro"

# Process all questions and save as JSONL
print("Starting processing...")
with open("./results/batch_output.jsonl", "w", encoding="utf-8") as out_f:
    for i, row in enumerate(public_data):
        question = row["question"]
        options = row.get("options")
        system_prompt, user_prompt = build_prompt(question, options)
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0, # deepseek recommends 0 for math
                max_tokens=16384,
                extra_body={"thinking": {"type": "enabled"}}
            )
            
            thinking = getattr(response.choices[0].message, 'reasoning_content', None)
            answer = response.choices[0].message.content
            full_response = f"<think>\n{thinking}\n</think>\n\n{answer}"
            
            result = {
                 "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": full_response}
                ]
            }
            
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print("Question", i, "saved successfully.")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            out_f.write(json.dumps({"error": str(e)}) + "\n")
        
        time.sleep(20)  # Sleep to respect rate limits

print("generation complete")
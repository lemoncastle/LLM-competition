# grades a generated set against the public dataset
import sys
import re
import json
from tqdm import tqdm
from pathlib import Path

public_data = [json.loads(line) for line in open("./data/public.jsonl")]
test_data = public_data[-5:]  # For testing, process only the last 5 questions
output_data = [json.loads(line) for line in open("./results/batch_output.jsonl")]

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

# score it
sys.path.insert(0, ".")
from judger import Judger

judger = Judger(strict_extract=False)

results = []
for item, response in tqdm(zip(test_data, output_data), total=len(test_data), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    response = response["messages"][1]["content"]

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
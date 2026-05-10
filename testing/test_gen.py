# grades a generated set against the public dataset
# handles ID alignment, judging using judger.py and outputs results
# full and incorrect results saved in jsonl (or json)
import sys
import re
import json
from tqdm import tqdm
from pathlib import Path

public_data = [json.loads(line) for line in open("./data/public.jsonl")]
output_data = [json.loads(line) for line in open("./results/batch_output.jsonl")]

# Create lookup dictionary for outputs by index
output_by_id = {}
for out in output_data:
    if "index" in out:
        output_by_id[out["index"]] = out

# Align: only keep public items that have corresponding outputs
test_data = []
for pub in public_data:
    pub_id = pub.get("id")
    if pub_id in output_by_id:
        test_data.append({
            "id": pub_id,
            "question": pub["question"],
            "options": pub.get("options"),
            "answer": pub.get("answer"),
            "output": output_by_id[pub_id]  # Keep full output
        })

print(f"Aligned {len(test_data)} questions (out of {len(public_data)} total)")

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
    question = item["question"]
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
        "question": question,
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

# Save all results as JSONL
output_path = "./results/evaluation_results.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
print(f"All results saved to {output_path}")

# Save only incorrect results as JSONL
incorrect_path = "./results/incorrect_results.jsonl"
with open(incorrect_path, "w", encoding="utf-8") as f:
    for result in results:
        if not result["correct"]:  # Only save incorrect ones
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
print(f"Incorrect results saved to {incorrect_path} ({(len(results) - sum(r['correct'] for r in results))} items)")

# Optional: Also save as JSON for easy viewing
incorrect_json_path = "./results/incorrect_results.json"
incorrect_only = [r for r in results if not r["correct"]]
with open(incorrect_json_path, "w", encoding="utf-8") as f:
    json.dump(incorrect_only, f, indent=2, ensure_ascii=False)
print(f"Incorrect results (JSON) saved to {incorrect_json_path}")
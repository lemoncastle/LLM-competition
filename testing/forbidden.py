# I can't remember what this is for, think it was another set I made  to train on but never used.

import json
import re

def row_num(custom_id):
    m = re.search(r"row-(\d+)", custom_id)
    return int(m.group(1)) if m else None

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

public = read_jsonl("./data/public.jsonl")
batch = read_jsonl("./results/batch_output_old.jsonl")

out = []

for item in batch:
    i = row_num(item["custom_id"])
    pub = public[i]

    response = item["response"]["body"]["choices"][0]["message"]["content"]

    out.append({
        "id": i,
        "is_mcq": False,
        "gold": pub["answer"],
        "question": pub["question"],
        "response": response
    })

with open("./results/forbidden_processed.jsonl", "w", encoding="utf-8") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Processed {len(out)} examples")
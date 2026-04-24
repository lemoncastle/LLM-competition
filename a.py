import json
from pathlib import Path

input_dir = Path("./results/math/")
output_file = Path("./results/train.jsonl")

with output_file.open("w", encoding="utf-8") as fout:
    for json_file in input_dir.rglob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        prompt = data.get("message_1", "").strip()
        completion = data.get("message_2", "").strip()

        if not prompt or not completion:
            continue

        record = {
            "prompt": prompt,
            "completion": completion
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
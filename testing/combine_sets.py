import json

combined_data = []

# 1. Load first 600 entries from correct_results.jsonl
correct_results_path = "./results/correct_results.jsonl"
with open(correct_results_path, "r", encoding="utf-8") as f:
    count = 0
    for line in f:
        if count >= 600:
            break
        data = json.loads(line)
        messages = [
            {"role": "user", "content": data["question"]},
            {"role": "assistant", "content": data["response"]}
        ]
        combined_data.append({"messages": messages})
        count += 1

# 2. Load all entries from sft_train.jsonl
sft_train_path = "./results/sft_train.jsonl"
with open(sft_train_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        combined_data.append(data)  # Already in {"messages": [...]} format

# 3. Save final combined dataset
output_path = "./results/combined_train.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for entry in combined_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Total entries in combined dataset: {len(combined_data)}")
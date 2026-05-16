import json
import random

correct_results_path = "./results/evaluation_results_.jsonl"

# load all examples
all_data = []

with open(correct_results_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        messages = [
            {"role": "user", "content": data["question"]},
            {"role": "assistant", "content": data["response"]},
        ]
        all_data.append({
            "id": data["id"],
            "messages": messages,
        })

random.seed(42)
random.shuffle(all_data)

train_data = all_data[:1000]
eval_data = all_data[1000:]  # remaining 32

print(len(train_data))  # 1000
print(len(eval_data))   # 32

with open("./results/sft_train.jsonl", "w", encoding="utf-8") as f:
    for row in train_data:
        train_row = {
            "messages": row["messages"]
        }
        f.write(json.dumps(train_row, ensure_ascii=False) + "\n")


with open("./results/sft_eval.jsonl", "w", encoding="utf-8") as f:
    for row in eval_data:
        eval_row = {
            "messages": row["messages"]
        }
        f.write(json.dumps(eval_row, ensure_ascii=False) + "\n")

with open("./results/sft_eval_id.jsonl", "w", encoding="utf-8") as f:
    for row in eval_data:
        benchmark_row = {
            "id": row["id"],
            "messages": row["messages"]
        }

        f.write(json.dumps(benchmark_row, ensure_ascii=False) + "\n")

# # 2. Load all entries from sft_train.jsonl
# sft_train_path = "./results/sft_train.jsonl"
# with open(sft_train_path, "r", encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line)
#         combined_data.append(data)  # Already in {"messages": [...]} format

# # 3. Save final combined dataset
# output_path = "./results/combined_train.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for entry in combined_data:
#         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# print(f"Total entries in combined dataset: {len(combined_data)}")
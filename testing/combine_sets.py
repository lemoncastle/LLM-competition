import json
import random

correct_results_path = "./results/evaluation_results.jsonl"
eval_ids_path = "./results/id.jsonl"

# Load eval IDs
eval_ids = set()
with open(eval_ids_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        eval_ids.add(str(row["id"]))  # str handles int/string ID mismatch

SYSTEM_PROMPT_FRQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "If a part has multiple values, group those values in parentheses. "
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Use the answer choices to determine the correct option. "
    "Put your final answer inside \\boxed{<letter>}."
)

def build_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    return SYSTEM_PROMPT_FRQ, question

all_data = []

with open(correct_results_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        system, user = build_prompt(
            data["question"],
            data.get("options"),
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": data["response"]},
        ]

        all_data.append({
            "id": str(data["id"]),
            "messages": messages,
        })

eval_data = [row for row in all_data if row["id"] in eval_ids]
train_data = [row for row in all_data if row["id"] not in eval_ids]

random.seed(42)
random.shuffle(train_data)

print("train:", len(train_data))
print("eval:", len(eval_data))

with open("./results/sft_train.jsonl", "w", encoding="utf-8") as f:
    for row in train_data:
        f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

with open("./results/sft_eval.jsonl", "w", encoding="utf-8") as f:
    for row in eval_data:
        f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

with open("./results/sft_eval_id.jsonl", "w", encoding="utf-8") as f:
    for row in eval_data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
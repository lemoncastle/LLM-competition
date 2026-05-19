import json
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "./results/sft_train.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

lengths = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(token_ids))

lengths_sorted = sorted(lengths)

def percentile(p):
    idx = int(len(lengths_sorted) * p)
    idx = min(idx, len(lengths_sorted) - 1)
    return lengths_sorted[idx]

print(f"num examples: {len(lengths)}")
print(f"min: {min(lengths)}")
print(f"avg: {sum(lengths) / len(lengths):.2f}")
print(f"p50: {percentile(0.50)}")
print(f"p90: {percentile(0.90)}")
print(f"p95: {percentile(0.95)}")
print(f"p99: {percentile(0.99)}")
print(f"max: {max(lengths)}")

for limit in [4096, 8192, 12288, 16384]:
    over = sum(x > limit for x in lengths)
    print(f">{limit}: {over} examples ({over / len(lengths) * 100:.2f}%)")
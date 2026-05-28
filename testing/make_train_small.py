# generates a dataset using https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
# the dataset is quite good but very verbose, I didn't have enough time to play around with this dataset unfortunately.
def main():
    from datasets import load_dataset
    import json
    import re
    from pathlib import Path

    dataset = load_dataset(
        "parquet",
        data_files="./results/math/*.parquet",
    )

    sampled = dataset["train"].shuffle(seed=42).select(range(1000))

    with open("./results/train.jsonl", "w", encoding="utf-8") as f:
        for example in sampled:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    input_path = Path("./results/train.jsonl")
    output_path = Path("./results/sft_train.jsonl")

    def clean_answer(answer: str) -> str:
        answer = answer.strip()

        # Remove existing boxed wrapper if present
        m = re.search(r"\\boxed\{(.+)\}", answer)
        if m:
            answer = m.group(1).strip()

        # Optional cleanup for weird LaTeX units
        answer = answer.replace(r"\mathrm{~}/\mathrm{}", "")
        answer = answer.replace(r"\mathrm{~}", " ")

        return rf"\boxed{{{answer}}}"

    def extract_reasoning(generation: str) -> str:
        generation = generation.strip()

        # If generation already has think tags, use inner content
        m = re.search(r"<think>(.*?)</think>", generation, flags=re.S)
        if m:
            return m.group(1).strip()

        # Remove final boxed answer from generation if present
        generation = re.sub(r"\\boxed\{.*?\}", "", generation, flags=re.S).strip()

        return generation

    written = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, \
        output_path.open("w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            problem = row.get("problem", "").strip()
            answer = clean_answer(row.get("answer", ""))

            generations = row.get("generations", [])
            correctness = row.get("correctness_math_verify", [])

            if not problem or not answer or not generations:
                skipped += 1
                continue

            for gen, is_correct in zip(generations, correctness):
                if not is_correct:
                    continue

                reasoning = extract_reasoning(gen)

                if not reasoning:
                    skipped += 1
                    continue

                assistant_content = (
                    f"<think>\n{reasoning}\n</think>\n\n"
                    f"{answer}"
                )

                example = {
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }

                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} SFT examples to {output_path}")
    print(f"Skipped {skipped} rows/generations")

if __name__ == "__main__":
    main()
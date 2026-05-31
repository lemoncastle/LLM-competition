import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve carefully. "
    "End with exactly one final line in this format: Final: \\boxed{...}. "
    "For multiple answers, put them inside one \\boxed{} separated by commas. "
    "Do not output anything after the final line."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quick SFT dataset from public.jsonl for LoRA fine-tuning."
    )
    parser.add_argument("--input", default="./data/public.jsonl")
    parser.add_argument("--output", default="./results/public_sft.jsonl")
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include explicit system prompt in each training example.",
    )
    return parser.parse_args()


def strip_boxed(s: str) -> str:
    s = s.strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", s)
    if m:
        return m.group(1).strip()
    return s


def format_gold_answer(item: Dict) -> str:
    ans = item["answer"]
    if isinstance(ans, list):
        parts = [strip_boxed(str(x)) for x in ans]
        joined = ", ".join(parts)
        return f"Final: \\boxed{{{joined}}}"
    return f"Final: \\boxed{{{strip_boxed(str(ans))}}}"


def build_user_text(item: Dict) -> str:
    question = item["question"]
    options = item.get("options")
    if not options:
        return question
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    return f"{question}\n\nOptions:\n{opts_text}"


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in rows:
            user_text = build_user_text(item)
            assistant_text = format_gold_answer(item)

            messages = []
            if args.include_system:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

            ex = {"messages": messages}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} SFT rows to {out_path}")


if __name__ == "__main__":
    main()

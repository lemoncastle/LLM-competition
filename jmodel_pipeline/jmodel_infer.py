import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from jmodel_normalizer import normalize_assistant_output

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

SYSTEM_PROMPT_FRQ = (
    "You are an expert mathematician. Solve carefully. "
    "Keep reasoning concise but correct. "
    "For multiple [ANS] placeholders, provide answers in that exact order. "
    "End with exactly one final line in this format: Final: \\boxed{...}. "
    "If multiple answers are needed, put them in one \\boxed{} separated by commas. "
    "Do not output anything after the final line."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve carefully and select one option. "
    "End with exactly one final line in this format: Final: \\boxed{<LETTER>}. "
    "Use only a single capital letter inside \\boxed{}. "
    "Do not output anything after the final line."
)

FINALIZE_SYSTEM_PROMPT = (
    "You are a strict answer formatter. "
    "Given a question and a draft solution, output exactly one line only. "
    "Required format: Final: \\boxed{...}. "
    "No extra words or lines."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="jmodel inference/eval script for CSE151B Kaggle competition."
    )
    parser.add_argument(
        "--mode",
        choices=["public_eval", "private_submit"],
        required=True,
        help="public_eval scores on data with answers; private_submit writes Kaggle CSV.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="JSONL input path. Defaults: ./data/public.jsonl or ./data/private.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path. Defaults: ./results/jmodel_public_eval.jsonl or ./results/jmodel_submission.csv",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping IDs already present in output file (private_submit only).",
    )
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.15,
        help="Small positive value reduces repetitive loops.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9, help="vLLM setting"
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--enable-prefix-caching", action="store_true", help="Enable vLLM prefix caching."
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA adapter inference in vLLM.",
    )
    parser.add_argument(
        "--lora-path",
        default="./qwen_math_sft/public_lora_v1/final",
        help="Path to LoRA adapter directory (if --use-lora).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Rank passed to LoRARequest (if --use-lora).",
    )
    parser.add_argument(
        "--finalize-missing-box",
        action="store_true",
        help="Run a second short generation pass when response has no \\boxed{}.",
    )
    parser.add_argument(
        "--finalize-max-tokens",
        type=int,
        default=192,
        help="Max tokens in finalization pass.",
    )
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="Normalize final response formatting (recommended for private submission).",
    )
    parser.add_argument(
        "--no-normalize-output",
        action="store_true",
        help="Disable output normalization.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_mcq(item: Dict) -> bool:
    return bool(item.get("options"))


def build_prompt_fields(question: str, options: Optional[Sequence[str]]) -> Tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_FRQ, question


def build_primary_prompt(tokenizer, item: Dict) -> str:
    system_prompt, user_prompt = build_prompt_fields(item["question"], item.get("options"))
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_finalize_prompt(tokenizer, item: Dict, draft_response: str) -> str:
    _, user_prompt = build_prompt_fields(item["question"], item.get("options"))
    final_user = (
        f"Question:\n{user_prompt}\n\n"
        f"Draft solution:\n{draft_response}\n\n"
        "Return only one line in this exact format: Final: \\boxed{...}"
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": final_user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def has_boxed_answer(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))


def extract_letter(text: str) -> str:
    boxed_matches = re.findall(r"\\boxed\{([A-Za-z])\}", text)
    if boxed_matches:
        return boxed_matches[-1].upper()
    answer_match = re.search(r"answer\s+is\s+([A-Za-z])", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    line_match = re.search(r"^\s*([A-Z])\s*$", text, re.MULTILINE)
    if line_match:
        return line_match.group(1).upper()
    return ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == str(gold_letter).strip().upper()


def acc(records: Sequence[Dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["correct"]) / len(records) * 100.0


def read_done_ids_csv(path: Path) -> set:
    if not path.exists():
        return set()
    done_ids = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done_ids.add(int(row["id"]))
    return done_ids


def write_private_rows(path: Path, rows: Sequence[Dict], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or not append
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "response": row["response"]})


def write_public_eval(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.mode == "private_submit":
        args.normalize_output = True
    if args.no_normalize_output:
        args.normalize_output = False

    if args.data_path is None:
        args.data_path = (
            "./data/public.jsonl" if args.mode == "public_eval" else "./data/private.jsonl"
        )
    if args.output_path is None:
        args.output_path = (
            "./results/jmodel_public_eval.jsonl"
            if args.mode == "public_eval"
            else "./results/jmodel_submission.csv"
        )

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = load_jsonl(data_path)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    total_mcq = sum(1 for d in data if is_mcq(d))
    total_free = len(data) - total_mcq
    print(f"Loaded {len(data)} rows from {data_path} ({total_mcq} MCQ, {total_free} free-form)")

    done_ids = set()
    if args.mode == "private_submit" and args.resume:
        done_ids = read_done_ids_csv(output_path)
        print(f"Resume mode: found {len(done_ids)} existing IDs in {output_path}")
        data = [item for item in data if int(item["id"]) not in done_ids]
        print(f"Remaining rows after resume filter: {len(data)}")
        if not data:
            print("Nothing to do.")
            return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=args.enable_prefix_caching,
        enable_lora=args.use_lora,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    lora_request = None
    if args.use_lora:
        lora_request = LoRARequest("jmodel_lora", args.lora_rank, args.lora_path)
        print(f"Using LoRA adapter: {args.lora_path} (rank={args.lora_rank})")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        presence_penalty=args.presence_penalty,
    )

    finalize_params = SamplingParams(
        max_tokens=args.finalize_max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
    )

    print("Model loaded.")

    public_results: List[Dict] = []
    private_rows: List[Dict] = []

    judger = None
    if args.mode == "public_eval":
        sys.path.insert(0, ".")
        from judger import Judger

        judger = Judger(strict_extract=False)

    if args.mode == "private_submit":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume and output_path.exists():
            output_path.unlink()
        if not output_path.exists():
            write_private_rows(output_path, [], append=False)

    for start in range(0, len(data), args.batch_size):
        batch = data[start : start + args.batch_size]
        prompts = [build_primary_prompt(tokenizer, item) for item in batch]
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
        responses = [out.outputs[0].text.strip() for out in outputs]

        missing_idx = [i for i, r in enumerate(responses) if not has_boxed_answer(r)]
        if args.finalize_missing_box and missing_idx:
            finalize_prompts = [
                build_finalize_prompt(tokenizer, batch[i], responses[i]) for i in missing_idx
            ]
            finalize_outputs = llm.generate(
                finalize_prompts,
                sampling_params=finalize_params,
                lora_request=lora_request,
            )
            finalized = [out.outputs[0].text.strip() for out in finalize_outputs]
            for idx, final_text in zip(missing_idx, finalized):
                responses[idx] = f"{responses[idx]}\n\n{final_text}"

        if args.mode == "private_submit":
            if args.normalize_output:
                responses = [normalize_assistant_output(r) for r in responses]
            private_rows.extend(
                [{"id": item["id"], "response": resp} for item, resp in zip(batch, responses)]
            )
            write_private_rows(output_path, private_rows, append=True)
            print(
                f"[{start + len(batch):4d}/{len(data)}] wrote {len(private_rows)} rows "
                f"(batch missing-box fixed: {len(missing_idx)})"
            )
            private_rows.clear()
            continue

        for item, response in zip(batch, responses):
            assert judger is not None
            this_is_mcq = is_mcq(item)
            gold = item["answer"]
            if this_is_mcq:
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

            public_results.append(
                {
                    "id": item["id"],
                    "is_mcq": this_is_mcq,
                    "gold": gold,
                    "response": response,
                    "correct": correct,
                }
            )

        print(
            f"[{start + len(batch):4d}/{len(data)}] scored {len(batch)} rows "
            f"(batch missing-box fixed: {len(missing_idx)})"
        )

    if args.mode == "private_submit":
        total_written = len(data)
        print(f"Done. Wrote {total_written} submission rows to {output_path}")
        return

    write_public_eval(output_path, public_results)
    mcq_rows = [r for r in public_results if r["is_mcq"]]
    free_rows = [r for r in public_results if not r["is_mcq"]]
    boxed_rows = sum(1 for r in public_results if has_boxed_answer(r["response"]))
    print("=" * 56)
    print("JMODEL PUBLIC EVALUATION")
    print("=" * 56)
    print(f"MCQ       : {sum(r['correct'] for r in mcq_rows):4d} / {len(mcq_rows):4d} ({acc(mcq_rows):.2f}%)")
    print(f"Free-form : {sum(r['correct'] for r in free_rows):4d} / {len(free_rows):4d} ({acc(free_rows):.2f}%)")
    print(
        f"Overall   : {sum(r['correct'] for r in public_results):4d} / {len(public_results):4d} "
        f"({acc(public_results):.2f}%)"
    )
    print(f"Boxed rate: {boxed_rows:4d} / {len(public_results):4d} ({boxed_rows / len(public_results) * 100:.2f}%)")
    print(f"Saved eval rows to: {output_path}")


if __name__ == "__main__":
    main()

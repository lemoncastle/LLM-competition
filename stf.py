import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
def main():
    import json
    import re
    import sys
    from pathlib import Path
    from typing import Optional

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from tqdm import tqdm

    MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
    OUTPUT_PATH = "./results/basesft.jsonl"

    # load dataset
    public_data = [json.loads(line) for line in open("./data/public.jsonl")]

    n_mcq  = sum(bool(d.get("options")) for d in public_data)
    n_free = sum(not d.get("options")   for d in public_data)
    print(f"Loaded {len(public_data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # prompts for free response and MCQ problems
    SYSTEM_PROMPT_FRQ = (
        "You are an expert mathematician. "
        "Solve the problem carefully. "
        "Use exact values unless a decimal is required. "
        "Round final answers to 8 decimal places if needed. "
        "Verify your result before answering. "
        "After solving, output a final answer section only. "
        "The final answer must be exactly one line in this form: Final: \\boxed{...}. "
        "If there are multiple answers, put them all inside the same \\boxed{} separated by commas. "
    )

    SYSTEM_PROMPT_MCQ = (
        "You are an expert mathematician. "
        "Solve the problem carefully and choose the single best answer. "
        "Verify your result against the choices. "
        "After solving, output exactly one final line and nothing else after it. "
        "Final line format: Final: \\boxed{<letter>}. "
    )

    def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
        """ determine if free response or MCQ problem and return (system_prompt, user_prompt)"""
        if options:
            labels    = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_FRQ, question

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    dataset = load_dataset("json", data_files="./results/train1.jsonl", split="train")
    dataset = dataset.select(range(min(800, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Important for QLoRA / k-bit training
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir="./qwen_math_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,

        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_length=2048,
        packing=False,
        assistant_only_loss=True,

        logging_steps=10,
        save_steps=150,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("./qwen_math_sft/test")
    tokenizer.save_pretrained("./qwen_math_sft/test")

    # load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.88,
        max_model_len=16384, # could increase a little, but watch out for OOM
        trust_remote_code=True,
        max_num_seqs=4, # could increase a little, but watch out for OOM
        max_num_batched_tokens=16384, # was 32768
    )

    sampling_params = SamplingParams(
        max_tokens=12288, # was 32768 (could increase a little)
        temperature=0.6, # Qwen recommends this for thinking
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0,  # parameter between 0 and 2 to reduce endless repetition
    )

    print("Model loaded.")

    # Build prompts for last 10 entries
    test_data = public_data[-10:]
    prompts = []
    for item in test_data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # Generate
    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=LoRARequest("math_sft", 1, "./qwen_math_sft/test"))

    responses = [out.outputs[0].text.strip() for out in outputs]

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


    # Load Judger for free-form scoring
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    results = []
    for item, response in tqdm(zip(test_data, responses), total=len(test_data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold   = item["answer"]

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

    SAVE_EVAL = True   # Set to False when running on the private test set

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            if SAVE_EVAL:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                        "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} records to {out_path}")

if __name__ == "__main__":
    main()
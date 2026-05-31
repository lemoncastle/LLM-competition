import argparse

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick (<2h target) LoRA SFT training on public-derived chat dataset."
    )
    parser.add_argument("--model-name", default="unsloth/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data-path", default="./results/quick_public_sft.jsonl")
    parser.add_argument("--output-dir", default="./qwen_math_sft/quick_public_lora")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=320,
        help="Hard cap for run-time control; lower this if training exceeds your time budget.",
    )
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    splits = dataset.train_test_split(test_size=0.05, seed=args.seed, shuffle=True)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.print_trainable_parameters()

    def formatting_func(examples):
        # Single row case
        if isinstance(examples["messages"], list) and isinstance(examples["messages"][0], dict):
            return [
                tokenizer.apply_chat_template(
                    examples["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            ]

        # Batch case
        return [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            for msgs in examples["messages"]
        ]

    train_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        max_length=args.max_seq_length,
        packing=False,
        assistant_only_loss=True,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_steps=80,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()

    final_dir = f"{args.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved quick LoRA adapter to: {final_dir}")


if __name__ == "__main__":
    main()

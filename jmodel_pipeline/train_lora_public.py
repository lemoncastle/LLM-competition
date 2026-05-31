import argparse

import torch
from datasets import load_dataset
import unsloth  # noqa: F401  # must be imported before transformers/peft patching
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA training using Transformers Trainer (stable path)."
    )
    parser.add_argument("--model-name", default="unsloth/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data-path", default="./results/public_sft.jsonl")
    parser.add_argument("--output-dir", default="./qwen_math_sft/public_lora_v1")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--save-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_steps % args.eval_steps != 0:
        raise ValueError(
            f"Invalid config: save_steps ({args.save_steps}) must be a multiple of eval_steps ({args.eval_steps})."
        )

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

    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|im_end|>"

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

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = text + tokenizer.eos_token
        return {"text": text}

    train_dataset = train_dataset.map(to_text, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(to_text, remove_columns=eval_dataset.column_names)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    final_dir = f"{args.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved LoRA adapter to: {final_dir}")


if __name__ == "__main__":
    main()

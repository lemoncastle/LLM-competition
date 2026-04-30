def main():
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    dataset = load_dataset("json", data_files="./results/train1.jsonl", split="train")
    splits = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)

    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    MAX_SEQ_LENGTH = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Thinking-2507",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules="all-linear",
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir="./qwen_math_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,

        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        bf16=True,

        max_length=MAX_SEQ_LENGTH,
        packing=False,
        assistant_only_loss=True,

        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_steps=150,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained("./qwen_math_sft/test")
    tokenizer.save_pretrained("./qwen_math_sft/test")

if __name__ == "__main__":
    main()  
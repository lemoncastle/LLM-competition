import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen3-4B-Thinking-2507"
DATA_PATH = "./results/sft_train.jsonl"
OUTPUT_DIR = "./qwen_math_sft"
MAX_SEQ_LENGTH = 8192

def load_data():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    splits = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    return splits["train"], splits["test"]

def main():
    train_dataset, eval_dataset = load_data()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
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

    model.save_pretrained(OUTPUT_DIR + "/test")
    tokenizer.save_pretrained(OUTPUT_DIR + "/test")

if __name__ == "__main__":
    main()
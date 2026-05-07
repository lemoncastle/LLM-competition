# file to train the model using the generated SFT data
# other stats found here like data used, hardware, training time etc..

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Qwen3-4B-Thinking-2507"
DATA_PATH = "./results/sft_train.jsonl"
OUTPUT_DIR = "./qwen_math_sft"
MAX_SEQ_LENGTH = 16384 # gen responses have 8-16k tokens (but OOM rip) # 8192 or 16384

def load_data():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    splits = dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
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
        r=16, # try 8 0r 32
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            ], # try with just ["q_proj","k_proj","v_proj","o_proj"]
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
        gradient_accumulation_steps=8, # try 16
        num_train_epochs=1.5,

        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        bf16=True,

        max_length=MAX_SEQ_LENGTH,
        packing=False,
        assistant_only_loss=True,

        eval_strategy="steps",
        eval_steps=200,
        logging_steps=30,
        save_steps=200,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
    )
    # I have no idea how this works so the llm did the work 
    def formatting_func(examples):
        # Case 1: Unsloth passes a single row
        if isinstance(examples["messages"], list) and isinstance(examples["messages"][0], dict):
            return [
                tokenizer.apply_chat_template(
                    examples["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            ]

        # Case 2: Unsloth passes a batch
        return [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in examples["messages"]
        ]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR + "/test")
    tokenizer.save_pretrained(OUTPUT_DIR + "/test")

if __name__ == "__main__":
    main()
# Model Run Plan

This path trains a small LoRA adapter quickly and runs it with the improved `jmodel` inference pipeline.

## 1) Build SFT data from public set

```bash
python jmodel_pipeline/build_public_sft_dataset.py --include-system
```

Output:
- `./results/public_sft.jsonl`

## 2) Train a time-capped LoRA adapter

```bash
python jmodel_pipeline/train_lora_public.py \
  --data-path ./results/public_sft.jsonl \
  --output-dir ./qwen_math_sft/public_lora_v1 \
  --max-seq-length 4096 \
  --lora-r 8 \
  --lora-alpha 16 \
  --epochs 2 \
  --max-steps 320 \
  --batch-size 1 \
  --grad-accum 16 \
  --learning-rate 8e-5
```

If this exceeds your time budget, rerun with:
- `--max-steps 220`

## 3) Evaluate fast on public subset

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode public_eval \
  --limit 150 \
  --batch-size 16 \
  --enable-prefix-caching \
  --finalize-missing-box \
  --use-lora \
  --lora-path ./qwen_math_sft/public_lora_v1/final \
  --output-path ./results/jmodel_lora_eval.jsonl
```

## 4) Generate full private submission with LoRA

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode private_submit \
  --batch-size 24 \
  --enable-prefix-caching \
  --finalize-missing-box \
  --use-lora \
  --lora-path ./qwen_math_sft/public_lora_v1/final \
  --output-path ./results/jmodel_lora_submission.csv
```

If interrupted:

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode private_submit \
  --batch-size 24 \
  --enable-prefix-caching \
  --finalize-missing-box \
  --use-lora \
  --lora-path ./qwen_math_sft/public_lora_v1/final \
  --resume \
  --output-path ./results/jmodel_lora_submission.csv
```

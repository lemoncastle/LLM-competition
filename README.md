# CSE151B Kaggle Math Competition (JModel Pipeline)

This repository contains our competition workflow for improving mathematical reasoning with:

- Base model: `Qwen/Qwen3-4B-Thinking-2507`
- Training method: LoRA supervised fine-tuning on public-set formatted data
- Inference method: strict boxed-answer prompting + fallback finalization + output normalization

## Repository Layout

| Path | Purpose |
|---|---|
| `data/` | Competition files (`public.jsonl`, `private.jsonl`) |
| `jmodel_pipeline/` | Main training/inference pipeline used for our method |
| `judger.py`, `utils.py` | Public-set local evaluation utilities |
| `real_run/`, `testing/` | Earlier/alternate scripts and experiments |

## JModel Method

The method is implemented in `jmodel_pipeline/`:

- `build_public_sft_dataset.py`
  - Builds chat-format SFT data from `data/public.jsonl`
  - Produces `./results/public_sft.jsonl`

- `train_lora_public.py`
  - Trains a LoRA adapter on public-derived SFT data
  - Saves adapter to `./qwen_math_sft/public_lora_v1/final`

- `jmodel_infer.py`
  - Runs inference for either:
    - `public_eval` (local scoring)
    - `private_submit` (Kaggle CSV generation)
  - Includes:
    - strict final answer formatting (`Final: \boxed{...}`)
    - optional fallback finalization when boxed answer is missing
    - resume support for interrupted private runs

- `jmodel_normalizer.py`
  - Normalizes final answers for robust extraction and submission formatting.

## End-to-End Commands

Run from repository root.

### 1) Build SFT dataset

```bash
python jmodel_pipeline/build_public_sft_dataset.py --include-system
```

### 2) Train LoRA adapter

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
  --learning-rate 8e-5 \
  --eval-steps 40 \
  --save-steps 80
```

### 3) Generate final Kaggle file (`submission.csv`)

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode private_submit \
  --batch-size 24 \
  --enable-prefix-caching \
  --finalize-missing-box \
  --use-lora \
  --lora-path ./qwen_math_sft/public_lora_v1/final \
  --output-path ./results/submission.csv
```

If interrupted, resume:

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode private_submit \
  --batch-size 24 \
  --enable-prefix-caching \
  --finalize-missing-box \
  --use-lora \
  --lora-path ./qwen_math_sft/public_lora_v1/final \
  --resume \
  --output-path ./results/submission.csv
```

## Public Evaluation (Optional)

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


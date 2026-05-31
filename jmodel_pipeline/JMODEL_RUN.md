# JModel Run Guide

This branch adds a separate script `jmodel_pipeline/jmodel_infer.py` so your partner's scripts stay untouched.

## 1) Start a GPU session (DSMLP)

Use a GPU that supports bitsandbytes quantization (for example A30 or RTX 2080Ti).

```bash
launch.sh -c 8 -m 32 -g 1 -v a30
```

## 2) Open repo

```bash
cd /path/to/cse151b
git checkout jmodel
```

## 3) Install Python packages

Install the same stack used by your current scripts (`transformers`, `vllm`, `torch`, etc.).
If you already have a working env from your partner's setup, reuse it.

## 4) Public evaluation (for local validation)

Runs on `data/public.jsonl`, scores with `judger.py`, and writes JSONL results.

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode public_eval \
  --enable-prefix-caching \
  --finalize-missing-box \
  --batch-size 24 \
  --output-path ./results/jmodel_public_eval.jsonl
```

Optional quick smoke test:

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode public_eval \
  --limit 80 \
  --enable-prefix-caching \
  --finalize-missing-box \
  --batch-size 16
```

## 5) Private submission generation

Writes Kaggle file `id,response` as CSV.

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode private_submit \
  --enable-prefix-caching \
  --finalize-missing-box \
  --batch-size 24 \
  --output-path ./results/jmodel_submission.csv
```

If interrupted, resume:

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode private_submit \
  --enable-prefix-caching \
  --finalize-missing-box \
  --batch-size 24 \
  --resume \
  --output-path ./results/jmodel_submission.csv
```

## 6) Using your LoRA adapter (optional)

If you want inference with an adapter:

```bash
python jmodel_pipeline/jmodel_infer.py \
  --mode private_submit \
  --use-lora \
  --lora-path ./qwen_math_sft/test \
  --enable-prefix-caching \
  --finalize-missing-box \
  --output-path ./results/jmodel_submission_lora.csv
```

## Notes

- `--finalize-missing-box` runs a short second pass only when no `\boxed{}` appears in the first output.
- This is designed to reduce extraction failures that hurt leaderboard score.
- Default `max_tokens` is lower than your old scripts to reduce never-ending generations.

## Pushing the Boundary of LLM Mathematical Reasoning
CSE 151B Spring 2026 Competition

This repository covers code to build a small llm (Qwen3-4B-Thinking-2507) to do good on a math set.

| File | Description |
|---|---|
| `data/` | Public and Private datasets for testing and inference |
| `results/` | Output JSONL files written at runtime |
| `testing/` | Various scripts used for testing |
| `real_run/` | training and inference scripts used for final submission |
| `starter_code.ipynb` | Walks through environment setup and first generation |
| `judger.py` | Response scoring logic on public set |
| `utils.py` | Utilities used by `judger.py` |

### Replication
---
Clone the repository ```git clone https://github.com/lemoncastle/cse151b.git```

Set up a virtual environment (can take 40+ minutes)
```rm -rf .venv
python -m venv .venv
source ./.venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install --no-cache-dir -r requirements.txt
python -m ipykernel install --user --name cse151b --display-name "Python (151B)"
```

Run ```full.py```

This runs a single function ```run_inference()``` that loads the model, runs inference, applies post processing and outputs the final submission as ```submission.csv```

Which gets saved in ```./results/``` Make sure your ```private.jsonl``` is in ```./data/```

### Inference
---
Final submission inference was done on DSMLP using RTX pro 6000 MIG to 24gb with 8 cpu and 32gb ram
- ```K8S_TIMEOUT_SECONDS=43200 launch-sp26-cuda128.sh -b -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32```

Inference time took 4 days restarting every 12 hours.

```full.py``` has a slightly modified version with max_num_seqs of 8 (instead of 2) which requires a 48gb GPU (A6000 GPU or equivalent) lowering inference time to ~9 hours 

Training time was done on runpod using A6000 GPU using template ```meloncastle/runpod-template-151:v2``` Taking 2 hours.

### Costs
---
Total $60 (split)

### Scores
---
1. 0.558 -> 0.516 (local inference)
2. 0.558 -> 0.510 (dsmlp)
3. 0.607 -> 0.560 (dsmlp)
4. 0.628 -> 0.565 (resubmit with normalization)
5. 0.646 -> 0.577 (resubmit with normalization and updated judger)
6. 0.636 -> 0.569 (heavy distilled training set on runpod)
7. **POST SCORES** .703 -> ?? (didn't upload)

Current leaderboard rank 40/78 -> Final rank 61/110 -> 3 months update """rank""" 40/110 

Teammate's scores: ([branch](https://github.com/lemoncastle/LLM-competition/tree/jmodel))
1. 0.321 -> 0.289
2. 0.208 -> 0.175

### 3 Months Update
---
So I had $9 left in my runpod account and I have still have access to schools servers and their GPUs so I decided to do another run. It was also haunting me how bad the scores I got were.
- I used the base model, as I realized that doing lora and supervised fine tuning was making the model forget and it was just a mess to deal with (where I originally got stuck).
- Use multiple prompts, and be more aggressive, force 1e8 precision for questions.
    - If response doesn't have a /boxed{} regen again.
- Post processing remained the same

After just changing some prompts around I got a + .056 bump lol. that would've got me +30 ranks on the leaderboard lol.

Some thing I could improve is doing multiple generations for each question. like I split mcq/frq like I did but mcq has 12 different prompts or something.
- Or like use a teacher LLM to assign a category to each question like (statistics), or (algebra) so I can write more focused prompts.
    - Cause I found long prompts waste tokens and the model just thinks about the prompts forever.

Well yeah that's it for this, that generation took probably 20 hours which split 12 hours on DSMLP and the rest on runpod with the money I had left

### Notes
---
All I could figure out was doing supervised fine tuning.
- I was having a lot of trouble getting good outputs so I spent lots of time looking at outputs and cleaning dataset, and generated response which was honestly a waste of time as a 'team' of 1 as I quickly ran out of time once I understood what was going on.
I have a teammate jgu0453 but their only contribution was panic spamming submissions on last day. His response was "Ive been a bit busy, and the internet is my dorm is absolute garbage."

Nice try though lots to learn we'll get them next time. 

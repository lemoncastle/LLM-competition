## Pushing the Boundary of LLM Mathematical Reasoning
CSE 151B Spring 2026 Competition

This repository covers code to build a small llm (Qwen3-4B-Thinking-2507) to do good on a math set.

| File | Description |
|---|---|
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

### Inference
Final submission inference was done on DSMLP using RTX pro 6000 MIG to 24gb with 8 cpu and 32gb ram
- ```K8S_TIMEOUT_SECONDS=43200 launch-sp26-cuda128.sh -b -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32```

Inference time took 4 days restarting every 12 hours.

Training time was done on runpod using A6000 GPU using template ```meloncastle/runpod-template-151:v2``` Taking 2 hours.

### Costs
- OpenAI - $5 (developing training set)
- Runpod - $30 (training and inference)
- Deepseek - $15 (developing training set)
- Electricity - $5.15 (local AI inference)
Total $52 (went over budget :( )

### Scores
1. 0.558 (local inference)
2. 0.558 (dsmlp)
3. 0.607 (dsmlp)
4. 0.628 (resubmit with normalization)
5. 0.646 (resubmit with normalization and updated judger)
6. 0.636 (heavy distilled training set on runpod)

Current leaderboard rank 40/70 :(

### Notes
All I could figure out was doing supervised fine tuning.
- I was having a lot of trouble getting good outputs so I spent lots of time looking at outputs and cleaning dataset, and generated response which was honestly a waste of time as a 'team' of 1 as I quickly ran out of time once I understood what was going on. I also went over my self imposed $50 budget quickly as compute is quite expensive unfort :(

Best ways to improve the model and what I ***should*** have done
- Work strictly using the base model **FIRST**. The base model is already quite strong and do something like self consistency where you have multiple prompts and you generate a question with each prompt and compare. The current pipeline I have is splitting by MCQ and FRQ but we could've split even more like statistics, arithmetic, or even math level like hard, easy or whatever.
    - I noticed statistics questions like ones that ask about statistical significance answers are all high precision 1e-10+ and our judger scores on precision 1e-8 so you need a lot of precision but generally asking the model to output 1e-8 precision wastes a lot of tokens so having seperate prompts would have been really smart. 
- I needed a lot more time doing fine tuning, I was only able to run training 5 times.
- I was never able to get reinforcement learning to work. I don't really know how to code.
- Something that hurt was the reliance on AI to help me do research. alot of these concepts aren't terribly hard but there's lots of research papers that go into these and having to sift through them actually hurt my brain because I never understood them. 

I have a teammate jgu0453 but they didn't help unfortunately.

Nice try though lots to learn we'll get them next time. 
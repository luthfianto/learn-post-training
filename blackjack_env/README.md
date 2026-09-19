---
title: Blackjack Environment Server
emoji: 🎲
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Blackjack Environment

A simplified hit/stand Blackjack environment for [OpenEnv](https://github.com/huggingface/OpenEnv),
built for finetuning LLMs with GRPO.

## Rules

- 6-deck shoe, reshuffled at ~75% penetration.
- Dealer stands on all 17.
- Player may only `hit` or `stand`.
- Payouts (unit bet, emitted on the terminal step only):
  - player blackjack `+1.5`
  - win `+1`
  - push `0`
  - loss / bust `-1`

## Layout

```
blackjack_env/
├── models.py                        # Action / Observation / State
├── client.py                        # BlackjackEnv WebSocket client
├── strategy.py                      # hand scoring + basic-strategy policy
├── harness.py                       # session factory + GRPO rollout_func
├── eval_baseline.py                 # scripted-policy EV check
├── collect_sft.py                   # scripted-teacher SFT dataset
├── train_grpo.py                    # TRL GRPOTrainer finetuning
├── server/
│   ├── blackjack_env_environment.py # game logic
│   └── app.py                       # FastAPI + WebSocket server
└── tests/
```

## Install

```bash
pip install -e .
```

## Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# or
python -m server.app --port 8000
```

Then use the client:

```python
from blackjack_env import BlackjackAction, BlackjackEnv

with BlackjackEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=42)
    print(result.observation.message)
    while not result.done:
        result = env.step(BlackjackAction(action="hit"))
    print(result.observation.message, result.reward)
```

## Validate the reward signal

The scripted basic-strategy policy is near-optimal and should land near
break-even, clearly beating "always hit":

```bash
python -m blackjack_env.eval_baseline --episodes 50000
```

## GRPO finetuning

`train_grpo.py` plugs the environment into TRL's `GRPOTrainer` through a custom
`rollout_func`. Each dataset row is a deal seed; GRPO samples `num_generations`
completions per seed so the group advantage compares different play against the
same deal. Environment-observation tokens are masked out of the loss via
`env_mask`, so the policy is only trained on its own generated tokens.

Smoke test (CPU, tiny random model, verifies the pipeline):

```bash
python -m blackjack_env.train_grpo --smoke
```

Real run:

```bash
python -m blackjack_env.train_grpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --num-prompts 128 --num-generations 8 --max-steps 200
```

## Optional: scripted-teacher SFT data

Collect scripted rollouts in TRL `SFTTrainer`-compatible JSONL, e.g. for a
warm start before GRPO:

```bash
python -m blackjack_env.collect_sft --episodes 500 --output-dir data/sft --only-wins
```

## Tests

```bash
pytest blackjack_env/tests -q
```

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
├── arena.py                         # watch/compare models vs baselines
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

## Watch and evaluate the model (arena)

`arena.py` plays a model and any baselines over the **same seeded deals**, so
results are directly comparable. It prints per-hand transcripts (the model's raw
output, the parsed action, and whether it was valid) and a final scoreboard with
expected value, win/loss/push/blackjack rates and invalid-action counts.

```bash
# Watch a GRPO-finetuned checkpoint for 20 hands, show 5 transcripts
python -m blackjack_env.arena --model outputs/blackjack-grpo --episodes 20 --show-hands 5

# Also print the exact prompt sent to the model each turn
python -m blackjack_env.arena --model outputs/blackjack-grpo --show-hands 3 --show-prompt

# Write a full per-episode log (observation, prompt, response, reward) to JSONL
python -m blackjack_env.arena --model outputs/blackjack-grpo --episodes 200 \
  --log-file logs/arena.jsonl

# Compare against basic strategy and random play over 2000 shared deals
python -m blackjack_env.arena --model outputs/blackjack-grpo \
  --contestants model,basic,random --episodes 2000

# Baselines only (no model download)
python -m blackjack_env.arena --contestants basic,random --episodes 20000
```

Each JSONL log line is one episode:

```json
{
  "seed": 0, "contestant": "model", "reward": -1.0, "invalid": 0, "turns": 1,
  "transcript": [
    {
      "turn": 1,
      "observation": "Dealer shows: 7\nYour hand: [10, 3] ...",
      "prompt": "<s>[INST] <<SYS>> ... [/INST]",
      "response": "{\"action\": \"stand\"}",
      "action": "stand", "valid": true, "reward": -1.0, "done": true,
      "observation_after": "Dealer hand: [7, 9, 2] ..."
    }
  ]
}
```

Use `--temperature 0` (the default) for greedy play, or raise it to see the
model's sampling distribution. `--model` accepts a Hub id or the directory saved
by `train_grpo.py`.

To see prompts/completions during **training**, pass `--log-completions` to
`train_grpo.py` (TRL prints sampled completions and their rewards).

Example scoreboard:

```
contestant    hands       EV     win    loss    push  blackjack  invalid
basic         20000  -0.0215   43.3%   48.0%    8.7%       5.0%        0
random        20000  -0.3036   31.5%   64.3%    4.2%       5.0%        0
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

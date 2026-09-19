# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the TRL rollout function's token accounting.

A fake tokenizer/model/trainer lets us verify the shape and alignment of
``prompt_ids`` / ``completion_ids`` / ``logprobs`` / ``env_mask`` without
downloading a model.
"""

import random
from types import SimpleNamespace

import blackjack_env.harness as harness
from blackjack_env.harness import BlackjackSessionFactory, build_blackjack_rollout_func

HIT = '{"action": "hit"}'
STAND = '{"action": "stand"}'


class FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7, 8, 9]}

    def decode(self, ids, skip_special_tokens=True):
        mapping = {100: HIT, 101: STAND}
        return " ".join(mapping.get(i, "?") for i in ids)


class FakeModel:
    training = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def _fake_generate_turn(
    model, tokenizer, input_ids, *, max_new_tokens, temperature, top_p
):
    action_id = 100 if random.random() < 0.5 else 101
    return [action_id], [-0.5]


def test_rollout_token_accounting(monkeypatch):
    monkeypatch.setattr(harness, "_generate_turn", _fake_generate_turn)
    random.seed(0)

    rollout = build_blackjack_rollout_func(
        BlackjackSessionFactory(num_decks=1),
        max_turns=6,
        max_new_tokens=8,
    )
    trainer = SimpleNamespace(
        model=FakeModel(),
        processing_class=FakeTokenizer(),
        args=SimpleNamespace(temperature=1.0, top_p=1.0, max_completion_length=64),
        accelerator=None,
    )
    prompts = [[{"role": "user", "content": f"Seed: {i}"}] for i in range(12)]

    batch = rollout(prompts, trainer)

    for key in ("prompt_ids", "completion_ids", "logprobs", "env_mask", "env_reward"):
        assert len(batch[key]) == len(prompts), key

    for i in range(len(prompts)):
        completion = batch["completion_ids"][i]
        assert batch["prompt_ids"][i], "prompt ids must be non-empty"
        assert len(completion) == len(batch["logprobs"][i]) == len(batch["env_mask"][i])
        assert sum(batch["env_mask"][i]) > 0, "must contain policy tokens"
        assert all(mask in (0, 1) for mask in batch["env_mask"][i])

    assert len(set(batch["env_reward"])) >= 2, "actions should produce varied rewards"

# SPDX-License-Identifier: BSD-3-Clause

"""GRPO finetuning of an LLM on the Blackjack environment.

Uses OpenEnv's in-process Blackjack session as a custom TRL ``rollout_func``.
Each dataset row is a starting seed; GRPO repeats each seed ``num_generations``
times so the group advantage compares different play against the same deal.

Smoke test (CPU, tiny model):
    python -m blackjack_env.train_grpo --smoke

Real run:
    python -m blackjack_env.train_grpo --model Qwen/Qwen2.5-0.5B-Instruct \
        --num-prompts 128 --num-generations 8 --max-steps 200
"""

from __future__ import annotations

import argparse

from datasets import Dataset

try:  # pragma: no cover
    from .harness import (
        BlackjackSessionFactory,
        build_blackjack_rollout_func,
        env_reward_func,
    )
except ImportError:  # pragma: no cover
    from harness import (
        BlackjackSessionFactory,
        build_blackjack_rollout_func,
        env_reward_func,
    )


def build_prompt_dataset(num_prompts: int, seed_start: int = 0) -> Dataset:
    prompts = [
        [
            {
                "role": "user",
                "content": (f"Play a hand of blackjack. Seed: {seed_start + index}"),
            }
        ]
        for index in range(num_prompts)
    ]
    return Dataset.from_dict({"prompt": prompts})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO finetuning on Blackjack")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=str, default="outputs/blackjack-grpo")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-completion-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-decks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny CPU-friendly run with a random small model.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        args.num_prompts = 2
        args.num_generations = 2
        args.per_device_batch_size = 2
        args.max_steps = 2
        args.max_completion_length = 24
        args.max_new_tokens = 12
        args.max_turns = 3
    return args


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
    )

    train_dataset = build_prompt_dataset(args.num_prompts, seed_start=args.seed)

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.per_device_batch_size,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=use_cuda,
        seed=args.seed,
    )

    rollout_func = build_blackjack_rollout_func(
        BlackjackSessionFactory(num_decks=args.num_decks),
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[env_reward_func],
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        rollout_func=rollout_func,
    )

    print(f"Model: {args.model}")
    print(f"Prompts: {args.num_prompts} | generations/prompt: {args.num_generations}")
    print(f"Device: {'cuda' if use_cuda else 'cpu'} | max_steps: {args.max_steps}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()

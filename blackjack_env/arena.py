# SPDX-License-Identifier: BSD-3-Clause

"""Blackjack arena: watch a model play and compare it to baselines.

Runs one or more contestants over the *same* seeded deals so results are
directly comparable, prints per-hand transcripts for the model, and a final
scoreboard (expected value, win/loss/push/blackjack rates, invalid actions).

Every episode can be logged to JSONL, including the exact prompt sent to the
model and its raw response at each turn.

Examples:
    # Watch a GRPO-finetuned model for 20 hands, show the first 5 transcripts
    python -m blackjack_env.arena --model outputs/blackjack-grpo --episodes 20 --show-hands 5

    # Include the full prompt sent to the model in the printed transcripts
    python -m blackjack_env.arena --model outputs/blackjack-grpo --show-hands 3 --show-prompt

    # Write a full per-episode log (prompt + response + rewards) to JSONL
    python -m blackjack_env.arena --model outputs/blackjack-grpo --episodes 200 \
        --log-file logs/arena.jsonl

    # Compare a model against basic strategy and random play
    python -m blackjack_env.arena --model outputs/blackjack-grpo --episodes 2000 \
        --contestants model,basic,random
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Optional

try:  # pragma: no cover
    from .harness import (
        BlackjackSession,
        BlackjackSessionFactory,
        extract_state_from_messages,
        format_prompt,
        generate_action,
    )
    from .strategy import basic_strategy_action
except ImportError:  # pragma: no cover
    from harness import (
        BlackjackSession,
        BlackjackSessionFactory,
        extract_state_from_messages,
        format_prompt,
        generate_action,
    )
    from strategy import basic_strategy_action


_VALID_ACTION_RE = re.compile(r'"action"\s*:\s*"?\s*(hit|stand)\b', re.IGNORECASE)


def _decide_basic(state: dict[str, Any]) -> str:
    return basic_strategy_action(state["player_hand"], int(state["dealer_upcard"]))


def _decide_random(state: dict[str, Any], rng: random.Random) -> str:
    return rng.choice(["hit", "stand"])


def _next_state(tool_result) -> dict[str, Any]:
    return {
        "player_hand": tool_result.data["player_hand"],
        "dealer_upcard": tool_result.data["dealer_upcard"],
        "player_total": tool_result.data["player_total"],
    }


# --------------------------------------------------------------------------- #
# Episode runners
# --------------------------------------------------------------------------- #
def play_baseline_episode(
    session: BlackjackSession,
    decide: Callable[[dict[str, Any]], str],
    *,
    max_turns: int,
    contestant: str = "baseline",
    seed: Optional[int] = None,
) -> dict[str, Any]:
    initial_messages = session.initial_messages()
    current_observation = initial_messages[-1]["content"]
    state = extract_state_from_messages(initial_messages)

    transcript: list[dict[str, Any]] = []
    reward = 0.0

    for turn in range(1, max_turns + 1):
        action = decide(state)
        tool_result = session.call_tool("play", {"action": action})
        reward = float(tool_result.metadata.get("reward") or 0.0)
        transcript.append(
            {
                "turn": turn,
                "observation": current_observation,
                "prompt": None,
                "response": action,
                "action": action,
                "valid": True,
                "reward": tool_result.metadata.get("reward"),
                "done": tool_result.done,
                "observation_after": tool_result.data.get("message", ""),
            }
        )
        if tool_result.done:
            break
        current_observation = tool_result.data.get("message", "")
        state = _next_state(tool_result)

    return {
        "seed": seed,
        "contestant": contestant,
        "reward": reward,
        "invalid": 0,
        "turns": len(transcript),
        "transcript": transcript,
    }


def play_model_episode(
    model: Any,
    tokenizer: Any,
    session: BlackjackSession,
    *,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    contestant: str = "model",
    seed: Optional[int] = None,
) -> dict[str, Any]:
    initial_messages = session.initial_messages()
    prompt_text = format_prompt(tokenizer, initial_messages)
    input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    context_text = prompt_text
    current_observation = initial_messages[-1]["content"]

    transcript: list[dict[str, Any]] = []
    reward = 0.0
    invalid = 0

    for turn in range(1, max_turns + 1):
        prompt_for_turn = context_text
        action, raw_text, gen_ids = generate_action(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        valid = bool(_VALID_ACTION_RE.search(raw_text))
        invalid += 0 if valid else 1

        tool_result = session.call_tool("play", {"action": action})
        reward = float(tool_result.metadata.get("reward") or 0.0)
        transcript.append(
            {
                "turn": turn,
                "observation": current_observation,
                "prompt": prompt_for_turn,
                "response": raw_text.strip(),
                "action": action,
                "valid": valid,
                "reward": tool_result.metadata.get("reward"),
                "done": tool_result.done,
                "observation_after": tool_result.data.get("message", ""),
            }
        )
        if tool_result.done:
            break

        observation_text = "\n" + str(tool_result.data.get("message", "")) + "\n"
        obs_ids = tokenizer(observation_text, add_special_tokens=False)["input_ids"]
        input_ids = input_ids + gen_ids + obs_ids
        context_text = context_text + raw_text + observation_text
        current_observation = tool_result.data.get("message", "")

    return {
        "seed": seed,
        "contestant": contestant,
        "reward": reward,
        "invalid": invalid,
        "turns": len(transcript),
        "transcript": transcript,
    }


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #
def _load_model(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def _print_episode(result: dict[str, Any], *, show_prompt: bool) -> None:
    print(f"\n{'=' * 72}")
    print(f"Episode seed={result['seed']} [{result['contestant']}]  "
          f"turns={result['turns']}  reward={result['reward']:+.2f}")
    print("=" * 72)
    for entry in result["transcript"]:
        print(f"\n[turn {entry['turn']}] OBSERVATION (shown to model):")
        print(entry["observation"])
        if show_prompt and entry["prompt"]:
            print("\nPROMPT (sent to model):")
            print(entry["prompt"])
        marker = "" if entry["valid"] else "  (invalid -> defaulted)"
        print(f"\nRESPONSE: {entry['response']!r} -> action={entry['action']}{marker}")
        print(f"  reward={entry['reward']} done={entry['done']}")
    print(f"\nfinal reward: {result['reward']:+.2f}")


def _summarize(name: str, results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results)
    rewards = [r["reward"] for r in results]
    return {
        "contestant": name,
        "hands": float(n),
        "ev": sum(rewards) / n if n else 0.0,
        "win": sum(1 for r in rewards if r > 0) / n if n else 0.0,
        "loss": sum(1 for r in rewards if r < 0) / n if n else 0.0,
        "push": sum(1 for r in rewards if r == 0) / n if n else 0.0,
        "blackjack": sum(1 for r in rewards if r == 1.5) / n if n else 0.0,
        "invalid": sum(r["invalid"] for r in results),
    }


def _print_scoreboard(summaries: list[dict[str, float]]) -> None:
    print("\n" + "=" * 78)
    print(
        f"{'contestant':<12}{'hands':>7}{'EV':>9}{'win':>8}{'loss':>8}"
        f"{'push':>8}{'blackjack':>11}{'invalid':>9}"
    )
    print("-" * 78)
    for s in summaries:
        print(
            f"{s['contestant']:<12}{int(s['hands']):>7}{s['ev']:>+9.4f}"
            f"{s['win']:>8.1%}{s['loss']:>8.1%}{s['push']:>8.1%}"
            f"{s['blackjack']:>11.1%}{int(s['invalid']):>9}"
        )
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blackjack model arena")
    parser.add_argument("--model", type=str, default=None, help="HF model id or saved output dir.")
    parser.add_argument("--contestants", type=str, default="model,basic,random")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--num-decks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--show-hands",
        type=int,
        default=3,
        help="Print this many full episode logs.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Include the exact prompt sent to the model in printed episodes.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write every episode (prompt, response, reward) as JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contestants = [c.strip() for c in args.contestants.split(",") if c.strip()]

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = tokenizer = None
    if "model" in contestants:
        if not args.model:
            raise SystemExit("--model is required when 'model' is a contestant.")
        print(f"Loading model: {args.model} (device={device})")
        model, tokenizer = _load_model(args.model, device)

    factory = BlackjackSessionFactory(num_decks=args.num_decks)
    seeds = [args.seed + i for i in range(args.episodes)]
    summaries: list[dict[str, float]] = []
    shown = 0

    log_handle = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("w", encoding="utf-8")

    try:
        for name in contestants:
            results: list[dict[str, Any]] = []
            rng = random.Random(args.seed)
            for seed in seeds:
                session = factory.create(seed=seed)
                try:
                    if name == "model":
                        result = play_model_episode(
                            model,
                            tokenizer,
                            session,
                            max_turns=args.max_turns,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            contestant=name,
                            seed=seed,
                        )
                    elif name == "basic":
                        result = play_baseline_episode(
                            session,
                            _decide_basic,
                            max_turns=args.max_turns,
                            contestant=name,
                            seed=seed,
                        )
                    elif name == "random":
                        result = play_baseline_episode(
                            session,
                            lambda s, r=rng: _decide_random(s, r),
                            max_turns=args.max_turns,
                            contestant=name,
                            seed=seed,
                        )
                    else:
                        raise SystemExit(f"Unknown contestant: {name!r}")
                finally:
                    session.close()

                results.append(result)
                if log_handle is not None:
                    log_handle.write(json.dumps(result, default=str) + "\n")
                if shown < args.show_hands:
                    _print_episode(result, show_prompt=args.show_prompt)
                    shown += 1

            summaries.append(_summarize(name, results))
    finally:
        if log_handle is not None:
            log_handle.close()

    _print_scoreboard(summaries)
    if args.log_file:
        print(f"\nFull episode log written to: {args.log_file}")


if __name__ == "__main__":
    main()

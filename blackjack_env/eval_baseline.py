# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the Blackjack environment with scripted policies.

The basic-strategy policy is near-optimal and should land close to break-even
(expected value around -0.5%), which validates that the reward signal is
correct. "Always hit" is clearly worse and is printed for contrast.

Usage:
    python -m blackjack_env.eval_baseline --episodes 50000
"""

from __future__ import annotations

import argparse
from typing import Callable

try:  # pragma: no cover
    from .models import BlackjackAction, BlackjackObservation
    from .server.blackjack_env_environment import BlackjackEnvironment
    from .strategy import basic_strategy_action
except ImportError:  # pragma: no cover
    from models import BlackjackAction, BlackjackObservation
    from server.blackjack_env_environment import BlackjackEnvironment
    from strategy import basic_strategy_action


Policy = Callable[[BlackjackObservation], str]


def basic_strategy_policy(observation: BlackjackObservation) -> str:
    return basic_strategy_action(observation.player_hand, observation.dealer_upcard)


def always_hit_policy(observation: BlackjackObservation) -> str:
    return "hit"


def always_stand_policy(observation: BlackjackObservation) -> str:
    return "stand"


def run_episode(env: BlackjackEnvironment, policy: Policy, seed: int) -> float:
    observation = env.reset(seed=seed)
    guard = 0
    while not observation.done and guard < 30:
        observation = env.step(BlackjackAction(action=policy(observation)))
        guard += 1
    return float(observation.reward or 0.0)


def evaluate(
    policy: Policy,
    episodes: int,
    num_decks: int,
    base_seed: int,
) -> dict[str, float]:
    env = BlackjackEnvironment(num_decks=num_decks)
    rewards = [run_episode(env, policy, base_seed + i) for i in range(episodes)]

    wins = sum(1 for r in rewards if r > 0)
    losses = sum(1 for r in rewards if r < 0)
    pushes = sum(1 for r in rewards if r == 0)
    blackjacks = sum(1 for r in rewards if r == 1.5)

    return {
        "episodes": float(episodes),
        "expected_value": sum(rewards) / episodes,
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "push_rate": pushes / episodes,
        "blackjack_rate": blackjacks / episodes,
    }


def _print_report(name: str, report: dict[str, float]) -> None:
    print(f"\n{name}")
    print(f"  episodes      : {int(report['episodes'])}")
    print(f"  expected value: {report['expected_value']:+.4f} per hand")
    print(f"  win rate      : {report['win_rate']:.2%}")
    print(f"  loss rate     : {report['loss_rate']:.2%}")
    print(f"  push rate     : {report['push_rate']:.2%}")
    print(f"  blackjack rate: {report['blackjack_rate']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blackjack scripted-policy evaluation")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--num-decks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    basic = evaluate(basic_strategy_policy, args.episodes, args.num_decks, args.seed)
    hit = evaluate(always_hit_policy, args.episodes, args.num_decks, args.seed)
    stand = evaluate(always_stand_policy, args.episodes, args.num_decks, args.seed)

    _print_report("Basic strategy (near-optimal)", basic)
    _print_report("Always hit", hit)
    _print_report("Always stand", stand)

    if not (hit["expected_value"] < basic["expected_value"] < 0.01):
        raise SystemExit(
            "Basic strategy did not outperform always-hit as expected; "
            "the reward signal may be broken."
        )
    print("\nOK: basic strategy beats always-hit and stays near break-even.")


if __name__ == "__main__":
    main()

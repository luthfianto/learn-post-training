# SPDX-License-Identifier: BSD-3-Clause

"""Collect scripted-teacher rollouts as an SFT dataset.

Writes ``results.jsonl`` (TRL ``SFTTrainer``-compatible ``messages`` column)
and ``metadata.json`` via OpenEnv's ``RolloutSerializer``. Useful for
warm-starting a policy before GRPO, or as a reference dataset.

Usage:
    python -m blackjack_env.collect_sft --episodes 500 --output-dir data/sft
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openenv.core.harness import HarnessRunLimits, MCPHarnessAdapter
from openenv.core.harness.collect import CollectRunner, RolloutSerializer

try:  # pragma: no cover
    from .harness import BlackjackSessionFactory, build_scripted_model_step
except ImportError:  # pragma: no cover
    from harness import BlackjackSessionFactory, build_scripted_model_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect scripted blackjack rollouts")
    parser.add_argument("--episodes", "-n", type=int, default=500)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("data/sft"))
    parser.add_argument("--num-decks", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--only-wins",
        action="store_true",
        help="Keep only rollouts with reward >= 0 (drops net losses).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes already present in results.jsonl.",
    )
    args = parser.parse_args()

    serializer = RolloutSerializer(args.output_dir)
    serializer.write_metadata(
        {
            "env": "blackjack_env",
            "teacher": "scripted-basic-strategy",
            "num_decks": args.num_decks,
            "num_episodes_requested": args.episodes,
            "only_wins": args.only_wins,
        }
    )

    runner = CollectRunner(
        session_factory=BlackjackSessionFactory(num_decks=args.num_decks),
        harness_adapter=MCPHarnessAdapter(),
        serializer=serializer,
        limits=HarnessRunLimits(max_turns=args.max_turns),
    )

    should_keep = (lambda record: record.reward >= 0.0) if args.only_wins else None

    result = runner.run(
        model_step=build_scripted_model_step(),
        num_episodes=args.episodes,
        episode_id_prefix="bj",
        resume=args.resume,
        should_keep=should_keep,
    )

    print(
        f"collected={result.num_collected} skipped={result.num_skipped} "
        f"dropped={result.num_dropped} failed={result.num_failed}"
    )
    print(f"avg_reward={result.avg_reward:.3f} success_rate={result.success_rate:.0%}")
    print(f"dataset: {serializer.results_path}")


if __name__ == "__main__":
    main()

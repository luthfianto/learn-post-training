# SPDX-License-Identifier: BSD-3-Clause

from blackjack_env.arena import play_baseline_episode
from blackjack_env.harness import BlackjackSessionFactory
from blackjack_env.strategy import basic_strategy_action


def _basic(state):
    return basic_strategy_action(state["player_hand"], int(state["dealer_upcard"]))


def test_baseline_episode_terminates():
    session = BlackjackSessionFactory(num_decks=1).create(seed=5)
    try:
        result = play_baseline_episode(session, _basic, max_turns=8)
    finally:
        session.close()

    assert result["transcript"]
    assert result["transcript"][-1]["done"]
    assert result["reward"] in (-1.0, 0.0, 1.0, 1.5)
    assert result["invalid"] == 0

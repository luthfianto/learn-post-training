# SPDX-License-Identifier: BSD-3-Clause

from openenv.core.harness import HarnessRunLimits, MCPHarnessAdapter

from blackjack_env.harness import (
    BlackjackSessionFactory,
    build_scripted_model_step,
    env_reward_func,
    parse_action_from_text,
)
from blackjack_env.models import BlackjackAction
from blackjack_env.server.blackjack_env_environment import BlackjackEnvironment


def _env_with_draws(cards):
    env = BlackjackEnvironment()
    remaining = list(cards)
    env._draw = lambda: remaining.pop(0)
    return env


def test_reset_is_seed_reproducible():
    first = BlackjackEnvironment().reset(seed=123)
    second = BlackjackEnvironment().reset(seed=123)
    assert first.player_hand == second.player_hand
    assert first.dealer_upcard == second.dealer_upcard


def test_ace_reduction():
    env = _env_with_draws([11, 11, 10, 9])  # player A,A; dealer 10,9
    observation = env.reset()
    assert observation.player_total == 12
    assert observation.player_is_soft


def test_natural_blackjack_payout():
    env = _env_with_draws([11, 10, 5, 6])  # player A,10; dealer 5,6
    observation = env.reset()
    assert observation.done
    assert observation.terminal_reason == "player_blackjack"
    assert observation.reward == 1.5


def test_player_bust_reward():
    env = _env_with_draws([10, 5, 10, 6, 10])
    observation = env.reset()
    assert not observation.done
    observation = env.step(BlackjackAction(action="hit"))
    assert observation.done
    assert observation.terminal_reason == "player_bust"
    assert observation.reward == -1.0


def test_push():
    env = _env_with_draws([10, 10, 10, 10])
    observation = env.reset()
    observation = env.step(BlackjackAction(action="stand"))
    assert observation.done
    assert observation.terminal_reason == "push"
    assert observation.reward == 0.0


def test_dealer_bust_player_wins():
    env = _env_with_draws([10, 10, 7, 6, 10])
    observation = env.reset()
    observation = env.step(BlackjackAction(action="stand"))
    assert observation.done
    assert observation.terminal_reason == "dealer_bust"
    assert observation.reward == 1.0


def test_state_tracks_progress():
    env = BlackjackEnvironment()
    env.reset(seed=7)
    state = env.state
    assert state.player_total >= 2
    assert state.shoe_remaining >= 0
    assert state.step_count == 0


def test_parse_action_from_text():
    assert parse_action_from_text('{"action": "hit"}') == "hit"
    assert parse_action_from_text("I will stand now") == "stand"
    assert parse_action_from_text("garbage") == "stand"


def test_session_call_tool_and_verify():
    session = BlackjackSessionFactory(num_decks=1).create(seed=42)
    try:
        messages = session.initial_messages()
        assert messages[0]["role"] == "system"
        result = session.call_tool("play", {"action": "stand"})
        assert result.done
        verify = session.verify(transcript=[])
        assert verify.env_reward == result.metadata["reward"]
    finally:
        session.close()


def test_scripted_rollout_completes():
    factory = BlackjackSessionFactory(num_decks=1)
    session = factory.create(seed=3)
    try:
        rollout = MCPHarnessAdapter().run_white_box(
            model_step=build_scripted_model_step(),
            session=session,
            limits=HarnessRunLimits(max_turns=8),
        )
        assert rollout.done
        assert rollout.tool_trace
    finally:
        session.close()


def test_env_reward_func():
    assert env_reward_func(["a", "b"], env_reward=[1.0, -1.0]) == [1.0, -1.0]
    assert env_reward_func(["a"], env_reward=None) == [0.0]

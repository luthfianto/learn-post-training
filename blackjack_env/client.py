# SPDX-License-Identifier: BSD-3-Clause

"""Blackjack Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import BlackjackAction, BlackjackObservation, BlackjackState


class BlackjackEnv(EnvClient[BlackjackAction, BlackjackObservation, BlackjackState]):
    """WebSocket client for the Blackjack environment.

    Example:
        >>> with BlackjackEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...     result = client.step(BlackjackAction(action="hit"))
    """

    def _step_payload(self, action: BlackjackAction) -> Dict[str, Any]:
        return {"action": action.action}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[BlackjackObservation]:
        obs_data = payload.get("observation", {})
        observation = BlackjackObservation(
            player_hand=obs_data.get("player_hand", []),
            dealer_upcard=obs_data.get("dealer_upcard", 0),
            dealer_hand=obs_data.get("dealer_hand", []),
            player_total=obs_data.get("player_total", 0),
            player_is_soft=obs_data.get("player_is_soft", False),
            dealer_total=obs_data.get("dealer_total"),
            legal_actions=obs_data.get("legal_actions", ["hit", "stand"]),
            message=obs_data.get("message", ""),
            terminal_reason=obs_data.get("terminal_reason"),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> BlackjackState:
        return BlackjackState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            player_hand=payload.get("player_hand", []),
            dealer_hand=payload.get("dealer_hand", []),
            player_total=payload.get("player_total", 0),
            dealer_total=payload.get("dealer_total"),
            shoe_remaining=payload.get("shoe_remaining", 0),
            num_decks=payload.get("num_decks", 6),
            terminal=payload.get("terminal", False),
            terminal_reason=payload.get("terminal_reason"),
        )

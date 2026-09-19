# SPDX-License-Identifier: BSD-3-Clause

"""Blackjack environment implementation (simplified hit/stand).

A 6-deck shoe with dealer-stands-on-17 rules. The player may only ``hit`` or
``stand``. Reward is emitted only on the terminal step using a unit bet:

    +1.5  player natural blackjack
    +1    player wins
     0    push (tie) or both blackjack
    -1    player loses or busts
"""

from __future__ import annotations

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:  # pragma: no cover - import path depends on how the server is launched
    from ..models import BlackjackAction, BlackjackObservation, BlackjackState
    from ..strategy import (
        format_hand,
        hand_value,
        is_blackjack,
        new_shoe,
    )
except ImportError:  # pragma: no cover
    from models import BlackjackAction, BlackjackObservation, BlackjackState
    from strategy import (
        format_hand,
        hand_value,
        is_blackjack,
        new_shoe,
    )


class BlackjackEnvironment(Environment):
    """Simplified hit/stand blackjack environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, num_decks: int = 6, seed: Optional[int] = None):
        super().__init__()
        self.num_decks = num_decks
        self._rng = random.Random(seed)
        self._shoe: list[int] = []
        self._player: list[int] = []
        self._dealer: list[int] = []
        self._state = BlackjackState(episode_id=str(uuid4()), step_count=0)
        self._terminal = False
        self._terminal_reason: Optional[str] = None
        self._reward: float = 0.0

    # ------------------------------------------------------------------ #
    # Shoe management
    # ------------------------------------------------------------------ #
    def _reshuffle(self) -> None:
        self._shoe = new_shoe(self.num_decks, self._rng)

    def _draw(self) -> int:
        if not self._shoe:
            self._reshuffle()
        return self._shoe.pop()

    def _maybe_reshuffle(self) -> None:
        # Reshuffle at ~75% penetration.
        if len(self._shoe) < 0.25 * 52 * self.num_decks:
            self._reshuffle()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BlackjackObservation:
        if seed is not None:
            self._rng = random.Random(seed)
            self._shoe = []
        self._maybe_reshuffle()

        self._player = [self._draw(), self._draw()]
        self._dealer = [self._draw(), self._draw()]
        self._terminal = False
        self._terminal_reason = None
        self._reward = 0.0
        self._state = BlackjackState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            num_decks=self.num_decks,
        )

        player_bj = is_blackjack(self._player)
        dealer_bj = is_blackjack(self._dealer)
        if player_bj or dealer_bj:
            if player_bj and dealer_bj:
                self._finish("both_blackjack", 0.0)
            elif player_bj:
                self._finish("player_blackjack", 1.5)
            else:
                self._finish("dealer_blackjack", -1.0)

        self._sync_state()
        return self._observation()

    def step(
        self,
        action: BlackjackAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BlackjackObservation:
        if self._terminal:
            return self._observation()

        self._state.step_count += 1

        if action.action == "hit":
            self._player.append(self._draw())
            total, _ = hand_value(self._player)
            if total > 21:
                self._finish("player_bust", -1.0)
            elif total == 21:
                self._dealer_play()
        else:  # stand
            self._dealer_play()

        self._sync_state()
        return self._observation()

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _dealer_play(self) -> None:
        while hand_value(self._dealer)[0] < 17:  # stands on soft 17
            self._dealer.append(self._draw())

        player_total, _ = hand_value(self._player)
        dealer_total, _ = hand_value(self._dealer)

        if dealer_total > 21:
            self._finish("dealer_bust", 1.0)
        elif dealer_total > player_total:
            self._finish("dealer_wins", -1.0)
        elif dealer_total < player_total:
            self._finish("player_wins", 1.0)
        else:
            self._finish("push", 0.0)

    def _finish(self, reason: str, reward: float) -> None:
        self._terminal = True
        self._terminal_reason = reason
        self._reward = reward

    def _sync_state(self) -> None:
        self._state.player_hand = list(self._player)
        self._state.dealer_hand = list(self._dealer)
        self._state.player_total = hand_value(self._player)[0]
        self._state.dealer_total = (
            hand_value(self._dealer)[0] if self._terminal else None
        )
        self._state.shoe_remaining = len(self._shoe)
        self._state.terminal = self._terminal
        self._state.terminal_reason = self._terminal_reason

    def _observation(self) -> BlackjackObservation:
        player_total, player_soft = hand_value(self._player)
        revealed = self._terminal
        dealer_hand = list(self._dealer) if revealed else [self._dealer[0]]
        dealer_total = hand_value(self._dealer)[0] if revealed else None

        return BlackjackObservation(
            player_hand=list(self._player),
            dealer_upcard=self._dealer[0],
            dealer_hand=dealer_hand,
            player_total=player_total,
            player_is_soft=player_soft,
            dealer_total=dealer_total,
            legal_actions=[] if self._terminal else ["hit", "stand"],
            message=self._render_message(),
            terminal_reason=self._terminal_reason,
            done=self._terminal,
            reward=self._reward if self._terminal else 0.0,
        )

    def _render_message(self) -> str:
        player_total, player_soft = hand_value(self._player)
        lines = [
            f"Dealer shows: {self._dealer[0]}",
            f"Your hand: {format_hand(self._player)} "
            f"(total {player_total}{', soft' if player_soft else ''})",
        ]
        if self._terminal:
            dealer_total, _ = hand_value(self._dealer)
            lines.append(
                f"Dealer hand: {format_hand(self._dealer)} (total {dealer_total})"
            )
            lines.append(f"Result: {self._terminal_reason} (reward {self._reward:+g})")
        else:
            lines.append("Legal actions: hit, stand")
        return "\n".join(lines)


__all__ = ["BlackjackEnvironment", "BlackjackAction", "BlackjackObservation"]

# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the Blackjack environment."""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class BlackjackAction(Action):
    """Player action in a simplified hit/stand blackjack game."""

    action: Literal["hit", "stand"] = Field(..., description="Either 'hit' or 'stand'.")


class BlackjackObservation(Observation):
    """Observation returned after reset/step."""

    player_hand: List[int] = Field(
        default_factory=list, description="Player's cards (Ace = 11)."
    )
    dealer_upcard: int = Field(default=0, description="Dealer's face-up card.")
    dealer_hand: List[int] = Field(
        default_factory=list,
        description="Dealer's full hand; only revealed once the episode ends.",
    )
    player_total: int = Field(default=0, description="Best player total.")
    player_is_soft: bool = Field(
        default=False, description="Whether the player total counts an Ace as 11."
    )
    dealer_total: Optional[int] = Field(
        default=None, description="Dealer total, once revealed."
    )
    legal_actions: List[str] = Field(
        default_factory=lambda: ["hit", "stand"],
        description="Actions available to the player.",
    )
    message: str = Field(default="", description="Human/LLM-readable game state.")
    terminal_reason: Optional[str] = Field(
        default=None, description="Why the episode ended, if it did."
    )


class BlackjackState(State):
    """Internal environment state."""

    player_hand: List[int] = Field(default_factory=list)
    dealer_hand: List[int] = Field(default_factory=list)
    player_total: int = Field(default=0)
    dealer_total: Optional[int] = Field(default=None)
    shoe_remaining: int = Field(default=0)
    num_decks: int = Field(default=6)
    terminal: bool = Field(default=False)
    terminal_reason: Optional[str] = Field(default=None)
    metadata: Dict = Field(default_factory=dict)

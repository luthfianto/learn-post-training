# SPDX-License-Identifier: BSD-3-Clause

"""Shared Blackjack card / hand utilities and a scripted basic-strategy policy.

The environment, the evaluation baseline, and the scripted teacher all import
from this module so that hand scoring and strategy stay consistent.
"""

from __future__ import annotations

import random
from typing import Iterable, Literal, Sequence

Action = Literal["hit", "stand"]

# Cards are represented by their value only (suit is irrelevant for hit/stand
# blackjack): 2-10 keep their pip value, J/Q/K are 10, and an Ace is 11. Aces
# are reduced to 1 by ``hand_value`` when needed.
ACE = 11
FACE = 10


def new_shoe(num_decks: int = 6, rng: random.Random | None = None) -> list[int]:
    """Build a shuffled shoe of ``num_decks`` decks of 52 cards."""
    if num_decks < 1:
        raise ValueError("num_decks must be >= 1")
    rng = rng or random.Random()
    shoe: list[int] = []
    for _ in range(num_decks):
        for value in range(2, 10):
            shoe.extend([value] * 4)
        shoe.extend([FACE] * 16)  # 10, J, Q, K
        shoe.extend([ACE] * 4)
    rng.shuffle(shoe)
    return shoe


def hand_value(cards: Sequence[int]) -> tuple[int, bool]:
    """Return ``(total, is_soft)`` for a hand.

    ``is_soft`` is True when at least one Ace is still counted as 11, i.e. the
    hand can absorb a hit without busting.
    """
    total = sum(cards)
    aces = sum(1 for card in cards if card == ACE)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


def is_blackjack(cards: Sequence[int]) -> bool:
    """A natural blackjack is exactly two cards totalling 21."""
    return len(cards) == 2 and sum(cards) == 21


def basic_strategy_action(cards: Sequence[int], dealer_upcard: int) -> Action:
    """Hit/stand basic strategy for a 6-deck, dealer-stands-on-17 game.

    Doubling and splitting are unavailable, so double-down cells fall back to
    the better of hit/stand. This is a near-optimal policy suitable as a
    baseline and as a scripted teacher.
    """
    total, soft = hand_value(cards)
    if total >= 21:
        return "stand"

    if soft:
        # Soft totals: A,2..A,9 (13..20). A,7 (18) is the only soft hand that
        # hits, and only against 9/10/A.
        if total >= 19:
            return "stand"
        if total == 18:
            return "hit" if dealer_upcard in (9, 10, ACE) else "stand"
        return "hit"

    # Hard totals.
    if total >= 17:
        return "stand"
    if total <= 11:
        return "hit"
    if total == 12:
        return "stand" if 4 <= dealer_upcard <= 6 else "hit"
    # 13..16
    return "stand" if 2 <= dealer_upcard <= 6 else "hit"


def format_hand(cards: Iterable[int]) -> str:
    """Render a hand as ``[10, 6]`` for human/LLM-facing messages."""
    return "[" + ", ".join(str(card) for card in cards) + "]"

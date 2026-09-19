# SPDX-License-Identifier: BSD-3-Clause

from blackjack_env.strategy import (
    basic_strategy_action,
    hand_value,
    is_blackjack,
    new_shoe,
)


def test_hand_value_hard():
    assert hand_value([10, 6]) == (16, False)
    assert hand_value([10, 10, 5]) == (25, False)


def test_hand_value_soft_ace():
    assert hand_value([11, 6]) == (17, True)
    assert hand_value([11, 10]) == (21, True)


def test_hand_value_multiple_aces():
    assert hand_value([11, 11]) == (12, True)
    assert hand_value([11, 11, 9]) == (21, True)
    assert hand_value([11, 10, 10]) == (21, False)  # ace must count as 1


def test_is_blackjack():
    assert is_blackjack([11, 10])
    assert not is_blackjack([10, 10])
    assert not is_blackjack([7, 7, 7])


def test_shoe_composition():
    shoe = new_shoe(num_decks=1)
    assert len(shoe) == 52
    assert shoe.count(11) == 4
    assert shoe.count(10) == 16


def test_basic_strategy_hard():
    assert basic_strategy_action([10, 6], 10) == "hit"
    assert basic_strategy_action([10, 6], 6) == "stand"
    assert basic_strategy_action([10, 2], 4) == "stand"
    assert basic_strategy_action([10, 2], 2) == "hit"
    assert basic_strategy_action([10, 10], 11) == "stand"
    assert basic_strategy_action([5, 5], 10) == "hit"


def test_basic_strategy_soft():
    assert basic_strategy_action([11, 7], 9) == "hit"  # soft 18 vs 9
    assert basic_strategy_action([11, 7], 7) == "stand"  # soft 18 vs 7
    assert basic_strategy_action([11, 6], 5) == "hit"  # soft 17
    assert basic_strategy_action([11, 9], 11) == "stand"  # soft 20

import pytest
from game_rng import GameRNG
from utils.helpers import roll_dice


def test_roll_dice_requires_rng():
    with pytest.raises(ValueError):
        roll_dice("1d6", None)


def test_roll_dice_with_rng():
    rng = GameRNG(seed=1)
    assert roll_dice("2d4+1", rng) == 6

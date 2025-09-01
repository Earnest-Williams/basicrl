import sys
import types
import pytest

# Stub game_rng for deterministic dice rolls
module = types.ModuleType("game_rng")

class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed
    def get_int(self, a, b):
        return a

module.GameRNG = DummyRNG
sys.modules["game_rng"] = module

from utils.helpers import roll_dice


def test_roll_dice_requires_rng():
    with pytest.raises(ValueError):
        roll_dice("1d6", None)


def test_roll_dice_with_rng():
    rng = DummyRNG()
    # DummyRNG.get_int always returns the lower bound 'a'
    assert roll_dice("2d4+1", rng) == 3

import os
import sys
import types
import importlib
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import utils.helpers as helpers


@pytest.fixture
def roll_dice_with_dummy_rng(monkeypatch):
    module = types.ModuleType("game_rng")

    class DummyRNG:
        def __init__(self, seed=None):
            self.initial_seed = seed

        def get_int(self, a, b):
            return a

    module.GameRNG = DummyRNG
    monkeypatch.setitem(sys.modules, "game_rng", module)
    importlib.reload(helpers)
    yield helpers.roll_dice, DummyRNG
    importlib.reload(helpers)


def test_roll_dice_requires_rng(roll_dice_with_dummy_rng):
    roll_dice, _ = roll_dice_with_dummy_rng
    with pytest.raises(ValueError):
        roll_dice("1d6", None)


def test_roll_dice_with_rng(roll_dice_with_dummy_rng):
    roll_dice, DummyRNG = roll_dice_with_dummy_rng
    rng = DummyRNG()
    # DummyRNG.get_int always returns the lower bound 'a'
    assert roll_dice("2d4+1", rng) == 3

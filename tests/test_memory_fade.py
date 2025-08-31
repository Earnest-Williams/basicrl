import numpy as np
import pytest
import pathlib
import sys
import os
import types

# Import lights_dev modules directly via path since the directory isn't a package
lights_dev_path = pathlib.Path(__file__).resolve().parent.parent / "lights_dev"
sys.path.append(str(lights_dev_path))
import main_game  # type: ignore  # pylint: disable=import-error
import constants  # type: ignore  # pylint: disable=import-error


def test_memory_fade_bounds():
    height, width = 1, 1
    current_time = np.float32(100.0)

    last_seen = np.full((height, width), current_time, dtype=np.float32)
    memory_intensity = np.ones((height, width), dtype=np.float32)
    visible = np.zeros((height, width), dtype=bool)

    # Immediately after being seen, intensity should remain at 1.0
    main_game._update_memory_fade_internal(
        current_time, last_seen, memory_intensity, visible, height, width
    )
    assert memory_intensity[0, 0] == pytest.approx(1.0)

    # After MEMORY_DURATION seconds, intensity should decay to ~0
    last_seen[0, 0] = current_time - constants.MEMORY_DURATION
    memory_intensity[0, 0] = 1.0
    main_game._update_memory_fade_internal(
        current_time, last_seen, memory_intensity, visible, height, width
    )
    assert memory_intensity[0, 0] == pytest.approx(0.0, abs=1e-6)


# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Minimal stubs for external modules
module = types.ModuleType("game_rng")


class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed

    def randint(self, a, b):
        return a


module.GameRNG = DummyRNG
sys.modules["game_rng"] = module

ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

from game.world.game_map import GameMap
from game.game_state import GameState

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def create_game_state():
    game_map = GameMap(width=10, height=10)
    game_map.create_test_room()
    gs = GameState(
        existing_map=game_map,
        player_start_pos=(5, 5),
        player_glyph=ord("@"),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        entity_templates={},
        effect_definitions={},
        rng_seed=42,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    return gs


def test_memory_fade_decay_and_mask():
    gs = create_game_state()
    px, py = gs.player_position

    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()

    assert gs.game_map.memory_intensity[py, px] < 1.0
    assert gs.game_map.memory_fade_mask[py, px]


def test_memory_fade_skips_zero_intensity_tiles():
    gs = create_game_state()
    px, py = gs.player_position

    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()

    gm = gs.game_map
    gm.memory_intensity[py, px] = 0.0
    assert gm.memory_fade_mask[py, px]

    gm.update_memory_fade(
        float(gs.turn_count),
        gs.memory_fade_steepness,
        gs.memory_fade_midpoint,
        gs.memory_fade_duration,
    )
    assert not gm.memory_fade_mask[py, px]
    assert gm.memory_intensity[py, px] == 0.0

    gm.update_memory_fade(
        float(gs.turn_count) + 1.0,
        gs.memory_fade_steepness,
        gs.memory_fade_midpoint,
        gs.memory_fade_duration,
    )
    assert gm.memory_intensity[py, px] == 0.0
    assert not gm.memory_fade_mask[py, px]


import numpy as np
import pytest
import sys
import os
import types

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from game.world.fov import update_memory_fade

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
MEMORY_DURATION = MEMORY_FADE_CFG["duration"]
MEMORY_SIGMOID_MIDPOINT = MEMORY_FADE_CFG["midpoint"]
MEMORY_SIGMOID_STEEPNESS = MEMORY_FADE_CFG["steepness"]


def test_memory_fade_bounds():
    current_time = np.float32(100.0)

    last_seen = np.full((1, 1), current_time, dtype=np.float32)
    memory_intensity = np.ones((1, 1), dtype=np.float32)
    visible = np.zeros((1, 1), dtype=bool)
    mask = np.zeros((1, 1), dtype=bool)
    prev_visible = np.ones((1, 1), dtype=bool)
    memory_strength = np.zeros((1, 1), dtype=np.float32)
    tile_modifiers = np.ones((1, 1), dtype=np.float32)

    update_memory_fade(
        current_time,
        last_seen,
        memory_intensity,
        visible,
        mask,
        prev_visible,
        memory_strength,
        tile_modifiers,
        MEMORY_SIGMOID_STEEPNESS,
        MEMORY_SIGMOID_MIDPOINT,
    )
    expected = 1.0 / (
        1.0 + np.exp(MEMORY_SIGMOID_STEEPNESS * (0.0 - MEMORY_SIGMOID_MIDPOINT))
    )
    assert memory_intensity[0, 0] == pytest.approx(expected)
    assert mask[0, 0]

    last_seen[0, 0] = current_time - MEMORY_DURATION
    update_memory_fade(
        current_time,
        last_seen,
        memory_intensity,
        visible,
        mask,
        prev_visible,
        memory_strength,
        tile_modifiers,
        MEMORY_SIGMOID_STEEPNESS,
        MEMORY_SIGMOID_MIDPOINT,
    )
    expected_after = 1.0 / (
        1.0 + np.exp(
            MEMORY_SIGMOID_STEEPNESS * (MEMORY_DURATION - MEMORY_SIGMOID_MIDPOINT)
        )
    )
    assert memory_intensity[0, 0] == pytest.approx(expected_after)
    assert mask[0, 0]

    last_seen[0, 0] = current_time - (MEMORY_DURATION + 1000.0)
    update_memory_fade(
        current_time,
        last_seen,
        memory_intensity,
        visible,
        mask,
        prev_visible,
        memory_strength,
        tile_modifiers,
        MEMORY_SIGMOID_STEEPNESS,
        MEMORY_SIGMOID_MIDPOINT,
    )
    assert memory_intensity[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert not mask[0, 0]


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
    )
    assert not gm.memory_fade_mask[py, px]
    assert gm.memory_intensity[py, px] == 0.0

    gm.update_memory_fade(
        float(gs.turn_count) + 1.0,
        gs.memory_fade_steepness,
        gs.memory_fade_midpoint,
    )
    assert gm.memory_intensity[py, px] == 0.0
    assert not gm.memory_fade_mask[py, px]


def test_tile_type_modifiers_affect_fade():
    current_time = np.float32(10.0)
    last_seen = np.zeros((1, 2), dtype=np.float32)
    memory_intensity = np.ones((1, 2), dtype=np.float32)
    visible = np.zeros((1, 2), dtype=bool)
    mask = np.zeros((1, 2), dtype=bool)
    prev_visible = np.ones((1, 2), dtype=bool)
    memory_strength = np.zeros((1, 2), dtype=np.float32)
    tile_modifiers = np.array([[1.0, 2.0]], dtype=np.float32)

    update_memory_fade(
        current_time,
        last_seen,
        memory_intensity,
        visible,
        mask,
        prev_visible,
        memory_strength,
        tile_modifiers,
        MEMORY_SIGMOID_STEEPNESS,
        MEMORY_SIGMOID_MIDPOINT,
    )

    assert memory_intensity[0, 1] > memory_intensity[0, 0]
    assert mask[0, 0] and mask[0, 1]

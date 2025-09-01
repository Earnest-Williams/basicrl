import numpy as np
import pytest
import sys
import types
from game_rng import GameRNG
from game.world.fov import update_memory_fade

# Minimal stubs for external modules
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

from game.world.game_map import GameMap
from game.game_state import GameState


def test_memory_fade_bounds():
    current_time = np.float32(100.0)
    duration = 5.0
    midpoint = 2.5
    steepness = 1.2

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
        steepness,
        midpoint,
    )
    expected = 1.0 / (1.0 + np.exp(steepness * (0.0 - midpoint)))
    assert memory_intensity[0, 0] == pytest.approx(expected)
    assert mask[0, 0]

    last_seen[0, 0] = current_time - duration
    update_memory_fade(
        current_time,
        last_seen,
        memory_intensity,
        visible,
        mask,
        prev_visible,
        memory_strength,
        tile_modifiers,
        steepness,
        midpoint,
    )
    expected_after = 1.0 / (
        1.0 + np.exp(steepness * (duration - midpoint))
    )
    assert memory_intensity[0, 0] == pytest.approx(expected_after)
    assert mask[0, 0]

    last_seen[0, 0] = current_time - (duration + 1000.0)
    update_memory_fade(
        current_time,
        last_seen,
        memory_intensity,
        visible,
        mask,
        prev_visible,
        memory_strength,
        tile_modifiers,
        steepness,
        midpoint,
    )
    assert memory_intensity[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert not mask[0, 0]


def create_game_state(duration: float = 5.0, midpoint: float = 2.5, steepness: float = 1.2):
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
        memory_fade_config={
            "enabled": True,
            "duration": duration,
            "midpoint": midpoint,
            "steepness": steepness,
        },
    )
    return gs


def test_memory_fade_decay_and_mask():
    gs = create_game_state()
    px, py = gs.player_position
    initial_strength = gs.game_map.memory_strength[py, px]

    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()

    assert gs.game_map.memory_intensity[py, px] < 1.0
    assert gs.game_map.memory_fade_mask[py, px]

    # Memory strength should decay for tiles no longer in view.
    expected_strength = max(initial_strength - 1.0, 0.0)
    assert gs.game_map.memory_strength[py, px] == pytest.approx(expected_strength)


def test_memory_fade_skips_zero_intensity_tiles():
    gs = create_game_state()
    px, py = gs.player_position

    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()

    gm = gs.game_map
    gm.memory_intensity[py, px] = 0.0
    assert gm.memory_fade_mask[py, px]

    update_memory_fade(
        float(gs.turn_count),
        gm.last_seen_time,
        gm.memory_intensity,
        gm.visible,
        gm.memory_fade_mask,
        gm.prev_visible,
        gm.memory_strength,
        gm.tile_memory_modifiers,
        gs.memory_fade_steepness,
        gs.memory_fade_midpoint,
    )
    assert not gm.memory_fade_mask[py, px]
    assert gm.memory_intensity[py, px] == 0.0

    update_memory_fade(
        float(gs.turn_count) + 1.0,
        gm.last_seen_time,
        gm.memory_intensity,
        gm.visible,
        gm.memory_fade_mask,
        gm.prev_visible,
        gm.memory_strength,
        gm.tile_memory_modifiers,
        gs.memory_fade_steepness,
        gs.memory_fade_midpoint,
    )
    assert gm.memory_intensity[py, px] == 0.0
    assert not gm.memory_fade_mask[py, px]


def test_tile_type_modifiers_affect_fade():
    current_time = np.float32(10.0)
    duration = 5.0
    midpoint = 2.5
    steepness = 1.2

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
        steepness,
        midpoint,
    )

    assert memory_intensity[0, 1] > memory_intensity[0, 0]
    assert mask[0, 0] and mask[0, 1]

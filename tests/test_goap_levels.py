import sys
import types

import numpy as np
import polars as pl

# Dummy RNG module
module = types.ModuleType("game_rng")


class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed

    def randint(self, a, b):
        return a


module.GameRNG = DummyRNG
sys.modules["game_rng"] = module

# Minimal ai_system for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.game_state import GameState
from game.ai import goap

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def create_game_state():
    game_map = GameMap(width=3, height=3)
    game_map.tiles[:] = TILE_ID_FLOOR
    game_map.update_tile_transparency()
    gs = GameState(
        existing_map=game_map,
        player_start_pos=(0, 0),
        player_glyph=ord("@"),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        effect_definitions={},
        rng_seed=1,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    return gs


def perception(gs, los=None):
    noise = np.zeros((gs.map_height, gs.map_width), dtype=np.int16)
    scent = np.zeros_like(noise)
    if los is None:
        los = np.zeros_like(noise, dtype=bool)
    return noise, scent, los


def _enemy_row(gs, enemy_id):
    return gs.entity_registry.entities_df.filter(pl.col("entity_id") == enemy_id).row(0, named=True)


def test_plan_depth_one_defaults_to_move():
    gs = create_game_state()
    enemy_id = gs.entity_registry.create_entity(
        x=1,
        y=1,
        glyph=ord("e"),
        color_fg=(255, 0, 0),
        name="Enemy",
        ai_type="goap",
    )
    row = _enemy_row(gs, enemy_id)
    adapter = goap.get_goap_adapter(1)
    adapter(row, gs, gs.rng_instance, perception(gs))
    assert gs._last_goap_action == "_action_move_attack"
    assert not hasattr(gs, "last_coordination")


def test_plan_depth_two_can_seek_cover():
    gs = create_game_state()
    enemy_id = gs.entity_registry.create_entity(
        x=1,
        y=1,
        glyph=ord("e"),
        color_fg=(255, 0, 0),
        name="Enemy",
        ai_type="goap",
    )
    row = _enemy_row(gs, enemy_id)
    los = np.ones((gs.map_height, gs.map_width), dtype=bool)
    los[2, 1] = False  # cover tile
    adapter = goap.get_goap_adapter(2)
    adapter(row, gs, gs.rng_instance, perception(gs, los))
    pos = gs.entity_registry.get_position(enemy_id)
    assert (pos.x, pos.y) == (1, 2)
    assert gs._last_goap_action == "_action_seek_cover"
    assert not hasattr(gs, "last_coordination")


def test_plan_depth_three_coordinates():
    gs = create_game_state()
    enemy_id = gs.entity_registry.create_entity(
        x=1,
        y=1,
        glyph=ord("e"),
        color_fg=(255, 0, 0),
        name="Enemy",
        ai_type="goap",
    )
    row = _enemy_row(gs, enemy_id)
    adapter = goap.get_goap_adapter(3)
    adapter(row, gs, gs.rng_instance, perception(gs))
    assert gs._last_goap_action == "_action_coordinate"
    assert gs.last_coordination == enemy_id

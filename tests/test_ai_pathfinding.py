from game.ai import goap, community
from game.game_state import GameState
from game.world.game_map import GameMap, TILE_ID_FLOOR
import sys
import types

import numpy as np
import polars as pl

# Minimal ai_system module for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module


MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def create_game_state():
    game_map = GameMap(width=5, height=5)
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


def _perception(gs):
    noise = np.zeros((gs.map_height, gs.map_width), dtype=np.int16)
    scent = np.zeros_like(noise)
    los = np.zeros_like(noise, dtype=bool)
    return noise, scent, los


def test_goap_moves_toward_player():
    gs = create_game_state()
    enemy_id = gs.entity_registry.create_entity(
        x=4,
        y=4,
        glyph=ord("e"),
        color_fg=(255, 0, 0),
        name="Enemy",
        ai_type="goap",
    )
    entity_row = gs.entity_registry.entities_df.filter(
        pl.col("entity_id") == enemy_id
    ).row(0, named=True)
    initial_pos = gs.entity_registry.get_position(enemy_id)
    px, py = gs.player_position
    init_dist = abs(initial_pos.x - px) + abs(initial_pos.y - py)
    goap.take_turn(entity_row, gs, gs.rng_instance, _perception(gs))
    new_pos = gs.entity_registry.get_position(enemy_id)
    new_dist = abs(new_pos.x - px) + abs(new_pos.y - py)
    assert new_dist < init_dist


def test_community_moves_toward_player():
    gs = create_game_state()
    enemy_id = gs.entity_registry.create_entity(
        x=4,
        y=4,
        glyph=ord("e"),
        color_fg=(255, 0, 0),
        name="Villager",
        ai_type="community",
    )
    entity_row = gs.entity_registry.entities_df.filter(
        pl.col("entity_id") == enemy_id
    ).row(0, named=True)
    initial_pos = gs.entity_registry.get_position(enemy_id)
    px, py = gs.player_position
    init_dist = abs(initial_pos.x - px) + abs(initial_pos.y - py)
    community.take_turn(entity_row, gs, gs.rng_instance, _perception(gs))
    new_pos = gs.entity_registry.get_position(enemy_id)
    new_dist = abs(new_pos.x - px) + abs(new_pos.y - py)
    assert new_dist < init_dist

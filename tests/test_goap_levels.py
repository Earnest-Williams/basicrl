import numpy as np
import polars as pl

from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.game_state import GameState
from game.systems.ai_system import dispatch_ai

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
        intelligence=1,
    )
    row = _enemy_row(gs, enemy_id)
    dispatch_ai([row], gs, gs.rng_instance, perception(gs))
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
        intelligence=2,
    )
    row = _enemy_row(gs, enemy_id)
    los = np.ones((gs.map_height, gs.map_width), dtype=bool)
    los[2, 1] = False  # cover tile
    dispatch_ai([row], gs, gs.rng_instance, perception(gs, los))
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
        intelligence=3,
    )
    row = _enemy_row(gs, enemy_id)
    dispatch_ai([row], gs, gs.rng_instance, perception(gs))
    assert gs._last_goap_action == "_action_coordinate"
    assert gs.last_coordination == enemy_id

from game.systems.ai_system import dispatch_ai
from game.game_state import GameState
from game.world.game_map import GameMap, TILE_ID_FLOOR
import types
import sys
import polars as pl
import numpy as np
from pathlib import Path
import yaml

# Provide deterministic RNG
module = types.ModuleType('game_rng')


class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed

    def get_int(self, a, b):
        return a

    def randint(self, a, b):
        return a


module.GameRNG = DummyRNG
sys.modules['game_rng'] = module


MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0,
                   "midpoint": 2.5, "steepness": 1.2}


def create_game_state():
    gm = GameMap(width=5, height=5)
    gm.tiles[:] = TILE_ID_FLOOR
    gm.update_tile_transparency()
    cfg_path = Path(__file__).resolve().parent.parent / \
        "config" / "ai_mappings.yaml"
    with cfg_path.open() as f:
        ai_cfg = yaml.safe_load(f)
    gs = GameState(
        existing_map=gm,
        player_start_pos=(2, 2),
        player_glyph=ord('@'),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        effect_definitions={},
        rng_seed=1,
        ai_config=ai_cfg,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    gs.game_map.visible[:, :] = True
    return gs


def _perception(gs):
    size = (gs.map_height, gs.map_width)
    noise = np.zeros(size, dtype=np.int16)
    scent = np.zeros_like(noise)
    los = np.ones(size, dtype=bool)
    return noise, scent, los


def _row(gs, eid):
    return gs.entity_registry.entities_df.filter(pl.col('entity_id') == eid).row(0, named=True)


def test_insect_moves_toward_ally():
    gs = create_game_state()
    er = gs.entity_registry
    a1 = er.create_entity(x=1, y=1, glyph=1, color_fg=(
        255, 0, 0), name='Bug', species='insect')
    er.create_entity(x=3, y=1, glyph=1, color_fg=(
        255, 0, 0), name='Bug', species='insect')
    dispatch_ai([_row(gs, a1)], gs, gs.rng_instance, _perception(gs))
    pos = er.get_position(a1)
    assert (pos.x, pos.y) == (2, 1)


def test_bird_flies_two_tiles():
    gs = create_game_state()
    er = gs.entity_registry
    b = er.create_entity(x=1, y=1, glyph=1, color_fg=(
        255, 255, 255), name='Bird', species='bird')
    dispatch_ai([_row(gs, b)], gs, gs.rng_instance, _perception(gs))
    pos = er.get_position(b)
    assert (pos.x, pos.y) == (3, 1)


def test_mammal_moves_toward_player():
    gs = create_game_state()
    er = gs.entity_registry
    m = er.create_entity(x=2, y=4, glyph=1, color_fg=(
        255, 255, 0), name='Wolf', species='mammal')
    dispatch_ai([_row(gs, m)], gs, gs.rng_instance, _perception(gs))
    pos = er.get_position(m)
    assert (pos.x, pos.y) == (2, 3)


def test_reptile_ambushes_player():
    gs = create_game_state()
    er = gs.entity_registry
    r = er.create_entity(x=2, y=4, glyph=1, color_fg=(
        0, 255, 0), name='Snake', species='reptile')
    dispatch_ai([_row(gs, r)], gs, gs.rng_instance, _perception(gs))
    pos = er.get_position(r)
    assert (pos.x, pos.y) == (2, 3)
    dispatch_ai([_row(gs, r)], gs, gs.rng_instance, _perception(gs))
    hp = er.get_entity_component(gs.player_id, 'hp')
    assert hp < 10


def test_plant_attacks_without_moving():
    gs = create_game_state()
    er = gs.entity_registry
    p = er.create_entity(x=2, y=3, glyph=1, color_fg=(
        0, 255, 0), name='Vine', species='plant')
    dispatch_ai([_row(gs, p)], gs, gs.rng_instance, _perception(gs))
    pos = er.get_position(p)
    assert (pos.x, pos.y) == (2, 3)
    hp = er.get_entity_component(gs.player_id, 'hp')
    assert hp < 10

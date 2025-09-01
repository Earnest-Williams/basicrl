import types
import sys
import polars as pl
import numpy as np
from pathlib import Path
import yaml

# deterministic RNG
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

from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.game_state import GameState
from game.systems.ai_system import dispatch_ai
from game.ai import community, ADAPTERS

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def create_game_state():
    game_map = GameMap(width=3, height=3)
    game_map.tiles[:] = TILE_ID_FLOOR
    game_map.update_tile_transparency()
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "ai_mappings.yaml"
    with cfg_path.open() as f:
        ai_cfg = yaml.safe_load(f)
    gs = GameState(
        existing_map=game_map,
        player_start_pos=(0, 0),
        player_glyph=ord('@'),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        effect_definitions={},
        rng_seed=1,
        ai_config=ai_cfg,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    return gs


def perception(gs, los=None):
    noise = np.zeros((gs.map_height, gs.map_width), dtype=np.int16)
    scent = np.zeros_like(noise)
    if los is None:
        los = np.zeros_like(noise, dtype=bool)
    return noise, scent, los


def _row(gs, eid):
    return gs.entity_registry.entities_df.filter(pl.col("entity_id") == eid).row(0, named=True)


def test_species_mapping_selects_adapter(monkeypatch):
    gs = create_game_state()
    called = {}
    def fake_take_turn(*args, **kwargs):
        called['used'] = True
    monkeypatch.setitem(ADAPTERS, 'community', fake_take_turn)
    enemy_id = gs.entity_registry.create_entity(
        x=1, y=1, glyph=ord('h'), color_fg=(255,255,255), name='Villager', species='human')
    dispatch_ai([_row(gs, enemy_id)], gs, gs.rng_instance, perception(gs))
    assert called.get('used')


def test_intelligence_mapping_selects_goap_tier():
    gs = create_game_state()
    enemy_id = gs.entity_registry.create_entity(
        x=1, y=1, glyph=ord('e'), color_fg=(255,0,0), name='Enemy', intelligence=2)
    row = _row(gs, enemy_id)
    los = np.ones((gs.map_height, gs.map_width), dtype=bool)
    los[2,1] = False  # cover tile
    dispatch_ai([row], gs, gs.rng_instance, perception(gs, los))
    assert gs._last_goap_action == "_action_seek_cover"

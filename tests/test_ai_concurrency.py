import numpy as np
import polars as pl


from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.game_state import GameState
from game.systems.ai_system import dispatch_ai

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0,
                   "midpoint": 2.5, "steepness": 1.2}


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
    gs.entity_registry.set_entity_component(gs.player_id, "faction", "player")
    return gs


def _perception(gs):
    size = (gs.map_height, gs.map_width)
    noise = np.zeros(size, dtype=np.int16)
    scent = np.zeros_like(noise)
    los = np.ones(size, dtype=bool)
    return noise, scent, los


def test_parallel_strategy_moves_entities():
    gs = create_game_state()
    positions = [(4, 4), (4, 3), (3, 4), (2, 4)]
    ids = []
    for x, y in positions:
        eid = gs.entity_registry.create_entity(
            x=x,
            y=y,
            glyph=ord("e"),
            color_fg=(255, 0, 0),
            name="Enemy",
            ai_type="strategy",
            strategy_state="CHARGE",
            faction="kobold",
        )
        ids.append(eid)
    rows = [
        gs.entity_registry.entities_df.filter(
            pl.col("entity_id") == eid).row(0, named=True)
        for eid in ids
    ]
    dispatch_ai(rows, gs, gs.rng_instance, _perception(gs))
    for eid in ids:
        pos = gs.entity_registry.get_position(eid)
        assert 0 <= pos.x < gs.map_width
        assert 0 <= pos.y < gs.map_height

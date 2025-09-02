from engine.render_base_layers import prepare_base_layers
from game.ai.perception import gather_perception
from game.game_state import GameState
from game.world.game_map import GameMap
import numpy as np
import sys
import types

# Provide a minimal ai_system module for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module


MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0,
                   "midpoint": 2.5, "steepness": 1.2}


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


def test_update_fov_origin_visibility_and_memory():
    gs = create_game_state()
    px, py = gs.player_position
    assert gs.game_map.visible[py, px]
    assert gs.game_map.explored[py, px]
    assert gs.game_map.memory_intensity[py, px] == 1.0

    # Move player away and advance the turn to trigger memory fade
    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()

    assert gs.game_map.memory_intensity[py, px] < 1.0


def test_gather_perception_matches_visible():
    gs = create_game_state()
    los_map = gather_perception(gs)[2]
    assert np.array_equal(los_map, gs.game_map.visible)

    # After moving the player and advancing the turn, LOS map should match new visibility
    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()
    los_map2 = gather_perception(gs)[2]
    assert np.array_equal(los_map2, gs.game_map.visible)


def test_renderer_uses_visibility():
    gs = create_game_state()
    gm = gs.game_map

    max_defined_tile_id = 255
    tile_fg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_bg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_indices_render = np.zeros(max_defined_tile_id + 1, dtype=np.uint16)

    result = prepare_base_layers(
        gm,
        viewport_x=0,
        viewport_y=0,
        viewport_width=gm.width,
        viewport_height=gm.height,
        max_defined_tile_id=max_defined_tile_id,
        tile_fg_colors=tile_fg_colors,
        tile_bg_colors=tile_bg_colors,
        tile_indices_render=tile_indices_render,
    )

    map_visible_vp = result[6]
    assert np.array_equal(map_visible_vp, gm.visible)

import sys
import types
import numpy as np
from game_rng import GameRNG

# Provide a minimal ai_system module for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.game_state import GameState
from engine import renderer

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


def test_memory_fade_blend_and_glyph_substitution():
    gs = create_game_state()
    gm = gs.game_map
    px, py = gs.player_position

    # Make the player's original tile only remembered, not visible
    gm.visible[py, px] = False
    gm.explored[py, px] = True
    gm.memory_intensity[py, px] = 0.5
    gm.tiles[py, px] = TILE_ID_FLOOR

    max_defined_tile_id = 255
    tile_fg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_bg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_indices_render = np.zeros(max_defined_tile_id + 1, dtype=np.uint16)
    tile_fg_colors[TILE_ID_FLOOR] = [200, 200, 200]
    tile_bg_colors[TILE_ID_FLOOR] = [10, 10, 10]
    tile_indices_render[TILE_ID_FLOOR] = ord('.')

    (
        base_fg,
        base_bg,
        glyph_indices,
        visible_mask,
        drawn_mask,
        map_height_vp,
        map_visible_vp,
        map_memory_vp,
        map_tiles_vp,
        (vp_h, vp_w),
    ) = renderer._prepare_base_layers(
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

    final_fg = base_fg.copy()
    final_bg = base_bg.copy()
    glyphs = glyph_indices.copy()
    fade_color = np.array([100, 100, 100], dtype=np.uint8)

    renderer._apply_memory_fade(
        final_fg,
        final_bg,
        glyphs,
        map_memory_vp,
        map_tiles_vp,
        drawn_mask,
        visible_mask,
        fade_color,
        gs.rng_instance,
        viewport_x=0,
        viewport_y=0,
    )

    assert (final_fg[py, px] == np.array([150, 150, 150], dtype=np.uint8)).all()
    assert (final_bg[py, px] == np.array([55, 55, 55], dtype=np.uint8)).all()
    expected_glyph = renderer.MEMORY_FLOOR_GLYPHS[2]
    assert glyphs[py, px] == expected_glyph


def test_memory_fade_variance_and_noise_deterministic():
    gs = create_game_state()
    gm = gs.game_map
    px, py = gs.player_position

    gm.visible[py, px] = False
    gm.explored[py, px] = True
    gm.memory_intensity[py, px] = 0.5
    gm.tiles[py, px] = TILE_ID_FLOOR

    max_defined_tile_id = 255
    tile_fg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_bg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_indices_render = np.zeros(max_defined_tile_id + 1, dtype=np.uint16)
    tile_fg_colors[TILE_ID_FLOOR] = [200, 200, 200]
    tile_bg_colors[TILE_ID_FLOOR] = [10, 10, 10]
    tile_indices_render[TILE_ID_FLOOR] = ord('.')

    (
        base_fg,
        base_bg,
        glyph_indices,
        visible_mask,
        drawn_mask,
        map_height_vp,
        map_visible_vp,
        map_memory_vp,
        map_tiles_vp,
        (vp_h, vp_w),
    ) = renderer._prepare_base_layers(
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

    fade_color = np.array([120, 80, 60], dtype=np.uint8)

    # Baseline without variance/noise
    baseline_fg = base_fg.copy()
    baseline_bg = base_bg.copy()
    baseline_glyphs = glyph_indices.copy()
    renderer._apply_memory_fade(
        baseline_fg,
        baseline_bg,
        baseline_glyphs,
        map_memory_vp,
        map_tiles_vp,
        drawn_mask,
        visible_mask,
        fade_color,
        gs.rng_instance,
        viewport_x=0,
        viewport_y=0,
    )

    # With variance and noise
    final_fg = base_fg.copy()
    final_bg = base_bg.copy()
    glyphs = glyph_indices.copy()
    renderer._apply_memory_fade(
        final_fg,
        final_bg,
        glyphs,
        map_memory_vp,
        map_tiles_vp,
        drawn_mask,
        visible_mask,
        fade_color,
        gs.rng_instance,
        fade_color_variance=0.25,
        noise_level=1.0,
        viewport_x=0,
        viewport_y=0,
    )

    final_fg2 = base_fg.copy()
    final_bg2 = base_bg.copy()
    glyphs2 = glyph_indices.copy()
    renderer._apply_memory_fade(
        final_fg2,
        final_bg2,
        glyphs2,
        map_memory_vp,
        map_tiles_vp,
        drawn_mask,
        visible_mask,
        fade_color,
        gs.rng_instance,
        fade_color_variance=0.25,
        noise_level=1.0,
        viewport_x=0,
        viewport_y=0,
    )

    # Deterministic across runs
    assert np.array_equal(final_fg[py, px], final_fg2[py, px])
    assert glyphs[py, px] == glyphs2[py, px]

    # Colour differs from baseline and glyph uses noisy set
    assert not np.array_equal(final_fg[py, px], baseline_fg[py, px])
    level = int((1.0 - gm.memory_intensity[py, px]) * renderer.MEMORY_LEVEL_COUNT)
    assert glyphs[py, px] == renderer.NOISY_MEMORY_FLOOR_GLYPHS[level]

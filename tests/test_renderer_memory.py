from PIL import Image
from engine.renderer import RenderConfig, ViewportParams, render_viewport
from engine.render_lighting import (
    apply_memory_fade,
    MEMORY_FLOOR_GLYPHS,
    MEMORY_LEVEL_COUNT,
    NOISY_MEMORY_FLOOR_GLYPHS,
)
from engine.render_base_layers import prepare_base_layers
from game.game_state import GameState
from game.world.game_map import GameMap, TILE_ID_FLOOR
import sys
import types
import numpy as np

# Provide a minimal ai_system module for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module


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
    tile_indices_render[TILE_ID_FLOOR] = ord(".")

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
    ) = prepare_base_layers(
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

    apply_memory_fade(
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
    expected_glyph = MEMORY_FLOOR_GLYPHS[2]
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
    tile_indices_render[TILE_ID_FLOOR] = ord(".")

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
    ) = prepare_base_layers(
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
    apply_memory_fade(
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
    apply_memory_fade(
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
    apply_memory_fade(
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
    level = int((1.0 - gm.memory_intensity[py, px]) * MEMORY_LEVEL_COUNT)
    assert glyphs[py, px] == NOISY_MEMORY_FLOOR_GLYPHS[level]


def test_render_viewport_smoke():
    gs = create_game_state()
    gm = gs.game_map
    max_defined_tile_id = 255
    tile_fg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_bg_colors = np.zeros((max_defined_tile_id + 1, 3), dtype=np.uint8)
    tile_indices_render = np.zeros(max_defined_tile_id + 1, dtype=np.uint16)
    coord_arrays = {
        "tile_coord_y": np.zeros((gm.height, gm.width), dtype=np.int16),
        "tile_coord_x": np.zeros((gm.height, gm.width), dtype=np.int16),
    }
    viewport = ViewportParams(
        viewport_x=0,
        viewport_y=0,
        viewport_width=gm.width,
        viewport_height=gm.height,
        tile_arrays={},
        tile_fg_colors=tile_fg_colors,
        tile_bg_colors=tile_bg_colors,
        tile_indices_render=tile_indices_render,
        max_defined_tile_id=max_defined_tile_id,
        tile_w=1,
        tile_h=1,
        coord_arrays=coord_arrays,
    )
    render_config = RenderConfig(
        show_height_vis=False,
        vis_max_diff=0,
        vis_color_high_np=np.zeros(3, dtype=np.uint8),
        vis_color_mid_np=np.zeros(3, dtype=np.uint8),
        vis_color_low_np=np.zeros(3, dtype=np.uint8),
        vis_blend_factor=np.float32(0.0),
        lighting_ambient=np.float32(1.0),
        lighting_min_fov=np.float32(0.0),
        lighting_falloff=np.float32(1.0),
        fov_radius_sq=np.float32(1.0),
    )
    image = render_viewport(gs, viewport, render_config)
    assert isinstance(image, Image.Image)

"""Lighting and visual effect helpers for rendering."""

import colorsys
import math
import numpy as np
import structlog

try:
    from numba import float32, njit, uint8
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(func=None, **options):  # type: ignore
        if func:
            return func
        return lambda f: f

    uint8 = np.uint8  # type: ignore
    float32 = np.float32  # type: ignore

try:
    from game.world.game_map import TILE_ID_FLOOR, TILE_ID_WALL
except ImportError:
    try:
        from basicrl.game.world.game_map import TILE_ID_FLOOR, TILE_ID_WALL
    except ImportError:
        TILE_ID_FLOOR = 0  # type: ignore
        TILE_ID_WALL = 1  # type: ignore
        structlog.get_logger().error(
            "CRITICAL: Failed to import tile IDs in render_lighting.")

try:
    from game_rng import GameRNG
except ImportError:
    try:
        from basicrl.game_rng import GameRNG
    except ImportError:
        class GameRNG:  # type: ignore
            pass
        structlog.get_logger().error(
            "CRITICAL: Failed to import GameRNG in render_lighting.")

log = structlog.get_logger()

MEMORY_WALL_GLYPHS = np.array(
    [ord("▓"), ord("▒"), ord("░"), ord("⋅"), ord(" ")], dtype=np.uint16
)
MEMORY_FLOOR_GLYPHS = np.array(
    [ord("."), ord("·"), ord("⋅"), ord(" "), ord(" ")], dtype=np.uint16
)
NOISY_MEMORY_WALL_GLYPHS = np.array(
    [ord("█"), ord("▓"), ord("▒"), ord("░"), ord(" ")], dtype=np.uint16
)
NOISY_MEMORY_FLOOR_GLYPHS = np.array(
    [ord("#"), ord("~"), ord(":"), ord("."), ord(" ")], dtype=np.uint16
)
MEMORY_LEVEL_COUNT = MEMORY_WALL_GLYPHS.size


@njit(
    float32(float32, float32, float32, float32), cache=True, fastmath=True, nogil=True
)
def _calculate_light_intensity_scalar(
    dist_sq: np.float32,
    radius_sq: np.float32,
    falloff_power: np.float32,
    min_light_level: np.float32,
) -> np.float32:
    """Calculates light intensity based on distance squared."""
    if radius_sq < 1e-6:
        return np.float32(1.0) if dist_sq < 1e-6 else np.float32(0.0)
    if dist_sq >= radius_sq:
        return np.float32(0.0)

    dist = math.sqrt(dist_sq)
    radius = math.sqrt(radius_sq)
    falloff_ratio = dist / radius
    light_value = max(np.float32(0.0), np.float32(1.0) - falloff_ratio) ** falloff_power
    intensity = max(light_value, min_light_level)
    return max(np.float32(0.0), min(np.float32(1.0), intensity))


@njit("uint8[:](uint8[:], float32)", cache=True, fastmath=True, nogil=True)
def _interpolate_color_numba_vector(
    base_color: np.ndarray, intensity: np.float32
) -> np.ndarray:
    """Interpolates an RGB color towards black based on intensity."""
    if base_color.shape[0] < 3:
        return np.zeros(3, dtype=uint8)

    intensity_clamped = max(np.float32(0.0), min(np.float32(1.0), intensity))
    result = np.empty(3, dtype=uint8)
    for i in range(3):
        result[i] = max(0, min(255, int(base_color[i] * intensity_clamped)))
    return result


@njit(cache=True, nogil=True)
def calculate_lighting(
    base_fg: np.ndarray,
    base_bg: np.ndarray,
    visible_mask: np.ndarray,
    vp_h: int,
    vp_w: int,
    viewport_x: int,
    viewport_y: int,
    player_x: int,
    player_y: int,
    config_ambient: float32,
    config_min_fov: float32,
    config_falloff: float32,
    fov_radius_sq: float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates lighting intensity and applies it."""
    intensity_map = np.full((vp_h, vp_w), config_ambient, dtype=np.float32)

    if np.any(visible_mask):
        visible_y_coords, visible_x_coords = np.where(visible_mask)
        map_abs_x_visible = visible_x_coords + viewport_x
        map_abs_y_visible = visible_y_coords + viewport_y

        dx = map_abs_x_visible - player_x
        dy = map_abs_y_visible - player_y
        dist_sq_map_visible = (dx * dx + dy * dy).astype(np.float32)

        visible_intensities = np.empty_like(dist_sq_map_visible, dtype=np.float32)
        for i in range(dist_sq_map_visible.shape[0]):
            visible_intensities[i] = _calculate_light_intensity_scalar(
                dist_sq_map_visible[i],
                fov_radius_sq,
                config_falloff,
                config_min_fov,
            )

        for i in range(visible_y_coords.shape[0]):
            y = visible_y_coords[i]
            x = visible_x_coords[i]
            intensity_map[y, x] = visible_intensities[i]

    intensity_broadcast = intensity_map[..., None]
    lit_fg = (base_fg.astype(np.float32) * intensity_broadcast).astype(np.uint8)
    lit_bg = (base_bg.astype(np.float32) * intensity_broadcast).astype(np.uint8)
    return lit_fg, lit_bg, intensity_map


@njit(cache=True, nogil=True)
def apply_height_visualization(
    lit_fg: np.ndarray,
    lit_bg: np.ndarray,
    drawn_mask: np.ndarray,
    map_height_vp: np.ndarray,
    player_height: int,
    config_show_vis: bool,
    config_max_diff: int,
    config_color_high: np.ndarray,
    config_color_mid: np.ndarray,
    config_color_low: np.ndarray,
    config_blend_factor: float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies height visualization tinting."""
    if not config_show_vis:
        return lit_fg, lit_bg

    final_fg = lit_fg.copy()
    final_bg = lit_bg.copy()
    relative_height_vp = map_height_vp - player_height

    max_diff_f32 = np.float32(config_max_diff)
    blend = config_blend_factor

    drawn_and_valid_diff = drawn_mask & (max_diff_f32 > np.float32(0.0))

    high_mask = (
        drawn_and_valid_diff
        & (relative_height_vp > 0)
        & (relative_height_vp <= config_max_diff)
    )
    if np.any(high_mask):
        high_y_coords, high_x_coords = np.where(high_mask)
        target_color_high_f32 = config_color_high.astype(np.float32)

        for i in range(high_y_coords.shape[0]):
            y = high_y_coords[i]
            x = high_x_coords[i]
            t = np.float32(relative_height_vp[y, x]) / max_diff_f32
            blend_t = blend * t
            current_fg_f32 = final_fg[y, x].astype(np.float32)
            current_bg_f32 = final_bg[y, x].astype(np.float32)
            final_fg[y, x] = np.clip(
                current_fg_f32 * (np.float32(1.0) - blend_t)
                + target_color_high_f32 * blend_t,
                0,
                255,
            ).astype(np.uint8)
            final_bg[y, x] = np.clip(
                current_bg_f32 * (np.float32(1.0) - blend_t)
                + target_color_high_f32 * blend_t,
                0,
                255,
            ).astype(np.uint8)

    low_mask = (
        drawn_and_valid_diff
        & (relative_height_vp < 0)
        & (relative_height_vp >= -config_max_diff)
    )
    if np.any(low_mask):
        low_y_coords, low_x_coords = np.where(low_mask)
        target_color_low_f32 = config_color_low.astype(np.float32)
        for i in range(low_y_coords.shape[0]):
            y = low_y_coords[i]
            x = low_x_coords[i]
            t = np.float32(np.abs(relative_height_vp[y, x])) / max_diff_f32
            blend_t = blend * t
            current_fg_f32 = final_fg[y, x].astype(np.float32)
            current_bg_f32 = final_bg[y, x].astype(np.float32)
            final_fg[y, x] = np.clip(
                current_fg_f32 * (np.float32(1.0) - blend_t)
                + target_color_low_f32 * blend_t,
                0,
                255,
            ).astype(np.uint8)
            final_bg[y, x] = np.clip(
                current_bg_f32 * (np.float32(1.0) - blend_t)
                + target_color_low_f32 * blend_t,
                0,
                255,
            ).astype(np.uint8)

    return final_fg, final_bg


def apply_memory_fade(
    final_fg: np.ndarray,
    final_bg: np.ndarray,
    glyph_indices: np.ndarray,
    map_memory_vp: np.ndarray,
    map_tiles_vp: np.ndarray,
    drawn_mask: np.ndarray,
    visible_mask: np.ndarray,
    fade_color_np: np.ndarray,
    rng: GameRNG,
    fade_color_variance: float = 0.0,
    noise_level: float = 0.0,
    viewport_x: int = 0,
    viewport_y: int = 0,
) -> None:
    """Blend colors and swap glyphs for tiles remembered in the fog."""
    memory_mask = drawn_mask & (~visible_mask)
    if not np.any(memory_mask):
        return

    fade_vals = map_memory_vp[memory_mask][:, None]
    coords_y, coords_x = np.nonzero(memory_mask)
    world_x = coords_x + viewport_x
    world_y = coords_y + viewport_y

    if fade_color_variance > 0.0:
        count = fade_vals.shape[0]
        base_h, base_s, base_v = colorsys.rgb_to_hsv(*(fade_color_np / 255.0))
        hue_offsets = np.array(
            [rng.noise_2d(x, y, seed_offset=1) for x, y in zip(world_x, world_y)]
        ) * fade_color_variance
        sat_offsets = np.abs(
            np.array(
                [rng.noise_2d(x, y, seed_offset=2) for x, y in zip(world_x, world_y)]
            )
        ) * fade_color_variance
        hues = (base_h + hue_offsets) % 1.0
        sats = np.clip(base_s * (1.0 - sat_offsets), 0.0, 1.0)
        vals = np.full(count, base_v)
        fade_rgbs = (
            np.array(
                [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, sats, vals)],
                dtype=np.float32,
            )
            * 255.0
        )
    else:
        fade_rgbs = np.tile(fade_color_np.astype(np.float32), (fade_vals.shape[0], 1))

    final_fg[memory_mask] = (
        final_fg[memory_mask].astype(np.float32) * fade_vals
        + fade_rgbs * (1.0 - fade_vals)
    ).astype(np.uint8)
    final_bg[memory_mask] = (
        final_bg[memory_mask].astype(np.float32) * fade_vals
        + fade_rgbs * (1.0 - fade_vals)
    ).astype(np.uint8)

    tile_ids = map_tiles_vp[memory_mask]
    levels = np.clip(
        (1.0 - map_memory_vp[memory_mask]) * MEMORY_LEVEL_COUNT,
        0,
        MEMORY_LEVEL_COUNT - 1,
    ).astype(np.intp)

    new_glyphs = glyph_indices[memory_mask].copy()
    if noise_level > 0.0:
        noise_vals = np.array(
            [rng.noise_2d(x, y, seed_offset=3) for x, y in zip(world_x, world_y)]
        )
        noise_mask = ((noise_vals + 1.0) * 0.5) < noise_level
    else:
        noise_mask = np.zeros(levels.shape[0], dtype=bool)

    wall_mask = tile_ids == TILE_ID_WALL
    if np.any(wall_mask):
        clean = wall_mask & ~noise_mask
        noisy = wall_mask & noise_mask
        if np.any(clean):
            new_glyphs[clean] = MEMORY_WALL_GLYPHS[levels[clean]]
        if np.any(noisy):
            new_glyphs[noisy] = NOISY_MEMORY_WALL_GLYPHS[levels[noisy]]
    floor_mask = tile_ids == TILE_ID_FLOOR
    if np.any(floor_mask):
        clean = floor_mask & ~noise_mask
        noisy = floor_mask & noise_mask
        if np.any(clean):
            new_glyphs[clean] = MEMORY_FLOOR_GLYPHS[levels[clean]]
        if np.any(noisy):
            new_glyphs[noisy] = NOISY_MEMORY_FLOOR_GLYPHS[levels[noisy]]
    glyph_indices[memory_mask] = new_glyphs


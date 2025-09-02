"""Helper utilities for preparing base rendering layers."""

import numpy as np
import structlog

try:
    from game.world.game_map import GameMap
except ImportError:
    try:
        from basicrl.game.world.game_map import GameMap
    except ImportError:
        GameMap = object  # type: ignore
        structlog.get_logger().error(
            "CRITICAL: Failed to import GameMap in render_base_layers."
        )

log = structlog.get_logger()


def prepare_base_layers(
    game_map: GameMap,
    viewport_x: int,
    viewport_y: int,
    viewport_width: int,
    viewport_height: int,
    max_defined_tile_id: int,
    tile_fg_colors: np.ndarray,
    tile_bg_colors: np.ndarray,
    tile_indices_render: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """Prepares base color and glyph index arrays for the viewport."""
    if not isinstance(game_map, GameMap) or GameMap is object:
        log.error("prepare_base_layers called with invalid GameMap")
        dummy_shape = (max(0, viewport_height), max(0, viewport_width))
        return (
            np.zeros((*dummy_shape, 3), dtype=np.uint8),
            np.zeros((*dummy_shape, 3), dtype=np.uint8),
            np.zeros(dummy_shape, dtype=np.uint16),
            np.zeros(dummy_shape, dtype=bool),
            np.zeros(dummy_shape, dtype=bool),
            np.zeros(dummy_shape, dtype=np.int16),
            np.zeros(dummy_shape, dtype=bool),
            np.zeros(dummy_shape, dtype=np.float32),
            dummy_shape,
        )

    map_y_slice = slice(viewport_y, viewport_y + viewport_height)
    map_x_slice = slice(viewport_x, viewport_x + viewport_width)
    safe_y_slice = slice(
        max(0, map_y_slice.start), min(game_map.height, map_y_slice.stop)
    )
    safe_x_slice = slice(
        max(0, map_x_slice.start), min(game_map.width, map_x_slice.stop)
    )

    actual_vp_h = safe_y_slice.stop - safe_y_slice.start
    actual_vp_w = safe_x_slice.stop - safe_x_slice.start

    if actual_vp_h <= 0 or actual_vp_w <= 0:
        log.warning(
            "Viewport slice resulted in zero or negative size.",
            vp_slice_y=map_y_slice,
            vp_slice_x=map_x_slice,
            map_shape=(game_map.height, game_map.width),
        )
        dummy_shape = (max(0, actual_vp_h), max(0, actual_vp_w))
        return (
            np.zeros((*dummy_shape, 3), dtype=np.uint8),
            np.zeros((*dummy_shape, 3), dtype=np.uint8),
            np.zeros(dummy_shape, dtype=np.uint16),
            np.zeros(dummy_shape, dtype=bool),
            np.zeros(dummy_shape, dtype=bool),
            np.zeros(dummy_shape, dtype=np.int16),
            np.zeros(dummy_shape, dtype=bool),
            np.zeros(dummy_shape, dtype=np.float32),
            dummy_shape,
        )

    map_visible_vp = game_map.visible[safe_y_slice, safe_x_slice]
    map_explored_vp = game_map.explored[safe_y_slice, safe_x_slice]
    map_tiles_vp = game_map.tiles[safe_y_slice, safe_x_slice]
    map_height_vp = game_map.height_map[safe_y_slice, safe_x_slice]
    map_memory_vp = game_map.memory_intensity[safe_y_slice, safe_x_slice]
    vp_h, vp_w = map_visible_vp.shape

    visible_mask = map_visible_vp
    explored_mask = map_explored_vp & (~visible_mask)
    drawn_mask = visible_mask | explored_mask

    base_fg = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
    base_bg = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
    glyph_indices = np.zeros((vp_h, vp_w), dtype=np.uint16)

    render_data_valid = (
        tile_fg_colors is not None
        and tile_bg_colors is not None
        and tile_indices_render is not None
        and max_defined_tile_id >= 0
        and len(tile_fg_colors) > max_defined_tile_id
        and len(tile_bg_colors) > max_defined_tile_id
        and len(tile_indices_render) > max_defined_tile_id
    )

    if not render_data_valid:
        log.error("Render data arrays invalid in prepare_base_layers")
        drawn_mask.fill(False)
    elif np.any(drawn_mask):
        try:
            tile_ids_in_vp_raw = map_tiles_vp[drawn_mask]
            valid_tile_ids_in_vp = np.clip(tile_ids_in_vp_raw, 0, max_defined_tile_id)
            base_fg[drawn_mask] = tile_fg_colors[valid_tile_ids_in_vp]
            base_bg[drawn_mask] = tile_bg_colors[valid_tile_ids_in_vp]
            glyph_indices[drawn_mask] = tile_indices_render[valid_tile_ids_in_vp]
        except IndexError as e:
            log.error(
                "IndexError during color/glyph assignment in prepare_base_layers",
                error=str(e),
                exc_info=True,
            )
            drawn_mask.fill(False)
        except Exception as e:
            log.error(
                "Unexpected error during color/glyph assignment in prepare_base_layers",
                error=str(e),
                exc_info=True,
            )
            drawn_mask.fill(False)

    return (
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
    )

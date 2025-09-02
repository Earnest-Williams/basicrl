"""Rendering helpers for tiles, items, and entities."""

from typing import Dict as PyDict, List, cast

import numpy as np

try:
    from numba import njit, uint8
    from numba.typed import Dict as NumbaDict
    from numba import types as nb_types
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    NumbaDict = dict  # type: ignore
    nb_types = object  # type: ignore

    def njit(func=None, **options):  # type: ignore
        if func:
            return func
        return lambda f: f

    uint8 = np.uint8  # type: ignore

from .render_lighting import _interpolate_color_numba_vector

NJIT_SENTINEL_TILE_ARRAY_SHAPE = (0, 0, 4)


@njit(cache=True, nogil=True)
def render_map_tiles(
    output_image_array: np.ndarray,
    glyph_indices: np.ndarray,
    drawn_mask: np.ndarray,
    final_fg: np.ndarray,
    final_bg: np.ndarray,
    tile_arrays: NumbaDict,
    vp_h: int,
    vp_w: int,
    tile_h: int,
    tile_w: int,
) -> None:
    tile_buffer = np.empty((tile_h, tile_w, 4), dtype=uint8)
    for vp_y in range(vp_h):
        for vp_x in range(vp_w):
            if not drawn_mask[vp_y, vp_x]:
                continue

            tile_glyph_idx = glyph_indices[vp_y, vp_x]
            if tile_glyph_idx in tile_arrays:
                tile_rgba_array = tile_arrays[tile_glyph_idx]
                if (
                    tile_rgba_array.shape == (tile_h, tile_w, 4)
                    and tile_rgba_array.shape != NJIT_SENTINEL_TILE_ARRAY_SHAPE
                ):
                    glyph_alpha_channel = tile_rgba_array[:, :, 3]
                    glyph_draw_mask_2d = glyph_alpha_channel > 10
                    mask_rows, mask_cols = np.where(glyph_draw_mask_2d)
                    tile_fg_color_rgb = final_fg[vp_y, vp_x]
                    for i in range(mask_rows.shape[0]):
                        r, c = mask_rows[i], mask_cols[i]
                        tile_buffer[r, c, 0] = max(
                            0, min(255, tile_fg_color_rgb[0]))
                        tile_buffer[r, c, 1] = max(
                            0, min(255, tile_fg_color_rgb[1]))
                        tile_buffer[r, c, 2] = max(
                            0, min(255, tile_fg_color_rgb[2]))
                        tile_buffer[r, c, 3] = glyph_alpha_channel[r, c]
                else:
                    tile_buffer[:, :, :] = 0
                    tile_bg_color_rgb = final_bg[vp_y, vp_x]
                    tile_buffer[:, :, 0] = max(
                        0, min(255, tile_bg_color_rgb[0]))
                    tile_buffer[:, :, 1] = max(
                        0, min(255, tile_bg_color_rgb[1]))
                    tile_buffer[:, :, 2] = max(
                        0, min(255, tile_bg_color_rgb[2]))
                    tile_buffer[:, :, 3] = 255
            else:
                tile_bg_color_rgb = final_bg[vp_y, vp_x]
                tile_buffer[:, :, 0] = max(0, min(255, tile_bg_color_rgb[0]))
                tile_buffer[:, :, 1] = max(0, min(255, tile_bg_color_rgb[1]))
                tile_buffer[:, :, 2] = max(0, min(255, tile_bg_color_rgb[2]))
                tile_buffer[:, :, 3] = 255

            px_start_y = vp_y * tile_h
            px_start_x = vp_x * tile_w
            dest_slice_y = slice(px_start_y, px_start_y + tile_h)
            dest_slice_x = slice(px_start_x, px_start_x + tile_w)
            output_image_array[dest_slice_y,
                               dest_slice_x, :] = tile_buffer[:, :, :]


@njit(cache=True, nogil=True)
def render_ground_items(
    output_image_array: np.ndarray,
    items_to_render: List[PyDict],
    tile_arrays: NumbaDict,
    intensity_map: np.ndarray,
    viewport_x: int,
    viewport_y: int,
    vp_h: int,
    vp_w: int,
    tile_w: int,
    tile_h: int,
) -> None:
    for item_data in items_to_render:
        if not (
            "x" in item_data
            and "y" in item_data
            and "glyph" in item_data
            and "color_fg_r" in item_data
            and "color_fg_g" in item_data
            and "color_fg_b" in item_data
        ):
            continue

        map_x = cast(nb_types.int64, item_data["x"])
        map_y = cast(nb_types.int64, item_data["y"])
        item_glyph_idx = cast(nb_types.int66, item_data["glyph"])
        color_r = cast(nb_types.uint8, item_data["color_fg_r"])
        color_g = cast(nb_types.uint8, item_data["color_fg_g"])
        color_b = cast(nb_types.uint8, item_data["color_fg_b"])

        if item_glyph_idx <= 0:
            continue

        cons_x = map_x - viewport_x
        cons_y = map_y - viewport_y
        if not (0 <= cons_y < vp_h and 0 <= cons_x < vp_w):
            continue

        if item_glyph_idx in tile_arrays:
            item_tile_rgba_array = tile_arrays[item_glyph_idx]
            if (
                item_tile_rgba_array.shape == (tile_h, tile_w, 4)
                and item_tile_rgba_array.shape != NJIT_SENTINEL_TILE_ARRAY_SHAPE
            ):
                if 0 <= cons_y < intensity_map.shape[0] and 0 <= cons_x < intensity_map.shape[1]:
                    item_intensity = intensity_map[cons_y, cons_x]
                else:
                    item_intensity = np.float32(1.0)

                base_item_fg_rgb = np.array(
                    [color_r, color_g, color_b], dtype=np.uint8)
                lit_item_fg_rgb = _interpolate_color_numba_vector(
                    base_item_fg_rgb, item_intensity
                )

                px_start_y = cons_y * tile_h
                px_start_x = cons_x * tile_w
                dest_slice_y = slice(px_start_y, px_start_y + tile_h)
                dest_slice_x = slice(px_start_x, px_start_x + tile_w)
                target_pixel_block = output_image_array[dest_slice_y, dest_slice_x]
                item_alpha_channel = item_tile_rgba_array[:, :, 3]
                item_draw_mask = item_alpha_channel > 10
                target_pixel_block[item_draw_mask, 0] = lit_item_fg_rgb[0]
                target_pixel_block[item_draw_mask, 1] = lit_item_fg_rgb[1]
                target_pixel_block[item_draw_mask, 2] = lit_item_fg_rgb[2]
                target_pixel_block[item_draw_mask,
                                   3] = item_alpha_channel[item_draw_mask]


@njit(cache=True, nogil=True)
def render_entities(
    output_image_array: np.ndarray,
    entities_to_render: List[PyDict],
    tile_arrays: NumbaDict,
    intensity_map: np.ndarray,
    viewport_x: int,
    viewport_y: int,
    vp_h: int,
    vp_w: int,
    tile_w: int,
    tile_h: int,
) -> None:
    for entity_data in entities_to_render:
        if not (
            "x" in entity_data
            and "y" in entity_data
            and "glyph" in entity_data
            and "color_fg_r" in entity_data
            and "color_fg_g" in entity_data
            and "color_fg_b" in entity_data
        ):
            continue

        map_ex = cast(nb_types.int64, entity_data["x"])
        map_ey = cast(nb_types.int64, entity_data["y"])
        glyph_idx = cast(nb_types.int66, entity_data["glyph"])
        color_r = cast(nb_types.uint8, entity_data["color_fg_r"])
        color_g = cast(nb_types.uint8, entity_data["color_fg_g"])
        color_b = cast(nb_types.uint8, entity_data["color_fg_b"])

        if glyph_idx <= 0:
            continue

        cons_ex = map_ex - viewport_x
        cons_ey = map_ey - viewport_y
        if not (0 <= cons_ey < vp_h and 0 <= cons_ex < vp_w):
            continue

        if glyph_idx in tile_arrays:
            entity_tile_rgba_array = tile_arrays[glyph_idx]
            if (
                entity_tile_rgba_array.shape == (tile_h, tile_w, 4)
                and entity_tile_rgba_array.shape != NJIT_SENTINEL_TILE_ARRAY_SHAPE
            ):
                if 0 <= cons_ey < intensity_map.shape[0] and 0 <= cons_ex < intensity_map.shape[1]:
                    e_intensity_f32 = intensity_map[cons_ey, cons_ex]
                else:
                    e_intensity_f32 = np.float32(1.0)

                base_fg_e_rgb = np.array(
                    [color_r, color_g, color_b], dtype=np.uint8)
                lit_fg_e_rgb = _interpolate_color_numba_vector(
                    base_fg_e_rgb, e_intensity_f32
                )

                px_start_y = cons_ey * tile_h
                px_start_x = cons_ex * tile_w
                slice(px_start_y, px_start_y + tile_h)
                slice(px_start_x, px_start_x + tile_w)
                target_pixel_block = output_image_array[
                    px_start_y: px_start_y + tile_h, px_start_x: px_start_x + tile_w
                ]
                entity_alpha_channel = entity_tile_rgba_array[:, :, 3]
                entity_draw_mask = entity_alpha_channel > 10
                target_pixel_block[entity_draw_mask, 0] = lit_fg_e_rgb[0]
                target_pixel_block[entity_draw_mask, 1] = lit_fg_e_rgb[1]
                target_pixel_block[entity_draw_mask, 2] = lit_fg_e_rgb[2]
                target_pixel_block[entity_draw_mask,
                                   3] = entity_alpha_channel[entity_draw_mask]

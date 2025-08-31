# engine/renderer.py
"""
Handles rendering the game state to a PIL Image, using
pre-calculated data
and optimized techniques.
"""
# Standard Library Imports
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict as PyDict, List, cast

# Third-party Imports
import numpy as np
import polars as pl
import structlog
from PIL import Image

# Numba for acceleration
try:
    from numba import float32, njit, uint8
    from numba.typed import Dict as NumbaDict # For type hinting Numba dict
    from numba import types as nb_types # For Numba types

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    NumbaDict = dict # Fallback type hint
    nb_types = object # Fallback type hint
    def njit(func=None, **options):
        if func:
            return func
        return lambda f: f
    uint8 = np.uint8
    float32 = np.float32

# Local Application Imports
# Ensure GameState and GameMap are importable
try:
    from game.game_state import GameState
    from game.world.game_map import GameMap
except ImportError:
     # Attempt fallback imports if needed
     try:
          from basicrl.game.game_state import GameState
          from basicrl.game.world.game_map import GameMap
     except ImportError:
          GameState = object # Define dummies if import fails
          GameMap = object
          structlog.get_logger().error("CRITICAL: Failed to import GameState or GameMap in renderer.")


if TYPE_CHECKING:
    pass

log = structlog.get_logger()

# Define a sentinel value for tile arrays in Numba to represent missing tiles
# Using a specific shape that is not possible for actual tile arrays (height, width, 4)
# This allows checking for its presence within Numba code.
NJIT_SENTINEL_TILE_ARRAY_SHAPE = (0, 0, 4)


# --- Numba Helper Functions ---
# Marked with cache=True for performance
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
    if radius_sq < 1e-6: # Handle zero radius case
        return np.float32(1.0) if dist_sq < 1e-6 else np.float32(0.0)
    if dist_sq >= radius_sq:
        return np.float32(0.0)

    dist = math.sqrt(dist_sq)
    radius = math.sqrt(radius_sq)
    falloff_ratio = dist / radius
    # Ensure 1.0 - falloff_ratio doesn't go below 0 due to floating point
    light_value = max(np.float32(0.0), np.float32(1.0) - falloff_ratio) ** falloff_power
    intensity = max(light_value, min_light_level)
    return max(np.float32(0.0), min(np.float32(1.0), intensity))


# Removed: _calculate_light_intensity_vectorized = np.vectorize(...)
# This object cannot be called from inside an njit function.
# The scalar function will be called in a loop instead.


# Marked with cache=True for performance
@njit("uint8[:](uint8[:], float32)", cache=True, fastmath=True, nogil=True)
def _interpolate_color_numba_vector(
    base_color: np.ndarray, intensity: np.float32
) -> np.ndarray:
    """Interpolates an RGB color towards black based on intensity."""
    # Ensure color is 3 elements
    if base_color.shape[0] < 3:
        # Should not happen with current data, but safety
        return np.zeros(3, dtype=uint8)

    intensity_clamped = max(np.float32(0.0), min(np.float32(1.0), intensity))
    result = np.empty(3, dtype=uint8)
    # Manual loop for Numba compatibility
    for i in range(3):
        # Clamp intermediate and final values to uint8 range [0, 255]
        result[i] = max(0, min(255, int(base_color[i] * intensity_clamped)))

    return result

# --- End Numba Helpers ---


@dataclass
class RenderConfig:
    """Configuration settings passed to the renderer."""
    show_height_vis: bool
    vis_max_diff: int
    vis_color_high_np: np.ndarray
    vis_color_mid_np: np.ndarray
    vis_color_low_np: np.ndarray
    vis_blend_factor: np.float32
    lighting_ambient: np.float32
    lighting_min_fov: np.float32
    lighting_falloff: np.float32
    fov_radius_sq: np.float32


def _prepare_base_layers(
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
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, tuple[int, int],
]:
    """Prepares base color and glyph index arrays for the viewport."""
    if not isinstance(game_map, GameMap) or GameMap is object:
        log.error("_prepare_base_layers called with invalid GameMap")
        # Return zero-size dummy arrays for graceful failure
        dummy_shape = (max(0, viewport_height), max(0, viewport_width)) # Ensure at least 0 size
        return ( np.zeros((*dummy_shape, 3), dtype=np.uint8), np.zeros((*dummy_shape, 3), dtype=np.uint8),
                 np.zeros(dummy_shape, dtype=np.uint16), np.zeros(dummy_shape, dtype=bool),
                 np.zeros(dummy_shape, dtype=bool), np.zeros(dummy_shape, dtype=np.int16),
                 np.zeros(dummy_shape, dtype=bool), dummy_shape )

    map_y_slice = slice(viewport_y, viewport_y + viewport_height)
    map_x_slice = slice(viewport_x, viewport_x + viewport_width)
    # Clamp slices to map bounds
    safe_y_slice = slice(max(0, map_y_slice.start), min(game_map.height, map_y_slice.stop))
    safe_x_slice = slice(max(0, map_x_slice.start), min(game_map.width, map_x_slice.stop))

    actual_vp_h = safe_y_slice.stop - safe_y_slice.start
    actual_vp_w = safe_x_slice.stop - safe_x_slice.start

    if actual_vp_h <= 0 or actual_vp_w <= 0:
        log.warning("Viewport slice resulted in zero or negative size.", vp_slice_y=map_y_slice, vp_slice_x=map_x_slice, map_shape=(game_map.height, game_map.width))
        dummy_shape = (max(0, actual_vp_h), max(0, actual_vp_w)) # Ensure at least 0 size
        return ( np.zeros((*dummy_shape, 3), dtype=np.uint8), np.zeros((*dummy_shape, 3), dtype=np.uint8),
                 np.zeros(dummy_shape, dtype=np.uint16), np.zeros(dummy_shape, dtype=bool),
                 np.zeros(dummy_shape, dtype=bool), np.zeros(dummy_shape, dtype=np.int16),
                 np.zeros(dummy_shape, dtype=bool), dummy_shape )

    # Get the viewport data from the map arrays using the safe slices
    map_visible_vp = game_map.visible[safe_y_slice, safe_x_slice]
    map_explored_vp = game_map.explored[safe_y_slice, safe_x_slice]
    map_tiles_vp = game_map.tiles[safe_y_slice, safe_x_slice]
    map_height_vp = game_map.height_map[safe_y_slice, safe_x_slice]
    vp_h, vp_w = map_visible_vp.shape # Actual dimensions of the viewport data

    # Masks for visible, explored-but-not-visible, and total drawn tiles
    visible_mask = map_visible_vp
    explored_mask = map_explored_vp & (~visible_mask) # Explored tiles not currently visible
    drawn_mask = visible_mask | explored_mask # All tiles that should be drawn

    # Initialize output arrays for the viewport region
    base_fg = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
    base_bg = np.zeros((vp_h, vp_w, 3), dtype=np.uint8)
    glyph_indices = np.zeros((vp_h, vp_w), dtype=np.uint16)

    # Check if render data arrays are valid before indexing
    render_data_valid = (
         tile_fg_colors is not None and tile_bg_colors is not None and
         tile_indices_render is not None and max_defined_tile_id >= 0 and
         len(tile_fg_colors) > max_defined_tile_id and
         len(tile_bg_colors) > max_defined_tile_id and
         len(tile_indices_render) > max_defined_tile_id
    )

    if not render_data_valid:
         log.error("Render data arrays invalid in _prepare_base_layers")
         # If data is invalid, we can't look up colors/glyphs, so nothing is drawn
         drawn_mask.fill(False)
    elif np.any(drawn_mask): # Only proceed if there are tiles to potentially draw
        try:
            # Get the tile IDs for the cells that are marked to be drawn
            tile_ids_in_vp_raw = map_tiles_vp[drawn_mask]
            # Ensure tile IDs are within the valid range of loaded tile data
            valid_tile_ids_in_vp = np.clip(tile_ids_in_vp_raw, 0, max_defined_tile_id)

            # Assign foreground, background colors, and glyph indices using the valid IDs
            # This uses NumPy advanced indexing, which is efficient
            base_fg[drawn_mask] = tile_fg_colors[valid_tile_ids_in_iel_ids_in_vp]
            base_bg[drawn_mask] = tile_bg_colors[valid_tile_ids_in_vp]
            glyph_indices[drawn_mask] = tile_indices_render[valid_tile_ids_in_vp]
        except IndexError as e:
             # This might happen if max_defined_tile_id is wrong or arrays are malformed
             log.error("IndexError during color/glyph assignment in _prepare_base_layers", error=str(e), exc_info=True)
             # On error, invalidate the drawn mask for the affected area or entirely
             drawn_mask.fill(False) # Simpler to just clear the mask on error
        except Exception as e:
             # Catch any other unexpected errors during array indexing/assignment
             log.error("Unexpected error during color/glyph assignment in _prepare_base_layers", error=str(e), exc_info=True)
             drawn_mask.fill(False) # Clear the mask on error


    # Return all relevant intermediate data and the actual viewport dimensions
    return ( base_fg, base_bg, glyph_indices, visible_mask, drawn_mask,
             map_height_vp, map_visible_vp, (vp_h, vp_w) )


# Marked with cache=True for performance
@njit(cache=True, nogil=True)
def _calculate_lighting(
    base_fg: np.ndarray, base_bg: np.ndarray, visible_mask: np.ndarray,
    vp_h: int, vp_w: int, viewport_x: int, viewport_y: int,
    player_x: int, player_y: int, config_ambient: float32,
    config_min_fov: float32, config_falloff: float32, fov_radius_sq: float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates lighting intensity and applies it."""
    # Initialize intensity map with ambient light level for all tiles
    intensity_map = np.full((vp_h, vp_w), config_ambient, dtype=np.float32)

    # Only calculate dynamic light for visible tiles
    if np.any(visible_mask):
        # Create grid coordinates for the viewport region
        # Using np.where to get coordinates *only for visible pixels* is efficient in Numba
        visible_y_coords, visible_x_coords = np.where(visible_mask)

        # Calculate absolute map coordinates for these visible pixels
        map_abs_x_visible = visible_x_coords + viewport_x
        map_abs_y_visible = visible_y_coords + viewport_y

        # Calculate distance squared from the player position for visible pixels
        dx = map_abs_x_visible - player_x
        dy = map_abs_y_visible - player_y
        dist_sq_map_visible = (dx * dx + dy * dy).astype(np.float32)

        # Calculate light intensity for visible tiles by calling the scalar function in a loop.
        # Removed call to _calculate_light_intensity_vectorized
        visible_intensities = np.empty_like(dist_sq_map_visible, dtype=np.float32)
        # Manually apply the scalar Numba function element-wise
        for i in range(dist_sq_map_visible.shape[0]):
            visible_intensities[i] = _calculate_light_intensity_scalar(
                dist_sq_map_visible[i],
                fov_radius_sq,      # Scalar argument
                config_falloff,     # Scalar argument
                config_min_fov      # Scalar argument
            )

        # Update the intensity map *only for visible tiles* using the calculated intensities
        # --- CORRECTED: Use a loop for assignment instead of advanced indexing ---
        # NumbaTypeError: Using more than one non-scalar array index is unsupported.
        # Replace intensity_map[visible_y_coords, visible_x_coords] = visible_intensities
        for i in range(visible_y_coords.shape[0]):
            y = visible_y_coords[i]
            x = visible_x_coords[i]
            intensity_map[y, x] = visible_intensities[i]
        # --- END CORRECTED ---


    # Apply intensity by element-wise multiplication of colors (requires broadcasting intensity)
    # intensity_map has shape (vp_h, vp_w), base_fg/bg have shape (vp_h, vp_w, 3)
    # Broadcasting intensity_map to (vp_h, vp_w, 1) allows direct multiplication
    intensity_broadcast = intensity_map[..., None] # Add a new dimension

    # Apply lighting: color * intensity, then cast back to uint8, clamping values
    lit_fg = (base_fg.astype(np.float32) * intensity_broadcast).astype(np.uint8) # Cast to float for multiplication
    lit_bg = (base_bg.astype(np.float32) * intensity_broadcast).astype(np.uint8) # Cast to float for multiplication


    return lit_fg, lit_bg, intensity_map


# Marked with cache=True for performance
@njit(cache=True, nogil=True)
def _apply_height_visualization(
    lit_fg: np.ndarray, lit_bg: np.ndarray, drawn_mask: np.ndarray,
    map_height_vp: np.ndarray, player_height: int, config_show_vis: bool,
    config_max_diff: int, config_color_high: np.ndarray,
    config_color_mid: np.ndarray, config_color_low: np.ndarray,
    config_blend_factor: float32
) -> tuple[np.ndarray, np.ndarray]:
    """Applies height visualization tinting."""
    if not config_show_vis:
        return lit_fg, lit_bg

    # Work on copies to avoid modifying the input lit_fg/lit_bg in place
    final_fg = lit_fg.copy()
    final_bg = lit_bg.copy()

    # Calculate relative height difference from the player's height
    relative_height_vp = map_height_vp - player_height

    max_diff_f32 = np.float32(config_max_diff)
    blend = config_blend_factor

    # Mask for pixels that are drawn AND have a valid max difference for visualization
    # Note: Numba supports multi-dimensional boolean masks for *generating* coordinates with np.where
    drawn_and_valid_diff = drawn_mask & (max_diff_f32 > np.float32(0.0))

    # --- Apply tinting for areas higher than the player ---
    # Mask for drawn tiles that are higher than the player, within max_diff
    high_mask = ( drawn_and_valid_diff & (relative_height_vp > 0) & (relative_height_vp <= config_max_diff) )
    if np.any(high_mask):
        # --- CORRECTED: Loop over coordinates instead of using multi-dim boolean indexing ---
        high_y_coords, high_x_coords = np.where(high_mask)
        target_color_high_f32 = config_color_high.astype(np.float32)

        for i in range(high_y_coords.shape[0]):
            y = high_y_coords[i]
            x = high_x_coords[i]

            # Calculate interpolation factor 't' for this single tile [0.0, 1.0]
            # 0.0 at player height, 1.0 at max_diff above player
            t = np.float32(relative_height_vp[y, x]) / max_diff_f32 # Get scalar value
            # Calculate blend factor based on 't' and overall blend intensity for this tile
            blend_t = blend * t

            # Get current colors for this tile
            current_fg_f32 = final_fg[y, x].astype(np.float32)
            current_bg_f32 = final_bg[y, x].astype(np.float32)

            # Perform linear interpolation: color * (1-blend_t) + target_color * blend_t
            # Assign blended color back to the corresponding pixel in the output arrays
            final_fg[y, x] = np.clip( current_fg_f32 * (np.float32(1.0) - blend_t) + target_color_high_f32 * blend_t, 0, 255 ).astype(np.uint8)
            final_bg[y, x] = np.clip( current_bg_f32 * (np.float32(1.0) - blend_t) + target_color_high_f32 * blend_t, 0, 255 ).astype(np.uint8)
        # --- END CORRECTED ---


    # --- Apply tinting for areas lower than the player ---
    # Mask for drawn tiles that are lower than the player, within max_diff
    low_mask = ( drawn_and_valid_diff & (relative_height_vp < 0) & (relative_height_vp >= -config_max_diff) )
    if np.any(low_mask):
         # --- CORRECTED: Loop over coordinates instead of using multi-dim boolean indexing ---
        low_y_coords, low_x_coords = np.where(low_mask)
        target_color_low_f32 = config_color_low.astype(np.float32)

        for i in range(low_y_coords.shape[0]):
            y = low_y_coords[i]
            x = low_x_coords[i]

            # Calculate interpolation factor 't' for low areas [0.0, 1.0]
            # 0.0 at player height, 1.0 at max_diff below player
            # Use absolute relative height for distance calculation for this single tile
            t = np.float32(np.abs(relative_height_vp[y, x])) / max_diff_f32 # Get scalar value
            # Calculate blend factor based on 't' and overall blend intensity for this tile
            blend_t = blend * t

            # Get current colors for this tile
            current_fg_f32 = final_fg[y, x].astype(np.float32)
            current_bg_f32 = final_bg[y, x].astype(np.float32)

            # Perform linear interpolation
            # Assign blended color back to the corresponding pixel in the output arrays
            final_fg[y, x] = np.clip( current_fg_f32 * (np.float32(1.0) - blend_t) + target_color_low_f32 * blend_t, 0, 255 ).astype(np.uint8)
            final_bg[y, x] = np.clip( current_bg_f32 * (np.float32(1.0) - blend_t) + target_color_low_f32 * blend_t, 0, 255 ).astype(np.uint8)
        # --- END CORRECTED ---


    # Note: Tiles exactly at player height (relative_height_vp == 0) or outside the +/- max_diff range are not tinted by this logic.
    # If a 'mid' color blend is desired, that would require additional masking/interpolation steps.

    return final_fg, final_bg

# Marked with cache=True for performance
# This function draws the map tiles onto the pre-filled background buffer.
# It needs to access the tile image data, which is passed as a Numba-typed dictionary.
# --- MODIFIED: Replace .get() with 'in' and [] ---
@njit(cache=True, nogil=True)
def _render_map_tiles(
    output_image_array: np.ndarray, # The final pixel buffer (h*tile_h, w*tile_w, 4)
    glyph_indices: np.ndarray, # Array of glyph IDs for each tile in the viewport (vp_h, vp_w)
    drawn_mask: np.ndarray, # Boolean mask indicating which tiles in the viewport should be drawn (vp_h, vp_w)
    final_fg: np.ndarray, # Final foreground colors after lighting/effects (vp_h, vp_w, 3)
    final_bg: np.ndarray, # Final background colors after lighting/effects (vp_h, vp_w, 3) - Note: BG already in output buffer
    tile_arrays: NumbaDict, # Numba Dict {tile_index: np.ndarray[tile_h, tile_w, 4]}
    vp_h: int, # Viewport height in tiles
    vp_w: int, # Viewport width in tiles
    tile_h: int, # Height of a single tile in pixels
    tile_w: int, # Width of a single tile in pixels
) -> None:
    # Create a temporary buffer for rendering a single tile
    tile_buffer = np.empty((tile_h, tile_w, 4), dtype=uint8)

    # Iterate over each tile position in the viewport
    for vp_y in range(vp_h):
        for vp_x in range(vp_w):
            # Only process tiles that are marked to be drawn
            if not drawn_mask[vp_y, vp_x]:
                continue

            # The background color for this tile is already set in the output_image_array
            # when the buffer was initialized.

            # Get the glyph index for this tile
            tile_glyph_idx = glyph_indices[vp_y, vp_x]

            # --- Numba-safe tile array access ---
            # Check if the glyph index exists in the tile_arrays dictionary
            if tile_glyph_idx in tile_arrays:
                tile_rgba_array = tile_arrays[tile_glyph_idx]

                # Check if the retrieved array is a valid tile array (not the sentinel)
                # and has the expected dimensions
                if (
                    tile_rgba_array.shape == (tile_h, tile_w, 4)
                    and tile_rgba_array.shape != NJIT_SENTINEL_TILE_ARRAY_SHAPE
                ):
                    # Get the alpha channel of the glyph
                    glyph_alpha_channel = tile_rgba_array[:, :, 3]

                    # Create a mask for pixels in the glyph that are mostly opaque
                    glyph_draw_mask_2d = glyph_alpha_channel > 10 # Threshold alpha

                    # Find the coordinates of opaque pixels within the tile
                    mask_rows, mask_cols = np.where(glyph_draw_mask_2d)

                    # Get the final foreground color for this tile
                    tile_fg_color_rgb = final_fg[vp_y, vp_x]

                    # Manually draw the opaque pixels of the glyph onto the buffer
                    # This loop copies the foreground color and alpha for the visible pixels
                    for i in range(mask_rows.shape[0]):
                        r, c = mask_rows[i], mask_cols[i] # Relative pixel coords within the tile
                        # Ensure pixel values are within uint8 range [0, 255]
                        tile_buffer[r, c, 0] = max(0, min(255, tile_fg_color_rgb[0]))
                        tile_buffer[r, c, 1] = max(0, min(255, tile_fg_color_rgb[1]))
                        tile_buffer[r, c, 2] = max(0, min(255, tile_fg_color_rgb[2]))
                        # Keep the original alpha from the glyph
                        tile_buffer[r, c, 3] = glyph_alpha_channel[r, c]
                else:
                    # If tile array is missing or invalid, fill tile buffer with transparent BG color
                    # This case should ideally not be reached if BG is filled first,
                    # but as a fallback, ensure tile_buffer is clear before copying.
                    tile_buffer[:, :, :] = 0 # Fill with transparent black
                    # Copy the final background color to the buffer as a fallback if glyph is invalid
                    tile_bg_color_rgb = final_bg[vp_y, vp_x]
                    tile_buffer[:, :, 0] = max(0, min(255, tile_bg_color_rgb[0]))
                    tile_buffer[:, :, 1] = max(0, min(255, tile_bg_color_rgb[1]))
                    tile_buffer[:, :, 2] = max(0, min(255, tile_bg_color_rgb[2]))
                    tile_buffer[:, :, 3] = 255 # Fully opaque background if glyph failed
            else:
                 # If glyph index is not in the dictionary, the tile array is missing.
                 # Fill the tile buffer with the final background color for this tile.
                 tile_bg_color_rgb = final_bg[vp_y, vp_x]
                 tile_buffer[:, :, 0] = max(0, min(255, tile_bg_color_rgb[0]))
                 tile_buffer[:, :, 1] = max(0, min(255, tile_bg_color_rgb[1]))
                 tile_buffer[:, :, 2] = max(0, min(255, tile_bg_color_rgb[2]))
                 tile_buffer[:, :, 3] = 255 # Fully opaque


            # Calculate the pixel coordinates for the top-left corner of the tile in the output buffer
            px_start_y = vp_y * tile_h
            px_start_x = vp_x * tile_w

            # Define the slice in the output buffer where this tile will be drawn
            dest_slice_y = slice(px_start_y, px_start_y + tile_h)
            dest_slice_x = slice(px_start_x, px_start_x + tile_w)

            # Copy the contents of the tile buffer to the corresponding location in the output image array
            output_image_array[dest_slice_y, dest_slice_x, :] = tile_buffer[:, :, :]


# Marked with cache=True for performance
# This function draws ground items. It also needs to access the tile data via the Numba dictionary.
# --- MODIFIED: Replace .get() with 'in' and [] ---
@njit(cache=True, nogil=True)
def _render_ground_items(
    output_image_array: np.ndarray, # The pixel buffer
    items_to_render: List[PyDict], # List of dictionaries (Numba supports List/Dict of supported types)
    tile_arrays: NumbaDict, # Numba Dict {tile_index: np.ndarray[tile_h, tile_w, 4]}
    intensity_map: np.ndarray, # Light intensity map (vp_h, vp_w)
    viewport_x: int, # Viewport X offset in map tiles
    viewport_y: int, # Viewport Y offset in map tiles
    vp_h: int, # Viewport height in tiles
    vp_w: int, # Viewport width in tiles
    tile_w: int, # Tile width in pixels
    tile_h: int, # Tile height in pixels
) -> None:
    """Draws ground items onto the output buffer."""
    # Iterate over the list of item data dictionaries
    for item_data in items_to_render:
        # Safely get item data, checking for expected keys and types within Numba
        # Accessing dictionary items by key is supported
        # Check if required keys exist and values are of expected types (basic checks)
        if not ('x' in item_data and 'y' in item_data and 'glyph' in item_data and
                'color_fg_r' in item_data and 'color_fg_g' in item_data and 'color_fg_b' in item_data):
                continue # Skip item if essential data is missing

        # Cast values to expected types
        map_x = cast(nb_types.int64, item_data['x'])
        map_y = cast(nb_types.int64, item_data['y'])
        item_glyph_idx = cast(nb_types.int66, item_data['glyph']) # Use int66 to be safe, or specific UInt16 if guaranteed
        color_r = cast(nb_types.uint8, item_data['color_fg_r'])
        color_g = cast(nb_types.uint8, item_data['color_fg_g'])
        color_b = cast(nb_types.uint8, item_data['color_fg_b'])

        # Check for valid glyph index (must be > 0 and a defined tile)
        if item_glyph_idx <= 0:
             continue

        # Convert map coordinates to console/viewport coordinates
        cons_x = map_x - viewport_x
        cons_y = map_y - viewport_y

        # Check if the item is within the current viewport bounds
        if not (0 <= cons_y < vp_h and 0 <= cons_x < vp_w):
            continue

        # --- Numba-safe tile array access for the item glyph ---
        # Replace .get() with 'in' check and direct indexing []
        if item_glyph_idx in tile_arrays:
             item_tile_rgba_array = tile_arrays[item_glyph_idx]

             # Check if the retrieved array is a valid tile array (not the sentinel)
             # and has the expected dimensions
             if (
                 item_tile_rgba_array.shape == (tile_h, tile_w, 4)
                 and item_tile_rgba_array.shape != NJIT_SENTINEL_TILE_ARRAY_SHAPE
             ):

                 # Get the lighting intensity for the tile the item is on
                 # Ensure indexing is within bounds of intensity_map
                 if 0 <= cons_y < intensity_map.shape[0] and 0 <= cons_x < intensity_map.shape[1]:
                      item_intensity = intensity_map[cons_y, cons_x]
                 else:
                      # Fallback intensity if console coords somehow invalid for intensity map
                      item_intensity = np.float32(1.0) # Assume full light

                 # Calculate the lit foreground color for the item
                 base_item_fg_rgb = np.array([color_r, color_g, color_b], dtype=np.uint8)
                 lit_item_fg_rgb = _interpolate_color_numba_vector( base_item_fg_rgb, item_intensity )

                 # Calculate the pixel coordinates for the top-left corner of the tile
                 px_start_y = cons_y * tile_h
                 px_start_x = cons_x * tile_w

                 # Define the slice in the output buffer
                 dest_slice_y = slice(px_start_y, px_start_y + tile_h)
                 dest_slice_x = slice(px_start_x, px_start_x + tile_w)

                 # Get the pixel block in the output array corresponding to this tile
                 target_pixel_block = output_image_array[dest_slice_y, dest_slice_x]

                 # Get the alpha channel of the item glyph
                 item_alpha_channel = item_tile_rgba_array[:, :, 3]

                 # Create a mask for opaque pixels in the item glyph
                 item_draw_mask = item_alpha_channel > 10

                 # Manually draw the opaque pixels of the item glyph over the existing tile
                 # Iterate over the opaque pixels and set the RGB color and Alpha
                 # Clamping is handled by the interpolation function for RGB
                 # Alpha is copied directly
                 target_pixel_block[item_draw_mask, 0] = lit_item_fg_rgb[0]
                 target_pixel_block[item_draw_mask, 1] = lit_item_fg_rgb[1]
                 target_pixel_block[item_draw_mask, 2] = lit_item_fg_rgb[2]
                 target_pixel_block[item_draw_mask, 3] = item_alpha_channel[item_draw_mask]

        # If item_glyph_idx is NOT in tile_arrays or the retrieved array is invalid,
        # the item simply won't be drawn over the tile background. This is correct behavior.


# Marked with cache=True for performance
# This function draws entities. It also needs to access the tile data via the Numba dictionary.
# --- MODIFIED: Replace .get() with 'in' and [] ---
@njit(cache=True, nogil=True)
def _render_entities(
    output_image_array: np.ndarray, # The pixel buffer
    entities_to_render: List[PyDict], # List of dictionaries (Numba supports List/Dict of supported types)
    tile_arrays: NumbaDict, # Numba Dict {tile_index: np.ndarray[tile_h, tile_w, 4]}
    intensity_map: np.ndarray, # Light intensity map (vp_h, vp_w)
    viewport_x: int, # Viewport X offset in map tiles
    viewport_y: int, # Viewport Y offset in map tiles
    vp_h: int, # Viewport height in tiles
    vp_w: int, # Viewport width in tiles
    tile_w: int, # Tile width in pixels
    tile_h: int, # Tile height in pixels
) -> None:
    """Draws entities onto the output buffer."""
    # Iterate over the list of entity data dictionaries
    for entity_data in entities_to_render:
        # Safely get entity data, checking for expected keys and types within Numba
        if not ('x' in entity_data and 'y' in entity_data and 'glyph' in entity_data and
                'color_fg_r' in entity_data and 'color_fg_g' in entity_data and 'color_fg_b' in entity_data):
                continue # Skip entity if essential data is missing

        # Cast values to expected types
        map_ex = cast(nb_types.int64, entity_data['x'])
        map_ey = cast(nb_types.int64, entity_data['y'])
        glyph_idx = cast(nb_types.int66, entity_data['glyph']) # Use int66 to be safe, or specific UInt16 if guaranteed
        color_r = cast(nb_types.uint8, entity_data['color_fg_r'])
        color_g = cast(nb_types.uint8, entity_data['color_fg_g'])
        color_b = cast(nb_types.uint8, entity_data['color_fg_b'])


        # Check for valid glyph index (must be > 0 and a defined tile)
        if glyph_idx <= 0:
             continue

        # Convert map coordinates to console/viewport coordinates
        cons_ex = map_ex - viewport_x
        cons_ey = map_ey - viewport_y

        # Check if the entity is within the current viewport bounds
        if not (0 <= cons_ey < vp_h and 0 <= cons_ex < vp_w):
            continue

        # --- Numba-safe tile array access for the entity glyph ---
        # Replace .get() with 'in' check and direct indexing []
        if glyph_idx in tile_arrays:
            entity_tile_rgba_array = tile_arrays[glyph_idx]

            # Check if the retrieved array is a valid tile array (not the sentinel)
            # and has the expected dimensions
            if (
                entity_tile_rgba_array.shape == (tile_h, tile_w, 4)
                and entity_tile_rgba_array.shape != NJIT_SENTINEL_TILE_ARRAY_SHAPE
            ):

                # Get the lighting intensity for the tile the entity is on
                # Ensure indexing is within bounds of intensity_map
                if 0 <= cons_ey < intensity_map.shape[0] and 0 <= cons_ex < intensity_map.shape[1]:
                     e_intensity_f32 = intensity_map[cons_ey, cons_ex]
                else:
                     # Fallback intensity if console coords somehow invalid for intensity map
                     e_intensity_f32 = np.float32(1.0) # Assume full light


                # Calculate the lit foreground color for the entity
                base_fg_e_rgb = np.array( [ color_r, color_g, color_b ], dtype=np.uint8 )
                lit_fg_e_rgb = _interpolate_color_numba_vector(base_fg_e_rgb, e_intensity_f32)

                # Calculate the pixel coordinates for the top-left corner of the tile
                px_start_y = cons_ey * tile_h
                px_start_x = cons_ex * tile_w

                # Define the slice in the output buffer
                dest_slice_y = slice(px_start_y, px_start_y + tile_h)
                dest_slice_x = slice(px_start_x, px_start_x + tile_w)

                # Get the pixel block in the output array corresponding to this tile
                 # Use direct indexing [cons_ey, cons_ex] to get the tile location, then slice within that tile
                 target_pixel_block = output_image_array[px_start_y : px_start_y + tile_h, px_start_x : px_start_x + tile_w]

                # Get the alpha channel of the entity glyph
                entity_alpha_channel = entity_tile_rgba_array[:, :, 3]

                # Create a mask for opaque pixels in the entity glyph
                entity_draw_mask = entity_alpha_channel > 10

                # Manually draw the opaque pixels of the entity glyph over the existing tile/item
                # Iterate over the opaque pixels and set the RGB color and Alpha
                # Clamping is handled by the interpolation function for RGB
                # Alpha is copied directly
                target_pixel_block[entity_draw_mask, 0] = lit_fg_e_rgb[0]
                target_pixel_block[entity_draw_mask, 1] = lit_fg_e_rgb[1]
                target_pixel_block[entity_draw_mask, 2] = lit_fg_e_rgb[2]
                target_pixel_block[entity_draw_mask, 3] = entity_alpha_channel[entity_draw_mask]

        # If glyph_idx is NOT in tile_arrays or the retrieved array is invalid,
        # the entity simply won't be drawn over the tile/item background. This is correct behavior.


# --- Main Rendering Function ---
def render_viewport(
    game_state: GameState,
    tile_arrays: NumbaDict | PyDict, # Can be Python dict if Numba not available
    tile_fg_colors: np.ndarray,
    tile_bg_colors: np.ndarray,
    tile_indices_render: np.ndarray,
    max_defined_tile_id: int,
    tile_w: int,
    tile_h: int,
    viewport_x: int,
    viewport_y: int,
    viewport_width: int,
    viewport_height: int,
    coord_arrays: PyDict[str, np.ndarray], # These are NumPy arrays for indexing
    render_config: RenderConfig,
) -> Image.Image | None:
    """
    Renders the visible portion of the game world to a PIL Image.
    Orchestrates the rendering pipeline, calling Numba helpers where applicable.
    """
    log.debug("render_viewport called",
              vp_x=viewport_x, vp_y=viewport_y, vp_w=viewport_width, vp_h=viewport_height,
              tile_dims=f"{tile_w}x{tile_h}", max_tile_id=max_defined_tile_id,
              tile_arrays_type=type(tile_arrays), tile_arrays_len=len(tile_arrays) if tile_arrays is not None else 'None',
              fg_colors_shape=tile_fg_colors.shape if tile_fg_colors is not None else 'None',
              bg_colors_shape=tile_bg_colors.shape if tile_bg_colors is not None else 'None',
              indices_shape=tile_indices_render.shape if tile_indices_render is not None else 'None',
              coord_keys=list(coord_arrays.keys()) if coord_arrays is not None else 'None',
              render_config_details=render_config)


    # --- Input Validation ---
    # Basic checks to prevent crashes early
    if not isinstance(game_state, GameState) or GameState is object:
        log.error("render_viewport called with invalid GameState object")
        # Return a dummy error image with appropriate size
        pw = max(1, viewport_width * max(1, tile_w));
        ph = max(1, viewport_height * max(1, tile_h))
        return Image.new("RGBA", (pw, ph), (255,0,0,255))

    gs = game_state; gm = gs.game_map # Shortcuts

    if tile_w <= 0 or tile_h <= 0 or tile_arrays is None or max_defined_tile_id < 0:
        log.warning( "Cannot render: Invalid params/cache", tile_w=tile_w, tile_h=tile_h,
                     has_arrays=(tile_arrays is not None), max_id=max_defined_tile_id )
        # Return a blank image based on nominal viewport size
        pw = max(1, viewport_width * max(1, tile_w));
        ph = max(1, viewport_height * max(1, tile_h))
        return Image.new("RGBA", (pw, ph), (0, 0, 0, 255))

    player_pos = gs.player_position
    if player_pos is None:
        log.warning("Cannot render: Player position not found")
        # Return a blank image based on nominal viewport size
        pw = max(1, viewport_width * max(1, tile_w));
        ph = max(1, viewport_height * max(1, tile_h))
        return Image.new("RGBA", (pw, ph), (0, 0, 0, 255))

    player_x, player_y = player_pos
    # Get player height safely
    player_height = 0 # Default to 0 if player pos is somehow out of bounds for height map
    try:
        if gm.in_bounds(player_x, player_y):
            player_height = int(gm.height_map[player_y, player_x])
        else:
             # This case should ideally be caught by player_pos check, but safety log
             log.error("Player position out of map bounds during height lookup", player_pos=player_pos, map_shape=(gm.height, gm.width));
             player_height = 0
    except IndexError:
         # Should not happen if in_bounds check passed, but defensive
         log.error("IndexError getting player height in renderer", player_pos=player_pos, map_shape=(gm.height, gm.width));
         player_height = 0
    except Exception as e:
         # Catch any other unexpected errors during height lookup
         log.error("Unexpected error getting player height in renderer", error=str(e), exc_info=True, player_pos=player_pos);
         player_height = 0


    # --- Prepare Base Data (Calls Python helper, returns NumPy arrays) ---
    try:
        ( base_fg, base_bg, glyph_indices, visible_mask, drawn_mask,
          map_height_vp, map_visible_vp, (vp_h, vp_w),
        ) = _prepare_base_layers(
            gm, viewport_x, viewport_y, viewport_width, viewport_height,
            max_defined_tile_id, tile_fg_colors, tile_bg_colors, tile_indices_render
        )
        # Check if the prepared viewport data is valid
        if vp_h <= 0 or vp_w <= 0:
            log.warning("Calculated actual viewport dimensions are zero or negative after slicing.", vp_h=vp_h, vp_w=vp_w)
            # Return a minimal blank image
            return Image.new("RGBA", (max(1, viewport_width * max(1, tile_w)), max(1, viewport_height * max(1, tile_h))), (0,0,0,0))
    except Exception as e:
        log.error("Error during _prepare_base_layers call or unpacking results", error=str(e), exc_info=True)
        # Return a dummy error image based on nominal viewport size
        pw = max(1, viewport_width * max(1, tile_w));
        ph = max(1, viewport_height * max(1, tile_h))
        return Image.new("RGBA", (pw, ph), (255,0,0,255))

    # --- Apply Lighting and Height Visualization (Calls Numba helpers) ---
    # Pass config values as individual Numba-compatible types
    # Call _calculate_lighting, which now performs the element-wise calls internally
    lit_fg, lit_bg, intensity_map = _calculate_lighting(
        base_fg, base_bg, visible_mask, vp_h, vp_w, viewport_x, viewport_y,
        player_x, player_y, render_config.lighting_ambient,
        render_config.lighting_min_fov, render_config.lighting_falloff,
        render_config.fov_radius_sq
    )

    final_fg, final_bg = _apply_height_visualization(
        lit_fg, lit_bg, drawn_mask, map_height_vp, player_height,
        render_config.show_height_vis, render_config.vis_max_diff,
        render_config.vis_color_high_np, render_config.vis_color_mid_np,
        render_config.vis_color_low_np, render_config.vis_blend_factor
    )

    # --- Prepare Output Buffer (NumPy array for PIL Image) ---
    output_pixel_h = vp_h * tile_h;
    output_pixel_w = vp_w * tile_w
    if output_pixel_h <= 0 or output_pixel_w <= 0:
        log.warning("Calculated output pixel size is zero or negative", w=output_pixel_w, h=output_pixel_h)
        # Return a minimal blank image
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    output_image_array = np.zeros((output_pixel_h, output_pixel_w, 4), dtype=np.uint8)


    # --- Fill Background Layer ---
    # Use the pre-calculated coordinate arrays to map pixels to viewport tile colors
    try:
        if not coord_arrays or "tile_coord_y" not in coord_arrays or "tile_coord_x" not in coord_arrays:
             # This should be handled by WindowManager creating the cache, but safety check
             raise KeyError("Coordinate arrays missing or invalid.")
        tile_coord_y = coord_arrays["tile_coord_y"];
        tile_coord_x = coord_arrays["tile_coord_x"]
        # Validate shape of coordinate arrays
        if tile_coord_y.shape != (output_pixel_h, output_pixel_w) or tile_coord_x.shape != (output_pixel_h, output_pixel_w):
             raise ValueError(f"Coordinate array shape mismatch. Expected:{(output_pixel_h, output_pixel_w)}, Got: y{tile_coord_y.shape},x{tile_coord_x.shape}")
        # Validate shape of background color array
        if final_bg.shape != (vp_h, vp_w, 3):
             raise ValueError(f"Background color shape mismatch. Expected:{(vp_h, vp_w, 3)}, Got:{final_bg.shape}")

        # Use NumPy advanced indexing to fill the background based on the coordinate maps
        # This efficiently maps the final background colors for each tile to all pixels within that tile
        output_image_array[:, :, :3] = final_bg[tile_coord_y, tile_coord_x]
        output_image_array[:, :, 3] = 255 # Set alpha channel to fully opaque for background
    except (KeyError, IndexError, ValueError) as e:
        log.error( "Error preparing output buffer with background (coordinate mapping)", error=str(e), exc_info=True,
                   final_bg_shape=final_bg.shape if final_bg is not None else 'None', output_shape=output_image_array.shape,
                   coord_y_shape=coord_arrays.get("tile_coord_y", np.array([])).shape, coord_x_shape=coord_arrays.get("tile_coord_x", np.array([])).shape,
                   vp_dims=(vp_h, vp_w) )
        # Return a purple error image indicating a background fill issue
        return Image.new("RGBA", (max(1, output_pixel_w), max(1, output_pixel_h)), (100, 0, 100, 255))

    except Exception as e:
         # Catch any other unexpected errors during background fill
         log.error("Unexpected error during background fill", error=str(e), exc_info=True)
         return Image.new("RGBA", (max(1, output_pixel_w), max(1, output_pixel_h)), (100, 0, 100, 255))


    # --- Render Map Tiles (Draws glyphs over background, calls Numba helper) ---
    try:
        # Only call the Numba tile rendering function if there are tile arrays and tiles to draw
        if not tile_arrays:
            log.warning("_render_map_tiles skipped: tile_arrays dictionary is empty.")
        elif np.sum(drawn_mask) == 0:
            log.debug("_render_map_tiles skipped: drawn_mask is all False (no tiles to draw).")
        elif _NUMBA_AVAILABLE and isinstance(tile_arrays, NumbaDict):
             # Pass arguments to the Numba compiled function
             _render_map_tiles( output_image_array, glyph_indices, drawn_mask, final_fg, final_bg, # final_bg passed for potential fallback
                                tile_arrays, vp_h, vp_w, tile_w, tile_h )
        else:
            # Fallback if Numba is not available or tile_arrays is not a NumbaDict
            log.warning("_render_map_tiles skipped: Numba not available or tile_arrays is not NumbaDict.")
            # A Python fallback for rendering tiles could be implemented here if needed,
            # but would be significantly slower. For now, we just log and skip.

    except Exception as e:
        # Catch errors specifically during the Numba tile rendering call
        log.error("Error during _render_map_tiles Numba call", error=str(e), exc_info=True)
        # Continue rendering other layers, but tiles might be missing/incorrect


    # --- Prepare Ground Item Data (Query using Polars) ---
    items_to_render_list: List[PyDict] = []
    # Only query items if there is any visible area on the map
    if np.any(map_visible_vp):
         try:
             # Define the absolute map bounds for the current viewport
             vp_x_min, vp_y_min = viewport_x, viewport_y;
             vp_x_max, vp_y_max = viewport_x + vp_w, viewport_y + vp_h

             # Filter items: active, on the ground, within viewport bounds, and have a valid glyph
             items_in_vp_df = gs.item_registry.items_df.filter(
                 (pl.col("x") >= vp_x_min) & (pl.col("x") < vp_x_max) &
                 (pl.col("y") >= vp_y_min) & (pl.col("y") < vp_y_max) &
                 (pl.col("location_type") == "ground") & pl.col("is_active") &
                 (pl.col("glyph") > 0)
             )

             if items_in_vp_df.height > 0:
                 # For now, only render the "top" item at each stack location
                 # group_by(["x", "y"]).last() selects the last item added/processed at each location
                 top_items_df = items_in_vp_df.group_by(["x", "y"]).last()

                 # Convert absolute map item coordinates to relative viewport coordinates
                 item_coords_x_abs = top_items_df["x"].to_numpy();
                 item_coords_y_abs = top_items_df["y"].to_numpy()
                 item_coords_x_vp = item_coords_x_abs - viewport_x;
                 item_coords_y_vp = item_coords_y_abs - viewport_y

                 # Create a mask to check if the item's console coordinates are within the *actual* viewport dimensions
                 valid_indices_mask = (item_coords_x_vp >= 0) & (item_coords_x_vp < vp_w) & (item_coords_y_vp >= 0) & (item_coords_y_vp < vp_h)

                 # Create a mask to check if the item's location is visible based on the viewport visibility mask
                 visible_item_mask = np.zeros(len(item_coords_x_vp), dtype=bool)
                 if np.any(valid_indices_mask):
                      # Get the relative viewport coordinates for the valid items
                      valid_item_rel_y = valid_item_rel_y_np = item_coords_y_vp[valid_indices_mask].astype(np.intp); # Use .astype(np.intp) for indexing
                      valid_item_rel_x = valid_item_rel_x_np = item_coords_x_vp[valid_indices_mask].astype(np.intp); # Use .astype(np.intp) for indexing


                      # Ensure indexing into map_visible_vp is safe
                      if map_visible_vp.shape == (vp_h, vp_w):
                           # Check visibility using the viewport visibility mask
                           visible_item_mask[valid_indices_mask] = map_visible_vp[valid_item_rel_y_np, valid_item_rel_x_np]
                      else:
                          log.warning("map_visible_vp shape mismatch, cannot accurately check item visibility.", map_vis_shape=map_visible_vp.shape, expected=(vp_h, vp_w))
                          # Fallback: Assume visible if coordinates are valid for the viewport data
                          visible_item_mask[valid_indices_mask] = True # Less accurate, but prevents skipping all items


                 # Filter the top items dataframe to include only items that are within the viewport AND visible
                 visible_items_df = top_items_df.filter(pl.Series(values=visible_item_mask))

                 # Convert the filtered DataFrame to a list of dictionaries for use in Numba
                 if visible_items_df.height > 0:
                     # Select necessary columns for rendering (position, glyph, color)
                     # Note: item_id is included for potential debugging/logging within Numba, but not used for drawing
                     items_to_render_list = visible_items_df.select(
                         ["x", "y", "glyph", "color_fg_r", "color_fg_g", "color_fg_b", "item_id"]
                     ).to_dicts()

         except Exception as e:
             # Catch any errors during Polars query or data processing for items
             log.error("Error querying/processing ground items for rendering", error=str(e), exc_info=True)

    # --- Render Ground Items (Draws items over tiles, calls Numba helper) ---
    if items_to_render_list:
         try:
              # Only call the Numba item rendering function if Numba is available and tile_arrays is a NumbaDict
              if _NUMBA_AVAILABLE and isinstance(tile_arrays, NumbaDict):
                   _render_ground_items( output_image_array, items_to_render_list, tile_arrays, intensity_map,
                                        viewport_x, viewport_y, vp_h, vp_w, tile_w, tile_h )
              else:
                   # Fallback if Numba is not available or tile_arrays is not a NumbaDict
                   log.warning("_render_ground_items skipped: Numba not available or tile_arrays is not NumbaDict.")
                   # A Python fallback could be added here if needed

         except Exception as e:
             # Catch errors specifically during the Numba item rendering call
             log.error("Error during _render_ground_items Numba call", error=str(e), exc_info=True)


    # --- Prepare Entity Data (Query using Polars) ---
    entities_to_render_list: List[PyDict] = []
    if isinstance(gs.entity_registry, object) and gs.entity_registry is not object: # Check if dummy object was used
        try:
            # Filter active entities that are within the viewport bounds and have a valid glyph
            # No visibility check here yet, visibility is handled within the Numba rendering function for efficiency
            active_entities_df = gs.entity_registry.get_active_entities()
            if active_entities_df.height > 0:
                 entities_in_vp_bounds = active_entities_df.filter(
                     (pl.col("x") >= viewport_x) & (pl.col("x") < viewport_x + vp_w) &
                     (pl.col("y") >= vp_y_min) & (pl.col("y") < vp_y_max) & # Use vp_y_min/max from item query for consistency
                     (pl.col("glyph") > 0) # Only include entities with a valid glyph
                 )

                 # Convert the filtered DataFrame to a list of dictionaries for use in Numba
                 if entities_in_vp_bounds.height > 0:
                     # Select necessary columns for rendering (position, glyph, color)
                     entities_to_render_list = entities_in_vp_bounds.select(
                         ["entity_id", "x", "y", "glyph", "color_fg_r", "color_fg_g", "color_fg_b"]
                     ).to_dicts()

        except Exception as e:
            # Catch any errors during Polars query or data processing for entities
            log.error("Error querying/processing entities for rendering", error=str(e), exc_info=True)


    # --- Render Entities (Draws entities over items/tiles, calls Numba helper) ---
    if entities_to_render_list:
         try:
             # Only call the Numba entity rendering function if Numba is available and tile_arrays is a NumbaDict
              if _NUMBA_AVAILABLE and isinstance(tile_arrays, NumbaDict):
                 # Note: Visibility check for entities is done *inside* the Numba function
                 _render_entities( output_image_array, entities_to_render_list, tile_arrays, intensity_map,
                                  viewport_x, viewport_y, vp_h, vp_w, tile_w, tile_h )
              else:
                  # Fallback if Numba is not available or tile_arrays is not a NumbaDict
                   log.warning("_render_entities skipped: Numba not available or tile_arrays is not NumbaDict.")
                   # A Python fallback could be added here if needed

         except Exception as e:
              # Catch errors specifically during the Numba entity rendering call
             log.error("Error during _render_entities Numba call", error=str(e), exc_info=True)


    # --- Final Image Conversion ---
    # Convert the final NumPy pixel array to a PIL Image
    try:
        # Ensure output array has positive dimensions before creating Image
        if output_pixel_h > 0 and output_pixel_w > 0:
             final_image = Image.fromarray(output_image_array, "RGBA")
             return final_image
        else:
             log.warning("Output pixel dimensions were non-positive after processing.", w=output_pixel_w, h=output_pixel_h)
             # Return a minimal blank image
             return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    except Exception as e:
        # Catch errors during the final conversion step
        log.error( "Error converting final array to PIL Image", error=str(e), exc_info=True )
        # Return a dark grey error image
        return Image.new("RGBA", (max(1, output_pixel_w), max(1, output_pixel_h)), (50, 50, 50, 255))

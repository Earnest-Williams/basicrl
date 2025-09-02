# engine/renderer.py
"""
Handles rendering the game state to a PIL Image, using
pre-calculated data
and optimized techniques.
"""
# Standard Library Imports
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict as PyDict, List, cast

# Third-party Imports
import numpy as np
import polars as pl
import structlog
from PIL import Image

# Numba for acceleration
try:
    from numba.typed import Dict as NumbaDict  # For type hinting Numba dict
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    NumbaDict = dict  # Fallback type hint

# Local Application Imports
# Ensure GameState and GameMap are importable
try:
    from game.game_state import GameState
    from game.world.game_map import GameMap
    from game.world.fov import compute_light_color_array
except ImportError:
    # Attempt fallback imports if needed
    try:
        from basicrl.game.game_state import GameState
        from basicrl.game.world.game_map import GameMap
        from basicrl.game.world.fov import compute_light_color_array
    except ImportError:
        GameState = object  # Define dummies if import fails
        GameMap = object
        compute_light_color_array = lambda *args, **kwargs: None
        structlog.get_logger().error(
            "CRITICAL: Failed to import GameState or GameMap in renderer."
        )

# Require GameRNG; abort if unavailable
try:
    from game_rng import GameRNG
except ImportError:
    try:
        from basicrl.game_rng import GameRNG
    except ImportError as exc:
        structlog.get_logger().critical(
            "CRITICAL: Failed to import GameRNG in renderer."
        )
        raise SystemExit from exc


if TYPE_CHECKING:
    pass

log = structlog.get_logger()

from .render_base_layers import prepare_base_layers
from .render_lighting import (
    calculate_lighting,
    apply_height_visualization,
    apply_memory_fade,
)
from .render_entities import (
    render_map_tiles,
    render_ground_items,
    render_entities,
)


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
    enable_memory_fade: bool = True
    enable_colored_lights: bool = True
    memory_fade_color_np: np.ndarray = field(
        default_factory=lambda: np.array([128, 128, 128], dtype=np.uint8)
    )
    memory_fade_variance: np.float32 = np.float32(0.0)
    memory_noise_level: np.float32 = np.float32(0.0)


@dataclass
class ViewportParams:
    """Collects parameters required for viewport rendering."""
    viewport_x: int
    viewport_y: int
    viewport_width: int
    viewport_height: int
    tile_arrays: NumbaDict | PyDict
    tile_fg_colors: np.ndarray
    tile_bg_colors: np.ndarray
    tile_indices_render: np.ndarray
    max_defined_tile_id: int
    tile_w: int
    tile_h: int
    coord_arrays: PyDict[str, np.ndarray]

# --- Main Rendering Function ---
def render_viewport(
    game_state: GameState,
    viewport: ViewportParams,
    render_config: RenderConfig,
) -> Image.Image | None:
    """Render the visible portion of the game world to a PIL Image."""
    log.debug(
        "render_viewport called",
        vp_x=viewport.viewport_x,
        vp_y=viewport.viewport_y,
        vp_w=viewport.viewport_width,
        vp_h=viewport.viewport_height,
        tile_dims=f"{viewport.tile_w}x{viewport.tile_h}",
        max_tile_id=viewport.max_defined_tile_id,
        tile_arrays_type=type(viewport.tile_arrays),
        tile_arrays_len=len(viewport.tile_arrays)
        if viewport.tile_arrays is not None
        else "None",
        fg_colors_shape=viewport.tile_fg_colors.shape
        if viewport.tile_fg_colors is not None
        else "None",
        bg_colors_shape=viewport.tile_bg_colors.shape
        if viewport.tile_bg_colors is not None
        else "None",
        indices_shape=viewport.tile_indices_render.shape
        if viewport.tile_indices_render is not None
        else "None",
        coord_keys=list(viewport.coord_arrays.keys())
        if viewport.coord_arrays is not None
        else "None",
        render_config_details=render_config,
    )


    viewport_x = viewport.viewport_x
    viewport_y = viewport.viewport_y
    viewport_width = viewport.viewport_width
    viewport_height = viewport.viewport_height
    tile_w = viewport.tile_w
    tile_h = viewport.tile_h
    tile_arrays = viewport.tile_arrays
    tile_fg_colors = viewport.tile_fg_colors
    tile_bg_colors = viewport.tile_bg_colors
    tile_indices_render = viewport.tile_indices_render
    max_defined_tile_id = viewport.max_defined_tile_id
    coord_arrays = viewport.coord_arrays

    # --- Input Validation ---
    if not isinstance(game_state, GameState) or GameState is object:
        log.error("render_viewport called with invalid GameState object")
        pw = max(1, viewport_width * max(1, tile_w))
        ph = max(1, viewport_height * max(1, tile_h))
        return Image.new("RGBA", (pw, ph), (255, 0, 0, 255))

    gs = game_state
    gm = gs.game_map

    if tile_w <= 0 or tile_h <= 0 or tile_arrays is None or max_defined_tile_id < 0:
        log.warning(
            "Cannot render: Invalid params/cache",
            tile_w=tile_w,
            tile_h=tile_h,
            has_arrays=(tile_arrays is not None),
            max_id=max_defined_tile_id,
        )
        pw = max(1, viewport_width * max(1, tile_w))
        ph = max(1, viewport_height * max(1, tile_h))
        return Image.new("RGBA", (pw, ph), (0, 0, 0, 255))

    player_pos = gs.player_position
    if player_pos is None:
        log.warning("Cannot render: Player position not found")
        pw = max(1, viewport_width * max(1, tile_w))
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
            viewport_x,
            viewport_y,
            viewport_width,
            viewport_height,
            max_defined_tile_id,
            tile_fg_colors,
            tile_bg_colors,
            tile_indices_render,
        )
        # Check if the prepared viewport data is valid
        if vp_h <= 0 or vp_w <= 0:
            log.warning("Calculated actual viewport dimensions are zero or negative after slicing.", vp_h=vp_h, vp_w=vp_w)
            # Return a minimal blank image
            return Image.new("RGBA", (max(1, viewport_width * max(1, tile_w)), max(1, viewport_height * max(1, tile_h))), (0,0,0,0))
    except Exception as e:
        log.error("Error during prepare_base_layers call or unpacking results", error=str(e), exc_info=True)
        # Return a dummy error image based on nominal viewport size
        pw = max(1, viewport_width * max(1, tile_w));
        ph = max(1, viewport_height * max(1, tile_h))
        return Image.new("RGBA", (pw, ph), (255,0,0,255))

    # --- Apply Lighting and Height Visualization (Calls Numba helpers) ---
    # Pass config values as individual Numba-compatible types
    # Call calculate_lighting, which now performs the element-wise calls internally
    lit_fg, lit_bg, intensity_map = calculate_lighting(
        base_fg, base_bg, visible_mask, vp_h, vp_w, viewport_x, viewport_y,
        player_x, player_y, render_config.lighting_ambient,
        render_config.lighting_min_fov, render_config.lighting_falloff,
        render_config.fov_radius_sq
    )

    final_fg, final_bg = apply_height_visualization(
        lit_fg, lit_bg, drawn_mask, map_height_vp, player_height,
        render_config.show_height_vis, render_config.vis_max_diff,
        render_config.vis_color_high_np, render_config.vis_color_mid_np,
        render_config.vis_color_low_np, render_config.vis_blend_factor
    )

    # --- Apply colored lights ---
    if (
        render_config.enable_colored_lights
        and hasattr(gs, "light_sources")
        and len(gs.light_sources) > 0
    ):
        light_rgb_map = np.zeros((gm.height, gm.width, 3), dtype=np.float32)
        opaque_grid = ~gm.transparent
        for ls in gs.light_sources:
            try:
                origin_h = int(gm.height_map[ls.y, ls.x])
                compute_light_color_array(
                    (ls.x, ls.y),
                    ls.radius,
                    opaque_grid,
                    gm.height_map,
                    gm.ceiling_map,
                    origin_h,
                    light_rgb_map,
                    ls.color,
                )
            except Exception as e:
                log.error("Error computing light source", error=str(e))
        light_rgb_vp = light_rgb_map[
            viewport_y : viewport_y + vp_h, viewport_x : viewport_x + vp_w
        ]
        final_fg = np.clip(
            final_fg.astype(np.float32) + light_rgb_vp, 0, 255
        ).astype(np.uint8)
        final_bg = np.clip(
            final_bg.astype(np.float32) + light_rgb_vp, 0, 255
        ).astype(np.uint8)

    # --- Apply memory fade ---
    if render_config.enable_memory_fade:
        apply_memory_fade(
            final_fg,
            final_bg,
            glyph_indices,
            map_memory_vp,
            map_tiles_vp,
            drawn_mask,
            visible_mask,
            render_config.memory_fade_color_np,
            gs.rng_instance,
            float(render_config.memory_fade_variance),
            float(render_config.memory_noise_level),
            viewport_x,
            viewport_y,
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
            log.warning("render_map_tiles skipped: tile_arrays dictionary is empty.")
        elif np.sum(drawn_mask) == 0:
            log.debug("render_map_tiles skipped: drawn_mask is all False (no tiles to draw).")
        elif _NUMBA_AVAILABLE and isinstance(tile_arrays, NumbaDict):
             # Pass arguments to the Numba compiled function
             render_map_tiles( output_image_array, glyph_indices, drawn_mask, final_fg, final_bg, # final_bg passed for potential fallback
                               tile_arrays, vp_h, vp_w, tile_w, tile_h )
        else:
            # Fallback if Numba is not available or tile_arrays is not a NumbaDict
            log.warning("render_map_tiles skipped: Numba not available or tile_arrays is not NumbaDict.")
            # A Python fallback for rendering tiles could be implemented here if needed,
            # but would be significantly slower. For now, we just log and skip.

    except Exception as e:
        # Catch errors specifically during the Numba tile rendering call
        log.error("Error during render_map_tiles Numba call", error=str(e), exc_info=True)
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
                   render_ground_items( output_image_array, items_to_render_list, tile_arrays, intensity_map,
                                         viewport_x, viewport_y, vp_h, vp_w, tile_w, tile_h )
              else:
                   # Fallback if Numba is not available or tile_arrays is not a NumbaDict
                   log.warning("render_ground_items skipped: Numba not available or tile_arrays is not NumbaDict.")
                   # A Python fallback could be added here if needed

         except Exception as e:
             # Catch errors specifically during the Numba item rendering call
             log.error("Error during render_ground_items Numba call", error=str(e), exc_info=True)


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
                 render_entities( output_image_array, entities_to_render_list, tile_arrays, intensity_map,
                                  viewport_x, viewport_y, vp_h, vp_w, tile_w, tile_h )
              else:
                  # Fallback if Numba is not available or tile_arrays is not a NumbaDict
                   log.warning("render_entities skipped: Numba not available or tile_arrays is not NumbaDict.")
                   # A Python fallback could be added here if needed

         except Exception as e:
              # Catch errors specifically during the Numba entity rendering call
             log.error("Error during render_entities Numba call", error=str(e), exc_info=True)


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

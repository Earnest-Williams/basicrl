# engine/main_loop.py
import polars as pl
import math
import numpy as np
from PIL import Image  # Keep for final conversion

from game.game_state import GameState
from game.world.game_map import TILE_TYPES

from typing import TYPE_CHECKING, Any, Self

try:
    from numba import njit, float32, uint8

    _NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: Numba not installed. Lighting calculation might be slower.")

    # Dummy decorator
    def njit(func=None, **options):
        if func:
            return func
        else:

            def decorator(f):
                return f

            return decorator

    _NUMBA_AVAILABLE = False


if TYPE_CHECKING:
    from engine.window_manager import WindowManager

# --- Tunable Lighting Parameters ---
AMBIENT_LIGHT_LEVEL = np.float32(0.15)
MIN_FOV_LIGHT_LEVEL = np.float32(0.25)
LIGHT_FALLOFF_POWER = np.float32(1.5)
# ---------------------------------


# --- Numba Helper Functions (Lighting - unchanged) ---
@njit(
    float32(float32, float32, float32, float32), cache=True, fastmath=True, nogil=True
)
def _calculate_light_intensity_scalar(
    dist_sq: np.float32,
    radius_sq: np.float32,
    falloff_power: np.float32,
    min_light_level: np.float32,
) -> np.float32:
    if dist_sq > radius_sq >= 0:
        return np.float32(0.0)
    if radius_sq <= 0:
        return np.float32(1.0)
    dist = math.sqrt(dist_sq)
    radius = math.sqrt(radius_sq)
    falloff_ratio = dist / radius
    light_value = max(np.float32(0.0), np.float32(1.0 - falloff_ratio)) ** falloff_power
    intensity = max(min_light_level if light_value > 1e-6 else 0.0, light_value)
    return max(np.float32(0.0), min(np.float32(1.0), intensity))


_calculate_light_intensity_vectorized = np.vectorize(
    _calculate_light_intensity_scalar,
    otypes=[np.float32],
    excluded=["radius_sq", "falloff_power", "min_light_level"],
)


@njit("uint8[:](uint8[:], float32)", cache=True, fastmath=True, nogil=True)
def _interpolate_color_numba_vector(
    base_color: np.ndarray, intensity: np.float32
) -> np.ndarray:
    intensity = max(np.float32(0.0), min(np.float32(1.0), intensity))
    result = np.empty(3, dtype=np.uint8)
    for i in range(3):
        result[i] = max(0, min(255, int(base_color[i] * intensity)))
    return result


# --- Main Loop Class ---
class MainLoop:
    def __init__(self: Self, game_state: GameState, window: "WindowManager"):
        # (Constructor remains the same as previous version)
        self.game_state: GameState = game_state
        self.window: "WindowManager" = window
        print("MainLoop initialized successfully")
        # Determine max ID *used* in the definition, assumes IDs start from 0 or positive
        self.max_defined_tile_id = max(TILE_TYPES.keys()) if TILE_TYPES else -1
        # Size arrays +1 to accommodate the max ID
        array_size = self.max_defined_tile_id + 1

        self._tile_fg_colors = np.zeros((array_size, 3), dtype=np.uint8)
        self._tile_bg_colors = np.zeros((array_size, 3), dtype=np.uint8)
        self._tile_indices_render = np.zeros(
            array_size, dtype=np.uint16
        )  # Stores glyph index
        if array_size > 0:  # Check if TILE_TYPES was not empty
            for tile_id, tile_type in TILE_TYPES.items():
                if (
                    0 <= tile_id <= self.max_defined_tile_id
                ):  # Ensure ID is within bounds
                    self._tile_fg_colors[tile_id] = tile_type.color_fg
                    self._tile_bg_colors[tile_id] = tile_type.color_bg
                    self._tile_indices_render[tile_id] = tile_type.tile_index
                else:
                    print(
                        f"Warning: Tile ID {tile_id} from TILE_TYPES is outside expected range [0, {self.max_defined_tile_id}]"
                    )
        else:
            print(
                "Warning: TILE_TYPES is empty or contains no valid IDs. Tile rendering might be blank."
            )

    def handle_action(self: Self, action: dict[str, Any]) -> bool:
        # (Implementation remains the same as previous version)
        action_type = action.get("type")
        player_acted = False
        match action_type:
            case "move":
                dx, dy = action.get("dx", 0), action.get("dy", 0)
                if dx != 0 or dy != 0:
                    player_acted = self.move_player(dx, dy)
            case "wait":
                player_acted = True
                self.game_state.add_message("You wait.", (128, 128, 128))
            case _:
                return False
        if player_acted:
            self.game_state.advance_turn()
            player_pos = self.game_state.player_position
            if player_pos:
                self.game_state.update_fov()  # FOV updates visibility map
            return True
        return False

    def move_player(self: Self, dx: int, dy: int) -> bool:
        # (Implementation remains the same as previous version)
        player_id = self.game_state.player_id
        current_pos = self.game_state.entity_registry.get_position(player_id)
        if current_pos is None:
            return False

        new_x, new_y = current_pos[0] + dx, current_pos[1] + dy

        if not self.game_state.game_map.in_bounds(new_x, new_y):
            self.game_state.add_message("You can't move there.", (255, 127, 0))
            return False
        if not self.game_state.game_map.is_walkable(new_x, new_y):
            self.game_state.add_message("That way is blocked.", (255, 127, 0))
            return False

        blocking_id = self.game_state.entity_registry.get_blocking_entity_at(
            new_x, new_y
        )
        if blocking_id is not None and blocking_id != player_id:
            blocker_name = (
                self.game_state.entity_registry.get_entity_component(
                    blocking_id, "name"
                )
                or "something"
            )
            self.game_state.add_message(
                f"The {blocker_name} blocks your way.", (255, 255, 0)
            )
            return False

        success = self.game_state.entity_registry.set_position(player_id, new_x, new_y)
        if not success:
            return False

        return True

    # --- REWRITTEN & CORRECTED TWICE: update_console ---
    def update_console(
        self: Self,
        viewport_x: int,
        viewport_y: int,
        viewport_width: int,
        viewport_height: int,
    ) -> Image.Image | None:
        """
        Updates viewport state using NumPy vectorization and returns a rendered PIL Image.
        """
        gs = self.game_state
        gm = gs.game_map
        tile_w = self.window.tile_width
        tile_h = self.window.tile_height
        tile_arrays = self.window.tile_arrays  # Access the cached NumPy arrays

        if (
            tile_w <= 0
            or tile_h <= 0
            or not tile_arrays
            or self.max_defined_tile_id < 0
        ):
            # Added check for max_defined_tile_id to prevent errors if TILE_TYPES was empty
            print(
                "Warning: Cannot render - invalid tile dimensions, empty tile cache, or TILE_TYPES empty."
            )
            return None

        player_pos = gs.player_position
        if player_pos is None:  # Handle missing player
            pw = viewport_width * tile_w
            ph = viewport_height * tile_h
            return Image.new("RGBA", (pw, ph), (0, 0, 0, 255))

        player_x, player_y = player_pos
        fov_radius = np.float32(gs.fov_radius)
        fov_radius_sq = fov_radius * fov_radius if fov_radius > 0 else np.float32(-1.0)

        # --- 1. Get Viewport Data Slices ---
        map_y_slice = slice(viewport_y, viewport_y + viewport_height)
        map_x_slice = slice(viewport_x, viewport_x + viewport_width)
        try:
            map_visible_vp = gm.visible[map_y_slice, map_x_slice]
            map_explored_vp = gm.explored[map_y_slice, map_x_slice]
            map_tiles_vp = gm.tiles[map_y_slice, map_x_slice]  # Tile IDs
        except IndexError:  # Handle potential out-of-bounds slice gracefully
            viewport_y = max(0, min(viewport_y, gm.height - viewport_height))
            viewport_x = max(0, min(viewport_x, gm.width - viewport_width))
            map_y_slice = slice(viewport_y, viewport_y + viewport_height)
            map_x_slice = slice(viewport_x, viewport_x + viewport_width)
            map_visible_vp = gm.visible[map_y_slice, map_x_slice]
            map_explored_vp = gm.explored[map_y_slice, map_x_slice]
            map_tiles_vp = gm.tiles[map_y_slice, map_x_slice]

        # Create coordinate grids for viewport calculations
        vp_rel_y, vp_rel_x = np.indices((viewport_height, viewport_width))
        map_abs_x_vp = vp_rel_x + viewport_x  # Absolute map X coordinates in viewport
        map_abs_y_vp = vp_rel_y + viewport_y  # Absolute map Y coordinates in viewport

        # --- 2. Visibility & Base Colors ---
        visible_mask = map_visible_vp
        explored_mask = map_explored_vp & (~visible_mask)
        drawn_mask = visible_mask | explored_mask  # Tiles to draw at all

        # Initialize arrays
        base_fg = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
        base_bg = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
        glyph_indices = np.zeros((viewport_height, viewport_width), dtype=np.uint16)

        if np.any(drawn_mask):  # Only lookup if there's something to draw
            # Get the raw tile IDs from the map for the drawn area
            tile_ids_in_vp_raw = map_tiles_vp[drawn_mask]

            # --- ADDED: Clip tile IDs to valid range ---
            # Ensure tile IDs used for indexing are within the bounds of the precomputed arrays
            valid_tile_ids_in_vp = np.clip(
                tile_ids_in_vp_raw, 0, self.max_defined_tile_id
            )
            # --- END CORRECTION ---

            # Use the *clipped* IDs for lookups
            base_fg[drawn_mask] = self._tile_fg_colors[valid_tile_ids_in_vp]
            base_bg[drawn_mask] = self._tile_bg_colors[valid_tile_ids_in_vp]
            glyph_indices[drawn_mask] = self._tile_indices_render[valid_tile_ids_in_vp]

        # --- 3. Lighting Calculation ---
        intensity_map = np.full(
            (viewport_height, viewport_width), AMBIENT_LIGHT_LEVEL, dtype=np.float32
        )
        if np.any(visible_mask):
            dx = map_abs_x_vp[visible_mask] - player_x
            dy = map_abs_y_vp[visible_mask] - player_y
            dist_sq_map = (dx * dx + dy * dy).astype(np.float32)

            visible_intensities = _calculate_light_intensity_vectorized(
                dist_sq_map, fov_radius_sq, LIGHT_FALLOFF_POWER, MIN_FOV_LIGHT_LEVEL
            )
            intensity_map[
                visible_mask
            ] = visible_intensities  # Apply calculated intensity only to visible tiles

        # --- 4. Apply Lighting to Base Colors ---
        intensity_broadcast = intensity_map[
            ..., None
        ]  # Shape (h, w, 1) for broadcasting
        lit_fg = (base_fg * intensity_broadcast).astype(np.uint8)
        lit_bg = (base_bg * intensity_broadcast).astype(np.uint8)

        # --- 5. Prepare Output Pixel Buffer ---
        output_pixel_h = viewport_height * tile_h
        output_pixel_w = viewport_width * tile_w
        output_image_array = np.repeat(
            np.repeat(lit_bg, tile_h, axis=0), tile_w, axis=1
        )
        output_image_array = np.dstack(
            (
                output_image_array,
                np.full((output_pixel_h, output_pixel_w), 255, dtype=np.uint8),
            )
        )

        # --- 6. Vectorized Map Tile Foreground Rendering ---
        # (Unchanged from previous corrected version)
        px_y, px_x = np.indices((output_pixel_h, output_pixel_w))
        tile_coord_y = px_y // tile_h
        tile_coord_x = px_x // tile_w
        pixel_in_tile_y = px_y % tile_h
        pixel_in_tile_x = px_x % tile_w
        pixel_glyph_indices = glyph_indices[tile_coord_y, tile_coord_x]
        map_fg_draw_mask = drawn_mask[tile_coord_y, tile_coord_x] & (
            pixel_glyph_indices > 0
        )
        unique_map_glyphs = np.unique(pixel_glyph_indices[map_fg_draw_mask])

        for glyph_idx in unique_map_glyphs:
            tile_rgba_array = tile_arrays.get(glyph_idx)
            if tile_rgba_array is None or tile_rgba_array.shape != (tile_h, tile_w, 4):
                continue
            current_glyph_pixel_mask = map_fg_draw_mask & (
                pixel_glyph_indices == glyph_idx
            )
            if not np.any(current_glyph_pixel_mask):
                continue

            glyph_rel_y = pixel_in_tile_y[current_glyph_pixel_mask]
            glyph_rel_x = pixel_in_tile_x[current_glyph_pixel_mask]
            tile_alpha_values = tile_rgba_array[glyph_rel_y, glyph_rel_x, 3]
            alpha_above_threshold = tile_alpha_values > 10

            final_pixel_mask = np.zeros_like(current_glyph_pixel_mask, dtype=bool)
            final_pixel_mask[current_glyph_pixel_mask] = alpha_above_threshold

            target_tile_coords_y = tile_coord_y[final_pixel_mask]
            target_tile_coords_x = tile_coord_x[final_pixel_mask]
            fg_colors_for_pixels = lit_fg[target_tile_coords_y, target_tile_coords_x]

            output_image_array[final_pixel_mask, :3] = fg_colors_for_pixels
            # Get the corresponding tile alpha values only for the final pixels
            # Need to re-apply the alpha_above_threshold mask to the 1D indices/values
            final_tile_alpha_values = tile_alpha_values[alpha_above_threshold]
            output_image_array[final_pixel_mask, 3] = final_tile_alpha_values

        # --- 7. Draw Entities (Vectorized Approach - unchanged from previous corrected version) ---
        entities_df = gs.entity_registry.get_active_entities()
        viewport_entities = entities_df.filter(
            (pl.col("x") >= viewport_x)
            & (pl.col("x") < viewport_x + viewport_width)
            & (pl.col("y") >= viewport_y)
            & (pl.col("y") < viewport_y + viewport_height)
            & (pl.col("glyph") > 0)  # Only consider entities with actual glyphs
        )
        player_id = gs.player_id
        sorted_entities = viewport_entities.sort(
            by=(pl.col("entity_id") == player_id), descending=False
        )

        for entity_row in sorted_entities.iter_rows(named=True):
            map_ex, map_ey = entity_row["x"], entity_row["y"]
            glyph_idx = entity_row["glyph"]
            is_visible = (
                gm.visible[map_ey, map_ex] if gm.in_bounds(map_ex, map_ey) else False
            )

            if not is_visible:
                continue

            entity_tile_array = tile_arrays.get(glyph_idx)
            if entity_tile_array is None or entity_tile_array.shape != (
                tile_h,
                tile_w,
                4,
            ):
                continue

            e_dx = map_ex - player_x
            e_dy = map_ey - player_y
            e_dist_sq = np.float32(e_dx * e_dx + e_dy * e_dy)
            e_intensity = _calculate_light_intensity_scalar(
                e_dist_sq, fov_radius_sq, LIGHT_FALLOFF_POWER, MIN_FOV_LIGHT_LEVEL
            )
            base_fg_e = np.array(
                [
                    entity_row["color_fg_r"],
                    entity_row["color_fg_g"],
                    entity_row["color_fg_b"],
                ],
                dtype=np.uint8,
            )
            lit_fg_e = _interpolate_color_numba_vector(base_fg_e, e_intensity)

            cons_ex = map_ex - viewport_x
            cons_ey = map_ey - viewport_y
            px_start_y, px_start_x = cons_ey * tile_h, cons_ex * tile_w
            px_end_y, px_end_x = px_start_y + tile_h, px_start_x + tile_w

            px_start_y_clip = max(0, px_start_y)
            px_start_x_clip = max(0, px_start_x)
            px_end_y_clip = min(output_pixel_h, px_end_y)
            px_end_x_clip = min(output_pixel_w, px_end_x)

            if px_start_y_clip >= px_end_y_clip or px_start_x_clip >= px_end_x_clip:
                continue

            tile_start_y = px_start_y_clip - px_start_y
            tile_start_x = px_start_x_clip - px_start_x
            tile_end_y = tile_start_y + (px_end_y_clip - px_start_y_clip)
            tile_end_x = tile_start_x + (px_end_x_clip - px_start_x_clip)

            pixel_block_view = output_image_array[
                px_start_y_clip:px_end_y_clip, px_start_x_clip:px_end_x_clip
            ]
            entity_tile_part = entity_tile_array[
                tile_start_y:tile_end_y, tile_start_x:tile_end_x
            ]

            if (
                pixel_block_view.shape[0] != entity_tile_part.shape[0]
                or pixel_block_view.shape[1] != entity_tile_part.shape[1]
            ):
                # This can happen if slicing calculation is off, add a warning/skip
                print(
                    f"Warning: Mismatched shapes for entity drawing at ({map_ex},{map_ey}). View: {pixel_block_view.shape}, Tile part: {entity_tile_part.shape}. Skipping."
                )
                continue

            tile_alpha = entity_tile_part[:, :, 3]
            alpha_mask = tile_alpha > 10

            pixel_block_view[alpha_mask, :3] = lit_fg_e
            pixel_block_view[alpha_mask, 3] = tile_alpha[alpha_mask]

        # --- 8. Convert final NumPy array to PIL Image ---
        final_image = Image.fromarray(output_image_array, "RGBA")
        return final_image

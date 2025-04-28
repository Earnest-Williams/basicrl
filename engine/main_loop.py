# engine/main_loop.py
import polars as pl
import math
import numpy as np
from PIL import Image
import structlog
import traceback

from game.game_state import GameState
from game.world.game_map import TILE_TYPES, GameMap

from typing import TYPE_CHECKING, Any, Self, List

try:
    from numba import njit, float32, uint8

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(func=None, **options):
        if func:
            return func
        else:

            def decorator(f):
                return f

            return decorator


if TYPE_CHECKING:
    from engine.window_manager import WindowManager

log = structlog.get_logger()


# --- Numba Helper Functions (Unchanged) ---
@njit(
    float32(float32, float32, float32, float32), cache=True, fastmath=True, nogil=True
)
def _calculate_light_intensity_scalar(
    dist_sq: np.float32,
    radius_sq: np.float32,
    falloff_power: np.float32,
    min_light_level: np.float32,
) -> np.float32:
    # (Implementation unchanged)
    if radius_sq < 0:
        return np.float32(1.0)
    if dist_sq > radius_sq:
        return np.float32(0.0)
    if dist_sq < 1e-6:
        return np.float32(1.0)
    if radius_sq <= 1e-6:
        return np.float32(0.0)
    dist = math.sqrt(dist_sq)
    radius = math.sqrt(radius_sq)
    falloff_ratio = dist / radius
    light_value = max(np.float32(0.0), np.float32(1.0 - falloff_ratio)) ** falloff_power
    intensity = max(light_value, min_light_level)
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
    # (Implementation unchanged)
    intensity = max(np.float32(0.0), min(np.float32(1.0), intensity))
    result = np.empty(3, dtype=np.uint8)
    for i in range(3):
        result[i] = max(0, min(255, int(base_color[i] * intensity)))
    return result


# --- End Numba Helpers ---


# --- Main Loop Class ---
class MainLoop:
    def __init__(
        self: Self,
        game_state: GameState,
        window: "WindowManager",
        vis_enabled_default: bool,
        vis_max_diff: int,
        vis_color_high: list,
        vis_color_mid: list,
        vis_color_low: list,
        vis_blend_factor: float,
        max_traversable_step: int,
        lighting_ambient: float,
        lighting_min_fov: float,
        lighting_falloff: float,
    ):
        # (Initialization unchanged)
        self.game_state: GameState = game_state
        self.window: "WindowManager" = window
        self.show_height_visualization: bool = vis_enabled_default
        self._cfg_height_vis_max_diff: int = vis_max_diff
        self._cfg_height_color_high_np = np.array(vis_color_high, dtype=np.uint8)
        self._cfg_height_color_mid_np = np.array(vis_color_mid, dtype=np.uint8)
        self._cfg_height_color_low_np = np.array(vis_color_low, dtype=np.uint8)
        self._cfg_height_vis_blend_factor: float = np.float32(
            max(0.0, min(1.0, vis_blend_factor))
        )
        self._cfg_max_traversable_step: int = max_traversable_step
        self._cfg_ambient_light = np.float32(lighting_ambient)
        self._cfg_min_fov_light = np.float32(lighting_min_fov)
        self._cfg_light_falloff = np.float32(lighting_falloff)
        # Tile Cache Setup... (logic unchanged)
        self.max_defined_tile_id = max(TILE_TYPES.keys()) if TILE_TYPES else -1
        array_size = self.max_defined_tile_id + 1
        self._tile_fg_colors = np.zeros((array_size, 3), dtype=np.uint8)
        self._tile_bg_colors = np.zeros((array_size, 3), dtype=np.uint8)
        self._tile_indices_render = np.zeros(array_size, dtype=np.uint16)
        if array_size > 0:
            valid_ids_loaded = 0
            for tile_id, tile_type in TILE_TYPES.items():
                if 0 <= tile_id <= self.max_defined_tile_id:
                    self._tile_fg_colors[tile_id] = tile_type.color_fg
                    self._tile_bg_colors[tile_id] = tile_type.color_bg
                    self._tile_indices_render[tile_id] = tile_type.tile_index
                    valid_ids_loaded += 1
                else:
                    log.warning(
                        "Tile ID out of range",
                        tile_id=tile_id,
                        expected_range=f"[0, {self.max_defined_tile_id}]",
                        source="TILE_TYPES",
                    )
            log.debug(
                "Tile render cache populated",
                loaded_ids=valid_ids_loaded,
                cache_size=array_size,
            )
        else:
            log.warning(
                "TILE_TYPES empty or invalid", detail="Tile rendering might be blank."
            )
        log.info(
            "MainLoop initialized successfully",
            initial_vis_mode=self.show_height_visualization,
            ambient=self._cfg_ambient_light,
            min_fov=self._cfg_min_fov_light,
            falloff=self._cfg_light_falloff,
        )

    def handle_action(self: Self, action: dict[str, Any]) -> bool:
        # (Unchanged)
        action_type = action.get("type")
        player_acted = False
        match action_type:
            case "move":
                dx, dy = action.get("dx", 0), action.get("dy", 0)
                player_acted = self.move_player(dx, dy) if dx or dy else False
            case "wait":
                player_acted = True
                self.game_state.add_message("You wait.", (128, 128, 128))
                log.debug("Player action", type="wait")
            case _:
                log.warning("Unknown action type", received_action=action_type)
                return False
        if player_acted:
            self.game_state.advance_turn()
            player_pos = self.game_state.player_position
            self.game_state.update_fov() if player_pos else None
            return True
        return False

    def move_player(self: Self, dx: int, dy: int) -> bool:
        # (Unchanged)
        player_id = self.game_state.player_id
        gs = self.game_state
        gm = gs.game_map
        current_pos = gs.entity_registry.get_position(player_id)
        if current_pos is None:
            log.warning("Move failed: Player position not found", player_id=player_id)
            return False
        current_x, current_y = current_pos
        new_x, new_y = current_x + dx, current_y + dy
        log_context = {
            "player_id": player_id,
            "from_pos": (current_x, current_y),
            "to_pos": (new_x, new_y),
        }
        if not gm.in_bounds(new_x, new_y):
            gs.add_message("You can't move there.", (255, 127, 0))
            log.debug("Movement blocked: Out of bounds", **log_context)
            return False
        if not gm.is_walkable(new_x, new_y):
            gs.add_message("That way is blocked.", (255, 127, 0))
            log.debug(
                "Movement blocked: Tile not walkable",
                tile_id=gm.tiles[new_y, new_x],
                **log_context,
            )
            return False
        try:  # Height Check
            h1 = gm.height_map[current_y, current_x]
            h2 = gm.height_map[new_y, new_x]
            delta_h = h2 - h1
            max_step = self._cfg_max_traversable_step
            if abs(delta_h) > max_step:
                msg = (
                    "That step is too high to climb."
                    if delta_h > max_step
                    else "That drop looks too steep."
                )
                gs.add_message(msg, (255, 127, 0))
                log.info(
                    "Movement blocked: Step too steep",
                    delta_h=int(delta_h),
                    max_step=max_step,
                    height1=int(h1),
                    height2=int(h2),
                    **log_context,
                )
                return False
        except IndexError:
            log.error("IndexError during height check", **log_context)
            return False
        blocking_id = gs.entity_registry.get_blocking_entity_at(new_x, new_y)
        if blocking_id is not None and blocking_id != player_id:
            blocker_name = (
                gs.entity_registry.get_entity_component(blocking_id, "name")
                or "something"
            )
            gs.add_message(f"The {blocker_name} blocks your way.", (255, 255, 0))
            log.debug(
                "Movement blocked: Entity",
                blocking_entity_id=blocking_id,
                name=blocker_name,
                **log_context,
            )
            return False
        success = gs.entity_registry.set_position(player_id, new_x, new_y)
        if success:
            log.debug("Player moved successfully", **log_context)
            return True
        else:
            log.error(
                "Setting player position failed unexpectedly",
                player_id=player_id,
                new_x=new_x,
                new_y=new_y,
            )
            return False

    # --- MODIFIED update_console ---
    def update_console(
        self: Self,
        viewport_x: int,
        viewport_y: int,
        viewport_width: int,
        viewport_height: int,
    ) -> Image.Image | None:
        # (Added explicit handling for relative_height == 0 in visualization)
        gs = self.game_state
        gm = gs.game_map
        tile_w = self.window.tile_width
        tile_h = self.window.tile_height
        tile_arrays = self.window.tile_arrays
        render_params = {
            "vp_x": viewport_x,
            "vp_y": viewport_y,
            "vp_w": viewport_width,
            "vp_h": viewport_height,
            "tile_w": tile_w,
            "tile_h": tile_h,
        }
        if (
            tile_w <= 0
            or tile_h <= 0
            or not tile_arrays
            or self.max_defined_tile_id < 0
        ):
            log.warning(
                "Cannot render: Invalid state",
                tile_arrays_empty=not tile_arrays,
                max_tile_id=self.max_defined_tile_id,
                **render_params,
            )
            return None
        player_pos = gs.player_position
        if player_pos is None:
            log.warning("Cannot render: Player position not found", **render_params)
            pw = viewport_width * tile_w
            ph = viewport_height * tile_h
            return Image.new("RGBA", (pw, ph), (0, 0, 0, 255))
        player_x, player_y = player_pos
        fov_radius = np.float32(gs.fov_radius)
        fov_radius_sq = fov_radius * fov_radius if fov_radius > 0 else np.float32(-1.0)
        ambient_light = self._cfg_ambient_light
        min_fov_light = self._cfg_min_fov_light
        falloff_power = self._cfg_light_falloff

        # Fetch map slices
        map_y_slice = slice(viewport_y, viewport_y + viewport_height)
        map_x_slice = slice(viewport_x, viewport_x + viewport_width)
        try:
            map_visible_vp = gm.visible[map_y_slice, map_x_slice]
            map_explored_vp = gm.explored[map_y_slice, map_x_slice]
            map_tiles_vp = gm.tiles[map_y_slice, map_x_slice]
            map_height_vp = gm.height_map[map_y_slice, map_x_slice]
            player_height = gm.height_map[player_y, player_x]
        except IndexError:
            log.warning("Viewport slice out of bounds, adjusting.", **render_params)
            viewport_y = max(0, min(viewport_y, gm.height - viewport_height))
            viewport_x = max(0, min(viewport_x, gm.width - viewport_width))
            map_y_slice = slice(viewport_y, viewport_y + viewport_height)
            map_x_slice = slice(viewport_x, viewport_x + viewport_width)
            map_visible_vp = gm.visible[map_y_slice, map_x_slice]
            map_explored_vp = gm.explored[map_y_slice, map_x_slice]
            map_tiles_vp = gm.tiles[map_y_slice, map_x_slice]
            map_height_vp = gm.height_map[map_y_slice, map_x_slice]
            # Need player height again in case player started OOB? Unlikely if start pos generation is correct.
            try:
                player_height = gm.height_map[player_y, player_x]
            except IndexError:
                log.error("Player height OOB after viewport adjust?")
                return None  # Bail out if player still OOB

        # Prepare data arrays
        vp_rel_y, vp_rel_x = np.indices((viewport_height, viewport_width))
        map_abs_x_vp = vp_rel_x + viewport_x
        map_abs_y_vp = vp_rel_y + viewport_y
        visible_mask = map_visible_vp
        explored_mask = map_explored_vp & (~visible_mask)
        drawn_mask = visible_mask | explored_mask
        base_fg = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
        base_bg = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
        glyph_indices = np.zeros((viewport_height, viewport_width), dtype=np.uint16)
        if np.any(drawn_mask):
            tile_ids_in_vp_raw = map_tiles_vp[drawn_mask]
            valid_tile_ids_in_vp = np.clip(
                tile_ids_in_vp_raw, 0, self.max_defined_tile_id
            )
            base_fg[drawn_mask] = self._tile_fg_colors[valid_tile_ids_in_vp]
            base_bg[drawn_mask] = self._tile_bg_colors[valid_tile_ids_in_vp]
            glyph_indices[drawn_mask] = self._tile_indices_render[valid_tile_ids_in_vp]

        # Calculate Lighting
        intensity_map = np.full(
            (viewport_height, viewport_width), ambient_light, dtype=np.float32
        )
        if np.any(visible_mask):
            dx = map_abs_x_vp[visible_mask] - player_x
            dy = map_abs_y_vp[visible_mask] - player_y
            dist_sq_map = (dx * dx + dy * dy).astype(np.float32)
            visible_intensities = _calculate_light_intensity_vectorized(
                dist_sq_map, fov_radius_sq, falloff_power, min_fov_light
            )
            intensity_map[visible_mask] = visible_intensities
        intensity_broadcast = intensity_map[..., None]
        lit_fg = (base_fg * intensity_broadcast).astype(np.uint8)
        lit_bg = (base_bg * intensity_broadcast).astype(np.uint8)

        # --- Height Visualization (Handles relative_height == 0 now) ---
        if self.show_height_visualization:
            log.debug("Applying height visualization")
            final_fg = lit_fg.copy()
            final_bg = lit_bg.copy()  # Start with lit colors
            relative_height_vp = map_height_vp - player_height
            max_diff = float(self._cfg_height_vis_max_diff)
            blend = self._cfg_height_vis_blend_factor
            drawn_and_valid = drawn_mask & (
                max_diff > 0
            )  # Can only divide if max_diff > 0

            # Define masks for high, low, and *mid* heights within range
            high_mask = (
                drawn_and_valid
                & (relative_height_vp > 0)
                & (relative_height_vp <= max_diff)
            )
            low_mask = (
                drawn_and_valid
                & (relative_height_vp < 0)
                & (relative_height_vp >= -max_diff)
            )
            mid_mask = drawn_mask & (relative_height_vp == 0)  # Tiles at player height

            # Blend Higher Tiles (towards Yellow)
            if np.any(high_mask):
                t = (relative_height_vp[high_mask] / max_diff).astype(np.float32)
                t = np.clip(t, 0.0, 1.0)[..., None]
                target_color = self._cfg_height_color_high_np.astype(np.float32)
                current_fg = final_fg[high_mask].astype(np.float32)
                current_bg = final_bg[high_mask].astype(np.float32)
                blended_fg = current_fg * (1.0 - blend * t) + target_color * (blend * t)
                blended_bg = current_bg * (1.0 - blend * t) + target_color * (blend * t)
                final_fg[high_mask] = np.clip(blended_fg, 0, 255).astype(np.uint8)
                final_bg[high_mask] = np.clip(blended_bg, 0, 255).astype(np.uint8)

            # Blend Lower Tiles (towards Blue)
            if np.any(low_mask):
                t = (-relative_height_vp[low_mask] / max_diff).astype(np.float32)
                t = np.clip(t, 0.0, 1.0)[..., None]
                target_color = self._cfg_height_color_low_np.astype(np.float32)
                current_fg = final_fg[low_mask].astype(np.float32)
                current_bg = final_bg[low_mask].astype(np.float32)
                blended_fg = current_fg * (1.0 - blend * t) + target_color * (blend * t)
                blended_bg = current_bg * (1.0 - blend * t) + target_color * (blend * t)
                final_fg[low_mask] = np.clip(blended_fg, 0, 255).astype(np.uint8)
                final_bg[low_mask] = np.clip(blended_bg, 0, 255).astype(np.uint8)

            # --- ADDED: Blend Mid Tiles (towards Green) ---
            if np.any(mid_mask):
                # Blend towards the mid color using the full blend factor
                target_color = self._cfg_height_color_mid_np.astype(np.float32)
                current_fg = final_fg[mid_mask].astype(np.float32)
                current_bg = final_bg[mid_mask].astype(np.float32)
                # LERP: current * (1-blend) + target * blend
                blended_fg = current_fg * (1.0 - blend) + target_color * blend
                blended_bg = current_bg * (1.0 - blend) + target_color * blend
                final_fg[mid_mask] = np.clip(blended_fg, 0, 255).astype(np.uint8)
                final_bg[mid_mask] = np.clip(blended_bg, 0, 255).astype(np.uint8)
            # --- END ADDED ---

        else:  # Visualization toggled off
            final_fg = lit_fg
            final_bg = lit_bg
        # --- End Height Visualization ---

        # --- Rendering Steps (Unchanged, use final_fg/final_bg) ---
        # 5. Prepare Output Buffer
        output_pixel_h = viewport_height * tile_h
        output_pixel_w = viewport_width * tile_w
        output_image_array = np.repeat(
            np.repeat(final_bg, tile_h, axis=0), tile_w, axis=1
        )
        output_image_array = np.dstack(
            (
                output_image_array,
                np.full((output_pixel_h, output_pixel_w), 255, dtype=np.uint8),
            )
        )
        # 6. Render Map Tiles
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
            fg_colors_for_pixels = final_fg[
                target_tile_coords_y, target_tile_coords_x
            ]  # Uses final_fg
            output_image_array[final_pixel_mask, :3] = fg_colors_for_pixels
            final_tile_alpha_values = tile_alpha_values[alpha_above_threshold]
            output_image_array[final_pixel_mask, 3] = final_tile_alpha_values
        # 7. Render Entities
        entities_df = gs.entity_registry.get_active_entities()
        viewport_entities = entities_df.filter(
            (pl.col("x") >= viewport_x)
            & (pl.col("x") < viewport_x + viewport_width)
            & (pl.col("y") >= viewport_y)
            & (pl.col("y") < viewport_y + viewport_height)
            & (pl.col("glyph") > 0)
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
            e_dx, e_dy = map_ex - player_x, map_ey - player_y
            e_dist_sq = np.float32(e_dx * e_dx + e_dy * e_dy)
            e_intensity = _calculate_light_intensity_scalar(
                e_dist_sq, fov_radius_sq, falloff_power, min_fov_light
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
            cons_ex, cons_ey = map_ex - viewport_x, map_ey - viewport_y
            px_start_y, px_start_x = cons_ey * tile_h, cons_ex * tile_w
            px_end_y, px_end_x = px_start_y + tile_h, px_start_x + tile_w
            px_start_y_clip, px_start_x_clip = max(0, px_start_y), max(0, px_start_x)
            px_end_y_clip, px_end_x_clip = min(output_pixel_h, px_end_y), min(
                output_pixel_w, px_end_x
            )
            if px_start_y_clip >= px_end_y_clip or px_start_x_clip >= px_end_x_clip:
                continue
            tile_start_y, tile_start_x = (
                px_start_y_clip - px_start_y,
                px_start_x_clip - px_start_x,
            )
            tile_end_y, tile_end_x = tile_start_y + (
                px_end_y_clip - px_start_y_clip
            ), tile_start_x + (px_end_x_clip - px_start_x_clip)
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
                log.warning(
                    "Mismatched shapes for entity drawing",
                    entity_id=entity_row.get("entity_id", -1),
                    map_pos=(map_ex, map_ey),
                    view_shape=pixel_block_view.shape,
                    tile_part_shape=entity_tile_part.shape,
                )
                continue
            tile_alpha = entity_tile_part[:, :, 3]
            alpha_mask = tile_alpha > 10
            pixel_block_view[alpha_mask, :3] = lit_fg_e
            pixel_block_view[alpha_mask, 3] = tile_alpha[alpha_mask]
        # 8. Convert final NumPy array to PIL Image
        final_image = Image.fromarray(output_image_array, "RGBA")
        return final_image

    # --- END MODIFIED update_console ---

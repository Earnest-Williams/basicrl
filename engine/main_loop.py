# engine/main_loop.py
import numpy as np
from PIL import Image
import structlog
import traceback
from typing import TYPE_CHECKING, Any, Self, List, Tuple

# Use absolute imports for game modules
from game.game_state import GameState
from game.world.game_map import (
    GameMap,
    TILE_TYPES,
)  # Keep TILE_TYPES only if needed elsewhere

# --- NEW: Import the new action_handler and renderer modules ---
from . import renderer  # Use relative import for sibling module
from . import action_handler

# --- End NEW ---

# Numba config removed - helpers are in renderer.py

if TYPE_CHECKING:
    # Relative import for sibling module within the same package ('engine')
    from .window_manager import WindowManager

log = structlog.get_logger()


class MainLoop:
    def __init__(
        self: Self,
        game_state: GameState,
        window: "WindowManager",
        # Rendering options passed through
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
        # Store core components
        self.game_state: GameState = game_state
        self.window: "WindowManager" = window
        self.show_height_visualization: bool = vis_enabled_default

        # Store config values needed by components managed here
        # (Renderer config is passed directly during render call)
        self._cfg_max_traversable_step: int = max_traversable_step
        # Store rendering configs needed by Renderer (to pass them later)
        self._cfg_vis_max_diff = vis_max_diff
        self._cfg_height_color_high_np = np.array(vis_color_high, dtype=np.uint8)
        self._cfg_height_color_mid_np = np.array(vis_color_mid, dtype=np.uint8)
        self._cfg_height_color_low_np = np.array(vis_color_low, dtype=np.uint8)
        self._cfg_vis_blend_factor = np.float32(vis_blend_factor)
        self._cfg_ambient_light = np.float32(lighting_ambient)
        self._cfg_min_fov_light = np.float32(lighting_min_fov)
        self._cfg_light_falloff = np.float32(lighting_falloff)

        # Tile cache arrays removed - managed by WindowManager

        log.info("MainLoop initialized successfully")  # Simplified log

    def handle_action(self: Self, action: dict[str, Any]) -> bool:
        """
        Receives an action, processes it via the action_handler,
        and updates game state if the action consumed a turn.
        """
        gs = self.game_state  # Alias
        print(f"MainLoop: Received action: {action}")

        # --- Delegate action processing ---
        try:
            player_acted = action_handler.process_player_action(
                action, gs, self._cfg_max_traversable_step
            )
            print(f"MainLoop: process_player_action returned: {player_acted}")
        except Exception as e:
            # Catch errors during action processing
            log.error(
                "Exception during action processing",
                action=action,
                error=str(e),
                exc_info=True,
            )
            gs.add_message("An internal error occurred.", (255, 0, 0))
            player_acted = False  # Error means no valid turn taken

        # --- Post-Action Updates (only if player acted) ---
        if player_acted:
            print("MainLoop: Player acted, advancing turn...")
            log.debug(
                "Player action successful, advancing turn",
                action_type=action.get("type"),
            )
            # Advance turn and update FOV
            gs.advance_turn()
            player_pos = gs.player_position
            if player_pos:
                gs.update_fov()
            else:
                log.warning("Player position lost after action, clearing FOV.")
                gs.game_map.visible[:] = False
            # Trigger redraw via WindowManager? Typically WindowManager's update_frame calls update_console.
            # No explicit redraw needed here if the input loop calls update_frame after handle_action.
            # Explicitly trigger a frame update after state changes
            self.window.update_frame()
            print("MainLoop: Player did not act.")
            return True

        # If player_acted is False, action failed or didn't consume a turn
        log.debug(
            "Player action did not result in turn", action_type=action.get("type")
        )
        return False

    # --- move_player method removed ---

    def update_console(self: Self) -> Image.Image | None:
        """
        Orchestrates rendering by gathering data and calling the renderer module.
        Called by WindowManager.update_frame(). Returns image or None on error.
        """
        gs = self.game_state
        win = self.window

        # Calculate Viewport (Same logic as before)
        label_w = win.label.width()
        label_h = win.label.height()
        if label_w <= 0 or label_h <= 0 or win.tile_width <= 0 or win.tile_height <= 0:
            log.warning("Cannot update console: Invalid window/tile dimensions")
            return None
        visible_cols = max(1, label_w // win.tile_width)
        visible_rows = max(1, label_h // win.tile_height)
        player_pos = gs.player_position
        cam_x, cam_y = (
            player_pos if player_pos else (gs.map_width // 2, gs.map_height // 2)
        )
        render_cols = min(visible_cols, gs.map_width)
        render_rows = min(visible_rows, gs.map_height)
        viewport_x = max(0, min(cam_x - render_cols // 2, gs.map_width - render_cols))
        viewport_y = max(0, min(cam_y - render_rows // 2, gs.map_height - render_rows))
        # Ensure final viewport dimensions are valid
        vp_width = min(render_cols, gs.map_width - viewport_x)
        vp_height = min(render_rows, gs.map_height - viewport_y)
        if vp_width <= 0 or vp_height <= 0:
            log.warning(
                "Calculated viewport has zero or negative dimensions",
                w=vp_width,
                h=vp_height,
            )
            # Return a blank image matching the label size maybe?
            pw = win.label.width()
            ph = win.label.height()
            return Image.new("RGBA", (pw, ph), (0, 0, 0, 255))

        # --- Gather Data for Renderer ---
        # Fetch tile cache arrays from WindowManager
        if (
            win._tile_fg_colors is None
            or win._tile_bg_colors is None
            or win._tile_indices_render is None
        ):
            log.error(
                "Cannot render: Tile cache arrays not initialized in WindowManager"
            )
            return None

        render_data = {
            "game_state": gs,
            "tile_arrays": win.tile_arrays,
            "tile_fg_colors": win._tile_fg_colors,
            "tile_bg_colors": win._tile_bg_colors,
            "tile_indices_render": win._tile_indices_render,
            "max_defined_tile_id": win.max_defined_tile_id,
            "tile_w": win.tile_width,
            "tile_h": win.tile_height,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_width": vp_width,  # Use calculated vp dimensions
            "viewport_height": vp_height,
            "show_height_vis": self.show_height_visualization,
            "vis_max_diff": self._cfg_vis_max_diff,
            "vis_color_high_np": self._cfg_height_color_high_np,
            "vis_color_mid_np": self._cfg_height_color_mid_np,
            "vis_color_low_np": self._cfg_height_color_low_np,
            "vis_blend_factor": self._cfg_vis_blend_factor,
            "lighting_ambient": self._cfg_ambient_light,
            "lighting_min_fov": self._cfg_min_fov_light,
            "lighting_falloff": self._cfg_light_falloff,
        }

        # --- Call Renderer ---
        try:
            image = renderer.render_viewport(**render_data)
            return image
        except Exception as e:
            log.error(
                "Error during renderer.render_viewport call",
                error=str(e),
                exc_info=True,
            )
            traceback.print_exc()
            pw = vp_width * win.tile_width
            ph = vp_height * win.tile_height
            return Image.new(
                "RGBA", (max(1, pw), max(1, ph)), (255, 0, 0, 255)
            )  # Red indicates error

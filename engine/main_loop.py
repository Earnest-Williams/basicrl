# engine/main_loop.py
# Added typing imports
from typing import TYPE_CHECKING, Any, Dict, Self

import numpy as np
import structlog
from PIL import Image

# Use absolute imports for game modules
from game.game_state import GameState

# Use relative import for sibling module within the same package ('engine')
# Import the renderer module and the RenderConfig dataclass
from . import action_handler, renderer
from .renderer import RenderConfig  # Import the dataclass

if TYPE_CHECKING:
    # Relative import for sibling module within the same package ('engine')
    from .window_manager import WindowManager

log = structlog.get_logger()


class MainLoop:
    """
    Coordinates the main game logic, including turn processing,
    action handling, and orchestrating rendering updates.
    """

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
        """
        Initializes the MainLoop.

        Args:
            game_state: The central GameState object.
            window: The WindowManager instance handling display and input.
            vis_enabled_default: Initial state for height visualization.
            vis_max_diff: Max height difference for visualization.
            vis_color_high: Color for high areas in height vis (RGB list).
            vis_color_mid: Color for mid areas in height vis (RGB list).
            vis_color_low: Color for low areas in height vis (RGB list).
            vis_blend_factor: Blend factor for height visualization.
            max_traversable_step: Max height difference walkable by entities.
            lighting_ambient: Ambient light level (0.0-1.0).
            lighting_min_fov: Minimum light level at FOV edge (0.0-1.0).
            lighting_falloff: Exponent for light falloff calculation.
        """
        # Store core components
        self.game_state: GameState = game_state
        self.window: "WindowManager" = window
        self.show_height_visualization: bool = vis_enabled_default

        # Store config values needed by components managed here
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

        log.info("MainLoop initialized successfully")

    def handle_action(self: Self, action: dict[str, Any]) -> bool:
        """
        Receives an action, processes it via the action_handler,
        and updates game state if the action consumed a turn.
        Returns True if the player acted and consumed a turn, False otherwise.
        """
        gs = self.game_state
        player_acted = False

        try:
            player_acted = action_handler.process_player_action(
                action, gs, self._cfg_max_traversable_step
            )
        except Exception as e:
            log.error(
                "Exception during action processing",
                action=action,
                error=str(e),
                exc_info=True,
            )
            gs.add_message("An internal error occurred.", (255, 0, 0))
            player_acted = False  # Error means no valid turn taken

        if player_acted:
            log.debug("Player action resulted in turn", action_type=action.get("type"))
            gs.advance_turn()
            player_pos = gs.player_position
            if player_pos:
                gs.update_fov()
            else:
                log.warning("Player position lost after action, clearing FOV.")
                gs.game_map.visible[:] = False
            # Trigger redraw via WindowManager after state changes
            self.window.update_frame()
            return True
        else:
            log.debug(
                "Player action did not result in turn", action_type=action.get("type")
            )
            return False

    # --- MODIFIED: Update Console Signature ---
    def update_console(
        self: Self,
        # Accept all args passed from WindowManager
        game_state: GameState,
        viewport_x: int,
        viewport_y: int,
        viewport_width: int,  # In tiles
        viewport_height: int,  # In tiles
        tile_arrays: Dict[int, np.ndarray | None],
        tile_fg_colors: np.ndarray,
        tile_bg_colors: np.ndarray,
        tile_indices_render: np.ndarray,
        max_defined_tile_id: int,
        tile_w: int,
        tile_h: int,
        coord_arrays: Dict[str, np.ndarray],
    ) -> Image.Image | None:
        # --- End MODIFIED Signature ---
        """
        Orchestrates rendering by gathering data and calling the renderer module.
        Called by WindowManager.update_frame(). Returns image or None on error.
        """
        gs = game_state  # Use the passed game_state

        # Calculate FOV radius squared (needed for RenderConfig)
        fov_radius = np.float32(gs.fov_radius)
        fov_radius_sq = fov_radius * fov_radius if fov_radius >= 0 else np.float32(-1.0)

        # --- Create RenderConfig instance ---
        render_config = RenderConfig(
            show_height_vis=self.show_height_visualization,
            vis_max_diff=self._cfg_vis_max_diff,
            vis_color_high_np=self._cfg_height_color_high_np,
            vis_color_mid_np=self._cfg_height_color_mid_np,
            vis_color_low_np=self._cfg_height_color_low_np,
            vis_blend_factor=self._cfg_vis_blend_factor,
            lighting_ambient=self._cfg_ambient_light,
            lighting_min_fov=self._cfg_min_fov_light,
            lighting_falloff=self._cfg_light_falloff,
            fov_radius_sq=fov_radius_sq,  # Pass pre-calculated value
        )
        # --- End Create RenderConfig ---

        # --- Call Renderer ---
        try:
            # Pass arguments to the renderer function
            image = renderer.render_viewport(
                game_state=gs,
                tile_arrays=tile_arrays,
                tile_fg_colors=tile_fg_colors,
                tile_bg_colors=tile_bg_colors,
                tile_indices_render=tile_indices_render,
                max_defined_tile_id=max_defined_tile_id,
                tile_w=tile_w,
                tile_h=tile_h,
                viewport_x=viewport_x,
                viewport_y=viewport_y,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                coord_arrays=coord_arrays,
                render_config=render_config,  # Pass the config object
            )
            return image
        except Exception as e:
            log.error(
                "Error during renderer.render_viewport call",
                error=str(e),
                exc_info=True,
            )
            # Return an error image (size calculation is tricky here, use WM size)
            pw = self.window.label.width()
            ph = self.window.label.height()
            return Image.new(
                "RGBA", (max(1, pw), max(1, ph)), (255, 0, 0, 255)
            )  # Red indicates error

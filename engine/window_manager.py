# engine/window_manager.py
# Standard library imports
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing import Dict as PyDict
from typing import List, Tuple

# Third-party imports
import numpy as np
from PIL import Image

# PySide6 imports
from PySide6.QtCore import Qt, QTimer, QRect # Added QRect
from PySide6.QtGui import (
    QAction, QColor, QCursor, QImage, QKeyEvent,
    QPalette, QPixmap, QResizeEvent, QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication, QLabel, QMenu, QMenuBar, QMessageBox,
    QSizePolicy, QVBoxLayout, QWidget, QScrollArea # Added QScrollArea
)

# --- Modularized Imports ---
from engine.window_manager_modules.input_handler import InputHandler
from engine.window_manager_modules.tileset_manager import TilesetManager
from engine.window_manager_modules.ui_overlay_manager import UIOverlayManager

# --- Type Checking ---
if TYPE_CHECKING:
    from engine.main_loop import MainLoop

# --- Logging Setup ---
import structlog
log = structlog.get_logger(__name__)
# ---

DEFAULT_MIN_TILE_SIZE = 4
DEFAULT_SCROLL_SCALE_DEBOUNCE_MS = 200
DEFAULT_RESIZE_DEBOUNCE_MS = 100
DEFAULT_INITIAL_WINDOW_WIDTH = 1024
DEFAULT_INITIAL_WINDOW_HEIGHT = 768

# lerp_color function
def lerp_color(self, color1: tuple, color2: tuple, t: float) -> tuple:
    """
    Linearly interpolate between two RGB colors.
    
    Args:
        color1: Starting RGB color tuple
        color2: Ending RGB color tuple
        t: Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated RGB color tuple
    """
    # Ensure t is clamped between 0 and 1
    t = max(0.0, min(1.0, t))
    
    # Interpolate each color component
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    
    # Clamp values between 0 and 255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return (r, g, b)


class WindowManager(QWidget):
    def __init__(
        self,
        app_config: PyDict[str, Any],
        keybindings_config: PyDict[str, Any],
        initial_tileset_path: str,
        initial_tile_width: int,
        initial_tile_height: int,
        map_width: int,
        map_height: int,
        min_tile_size_cfg: int = DEFAULT_MIN_TILE_SIZE,
        scroll_debounce_cfg: int = DEFAULT_SCROLL_SCALE_DEBOUNCE_MS,
        resize_debounce_cfg: int = DEFAULT_RESIZE_DEBOUNCE_MS,
    ):
        super().__init__()
        self.app_config = app_config
        self.keybindings_config = keybindings_config
        log.info("Initializing WindowManager...")

        self.min_tile_size = min_tile_size_cfg
        self.scroll_debounce_ms = scroll_debounce_cfg
        self.resize_debounce_ms = resize_debounce_cfg
        self.map_width = map_width
        self.map_height = map_height

        # Instantiate TilesetManager
        self.tileset_manager = TilesetManager(
            initial_tileset_path=initial_tileset_path,
            initial_tile_width=initial_tile_width,
            initial_tile_height=initial_tile_height,
            min_tile_size_cfg=min_tile_size_cfg,
        )

        # Rendering coord cache
        self._render_coord_cache: PyDict[str, np.ndarray] = {}
        self._cached_vp_pixel_dims: Tuple[int, int] | None = None
        self._cached_tile_dims: Tuple[int, int] | None = None

        # --- UI Setup ---
        self.setWindowTitle("Basic Roguelike")
        self.resize(DEFAULT_INITIAL_WINDOW_WIDTH, DEFAULT_INITIAL_WINDOW_HEIGHT)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.menu_bar = QMenuBar(self)
        self.layout.setMenuBar(self.menu_bar)
        self.build_menus()

        # Main display label within a scroll area
        self.scroll_area = QScrollArea() # Defined scroll_area
        self.scroll_area.setBackgroundRole(QPalette.ColorRole.Dark)
        self.scroll_area.setWidgetResizable(False) # Important for fixed size content
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.scroll_area) # Add scroll area to layout

        self.label = QLabel() # The widget that shows the rendered map
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Let it expand if needed
        self.label.setScaledContents(False) # Do not scale the pixmap
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center if smaller than area
        self.label.setAutoFillBackground(True)
        pal = self.label.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(Qt.GlobalColor.black))
        self.label.setPalette(pal)
        # Set the label *inside* the scroll area
        self.scroll_area.setWidget(self.label)
        # --- End UI Setup ---

        self.main_loop: "MainLoop" | None = None
        self.last_rendered_image: Image.Image | None = None

        # Timers
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(self.resize_debounce_ms)
        self._resize_timer.timeout.connect(self.update_frame)
        self._pending_tile_size_change: int = 0
        self._scroll_scale_timer = QTimer(self)
        self._scroll_scale_timer.setSingleShot(True)
        self._scroll_scale_timer.setInterval(self.scroll_debounce_ms)
        self._scroll_scale_timer.timeout.connect(self._apply_scroll_scaling)

        # Active keybindings
        self.active_keybinding_sets: List[str] = [
            "common", "modern", "numpad", "arrows",
        ]
        log.info("Active keybinding sets", sets=self.active_keybinding_sets)

        # Instantiate other handlers
        self.input_handler = InputHandler(self.keybindings_config, self)
        self.ui_overlay_manager = UIOverlayManager(self)

        log.debug("WindowManager __init__ complete")


    def _update_render_coord_cache(self, vp_pixel_w: int, vp_pixel_h: int) -> None:
        """Updates the cache mapping output pixels to viewport tile coords."""
        current_tile_w = self.tileset_manager.tile_width
        current_tile_h = self.tileset_manager.tile_height
        current_tile_dims = (current_tile_w, current_tile_h)
        current_vp_pixel_dims = (vp_pixel_w, vp_pixel_h)

        if ( self._cached_vp_pixel_dims == current_vp_pixel_dims and
             self._cached_tile_dims == current_tile_dims and
             self._render_coord_cache ):
            return # Cache is still valid

        log_context = {
             "vp_pixel_w": vp_pixel_w, "vp_pixel_h": vp_pixel_h,
             "tile_w": current_tile_w, "tile_h": current_tile_h
        }
        log.info("Updating render coordinate cache...", **log_context)

        start_time = time.perf_counter()
        try:
            # Ensure dimensions are valid before creating large arrays
            if vp_pixel_h <= 0 or vp_pixel_w <= 0:
                 raise ValueError("Viewport pixel dimensions must be positive.")
            if current_tile_h <= 0 or current_tile_w <= 0:
                 raise ValueError("Tile dimensions must be positive.")

            px_y_indices, px_x_indices = np.indices(
                (vp_pixel_h, vp_pixel_w), dtype=np.int16
            )
            tile_coord_y = (px_y_indices // current_tile_h).astype(np.int16)
            tile_coord_x = (px_x_indices // current_tile_w).astype(np.int16)

            self._render_coord_cache = {
                "tile_coord_y": tile_coord_y,
                "tile_coord_x": tile_coord_x,
            }
            self._cached_vp_pixel_dims = current_vp_pixel_dims
            self._cached_tile_dims = current_tile_dims

            # *** ADDED LOGGING ***
            log.debug(
                "Render coordinate cache updated",
                duration_ms=(time.perf_counter() - start_time) * 1000,
                coord_y_shape=tile_coord_y.shape,
                coord_x_shape=tile_coord_x.shape,
                **log_context
            )
            # *** END LOGGING ***

        except (ValueError, MemoryError, Exception) as e: # Catch potential errors
            log.error(
                "Failed to update render coordinate cache", error=e, exc_info=True, **log_context
            )
            # Reset cache on failure
            self._render_coord_cache = {}
            self._cached_vp_pixel_dims = None
            self._cached_tile_dims = None


    def build_menus(self) -> None:
        # (Implementation unchanged)
        log.debug("Building menus...")
        tileset_menu = QMenu("Tileset", self)
        try: script_dir = Path(__file__).parent.resolve(); project_root = script_dir.parent
        except NameError: project_root = Path(".")
        png_path_str = str(project_root / "fonts" / "classic_roguelike_sliced")
        use_png_action = QAction("Use PNG Tileset (8x8 base)", self)
        use_png_action.triggered.connect( lambda: self.handle_load_tileset_action(png_path_str, 8, 8) )
        tileset_menu.addAction(use_png_action)
        svg_path_str = str(project_root / "fonts" / "classic_roguelike_sliced_svgs")
        initial_svg_render_size = 16
        use_svg_action = QAction(f"Use SVG Tileset (@{initial_svg_render_size}x)", self)
        use_svg_action.triggered.connect( lambda: self.handle_load_tileset_action( svg_path_str, initial_svg_render_size, initial_svg_render_size ) )
        tileset_menu.addAction(use_svg_action)
        self.menu_bar.addMenu(tileset_menu)
        log.debug("Menus built")

    def handle_load_tileset_action(self, folder: str, width: int, height: int) -> None:
        """Callback for menu actions to load tilesets via the manager."""
        # (Implementation unchanged)
        success = self.tileset_manager.load_new_tileset(folder, width, height)
        if success:
            self._cached_vp_pixel_dims = None; self._cached_tile_dims = None
            self._render_coord_cache = {}; self.update_frame()
        else: QMessageBox.critical( self, "Tileset Error", f"Failed to load tileset from:\n{folder}" )

    def set_main_loop(self, main_loop: "MainLoop") -> None:
        # (Implementation unchanged)
        self.main_loop = main_loop
        log.info("MainLoop instance set in WindowManager")
        QTimer.singleShot(0, self.update_frame) # Trigger initial frame render

    def resizeEvent(self, event: QResizeEvent) -> None:
        # (Implementation unchanged - uses debounce timer)
        log.debug("Resize event detected", new_size=event.size())
        self._resize_timer.start() # Debounced update_frame call
        super().resizeEvent(event)

    def update_frame(self) -> None:
        """Updates and redraws the main display label."""
        frame_start_time = time.perf_counter()
        if ( not self.main_loop or not self.main_loop.game_state or
             not self.tileset_manager or not self.isVisible() ):
            log.debug( "Skipping frame update: Components not ready or window not visible.",
                       has_loop=(self.main_loop is not None),
                       has_gs=(hasattr(self.main_loop, 'game_state') if self.main_loop else False),
                       has_tsm=(self.tileset_manager is not None),
                       is_visible=self.isVisible() )
            # self.label.clear() # Maybe don't clear if just invisible?
            return

        # Use scroll area viewport size for calculations, not label size
        viewport_w = self.scroll_area.viewport().width()
        viewport_h = self.scroll_area.viewport().height()
        current_tile_w = self.tileset_manager.tile_width
        current_tile_h = self.tileset_manager.tile_height

        if viewport_w <= 0 or viewport_h <= 0 or current_tile_w <= 0 or current_tile_h <= 0:
            log.warning( "Skipping frame: Invalid viewport/tile dimensions",
                         vp_w=viewport_w, vp_h=viewport_h,
                         tile_w=current_tile_w, tile_h=current_tile_h )
            self.label.clear() # Clear display if dimensions are invalid
            return

        gs = self.main_loop.game_state
        # Calculate visible tiles based on viewport size
        visible_cols = max(1, viewport_w // current_tile_w)
        visible_rows = max(1, viewport_h // current_tile_h)

        # Calculate camera/viewport position based on player
        player_pos = gs.player_position
        cam_x, cam_y = ( player_pos if player_pos else (gs.map_width // 2, gs.map_height // 2) )

        # Calculate viewport tile coordinates (top-left corner)
        # Ensure viewport doesn't go out of map bounds
        render_cols = min(visible_cols, gs.map_width) # Cannot render more tiles than map width
        render_rows = min(visible_rows, gs.map_height)
        viewport_tile_x = max(0, min(cam_x - render_cols // 2, gs.map_width - render_cols))
        viewport_tile_y = max(0, min(cam_y - render_rows // 2, gs.map_height - render_rows))

        # Calculate actual number of tiles to render based on map limits
        vp_render_tile_w = min(render_cols, gs.map_width - viewport_tile_x)
        vp_render_tile_h = min(render_rows, gs.map_height - viewport_tile_y)

        if vp_render_tile_w <= 0 or vp_render_tile_h <= 0:
            log.warning("Calculated viewport tile dimensions are zero or negative",
                         w=vp_render_tile_w, h=vp_render_tile_h)
            self.label.clear()
            return

        # Calculate the required pixel size of the output image
        output_pixel_w = vp_render_tile_w * current_tile_w
        output_pixel_h = vp_render_tile_h * current_tile_h

        # Update coordinate cache if necessary
        try:
            self._update_render_coord_cache(output_pixel_w, output_pixel_h)
        except Exception as cache_err:
            log.error("Render coordinate cache update failed", error=cache_err, exc_info=True)
            # Attempt to display error on screen? For now, clear.
            self.label.clear()
            return # Cannot render without valid cache

        if not self._render_coord_cache:
            log.error("Render coordinate cache is empty, cannot render.")
            self.label.clear()
            return

        # Fetch render data from TilesetManager
        render_data = self.tileset_manager.get_render_data()
        if render_data["max_defined_tile_id"] < 0:
            log.error("TilesetManager reported invalid render data cache.")
            self.label.clear()
            return

        # Call MainLoop's update_console to get the rendered image
        try:
            rendered_image: Image.Image | None = self.main_loop.update_console(
                game_state=gs,
                viewport_x=viewport_tile_x, # Pass calculated viewport tile coords
                viewport_y=viewport_tile_y,
                viewport_width=vp_render_tile_w, # Pass calculated viewport tile dimensions
                viewport_height=vp_render_tile_h,
                # Pass data from TilesetManager
                tile_arrays=render_data["tile_arrays"],
                tile_fg_colors=render_data["tile_fg_colors"],
                tile_bg_colors=render_data["tile_bg_colors"],
                tile_indices_render=render_data["tile_indices_render"],
                max_defined_tile_id=render_data["max_defined_tile_id"],
                tile_w=render_data["tile_w"], # Current tile dimensions
                tile_h=render_data["tile_h"],
                # Pass coordinate cache
                coord_arrays=self._render_coord_cache,
            )
        except Exception as e:
            # Catch errors during the update_console call itself
            log.error("Error during main_loop.update_console call", error=e, exc_info=True)
            rendered_image = None # Ensure image is None on error

        # Process the result
        if rendered_image:
            self.last_rendered_image = rendered_image
            # Add overlays using UIOverlayManager
            img_with_overlays = self.ui_overlay_manager.render_overlays(
                self.last_rendered_image, gs, self.main_loop
            )
            # Display final image on the label
            try:
                img_rgba = img_with_overlays.convert("RGBA")
                data = img_rgba.tobytes("raw", "RGBA")
                qimg = QImage(
                    data, img_with_overlays.width, img_with_overlays.height,
                    QImage.Format.Format_RGBA8888,
                )
                if qimg.isNull():
                    log.error("QImage conversion failed.")
                    self.label.clear() # Clear label if conversion fails
                else:
                    # Resize the label to match the image size before setting pixmap
                    self.label.setFixedSize(qimg.width(), qimg.height())
                    self.label.setPixmap(QPixmap.fromImage(qimg))

            except Exception as e:
                 log.error("Error converting/displaying final image", error=e, exc_info=True)
                 self.label.clear()

        else:
            log.warning("MainLoop returned no base image.")
            self.label.clear() # Clear display if no image returned
            self.last_rendered_image = None

        log.debug( "Frame update finished", duration_ms=(time.perf_counter() - frame_start_time) * 1000 )


    def keyPressEvent(self, event: QKeyEvent) -> None:
        # (Implementation unchanged)
        if ( not self.main_loop or not self.main_loop.game_state or not self.input_handler ):
            event.ignore(); return
        key_handled = self.input_handler.process_key_event( event, self.main_loop.game_state,
                                                             self.main_loop, self.active_keybinding_sets )
        if not key_handled: super().keyPressEvent(event)

    def show_help_dialog(self) -> None:
        # (Implementation unchanged)
        bindings_sets = self.keybindings_config.get("bindings", {})
        help_text = "<h2>Controls Help</h2>"
        if not bindings_sets: help_text += "<p><i>Error: No keybindings found!</i></p>"
        else:
            def _format_control_string(binding_dict: PyDict[str, Any]) -> str:
                key_str = binding_dict.get("key", "?"); mods_list = binding_dict.get("mods", [])
                display_key = key_str.replace("KP_", "Numpad "); mods_str = "+".join(m.capitalize() for m in mods_list if m)
                prefix_parts = [mods_str] if mods_str else []; prefix_parts.append(f"'{display_key}'"); return " + ".join(prefix_parts)
            grouped_bindings: PyDict[str, List[str]] = {}
            for set_name, set_dict in bindings_sets.items():
                if not isinstance(set_dict, dict): continue
                for action_name, action_dict in set_dict.items():
                    if not isinstance(action_dict, dict): continue
                    description = action_dict.get("desc")
                    if description:
                        formatted_control = _format_control_string(action_dict)
                        if description not in grouped_bindings: grouped_bindings[description] = []
                        if formatted_control not in grouped_bindings[description]: grouped_bindings[description].append(formatted_control)
            help_text += "<ul>"
            for desc_key in sorted(grouped_bindings.keys()):
                controls_str = " / ".join(sorted(grouped_bindings[desc_key])); help_text += f"<li>{controls_str}: {desc_key}</li>"
            help_text += "</ul>"
        msg_box = QMessageBox(self); msg_box.setWindowTitle("Help - Controls"); msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(help_text); msg_box.setIcon(QMessageBox.Icon.Information); msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    # --- UI Callback methods ---
    def ui_open_inventory_view(self) -> None:
        # (Implementation unchanged)
        if self.main_loop and self.main_loop.game_state:
            self.main_loop.game_state.change_ui_state("INVENTORY_VIEW")
            self.ui_overlay_manager.reset_inventory_state()
            self.update_frame()

    def ui_toggle_height_visualization(self) -> None:
        # (Implementation unchanged)
        if self.main_loop:
            self.main_loop.show_height_visualization = ( not self.main_loop.show_height_visualization )
            log.info( "Height vis toggled", enabled=self.main_loop.show_height_visualization )
            self.update_frame()

    def ui_quit_game(self) -> None:
        # (Implementation unchanged)
        app_instance = QApplication.instance()
        if app_instance: app_instance.quit()

    def ui_return_to_player_turn(self) -> None:
        # (Implementation unchanged)
        if self.main_loop and self.main_loop.game_state:
            self.main_loop.game_state.change_ui_state("PLAYER_TURN")
            self.ui_overlay_manager.reset_inventory_state()
            self.update_frame()

    def ui_show_help_dialog(self) -> None:
        # (Implementation unchanged)
        self.show_help_dialog()

    # --- wheelEvent ---
    def wheelEvent(self, event: QWheelEvent) -> None:
        # (Implementation unchanged)
        if not self.main_loop: return
        modifiers = event.modifiers(); angle_delta = event.angleDelta()
        if modifiers & Qt.KeyboardModifier.ControlModifier: # Zooming
            delta_y = angle_delta.y(); change = 1 if delta_y > 0 else -1 if delta_y < 0 else 0
            if change != 0: self._pending_tile_size_change += change; self._scroll_scale_timer.start(self.scroll_debounce_ms)
        else: super().wheelEvent(event)

    def _apply_scroll_scaling(self) -> None:
        """Applies pending zoom changes."""
        # Ensure scroll_area exists before using it
        if not hasattr(self, 'scroll_area') or not self.scroll_area:
            log.error("_apply_scroll_scaling called without scroll_area")
            self._pending_tile_size_change = 0
            return
        # (Rest of implementation unchanged)
        if ( not self.main_loop or self._pending_tile_size_change == 0 or not self.tileset_manager ): return
        viewport_widget = self.scroll_area.viewport()
        if not viewport_widget: log.warning("Scroll area viewport missing for zoom"); self._pending_tile_size_change = 0; return
        h_bar = self.scroll_area.horizontalScrollBar(); v_bar = self.scroll_area.verticalScrollBar()
        mouse_pos_in_viewport = viewport_widget.mapFromGlobal(QCursor.pos())
        content_x_at_mouse_before = h_bar.value() + mouse_pos_in_viewport.x()
        content_y_at_mouse_before = v_bar.value() + mouse_pos_in_viewport.y()
        old_tile_w = self.tileset_manager.tile_width; old_tile_h = self.tileset_manager.tile_height
        if old_tile_w <= 0 or old_tile_h <= 0: log.error("Old tile size invalid"); return
        grid_x_at_mouse = content_x_at_mouse_before / old_tile_w; grid_y_at_mouse = content_y_at_mouse_before / old_tile_h
        target_width = old_tile_w + self._pending_tile_size_change; target_height = old_tile_h + self._pending_tile_size_change
        min_sz = self.min_tile_size; max_sz = self.app_config.get("max_tile_size", 64)
        new_width = max(min_sz, min(target_width, max_sz)); new_height = max(min_sz, min(target_height, max_sz))
        accumulated_change = self._pending_tile_size_change; self._pending_tile_size_change = 0
        if new_width != old_tile_w or new_height != old_tile_h:
            log.info( "Applying zoom", change=accumulated_change, old=f"{old_tile_w}x{old_tile_h}", new=f"{new_width}x{new_height}" )
            success = self.tileset_manager.load_new_tileset( self.tileset_manager.current_tileset_path, new_width, new_height )
            if success:
                self._cached_vp_pixel_dims = None; self._cached_tile_dims = None; self._render_coord_cache = {}
                # update_frame is triggered by load_new_tileset if successful
                def recenter_view_after_zoom():
                    QApplication.processEvents() # Allow resize events to process
                    current_h_bar = self.scroll_area.horizontalScrollBar(); current_v_bar = self.scroll_area.verticalScrollBar()
                    new_content_x_at_grid = grid_x_at_mouse * self.tileset_manager.tile_width
                    new_content_y_at_grid = grid_y_at_mouse * self.tileset_manager.tile_height
                    current_h_bar.setValue( int(new_content_x_at_grid - mouse_pos_in_viewport.x()) )
                    current_v_bar.setValue( int(new_content_y_at_grid - mouse_pos_in_viewport.y()) )
                    log.debug("Recentered view after successful zoom.")
                QTimer.singleShot(0, recenter_view_after_zoom)
            else: log.error("Zoom failed because tileset failed to load new size.")
        else: log.debug("Zoom resulted in no size change (min/max limits?).")

# engine/window_manager.py
import math
import traceback
import structlog
import time
from pathlib import Path
from PIL import Image, ImageDraw

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QMenuBar,
    QMenu,
    QSizePolicy,
    QMessageBox,
)
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QWheelEvent, QAction, QResizeEvent
from PySide6.QtCore import Qt, QTimer

# Use absolute import for engine module sibling
from engine.tileset_loader import load_tiles
import numpy as np
import polars as pl
from typing import TYPE_CHECKING, Dict, Any  # Added Dict, Any

# Use absolute import for game module
from game.world.game_map import TILE_TYPES  # Import TILE_TYPES here

if TYPE_CHECKING:
    from engine.main_loop import MainLoop  # Keep relative for TYPE_CHECKING

log = structlog.get_logger()

DEFAULT_MIN_TILE_SIZE = 4
DEFAULT_SCROLL_SCALE_DEBOUNCE_MS = 200
DEFAULT_RESIZE_DEBOUNCE_MS = 100
DEFAULT_INITIAL_WINDOW_WIDTH = 1024
DEFAULT_INITIAL_WINDOW_HEIGHT = 768


def lerp_color(color1, color2, factor):  # Keep helper
    factor = max(0.0, min(1.0, factor))
    r = int(color1[0] + (color2[0] - color1[0]) * factor)
    g = int(color1[1] + (color2[1] - color1[1]) * factor)
    b = int(color1[2] + (color2[2] - color1[2]) * factor)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


class WindowManager(QWidget):
    def __init__(
        self,
        app_config: dict,
        keybindings_config: dict,
        initial_tileset_path: str,
        initial_tiles: dict[int, Image.Image],
        initial_tile_width: int,
        initial_tile_height: int,
        map_width: int,  # Still useful for initial window sizing/checks
        map_height: int,
        min_tile_size_cfg: int = DEFAULT_MIN_TILE_SIZE,
        scroll_debounce_cfg: int = DEFAULT_SCROLL_SCALE_DEBOUNCE_MS,
        resize_debounce_cfg: int = DEFAULT_RESIZE_DEBOUNCE_MS,
    ):
        super().__init__()
        self.app_config = app_config  # Store the main config dictionary
        self.keybindings_config = keybindings_config  # Store the keybindings dictionary
        log.info("Initializing WindowManager...")
        self.current_tileset_path = initial_tileset_path
        self.tiles: dict[int, Image.Image] = initial_tiles
        self.tile_width: int = initial_tile_width
        self.tile_height: int = initial_tile_height
        # Store map dims if needed, but primarily handled by GameState now
        # self.map_width: int = map_width
        # self.map_height: int = map_height
        self.min_tile_size = min_tile_size_cfg
        self.scroll_debounce_ms = scroll_debounce_cfg
        self.resize_debounce_ms = resize_debounce_cfg
        log.debug("WindowManager config parameters set", ...)  # Abbreviated log
        self.tile_arrays: dict[int, np.ndarray | None] = {}

        # --- MOVED: Tile Cache Arrays from MainLoop ---
        self.max_defined_tile_id: int = -1  # Initialize
        self._tile_fg_colors: np.ndarray | None = None  # Initialize
        self._tile_bg_colors: np.ndarray | None = None
        self._tile_indices_render: np.ndarray | None = None
        self._update_tile_array_cache()  # Populate cache arrays
        # --- End MOVED ---

        self.setWindowTitle("Basic Roguelike")
        self.resize(DEFAULT_INITIAL_WINDOW_WIDTH, DEFAULT_INITIAL_WINDOW_HEIGHT)
        log.info("Window initialized")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.menu_bar = QMenuBar(self)
        self.layout.setMenuBar(self.menu_bar)
        self.build_menus()
        self.label = QLabel()
        self.label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)
        self.main_loop: "MainLoop" | None = None
        self.last_rendered_image: Image.Image | None = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(self.resize_debounce_ms)
        self._resize_timer.timeout.connect(self.update_frame)
        self._pending_tile_size_change: int = 0
        self._scroll_scale_timer = QTimer(self)
        self._scroll_scale_timer.setSingleShot(True)
        self._scroll_scale_timer.setInterval(self.scroll_debounce_ms)
        self._scroll_scale_timer.timeout.connect(self._apply_scroll_scaling)
        log.debug("Timers initialized")

    def _update_tile_array_cache(self) -> None:
        """Updates tile_arrays dict and populates color/index cache arrays."""
        log.debug("Updating tile array cache and render cache arrays...")
        self.tile_arrays.clear()
        count = 0
        if not self.tiles or self.tile_width <= 0 or self.tile_height <= 0:
            log.warning("Cannot update tile cache: Invalid tiles or dimensions.", ...)
            # Reset cache arrays if invalid
            self.max_defined_tile_id = -1
            self._tile_fg_colors = None
            self._tile_bg_colors = None
            self._tile_indices_render = None
            return

        # --- MOVED: Cache Array Population ---
        self.max_defined_tile_id = max(TILE_TYPES.keys()) if TILE_TYPES else -1
        array_size = self.max_defined_tile_id + 1

        self._tile_fg_colors = np.zeros((array_size, 3), dtype=np.uint8)
        self._tile_bg_colors = np.zeros((array_size, 3), dtype=np.uint8)
        self._tile_indices_render = np.zeros(array_size, dtype=np.uint16)

        if array_size > 0:
            valid_ids_loaded = 0
            # Use TILE_TYPES defined globally (or passed in if preferred)
            for tile_id, tile_type in TILE_TYPES.items():
                if 0 <= tile_id <= self.max_defined_tile_id:
                    self._tile_fg_colors[tile_id] = tile_type.color_fg
                    self._tile_bg_colors[tile_id] = tile_type.color_bg
                    self._tile_indices_render[tile_id] = tile_type.tile_index
                    valid_ids_loaded += 1
                # No else needed if TILE_TYPES is assumed correct
            log.debug(
                "Tile render cache populated",
                loaded_ids=valid_ids_loaded,
                cache_size=array_size,
            )
        else:
            log.warning(
                "TILE_TYPES dictionary is empty.",
                detail="Tile cache arrays not populated.",
            )
            self._tile_fg_colors = None
            self._tile_bg_colors = None
            self._tile_indices_render = None
        # --- End MOVED ---

        # Populate tile_arrays (NumPy RGBA versions of images)
        for tile_index, img in self.tiles.items():
            if img is None:
                self.tile_arrays[tile_index] = None
                continue
            try:
                if img.size != (self.tile_width, self.tile_height):
                    img = img.resize(
                        (self.tile_width, self.tile_height), Image.Resampling.NEAREST
                    )
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                self.tile_arrays[tile_index] = np.array(img, dtype=np.uint8)
                count += 1
            except Exception as e:
                log.warning(
                    "Could not convert tile to NumPy array",
                    tile_index=tile_index,
                    error=str(e),
                )
                self.tile_arrays[tile_index] = None
        log.info(
            "Tile array cache updated", cached_count=count, total_tiles=len(self.tiles)
        )

    def build_menus(self) -> None:
        # (Unchanged)
        log.debug("Building menus...")
        tileset_menu = QMenu("Tileset", self)
        png_path = "fonts/classic_roguelike_sliced"
        use_png_action = QAction("Use PNG Tileset (8x8 base)", self)
        use_png_action.triggered.connect(lambda: self.load_tileset(png_path, 8, 8))
        tileset_menu.addAction(use_png_action)
        svg_path = "fonts/classic_roguelike_sliced_svgs"
        initial_svg_render_size = 16
        use_svg_action = QAction(f"Use SVG Tileset (@{initial_svg_render_size}x)", self)
        use_svg_action.triggered.connect(
            lambda: self.load_tileset(
                svg_path, initial_svg_render_size, initial_svg_render_size
            )
        )
        tileset_menu.addAction(use_svg_action)
        self.menu_bar.addMenu(tileset_menu)
        log.debug("Menus built")

    def load_tileset(self, folder: str, width: int, height: int) -> None:
        # (Unchanged)
        if not self.main_loop:
            return
        try:
            clamped_width = max(self.min_tile_size, width)
            clamped_height = max(self.min_tile_size, height)
            if (
                clamped_width == self.tile_width
                and clamped_height == self.tile_height
                and folder == self.current_tileset_path
            ):
                log.info("Tileset unchanged, skipping reload.", path=folder)
                return
            log.info(
                "Loading tileset",
                path=folder,
                base_w=width,
                base_h=height,
                clamped_w=clamped_width,
                clamped_h=clamped_height,
            )
            # --- MODIFIED: Pass absolute path if needed ---
            # Assuming 'folder' is relative to script dir as handled in main.py
            # If not, path needs adjustment here or in main.py
            abs_folder_path = str(
                Path(folder).resolve()
            )  # Example if path needs resolving
            loaded_tiles, _ = load_tiles(abs_folder_path, clamped_width, clamped_height)
            # --- END MODIFIED ---
            self.current_tileset_path = folder  # Store original relative path maybe? Or absolute? Needs consistency. Store abs path for now.
            self.current_tileset_path = abs_folder_path
            self.tiles = loaded_tiles
            self.tile_width = clamped_width
            self.tile_height = clamped_height
            self._update_tile_array_cache()  # Regenerate tile arrays and cache arrays
            self.update_frame()
            log.info(
                "Tileset loaded successfully",
                path=abs_folder_path,
                final_w=self.tile_width,
                final_h=self.tile_height,
            )
        except Exception as e:
            log.error("Error loading tileset", path=folder, error=str(e), exc_info=True)

    def set_main_loop(self, main_loop: "MainLoop") -> None:
        # (Unchanged)
        self.main_loop = main_loop
        log.info("MainLoop instance set in WindowManager")
        QTimer.singleShot(0, self.update_frame)

    def resizeEvent(self, event: QResizeEvent) -> None:
        # (Unchanged)
        log.debug("Resize event detected", new_size=event.size())
        self._resize_timer.start()
        super().resizeEvent(event)

    def update_frame(self) -> None:
        """Requests a frame update from the MainLoop."""
        # This method no longer calls rendering directly.
        # It ensures MainLoop's update cycle (which includes calling the renderer) runs.
        frame_start_time = time.perf_counter()
        log.debug("Update frame requested in WindowManager...")
        if not self.main_loop:
            log.warning("Skipping frame update request: MainLoop not set.")
            return
        if self.label.width() <= 0 or self.label.height() <= 0:
            log.warning("Skipping frame update request: Invalid label size")
            self.label.clear()
            return

        try:
            # MainLoop's update_console method now orchestrates getting data
            # and calling the renderer.
            rendered_image = self.main_loop.update_console()  # Call simplified method

            if rendered_image:
                self.last_rendered_image = rendered_image
                img_with_debug = self.get_image_with_debug()  # Add overlays locally
                img_rgba = img_with_debug.convert("RGBA")
                data = img_rgba.tobytes("raw", "RGBA")
                qimg = QImage(
                    data,
                    img_with_debug.width,
                    img_with_debug.height,
                    QImage.Format.Format_RGBA8888,
                )
                if qimg.isNull():
                    log.error("QImage conversion resulted in a null image.")
                    self.label.clear()
                else:
                    pixmap = QPixmap.fromImage(qimg)
                    self.label.setPixmap(pixmap)
            else:
                log.warning("MainLoop returned no image to render.")
                self.label.clear()
                self.last_rendered_image = None
        except Exception as e:
            log.error(
                "Error during frame update cycle in WindowManager",
                error=str(e),
                exc_info=True,
            )
            self.last_rendered_image = None
            self.label.clear()  # Clear label on error

        frame_end_time = time.perf_counter()
        log.debug(
            "Frame update finished in WindowManager",
            duration_ms=(frame_end_time - frame_start_time) * 1000,
        )

    def get_image_with_debug(self) -> Image.Image:
        # (Implementation remains the same, using self.last_rendered_image)
        if not self.last_rendered_image:
            return Image.new(
                "RGBA", (self.label.width(), self.label.height()), (0, 0, 0, 255)
            )
        # ... (rest of debug text and height key overlay logic) ...
        img_copy = self.last_rendered_image.copy()
        draw = ImageDraw.Draw(img_copy)
        # --- Draw Debug Text ---
        debug_text = "Debug info unavailable"
        try:
            if self.main_loop and self.main_loop.game_state:
                gs = self.main_loop.game_state
                turn = gs.turn_count
                player_pos = gs.player_position
                pos_str = f"({player_pos[0]},{player_pos[1]})" if player_pos else "N/A"
                entities = "?"
                try:  # Safely access entity count
                    if gs.entity_registry and hasattr(
                        gs.entity_registry, "entities_df"
                    ):
                        entities = gs.entity_registry.entities_df.filter(
                            pl.col("is_active")
                        ).height
                except Exception:
                    pass
                label_w = self.label.width()
                label_h = self.label.height()
                vp_cols = (
                    max(1, label_w // self.tile_width) if self.tile_width > 0 else "?"
                )
                vp_rows = (
                    max(1, label_h // self.tile_height) if self.tile_height > 0 else "?"
                )
                vp_size_str = f"{vp_cols}x{vp_rows}"
                render_size = f"{self.tile_width}x{self.tile_height}"
                vis_mode = "H" if self.main_loop.show_height_visualization else "-"
                debug_text = f"T:{turn} | P:{pos_str} | E:{entities} | VP:{vp_size_str} | TR:{render_size} | V:{vis_mode}"
            else:
                log.warning("Cannot get debug info: MainLoop or GameState missing.")
        except Exception as e:
            log.error("Error getting debug info", error=str(e))
            debug_text = "Debug info error"
        text_font = None
        text_x = 5
        text_y = 5
        text_color = (255, 255, 0, 255)
        bg_color = (0, 0, 0, 180)
        box_height = 15
        try:
            draw.rectangle([(0, 0), (img_copy.width, box_height)], fill=bg_color)
            draw.text((text_x, text_y), debug_text, fill=text_color, font=text_font)
        except Exception as e:
            log.error("Error drawing debug text", error=str(e))
        # --- Draw Height Key Overlay ---
        if self.main_loop and self.main_loop.show_height_visualization:
            log.debug("Drawing height key overlay")
            try:
                max_diff_units = self.main_loop._cfg_vis_max_diff
                max_diff_meters = max_diff_units / 2.0
                color_high = tuple(self.main_loop._cfg_height_color_high_np)
                color_mid = tuple(self.main_loop._cfg_height_color_mid_np)
                color_low = tuple(self.main_loop._cfg_height_color_low_np)
                key_width = 15
                key_height = 100
                key_x = 10
                key_y = box_height + 10
                key_label_offset = 5
                for i in range(key_height):
                    t = ((key_height - 1 - i) / (key_height - 1)) * 2.0 - 1.0
                    line_color = (
                        lerp_color(color_mid, color_high, t)
                        if t > 0
                        else lerp_color(color_mid, color_low, -t)
                    )
                    draw.line(
                        [(key_x, key_y + i), (key_x + key_width - 1, key_y + i)],
                        fill=line_color + (255,),
                    )
                mid_y = key_y + key_height // 2
                draw.line(
                    [(key_x - 2, mid_y), (key_x + key_width + 1, mid_y)],
                    fill=(255, 255, 255, 255),
                    width=1,
                )
                label_x = key_x + key_width + key_label_offset
                label_color = (200, 200, 200, 255)
                draw.text(
                    (label_x, key_y - 4),
                    f"+{max_diff_meters:.1f}m",
                    fill=label_color,
                    font=text_font,
                )
                draw.text((label_x, mid_y - 4), "0m", fill=label_color, font=text_font)
                draw.text(
                    (label_x, key_y + key_height - 4),
                    f"-{max_diff_meters:.1f}m",
                    fill=label_color,
                    font=text_font,
                )
            except Exception as e:
                log.error("Error drawing height key", error=str(e))
        return img_copy

    def keyPressEvent(self, event: QKeyEvent) -> None:
        log.debug("Key pressed", key=event.key(), text=event.text())
        key = event.key()  # Get key code

        # --- Explicit F1 Check ---
        if key == Qt.Key.Key_F1:
            print("DEBUG: F1 key explicitly detected in keyPressEvent!")
            try:
                # Directly call the help dialog method
                self.show_help_dialog()
                event.accept()  # Mark event as handled
                return  # Stop further processing for this key
            except Exception as e:
                log.error(
                    "Error calling show_help_dialog directly",
                    error=str(e),
                    exc_info=True,
                )
                # event.ignore() # Allow potential fall-through if needed, but usually accept on error
        # --- END EXPLICIT F1 CHECK ---

        # --- Other Key Handling ---

        # Escape/Quit check
        if key == Qt.Key.Key_Escape or key == Qt.Key.Key_Q:
            log.info("Quit key pressed.")
            self.close()
            event.accept()  # Mark event as handled
            return

        # V key check
        elif key == Qt.Key.Key_V:
            if self.main_loop:
                self.main_loop.show_height_visualization = (
                    not self.main_loop.show_height_visualization
                )
                log.info(
                    "Height visualization toggled",
                    enabled=self.main_loop.show_height_visualization,
                )
                self.update_frame()  # Redraw after toggle
                event.accept()  # Mark event as handled
                return
            else:
                log.warning("Cannot toggle visualization: MainLoop not set.")
                event.ignore()  # Ignore if no main loop
                return

        # --- Action Mapping (Movement, Pickup, Wait, etc.) ---
        elif self.main_loop:
            action: dict | None = None
            # Map movement and other action keys
            if key in (Qt.Key.Key_Up, Qt.Key.Key_K, Qt.Key.Key_W):
                action = {"type": "move", "dx": 0, "dy": -1}
                # print("up") # Keep for debugging if needed
            elif key in (Qt.Key.Key_Down, Qt.Key.Key_J, Qt.Key.Key_S):
                action = {"type": "move", "dx": 0, "dy": 1}
            elif key in (Qt.Key.Key_Left, Qt.Key.Key_H, Qt.Key.Key_A):
                action = {"type": "move", "dx": -1, "dy": 0}
            elif key in (Qt.Key.Key_Right, Qt.Key.Key_L, Qt.Key.Key_D):
                action = {"type": "move", "dx": 1, "dy": 0}
            elif key == Qt.Key.Key_Home or key == Qt.Key.Key_Y or key == Qt.Key.Key_7:
                action = {"type": "move", "dx": -1, "dy": -1}
            elif key == Qt.Key.Key_End or key == Qt.Key.Key_B or key == Qt.Key.Key_1:
                action = {"type": "move", "dx": -1, "dy": 1}
            elif key == Qt.Key.Key_PageUp or key == Qt.Key.Key_U or key == Qt.Key.Key_9:
                action = {"type": "move", "dx": 1, "dy": -1}
            elif (
                key == Qt.Key.Key_PageDown or key == Qt.Key.Key_N or key == Qt.Key.Key_3
            ):
                action = {"type": "move", "dx": 1, "dy": 1}
            elif (
                key == Qt.Key.Key_Period
                or key == Qt.Key.Key_Space
                or key == Qt.Key.Key_5
            ):
                action = {"type": "wait"}
            elif key == Qt.Key.Key_G:
                action = {"type": "pickup"}

            # If an action was mapped, process it
            if action:
                print(f"WindowManager: Action created: {action}")  # Debug print
                # Let handle_action determine if turn was taken and if redraw needed
                self.main_loop.handle_action(action)
                # Assuming handle_action now calls update_frame if needed
                event.accept()  # Mark event as handled
                return
            else:
                log.debug(
                    "Unmapped key pressed while main_loop active",
                    key=key,
                    text=event.text(),
                )
                event.ignore()  # Pass unmapped keys through if main_loop active
                return
        else:
            # main_loop not set, ignore most keys
            log.warning("Key press ignored: MainLoop not set.")
            event.ignore()
            return

        # Fallback: If no specific handler accepted the event
        event.ignore()

    # Add this method to the WindowManager class
    def show_help_dialog(self):
        """Displays controls help based on keybindings_config."""
        # print("DEBUG: show_help_dialog method called!") # Can remove this now

        # Read controls from the stored keybindings config
        bindings_sets = self.keybindings_config.get(
            "bindings", {}
        )  # Get the dict of sets { 'common': {...}, 'modern': {...}, ...}
        if not bindings_sets:
            help_text = "<h2>Controls Help</h2><p><i>Error: No keybindings found! Check config/keybindings.toml</i></p>"
        else:
            help_text = "<h2>Controls Help</h2>"

            # Helper to format control strings from TOML data
            def _format_control_string(control_data):
                # Source: [source 230-233] - This helper function appears correct
                key = control_data.get("key", "?")
                mods = control_data.get("mods", [])
                desc = control_data.get("desc", "-")
                mod_str = "+".join(
                    m.capitalize() for m in mods if m
                )  # Filter empty mods
                parts = []
                if mod_str:
                    parts.append(mod_str)
                # Display Numpad keys nicely
                display_key = key.replace("KP_", "Numpad ")
                parts.append(f"'{display_key}'")
                return f"{' + '.join(parts)}: {desc}"

            # Group bindings by description or action_type for better readability
            # Grouping by description might consolidate alternate keys better
            grouped_bindings = {}
            # --- MODIFIED ITERATION ---
            for (
                _set_name,
                set_data,
            ) in (
                bindings_sets.items()
            ):  # Iterate through sets ('common', 'modern', etc.)
                if not isinstance(set_data, dict):
                    continue  # Skip if set_data isn't a dict

                for (
                    _action_name,
                    action_data,
                ) in (
                    set_data.items()
                ):  # Iterate through actions in the set ('move_n', 'pickup')
                    if not isinstance(action_data, dict):
                        continue  # Skip if action_data isn't a dict

                    desc = action_data.get(
                        "desc"
                    )  # Get description from the individual action
                    if not desc:
                        continue  # Skip actions without descriptions for the help dialog

                    # Pass the individual action's data to the formatter
                    fmt = _format_control_string(
                        action_data
                    )  # <-- CORRECTED data passed

                    if desc not in grouped_bindings:
                        grouped_bindings[desc] = []
                    # Avoid duplicate formatting strings if multiple keys map to the same action/description
                    if fmt not in grouped_bindings[desc]:
                        grouped_bindings[desc].append(fmt)
            # --- END MODIFIED ITERATION ---

            # Sort descriptions alphabetically for consistency
            sorted_descs = sorted(grouped_bindings.keys())

            help_text += "<ul>"
            for desc in sorted_descs:
                # Join multiple key options for the same description
                # The formatting is now done correctly before appending to the list
                keys_str = " / ".join(grouped_bindings[desc])
                # Display the description followed by the keys
                help_text += f"<li>{desc}: {keys_str.split(': ')[-1]}</li>"  # Simpler formatting assuming helper adds ': desc'
            help_text += "</ul>"

        # Display the message box
        msg = QMessageBox(self)
        msg.setWindowTitle("Help - Controls")
        msg.setTextFormat(Qt.TextFormat.RichText)  # Use RichText for HTML
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def wheelEvent(self, event: QWheelEvent) -> None:
        # (Unchanged)
        if not self.main_loop:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            change = 1 if delta > 0 else -1 if delta < 0 else 0
            log.debug("Wheel event for zoom", delta=delta, change=change)
            if change != 0:
                self._pending_tile_size_change += change
                self._scroll_scale_timer.start(self.scroll_debounce_ms)
        else:
            super().wheelEvent(event)

    def _apply_scroll_scaling(self) -> None:
        # (Unchanged)
        if not self.main_loop or self._pending_tile_size_change == 0:
            return
        target_width = self.tile_width + self._pending_tile_size_change
        target_height = self.tile_height + self._pending_tile_size_change
        new_width = max(self.min_tile_size, target_width)
        new_height = max(self.min_tile_size, target_height)
        accumulated_change = self._pending_tile_size_change
        self._pending_tile_size_change = 0
        if new_width != self.tile_width or new_height != self.tile_height:
            log.info(
                "Applying debounced scaling",
                accumulated_change=accumulated_change,
                old_w=self.tile_width,
                old_h=self.tile_height,
                target_w=target_width,
                target_h=target_height,
                new_w=new_width,
                new_h=new_height,
            )
            self.load_tileset(self.current_tileset_path, new_width, new_height)
        else:
            log.info(
                "Debounced scaling resulted in no size change.",
                accumulated_change=accumulated_change,
                current_size=f"{self.tile_width}x{self.tile_height}",
            )

# engine/window_manager.py
import math
import traceback
import structlog
import time
from pathlib import Path
from PIL import Image, ImageDraw  # Added ImageDraw

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QMenuBar, QMenu, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QWheelEvent, QAction, QResizeEvent
from PySide6.QtCore import Qt, QTimer

from engine.tileset_loader import load_tiles
import numpy as np
import polars as pl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.main_loop import MainLoop

log = structlog.get_logger()

# --- Defaults (Unchanged) ---
DEFAULT_MIN_TILE_SIZE = 4
DEFAULT_SCROLL_SCALE_DEBOUNCE_MS = 200
DEFAULT_RESIZE_DEBOUNCE_MS = 100
DEFAULT_INITIAL_WINDOW_WIDTH = 1024
DEFAULT_INITIAL_WINDOW_HEIGHT = 768


# --- Helper Function for Color Interpolation (LERP) ---
# (Could be placed elsewhere, e.g., utils, but keep here for now)
def lerp_color(
    color1: tuple[int, int, int], color2: tuple[int, int, int], factor: float
) -> tuple[int, int, int]:
    """Linearly interpolates between two RGB colors."""
    factor = max(0.0, min(1.0, factor))  # Clamp factor
    r = int(color1[0] + (color2[0] - color1[0]) * factor)
    g = int(color1[1] + (color2[1] - color1[1]) * factor)
    b = int(color1[2] + (color2[2] - color1[2]) * factor)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


# --- End Helper ---


class WindowManager(QWidget):
    def __init__(
        self,
        initial_tileset_path: str,
        initial_tiles: dict[int, Image.Image],
        initial_tile_width: int,
        initial_tile_height: int,
        map_width: int,
        map_height: int,
        min_tile_size_cfg: int = DEFAULT_MIN_TILE_SIZE,
        scroll_debounce_cfg: int = DEFAULT_SCROLL_SCALE_DEBOUNCE_MS,
        resize_debounce_cfg: int = DEFAULT_RESIZE_DEBOUNCE_MS,
    ):
        # (Initialization unchanged)
        super().__init__()
        log.info("Initializing WindowManager...")
        self.current_tileset_path = initial_tileset_path
        self.tiles: dict[int, Image.Image] = initial_tiles
        self.tile_width: int = initial_tile_width
        self.tile_height: int = initial_tile_height
        self.map_width: int = map_width
        self.map_height: int = map_height
        self.min_tile_size = min_tile_size_cfg
        self.scroll_debounce_ms = scroll_debounce_cfg
        self.resize_debounce_ms = resize_debounce_cfg
        log.debug(
            "WindowManager config parameters set",
            min_tile=self.min_tile_size,
            scroll_debounce=self.scroll_debounce_ms,
            resize_debounce=self.resize_debounce_ms,
        )
        self.tile_arrays: dict[int, np.ndarray | None] = {}
        self._update_tile_array_cache()
        self.setWindowTitle("Basic Roguelike")
        self.resize(DEFAULT_INITIAL_WINDOW_WIDTH, DEFAULT_INITIAL_WINDOW_HEIGHT)
        log.info(
            "Window initialized",
            title="Basic Roguelike",
            initial_width=DEFAULT_INITIAL_WINDOW_WIDTH,
            initial_height=DEFAULT_INITIAL_WINDOW_HEIGHT,
        )
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
        # (Implementation unchanged)
        log.debug("Updating tile array cache...")
        self.tile_arrays.clear()
        count = 0
        if not self.tiles or self.tile_width <= 0 or self.tile_height <= 0:
            log.warning(
                "Cannot update tile cache: Invalid tiles or dimensions.",
                has_tiles=bool(self.tiles),
                tile_w=self.tile_width,
                tile_h=self.tile_height,
            )
            return
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
        # (Implementation unchanged)
        log.debug("Building menus...")
        tileset_menu = QMenu("Tileset", self)
        png_path = "fonts/classic_roguelike_sliced"
        use_png_action = QAction("Use PNG Tileset (8x8 base)", self)
        use_png_action.triggered.connect(
            lambda checked=False, p=png_path: self.load_tileset(p, 8, 8)
        )
        tileset_menu.addAction(use_png_action)
        svg_path = "fonts/classic_roguelike_sliced_svgs"
        initial_svg_render_size = 16
        use_svg_action = QAction(
            f"Use SVG Tileset (render @ {initial_svg_render_size}x{initial_svg_render_size} base)",
            self,
        )
        use_svg_action.triggered.connect(
            lambda checked=False, p=svg_path, s=initial_svg_render_size: self.load_tileset(
                p, s, s
            )
        )
        tileset_menu.addAction(use_svg_action)
        self.menu_bar.addMenu(tileset_menu)
        log.debug("Menus built")

    def load_tileset(self, folder: str, width: int, height: int) -> None:
        # (Implementation unchanged)
        if not self.main_loop:
            log.warning("Cannot load tileset: MainLoop not set.")
            return
        try:
            clamped_width = max(self.min_tile_size, width)
            clamped_height = max(self.min_tile_size, height)
            if (
                clamped_width == self.tile_width
                and clamped_height == self.tile_height
                and folder == self.current_tileset_path
            ):
                log.info(
                    "Tileset unchanged, skipping reload.",
                    path=folder,
                    width=clamped_width,
                    height=clamped_height,
                )
                return
            log.info(
                "Loading tileset",
                path=folder,
                base_w=width,
                base_h=height,
                clamped_w=clamped_width,
                clamped_h=clamped_height,
            )
            loaded_tiles, _ = load_tiles(folder, clamped_width, clamped_height)
            self.current_tileset_path = folder
            self.tiles = loaded_tiles
            self.tile_width = clamped_width
            self.tile_height = clamped_height
            self._update_tile_array_cache()
            self.update_frame()
            log.info(
                "Tileset loaded successfully",
                path=folder,
                final_w=self.tile_width,
                final_h=self.tile_height,
            )
        except Exception as e:
            log.error("Error loading tileset", path=folder, error=str(e), exc_info=True)

    def set_main_loop(self, main_loop: "MainLoop") -> None:
        # (Implementation unchanged)
        self.main_loop = main_loop
        log.info("MainLoop instance set in WindowManager")
        QTimer.singleShot(0, self.update_frame)

    def resizeEvent(self, event: QResizeEvent) -> None:
        # (Implementation unchanged)
        log.debug("Resize event detected", new_size=event.size())
        self._resize_timer.start()
        super().resizeEvent(event)

    def update_frame(self) -> None:
        # (Implementation unchanged)
        frame_start_time = time.perf_counter()
        log.debug("Update frame requested...")
        if not self.main_loop or self.tile_width <= 0 or self.tile_height <= 0:
            log.warning(
                "Skipping frame update: Invalid state",
                has_main_loop=bool(self.main_loop),
                tile_w=self.tile_width,
                tile_h=self.tile_height,
            )
            return
        label_w = self.label.width()
        label_h = self.label.height()
        if label_w <= 0 or label_h <= 0:
            log.warning(
                "Skipping frame update: Invalid label size",
                width=label_w,
                height=label_h,
            )
            self.label.clear()
            return
        visible_cols = max(1, label_w // self.tile_width)
        visible_rows = max(1, label_h // self.tile_height)
        player_pos = self.main_loop.game_state.player_position
        cam_x, cam_y = (
            player_pos if player_pos else (self.map_width // 2, self.map_height // 2)
        )
        render_cols = min(visible_cols, self.map_width)
        render_rows = min(visible_rows, self.map_height)
        viewport_x = max(0, min(cam_x - render_cols // 2, self.map_width - render_cols))
        viewport_y = max(
            0, min(cam_y - render_rows // 2, self.map_height - render_rows)
        )
        log_context = {
            "vp_x": viewport_x,
            "vp_y": viewport_y,
            "vp_cols": render_cols,
            "vp_rows": render_rows,
        }
        log.debug("Calculated viewport", **log_context)
        try:
            rendered_image = self.main_loop.update_console(
                viewport_x, viewport_y, render_cols, render_rows
            )
            if rendered_image:
                self.last_rendered_image = rendered_image
                img_with_debug = self.get_image_with_debug()  # Draws overlays
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
                log.warning("MainLoop returned no image to render.", **log_context)
                self.label.clear()
                self.last_rendered_image = None
        except Exception as e:
            log.error(
                "Error during frame update cycle",
                error=str(e),
                exc_info=True,
                **log_context,
            )
            self.last_rendered_image = None
            self.label.clear()
        frame_end_time = time.perf_counter()
        log.debug(
            "Frame update finished",
            duration_ms=(frame_end_time - frame_start_time) * 1000,
        )

    # --- MODIFIED get_image_with_debug ---
    def get_image_with_debug(self) -> Image.Image:
        """Adds debug text overlay AND height key overlay to the last rendered image."""
        if not self.last_rendered_image:
            return Image.new(
                "RGBA", (self.label.width(), self.label.height()), (0, 0, 0, 255)
            )

        img_copy = self.last_rendered_image.copy()
        draw = ImageDraw.Draw(img_copy)  # Use one Draw object

        # --- Draw Debug Text (Existing Logic) ---
        debug_text = "Debug info unavailable"
        try:
            if self.main_loop and self.main_loop.game_state:
                gs = self.main_loop.game_state
                turn = gs.turn_count
                player_pos = gs.player_position
                pos_str = f"({player_pos[0]},{player_pos[1]})" if player_pos else "N/A"
                entities = "N/A"
                try:
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
            log.error("Error getting debug info", error=str(e), exc_info=True)
            debug_text = "Debug info error"

        # Draw debug text background and text
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
            log.error("Error drawing debug text", error=str(e), exc_info=True)
        # --- End Debug Text ---

        # --- ADDED: Draw Height Key Overlay ---
        if self.main_loop and self.main_loop.show_height_visualization:
            log.debug("Drawing height key overlay")
            try:
                # Get parameters from MainLoop (which got them from config)
                max_diff_units = self.main_loop._cfg_height_vis_max_diff
                # Convert units (0.5m) to meters for display labels
                max_diff_meters = max_diff_units / 2.0
                color_high = tuple(self.main_loop._cfg_height_color_high_np)
                color_mid = tuple(self.main_loop._cfg_height_color_mid_np)  # Green
                color_low = tuple(self.main_loop._cfg_height_color_low_np)

                # Define key dimensions and position (e.g., top-left)
                key_width = 15
                key_height = 100
                key_x = 10
                key_y = box_height + 10  # Below debug text
                key_label_offset = 5

                # Draw gradient rectangle
                for i in range(key_height):
                    # Map pixel row 'i' (0 to key_height-1) to interpolation factor 't' (-1 to +1)
                    t = (
                        (key_height - 1 - i) / (key_height - 1)
                    ) * 2.0 - 1.0  # t = 1 at top, 0 at mid, -1 at bottom
                    line_color: tuple[int, int, int]
                    if t > 0:  # Upper half (LERP between Mid and High)
                        line_color = lerp_color(color_mid, color_high, t)
                    else:  # Lower half (LERP between Mid and Low)
                        line_color = lerp_color(
                            color_mid, color_low, -t
                        )  # Use -t as factor

                    # Draw horizontal line for this color step
                    draw.line(
                        [(key_x, key_y + i), (key_x + key_width - 1, key_y + i)],
                        fill=line_color + (255,),  # Add alpha
                    )

                # Draw player marker (white line at midpoint)
                mid_y = key_y + key_height // 2
                draw.line(
                    [(key_x - 2, mid_y), (key_x + key_width + 1, mid_y)],
                    fill=(255, 255, 255, 255),
                    width=1,
                )

                # Draw labels
                label_x = key_x + key_width + key_label_offset
                label_color = (200, 200, 200, 255)
                # Top label (+X m)
                draw.text(
                    (label_x, key_y - 4),
                    f"+{max_diff_meters:.1f}m",
                    fill=label_color,
                    font=text_font,
                )
                # Mid label (0m)
                draw.text((label_x, mid_y - 4), "0m", fill=label_color, font=text_font)
                # Bottom label (-X m)
                draw.text(
                    (label_x, key_y + key_height - 4),
                    f"-{max_diff_meters:.1f}m",
                    fill=label_color,
                    font=text_font,
                )

            except Exception as e:
                log.error("Error drawing height key", error=str(e), exc_info=True)
        # --- End Height Key Overlay ---

        return img_copy

    # --- END MODIFIED get_image_with_debug ---

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # (Implementation unchanged - includes 'V' toggle)
        log.debug("Key pressed", key=event.key(), text=event.text())
        if event.key() == Qt.Key.Key_Escape or event.key() == Qt.Key.Key_Q:
            log.info("Quit key pressed.")
            self.close()
        elif event.key() == Qt.Key.Key_V:
            if self.main_loop:
                self.main_loop.show_height_visualization = (
                    not self.main_loop.show_height_visualization
                )
                log.info(
                    "Height visualization toggled",
                    enabled=self.main_loop.show_height_visualization,
                )
                self.update_frame()
            else:
                log.warning("Cannot toggle visualization: MainLoop not set.")
        elif self.main_loop:
            action: dict | None = None
            key = event.key()
            if key in (Qt.Key.Key_Up, Qt.Key.Key_K, Qt.Key.Key_W):
                action = {"type": "move", "dx": 0, "dy": -1}
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
            if action:
                processed = self.main_loop.handle_action(action)
                self.update_frame() if processed else None
            else:
                log.debug("Unmapped key pressed", key=event.key(), text=event.text())
        else:
            log.warning("Key press ignored: MainLoop not set.")

    def wheelEvent(self, event: QWheelEvent) -> None:
        # (Implementation unchanged)
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
        # (Implementation unchanged)
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
                "Debounced scaling resulted in no size change, skipping reload.",
                accumulated_change=accumulated_change,
                current_size=f"{self.tile_width}x{self.tile_height}",
            )

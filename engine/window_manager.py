# engine/window_manager.py
import math
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QMenuBar, QMenu, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QWheelEvent, QAction, QResizeEvent
from PySide6.QtCore import Qt, QTimer
from PIL import Image, ImageDraw

from engine.tileset_loader import load_tiles

import numpy as np
import polars as pl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.main_loop import MainLoop

# --- Default Configuration (used if not passed from config) ---
DEFAULT_MIN_TILE_SIZE = 4
DEFAULT_SCROLL_SCALE_DEBOUNCE_MS = 200
DEFAULT_RESIZE_DEBOUNCE_MS = 100
# --- ADDED: Default initial window size ---
DEFAULT_INITIAL_WINDOW_WIDTH = 1024
DEFAULT_INITIAL_WINDOW_HEIGHT = 768
# --- END ADDITION ---


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
        super().__init__()
        self.current_tileset_path = initial_tileset_path
        self.tiles: dict[int, Image.Image] = initial_tiles
        self.tile_width: int = initial_tile_width
        self.tile_height: int = initial_tile_height
        self.map_width: int = map_width
        self.map_height: int = map_height

        self.min_tile_size = min_tile_size_cfg
        self.scroll_debounce_ms = scroll_debounce_cfg
        self.resize_debounce_ms = resize_debounce_cfg

        self.tile_arrays: dict[int, np.ndarray | None] = {}
        self._update_tile_array_cache()

        self.setWindowTitle("Basic Roguelike")

        # --- Set a fixed initial window size ---
        # Removed calculation based on map size
        # initial_width = self.map_width * self.tile_width
        # initial_height = self.map_height * self.tile_height
        # self.resize(max(400, initial_width // 2), max(300, initial_height // 2))
        self.resize(DEFAULT_INITIAL_WINDOW_WIDTH, DEFAULT_INITIAL_WINDOW_HEIGHT)
        # Optional: Set a minimum size if desired, though usually not necessary
        # self.setMinimumSize(400, 300)
        # --- End fixed size ---

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

    # ... (rest of the class remains the same as the previous version) ...

    def _update_tile_array_cache(self) -> None:
        # (Implementation remains the same)
        self.tile_arrays.clear()
        if not self.tiles or self.tile_width <= 0 or self.tile_height <= 0:
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
            except Exception as e:
                print(
                    f"Warning: Could not convert tile {tile_index} to NumPy array: {e}"
                )
                self.tile_arrays[tile_index] = None

    def build_menus(self) -> None:
        # (Implementation remains the same)
        tileset_menu = QMenu("Tileset", self)
        png_path = "fonts/classic_roguelike_sliced"
        use_png_action = QAction("Use PNG Tileset (8x8 base)", self)
        use_png_action.triggered.connect(lambda: self.load_tileset(png_path, 8, 8))
        tileset_menu.addAction(use_png_action)
        svg_path = "fonts/classic_roguelike_sliced_svgs"
        initial_svg_render_size = 8
        use_svg_action = QAction(
            f"Use SVG Tileset (render @ {initial_svg_render_size}x{initial_svg_render_size} base)",
            self,
        )
        use_svg_action.triggered.connect(
            lambda: self.load_tileset(
                svg_path, initial_svg_render_size, initial_svg_render_size
            )
        )
        tileset_menu.addAction(use_svg_action)
        self.menu_bar.addMenu(tileset_menu)

    def load_tileset(self, folder: str, width: int, height: int) -> None:
        # Use self.min_tile_size for clamping
        if not self.main_loop:
            return
        try:
            # Clamp dimensions before loading using the configured minimum
            clamped_width = max(self.min_tile_size, width)
            clamped_height = max(self.min_tile_size, height)

            if clamped_width == self.tile_width and clamped_height == self.tile_height:
                print("Tileset size unchanged, skipping reload.")
                return

            print(f"Loading tileset '{folder}' at {clamped_width}x{clamped_height}")
            loaded_tiles, _ = load_tiles(folder, clamped_width, clamped_height)
            self.current_tileset_path = folder
            self.tiles = loaded_tiles
            self.tile_width = clamped_width
            self.tile_height = clamped_height
            self._update_tile_array_cache()
            self.update_frame()
            print(
                f"Tileset loaded. Tile dimensions: {self.tile_width}x{self.tile_height}"
            )
        except Exception as e:
            print(f"Error loading tileset '{folder}': {e}")
            import traceback

            traceback.print_exc()

    def set_main_loop(self, main_loop: "MainLoop") -> None:
        # (Implementation remains the same)
        self.main_loop = main_loop
        QTimer.singleShot(0, self.update_frame)

    def resizeEvent(self, event: QResizeEvent) -> None:
        # Start resize timer (uses configured interval)
        self._resize_timer.start()
        super().resizeEvent(event)

    def update_frame(self) -> None:
        # (Implementation remains the same as previous version)
        if not self.main_loop or self.tile_width <= 0 or self.tile_height <= 0:
            return

        label_w = self.label.width()
        label_h = self.label.height()

        if label_w <= 0 or label_h <= 0:
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

        try:
            rendered_image = self.main_loop.update_console(
                viewport_x,
                viewport_y,
                render_cols,
                render_rows,
            )

            if rendered_image:
                self.last_rendered_image = rendered_image
                img_with_debug = self.get_image_with_debug()

                img_rgba = img_with_debug.convert("RGBA")
                data = img_rgba.tobytes("raw", "RGBA")
                qimg = QImage(
                    data,
                    img_with_debug.width,
                    img_with_debug.height,
                    QImage.Format.Format_RGBA8888,
                )
                if qimg.isNull():
                    print("ERROR: QImage conversion resulted in a null image.")
                    self.label.clear()
                else:
                    pixmap = QPixmap.fromImage(qimg)
                    self.label.setPixmap(pixmap)

            else:
                self.label.clear()
                self.last_rendered_image = None

        except Exception as e:
            print(f"Error during frame update cycle: {e}")
            import traceback

            traceback.print_exc()
            self.last_rendered_image = None
            self.label.clear()

    def get_image_with_debug(self) -> Image.Image:
        # (Implementation remains the same as previous version)
        if (
            not self.last_rendered_image
            or not self.main_loop
            or not self.main_loop.game_state
        ):
            return (
                self.last_rendered_image
                if self.last_rendered_image
                else Image.new(
                    "RGBA", (self.label.width(), self.label.height()), (0, 0, 0, 255)
                )
            )
        img_copy = self.last_rendered_image.copy()
        draw = ImageDraw.Draw(img_copy)
        try:
            gs = self.main_loop.game_state
            turn = gs.turn_count
            player_pos = gs.player_position
            pos_str = f"({player_pos[0]},{player_pos[1]})" if player_pos else "N/A"
            entities = gs.entity_registry.entities_df.filter(pl.col("is_active")).height
            label_w = self.label.width()
            label_h = self.label.height()
            vp_cols = max(1, label_w // self.tile_width) if self.tile_width > 0 else "?"
            vp_rows = (
                max(1, label_h // self.tile_height) if self.tile_height > 0 else "?"
            )
            vp_size_str = f"{vp_cols}x{vp_rows}"
            render_size = f"{self.tile_width}x{self.tile_height}"
        except Exception as e:
            print(f"Error getting debug info: {e}")
            debug_text = "Debug info error"
        else:
            debug_text = (
                f"Turn: {turn} | Player: {pos_str} | Entities: {entities} | "
                f"Viewport: {vp_size_str} | Render Tiles: {render_size}"
            )
        text_font = None
        text_x = 5
        text_y = 5
        text_color = (255, 255, 0, 255)
        bg_color = (0, 0, 0, 180)
        box_height = 15
        draw.rectangle([(0, 0), (img_copy.width, box_height)], fill=bg_color)
        draw.text((text_x, text_y), debug_text, fill=text_color, font=text_font)
        return img_copy

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # (Implementation remains the same as previous version)
        if event.key() == Qt.Key.Key_Escape or event.key() == Qt.Key.Key_Q:
            self.close()
        elif self.main_loop:
            action: dict | None = None
            key = event.key()
            # Movement keys... (unchanged)
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
                if processed:
                    self.update_frame()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handles mouse wheel events for zooming, debounced."""
        # Use self.scroll_debounce_ms
        if not self.main_loop:
            return

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            change = 0
            if delta > 0:
                change = max(1, self.tile_width // 8)
            elif delta < 0:
                change = -max(1, self.tile_width // 8)

            if change != 0:
                self._pending_tile_size_change += change
                # Use configured debounce interval
                self._scroll_scale_timer.start(self.scroll_debounce_ms)
        else:
            super().wheelEvent(event)

    def _apply_scroll_scaling(self) -> None:
        """Applies the accumulated tile size change after debouncing."""
        # Use self.min_tile_size for clamping
        if not self.main_loop or self._pending_tile_size_change == 0:
            return

        target_width = self.tile_width + self._pending_tile_size_change
        target_height = self.tile_height + self._pending_tile_size_change

        # Clamp to minimum size using the configured value
        new_width = max(self.min_tile_size, target_width)
        new_height = max(self.min_tile_size, target_height)

        self._pending_tile_size_change = 0

        if new_width != self.tile_width or new_height != self.tile_height:
            print(f"Applying debounced scaling to {new_width}x{new_height}")
            # load_tileset already uses self.min_tile_size implicitly via its own logic
            self.load_tileset(self.current_tileset_path, new_width, new_height)
        else:
            print("Debounced scaling resulted in no size change, skipping reload.")

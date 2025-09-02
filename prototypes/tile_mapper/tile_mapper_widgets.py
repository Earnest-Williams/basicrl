# tile_mapper_widgets.py

import math

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap  # Now QKeyEvent is imported here
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

import structlog

# Import necessary components from other modules
# Assuming flat structure for now, adjust paths like 'from .tile_mapper_core ...' if needed
from tile_mapper_core import (
    TileMap,
    ctrl_click_fill,
    ctrl_shift_click_wall,
    draw_line,
    fill_rectangle,
    flood_fill_replace,
)
from tile_mapper_dialogs import EditTileDialog  # Import needed dialog
from tile_mapper_utils import (  # Assuming utils module exists/will exist
    parse_key,
    parse_modifier,
)


log = structlog.get_logger(__name__)


# --- Tile Editor Widget ---
class TileEditorWidget(QWidget):
    """The main widget for drawing and interacting with the tile map."""

    map_changed = Signal()
    zoom_changed = Signal()
    selected_tile_changed = Signal(str)
    selection_rect_changed = Signal(QRect)  # Signal for selection tool

    def __init__(
        self, tilemap: TileMap, scroll_area: QScrollArea, app_config: dict, parent=None
    ):
        super().__init__(parent)
        self.tilemap = tilemap
        self.scroll_area = scroll_area
        self.app_config = app_config  # Store config reference
        self._selected_tile = self._get_initial_selected_tile()
        self.start_drag_pos: QPoint | None = None
        self.current_mouse_pos = QPoint(0, 0)
        self.drawing_line = False
        self.drawing_rect = False
        self._tile_size = self.app_config.get("tile_size", 16)

        # Add state for selection tool
        self.selecting_region = False
        self.selection_start_pos: QPoint | None = None
        self.selection_rect: QRect | None = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.update_widget_size()
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Preferred)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window,
                     QColor(Qt.GlobalColor.darkGray))
        self.setPalette(pal)

    def _get_initial_selected_tile(self):
        """Determines a valid initial tile selection from config."""
        tiles = self.app_config.get("tiles", {})
        default_tile = self.app_config.get("default_tile", ".")
        wall_tile = self.app_config.get("wall_tile", "#")
        for char in tiles:
            if char != default_tile and char != wall_tile:
                return char
        for char in tiles:
            if char != default_tile:
                return char
        return (
            default_tile
            if default_tile in tiles
            else (list(tiles.keys())[0] if tiles else ".")
        )

    # --- Selection Tool Methods ---
    def start_selection(self):
        """Activates the region selection tool."""
        self.selecting_region = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        log.debug("Selection tool activated")
        # Clear previous selection visually
        if self.selection_rect:
            self.selection_rect = None
            self.selection_rect_changed.emit(QRect())  # Emit empty rect
            self.update()

    def stop_selection(self):
        """Deactivates the region selection tool."""
        self.selecting_region = False
        self.selection_start_pos = None
        # Don't clear self.selection_rect here, keep it until next selection starts
        self.setCursor(Qt.CursorShape.ArrowCursor)
        log.debug("Selection tool deactivated")

    def get_current_selection(self) -> QRect | None:
        """Returns the current valid selection rectangle (in grid coords)."""
        return (
            self.selection_rect
            if self.selection_rect and self.selection_rect.isValid()
            else None
        )

    # --- Standard Widget Methods ---
    def update_widget_size(self):
        """Updates minimum and preferred size based on map and tile size."""
        self._tile_size = self.app_config.get("tile_size", 16)
        width = self.tilemap.width * self._tile_size
        height = self.tilemap.height * self._tile_size
        self.setMinimumSize(max(1, width), max(1, height))
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        """Provide a preferred size based on current tile size."""
        current_tile_size = self.app_config.get("tile_size", 16)
        width = self.tilemap.width * current_tile_size
        height = self.tilemap.height * current_tile_size
        return QSize(max(1, width), max(1, height))

    def resize_map(self, new_width, new_height, new_default_tile, new_tiles_data=None):
        """Resizes the underlying tilemap instance and redraws."""
        # Logic remains mostly the same, just uses self.tilemap
        # ... (resize logic from previous version) ...
        new_width = max(0, new_width)
        new_height = max(0, new_height)
        old_tiles = (
            [row[:] for row in self.tilemap.tiles] if new_tiles_data is None else None
        )
        old_width = self.tilemap.width
        old_height = self.tilemap.height

        self.tilemap.width = new_width
        self.tilemap.height = new_height
        self.tilemap.default_tile = new_default_tile

        if new_tiles_data is not None:
            if len(new_tiles_data) == new_height and (
                new_height == 0 or all(
                    len(row) == new_width for row in new_tiles_data)
            ):
                self.tilemap.tiles = new_tiles_data
            else:
                log.error(
                    "Provided new_tiles_data dimensions mismatch. Re-initializing"
                )
                self.tilemap.tiles = [
                    [new_default_tile] * new_width for _ in range(new_height)
                ]
        else:  # Preserve old data
            new_grid = [[new_default_tile] *
                        new_width for _ in range(new_height)]
            if old_tiles:
                copy_height = min(old_height, new_height)
                copy_width = min(old_width, new_width)
                for y in range(copy_height):
                    # Bounds check
                    if y < len(old_tiles) and x < len(old_tiles[y]):
                        new_grid[y][:copy_width] = old_tiles[y][
                            :copy_width
                        ]  # Slice assignment
            self.tilemap.tiles = new_grid

        self.update_widget_size()
        self.map_changed.emit()  # Signal size change

    @property
    def selected_tile(self):
        return self._selected_tile

    @selected_tile.setter
    def selected_tile(self, tile_char):
        if tile_char in self.app_config.get("tiles", {}):
            if self._selected_tile != tile_char:
                self._selected_tile = tile_char
                log.info("Selected tile", tile=self._selected_tile)
                self.selected_tile_changed.emit(self._selected_tile)
        else:
            # Existing revert logic...
            log.warning("Attempted unknown tile", tile=tile_char)
            safe_tile = self._get_initial_selected_tile()
            if self._selected_tile != safe_tile:
                self._selected_tile = safe_tile
                log.info("Reverted to safe tile", tile=self._selected_tile)
                self.selected_tile_changed.emit(self._selected_tile)

    def pixel_to_grid(self, pos: QPoint) -> QPoint:
        """Converts pixel coordinates (QWidget) to grid coordinates."""
        current_tile_size = self.app_config.get("tile_size", 16)
        if current_tile_size <= 0:
            return QPoint(-1, -1)  # Indicate invalid
        x = pos.x() // current_tile_size
        y = pos.y() // current_tile_size
        # Clamp to valid grid indices, return invalid (-1) if map is zero size
        if self.tilemap.width <= 0 or self.tilemap.height <= 0:
            return QPoint(-1, -1)
        x = max(0, min(x, self.tilemap.width - 1))
        y = max(0, min(y, self.tilemap.height - 1))
        return QPoint(x, y)

    def grid_to_pixel(self, grid_pos: QPoint) -> QPoint:
        """Converts grid coordinates to top-left pixel coordinates."""
        current_tile_size = self.app_config.get("tile_size", 16)
        return QPoint(
            grid_pos.x() * current_tile_size, grid_pos.y() * current_tile_size
        )

    def paintEvent(self, event: QPaintEvent):
        """Handles painting the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        tile_configs = self.app_config.get("tiles", {})
        unknown_color = QColor(255, 0, 255)
        current_tile_size = self.app_config.get("tile_size", 16)
        if current_tile_size <= 0:
            return

        paint_rect = event.rect()
        start_col = max(0, paint_rect.left() // current_tile_size)
        end_col = min(
            self.tilemap.width,
            (paint_rect.right() + current_tile_size - 1) // current_tile_size,
        )
        start_row = max(0, paint_rect.top() // current_tile_size)
        end_row = min(
            self.tilemap.height,
            (paint_rect.bottom() + current_tile_size - 1) // current_tile_size,
        )

        # Draw Tiles
        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                tile_char = self.tilemap.get_tile(x, y)
                tile_data = tile_configs.get(tile_char)
                color = (
                    tile_data.get("color_qt", unknown_color)
                    if tile_data
                    else unknown_color
                )
                rect = QRect(
                    x * current_tile_size,
                    y * current_tile_size,
                    current_tile_size,
                    current_tile_size,
                )
                if rect.intersects(paint_rect):
                    painter.fillRect(rect, color)

        # Draw Grid
        if current_tile_size > 4:
            pen = QPen(QColor(180, 180, 180), 1, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            left = start_col * current_tile_size
            right = end_col * current_tile_size
            top = start_row * current_tile_size
            bottom = end_row * current_tile_size
            for x_coord in range(start_col, end_col + 1):
                px = x_coord * current_tile_size
                painter.drawLine(
                    px, max(top, paint_rect.top()), px, min(
                        bottom, paint_rect.bottom())
                )
            for y_coord in range(start_row, end_row + 1):
                py = y_coord * current_tile_size
                painter.drawLine(
                    max(left, paint_rect.left()), py, min(
                        right, paint_rect.right()), py
                )

        # Draw Previews (Line/Rect)
        if self.start_drag_pos and self.current_mouse_pos and not self.selecting_region:
            # ... (drawing preview logic - unchanged) ...
            end_pos = self.pixel_to_grid(self.current_mouse_pos)
            preview_color = self.app_config.get(
                "preview_line_color_qt", QColor(255, 0, 0)
            )
            preview_thickness = self.app_config.get(
                "preview_line_thickness", 1)
            preview_pen = QPen(
                preview_color, preview_thickness, Qt.PenStyle.DotLine)
            painter.setPen(preview_pen)
            if self.drawing_rect:
                x1, y1 = self.start_drag_pos.x(), self.start_drag_pos.y()
                x2, y2 = end_pos.x(), end_pos.y()
                min_x, max_x = min(x1, x2), max(x1, x2)
                min_y, max_y = min(y1, y2), max(y1, y2)
                preview_rect = QRect(
                    min_x * current_tile_size,
                    min_y * current_tile_size,
                    (max_x - min_x + 1) * current_tile_size,
                    (max_y - min_y + 1) * current_tile_size,
                )
                if preview_thickness > 1:
                    offset = preview_thickness / 2.0
                    preview_rect.adjust(offset, offset, -offset, -offset)
                painter.drawRect(preview_rect)
            elif self.drawing_line:
                start_pixel = QPoint(
                    self.start_drag_pos.x() * current_tile_size
                    + current_tile_size // 2,
                    self.start_drag_pos.y() * current_tile_size
                    + current_tile_size // 2,
                )
                painter.drawLine(start_pixel, self.current_mouse_pos)

        # Draw Selection Rectangle
        temp_selection_rect = None
        if (
            self.selecting_region
            and self.selection_start_pos
            and self.current_mouse_pos
        ):
            # Calculate temporary rect during drag
            end_pos = self.pixel_to_grid(self.current_mouse_pos)
            temp_selection_rect = QRect(
                self.selection_start_pos, end_pos).normalized()
        elif self.selection_rect and self.selection_rect.isValid():
            # Use finalized selection rect if valid
            temp_selection_rect = self.selection_rect

        if temp_selection_rect:
            pen = QPen(
                QColor(50, 150, 255), 2, Qt.PenStyle.DashLine
            )  # Blue dashed line
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Convert grid rect to pixel rect
            pixel_rect = QRect(
                temp_selection_rect.left() * current_tile_size,
                temp_selection_rect.top() * current_tile_size,
                temp_selection_rect.width() * current_tile_size,
                temp_selection_rect.height() * current_tile_size,
            )
            painter.drawRect(pixel_rect)

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse button presses, including selection start."""
        self.current_mouse_pos = event.position().toPoint()
        grid_pos = self.pixel_to_grid(self.current_mouse_pos)
        if grid_pos.x() < 0 or grid_pos.y() < 0:  # Check valid grid pos
            event.ignore()
            return

        # Handle Selection Tool first if active
        if self.selecting_region:
            if event.button() == Qt.MouseButton.LeftButton:
                self.selection_start_pos = grid_pos
                self.selection_rect = None  # Clear old finalized rect
                self.update()  # Redraw to show selection start/clear old
                event.accept()
            else:  # Ignore other buttons in selection mode
                event.ignore()
            return  # Don't process other tools if selecting

        # --- Normal Tool Handling ---
        # ... (Existing mousePressEvent logic using self.app_config) ...
        current_modifiers = event.modifiers()
        controls = self.app_config.get("controls", {})
        action_handled = False

        # Left Button
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_drag_pos = grid_pos
            self.drawing_line = False
            self.drawing_rect = False
            action_found = False

            for action_name, control in controls.items():
                if control.get("trigger") == "LeftClick":
                    required_modifier = parse_modifier(
                        control.get("modifier", "None"))
                    if current_modifiers == required_modifier:
                        log.debug("Matched press", action=action_name)
                        action_found = True
                redraw_needed = False
                if action_name == "flood_fill":
                    filled = flood_fill_replace(
                        self.tilemap,
                        grid_pos.x(),
                        grid_pos.y(),
                        self.selected_tile,
                    )
                    if filled > 0:
                        redraw_needed = True
                    self.start_drag_pos = None
                elif action_name == "fill_perimeter":
                    if ctrl_click_fill(
                        self.tilemap,
                        grid_pos.x(),
                        grid_pos.y(),
                        self.selected_tile,
                    ):
                        redraw_needed = True
                    self.start_drag_pos = None
                elif action_name == "wall_perimeter":
                    if ctrl_shift_click_wall(
                        self.tilemap,
                        grid_pos.x(),
                        grid_pos.y(),
                        self.selected_tile,
                        self.app_config,
                    ):
                        redraw_needed = True
                    self.start_drag_pos = None
                else:  # Start drag actions
                    if required_modifier == parse_modifier("Shift"):
                        self.drawing_rect = True
                    elif required_modifier == Qt.KeyboardModifier.NoModifier:
                        self.drawing_line = True
                if redraw_needed:
                    self.update()
                action_handled = True
                break

            if not action_found and current_modifiers == Qt.KeyboardModifier.NoModifier:
                if self.start_drag_pos:
                    self.drawing_line = True
                    action_handled = True

            if action_handled:
                event.accept()
            else:
                event.ignore()

        # Right Button
        elif event.button() == Qt.MouseButton.RightButton:
            # ... (Existing Right Button logic using self.app_config) ...
            for action_name, control in controls.items():
                if control.get("trigger") == "RightClick":
                    required_modifier = parse_modifier(
                        control.get("modifier", "None"))
                    if current_modifiers == required_modifier:
                        log.debug("Matched press", action=action_name)
                        if action_name == "erase_tile":
                            if self.tilemap.set_tile(
                                grid_pos.x(), grid_pos.y(), self.tilemap.default_tile
                            ):
                                self.update()
                                action_handled = True
                        break
            if action_handled:
                event.accept()
            else:
                event.ignore()

        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles mouse movement for drag previews and selection."""
        self.current_mouse_pos = event.position().toPoint()

        # Handle selection drag
        if (
            self.selecting_region
            and event.buttons() & Qt.MouseButton.LeftButton
            and self.selection_start_pos
        ):
            self.update()  # Update to show temporary selection rect
            event.accept()
            return

        # Handle tool drag previews
        if event.buttons() & Qt.MouseButton.LeftButton and self.start_drag_pos:
            if self.drawing_line or self.drawing_rect:
                self.update()
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handles mouse button releases, including selection end."""
        self.current_mouse_pos = (
            event.position().toPoint()
        )  # Ensure current pos is updated
        end_pos = self.pixel_to_grid(self.current_mouse_pos)
        if end_pos.x() < 0 or end_pos.y() < 0:  # Check valid grid pos
            event.ignore()
            return

        # Handle Selection Tool end
        if (
            self.selecting_region
            and event.button() == Qt.MouseButton.LeftButton
            and self.selection_start_pos
        ):
            self.selection_rect = QRect(
                self.selection_start_pos, end_pos).normalized()
            self.selection_start_pos = None  # End drag
            log.debug(
                "Selected region", coords=self.selection_rect.getRect()
            )
            self.selection_rect_changed.emit(
                self.selection_rect)  # Emit final rect
            self.update()  # Redraw finalized selection
            event.accept()
            # Optionally deactivate tool after selection? Depends on desired workflow.
            # self.stop_selection()
            return  # Don't process other tools

        # --- Normal Tool Handling ---
        # ... (Existing mouseReleaseEvent logic using self.app_config) ...
        current_modifiers = QApplication.keyboardModifiers()
        controls = self.app_config.get("controls", {})
        action_handled = False

        if event.button() == Qt.MouseButton.LeftButton and self.start_drag_pos:
            is_drag = self.start_drag_pos != end_pos
            _start_x, _start_y = self.start_drag_pos.x(), self.start_drag_pos.y()
            end_x, end_y = end_pos.x(), end_pos.y()

            action_to_perform = None
            trigger_type = "LeftDrag" if is_drag else "LeftClick"
            for action_name, control in controls.items():
                if control.get("trigger") == trigger_type:
                    required_modifier = parse_modifier(
                        control.get("modifier", "None"))
                    if current_modifiers == required_modifier:
                        if action_name not in [
                            "flood_fill",
                            "fill_perimeter",
                            "wall_perimeter",
                        ]:
                            action_to_perform = action_name
                            break

            place_tile_control = self._find_control_by_action(
                "place_tile_click")
            if (
                not is_drag
                and action_to_perform is None
                and place_tile_control
                and parse_modifier(place_tile_control.get("modifier", "None"))
                == Qt.KeyboardModifier.NoModifier
                and current_modifiers == Qt.KeyboardModifier.NoModifier
            ):
                action_to_perform = "place_tile_click"

            log.debug(
                "Release event",
                action=action_to_perform,
                is_drag=is_drag,
                modifiers=current_modifiers,
            )
            redraw_needed = False
            if action_to_perform == "draw_rect":
                if fill_rectangle(
                    self.tilemap, self.start_drag_pos, end_pos, self.selected_tile
                ):
                    redraw_needed = True
            elif action_to_perform == "draw_line":
                if draw_line(
                    self.tilemap, self.start_drag_pos, end_pos, self.selected_tile
                ):
                    redraw_needed = True
            elif action_to_perform == "place_tile_click":
                if self.tilemap.set_tile(end_x, end_y, self.selected_tile):
                    redraw_needed = True

            self.start_drag_pos = None
            self.drawing_line = False
            self.drawing_rect = False
            if redraw_needed:
                self.update()
            action_handled = True

        if action_handled:
            event.accept()
        else:
            event.ignore()

    def _find_control_by_action(self, action_name_to_find):
        """Helper to find a control definition by its name."""
        return self.app_config.get("controls", {}).get(action_name_to_find)

    def wheelEvent(self, event: QWheelEvent):
        """Handles mouse wheel events for zooming and panning."""
        # ... (Existing wheelEvent logic using self.app_config) ...
        self.current_mouse_pos = event.position().toPoint()
        current_modifiers = event.modifiers()
        controls = self.app_config.get("controls", {})
        angle = event.angleDelta().y()
        action_handled = False

        action_to_perform = None
        trigger_type = "ScrollUp" if angle > 0 else "ScrollDown"
        for action_name, control in controls.items():
            if control.get("trigger") == trigger_type:
                required_modifier = parse_modifier(
                    control.get("modifier", "None"))
                if current_modifiers == required_modifier:
                    action_to_perform = action_name
                    break

        log.debug(
            "Wheel event", action=action_to_perform, modifiers=current_modifiers
        )
        if not self.scroll_area:
            event.ignore()
            return

        if action_to_perform in ["zoom_in", "zoom_out"]:
            zoom_factor = self.app_config.get("zoom_step", 1.2)
            min_size = self.app_config.get("min_tile_size", 4)
            max_size = self.app_config.get("max_tile_size", 64)
            old_tile_size = self.app_config.get("tile_size", 16)

            if action_to_perform == "zoom_in":
                new_tile_size = math.ceil(old_tile_size * zoom_factor)
            else:
                new_tile_size = math.floor(old_tile_size / zoom_factor)
            new_tile_size = max(min_size, min(new_tile_size, max_size))

            if new_tile_size != old_tile_size:
                widget_point = self.mapFromGlobal(
                    event.globalPosition().toPoint())
                rel_x = widget_point.x() / self.width() if self.width() > 0 else 0.5
                rel_y = widget_point.y() / self.height() if self.height() > 0 else 0.5

                # Update config directly
                self.app_config["tile_size"] = new_tile_size
                log.debug("Zooming", new_tile_size=new_tile_size)
                self.update_widget_size()  # Resizes widget
                self.zoom_changed.emit()  # Notify palette

                def adjust_scrollbars():  # Closure to adjust scrollbars after resize
                    new_w, new_h = self.width(), self.height()
                    target_wx, target_wy = rel_x * new_w, rel_y * new_h
                    vp_mouse_pos = self.scroll_area.viewport().mapFromGlobal(
                        event.globalPosition().toPoint()
                    )
                    new_sx = int(target_wx - vp_mouse_pos.x())
                    new_sy = int(target_wy - vp_mouse_pos.y())
                    h_bar = self.scroll_area.horizontalScrollBar()
                    v_bar = self.scroll_area.verticalScrollBar()
                    h_bar.setValue(
                        max(h_bar.minimum(), min(new_sx, h_bar.maximum())))
                    v_bar.setValue(
                        max(v_bar.minimum(), min(new_sy, v_bar.maximum())))

                QTimer.singleShot(0, adjust_scrollbars)

            action_handled = True
            event.accept()

        elif action_to_perform in ["pan_left", "pan_right"]:
            h_bar = self.scroll_area.horizontalScrollBar()
            step = self.app_config.get("pan_step", 50)
            if action_to_perform == "pan_left":
                h_bar.setValue(h_bar.value() - step)
            else:
                h_bar.setValue(h_bar.value() + step)
            action_handled = True
            event.accept()

        if not action_handled:
            event.ignore()

    def keyPressEvent(self, event: QKeyEvent):
        """Handles key presses for changing tiles and other actions."""
        # ... (Existing keyPressEvent logic using self.app_config and parse_key/parse_modifier) ...
        current_modifiers = event.modifiers()
        current_key = event.key()
        controls = self.app_config.get("controls", {})
        available_tiles = sorted(self.app_config.get("tiles", {}).keys())
        action_handled = False

        for action_name, control in controls.items():
            if control.get("trigger") == "KeyPress":
                required_modifier = parse_modifier(
                    control.get("modifier", "None"))
                required_key_str = control.get("key")
                required_key_enum = parse_key(required_key_str)

                if (
                    required_key_enum is not None
                    and current_key == required_key_enum
                    and current_modifiers == required_modifier
                ):
                    log.debug("Matched keypress", action=action_name)
                    if action_name.startswith("select_tile_"):
                        try:
                            idx = int(action_name.split("_")[-1]) - 1
                            if 0 <= idx < len(available_tiles):
                                self.selected_tile = available_tiles[idx]
                                action_handled = True
                        except (ValueError, IndexError):
                            pass
                    # Add other KeyPress actions here if needed (e.g., toggle selection tool?)

                    if action_handled:
                        event.accept()
                        return

        if not action_handled:
            event.ignore()


# --- Tile Palette Widget ---
class TilePaletteWidget(QWidget):
    """Widget for tile selection, search, and info display."""

    palette_updated = Signal()

    def __init__(self, editor_widget: TileEditorWidget, app_config: dict, parent=None):
        super().__init__(parent)
        self.editor_widget = editor_widget
        self.app_config = app_config  # Store config reference
        self.buttons = {}  # {tile_char: QPushButton}
        self.init_ui()
        self.rebuild_palette()  # Initial population

        # Connect signals
        self.editor_widget.zoom_changed.connect(self.rebuild_palette)
        self.editor_widget.selected_tile_changed.connect(
            self.update_selection_visuals)
        self.editor_widget.selected_tile_changed.connect(
            self.update_selected_tile_info)
        self.search_edit.textChanged.connect(self._filter_palette)

    def init_ui(self):
        """Sets up the palette UI including search and info."""
        # ... (Layout setup from previous version) ...
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(3)
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Filter:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Char or desc...")
        self.search_edit.setClearButtonEnabled(True)
        top_layout.addWidget(self.search_edit)
        self.add_button = QPushButton("+")
        self.add_button.setFixedSize(QSize(25, 25))
        self.add_button.setToolTip("Add new tile")
        self.add_button.clicked.connect(self.add_new_tile)
        top_layout.addWidget(self.add_button)
        main_layout.addLayout(top_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.StyledPanel)
        self.button_container = QWidget()
        self.button_grid_layout = QGridLayout(self.button_container)
        self.button_grid_layout.setSpacing(2)
        self.button_grid_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.button_container.setLayout(self.button_grid_layout)
        self.scroll_area.setWidget(self.button_container)
        self.scroll_area.setMinimumHeight(60)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self.scroll_area, 1)

        self.selected_tile_label = QLabel("Selected: -")
        font = self.selected_tile_label.font()
        font.setPointSize(font.pointSize() - 1)
        self.selected_tile_label.setFont(font)
        self.selected_tile_label.setStyleSheet(
            "QLabel { padding: 3px; background-color: #f0f0f0; border-radius: 3px; border: 1px solid #cccccc; }"
        )
        self.selected_tile_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.selected_tile_label)
        self.setLayout(main_layout)

    def _create_tile_icon(self, tile_char, tile_data, icon_size):
        """Creates a QPixmap icon."""
        # ... (Icon creation logic - unchanged, uses tile_data['color_qt']) ...
        if (
            not isinstance(icon_size, QSize)
            or not icon_size.isValid()
            or icon_size.width() <= 0
        ):
            icon_size = QSize(16, 16)
        pixmap = QPixmap(icon_size)
        color = tile_data.get("color_qt", QColor(255, 0, 255))
        pixmap.fill(color)
        painter = QPainter(pixmap)
        lumi = 0.299 * color.redF() + 0.587 * color.greenF() + 0.114 * color.blueF()
        text_color = Qt.GlobalColor.black if lumi > 0.5 else Qt.GlobalColor.white
        painter.setPen(text_color)
        font_size = max(6, int(icon_size.height() * 0.6))
        font = QFont()
        font.setPointSize(font_size)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            pixmap.rect(), Qt.AlignmentFlag.AlignCenter, tile_char)
        painter.end()
        return QIcon(pixmap)

    def _filter_palette(self):
        """Hides/Shows tile buttons based on search text."""
        # ... (Filtering logic - unchanged, uses self.app_config) ...
        filter_text = self.search_edit.text().strip().lower()
        tile_configs = self.app_config.get("tiles", {})
        buttons_visible = 0
        total_buttons = len(self.buttons)
        for tile_char, button in self.buttons.items():
            if button:
                tile_data = tile_configs.get(tile_char, {})
                desc = tile_data.get("description", "").lower()
                match = (
                    not filter_text
                    or filter_text in tile_char.lower()
                    or filter_text in desc
                )
                button.setVisible(match)
                if match:
                    buttons_visible += 1
        self.search_edit.setStyleSheet(
            "QLineEdit { color: red; }"
            if (filter_text and buttons_visible == 0 and total_buttons > 0)
            else ""
        )

    def rebuild_palette(self):
        """Clears and rebuilds tile buttons based on config, respecting filter."""
        # ... (Rebuild logic - unchanged, uses self.app_config) ...
        log.debug("Rebuilding palette")
        while self.button_grid_layout.count():
            item = self.button_grid_layout.takeAt(0)
            if item:
                widget = item.widget()
            if widget:
                widget.deleteLater()
        self.buttons.clear()

        tile_size = self.app_config.get("tile_size", 16)
        icon_qsize = QSize(tile_size, tile_size)
        button_width = max(32, tile_size + 10)
        button_height = max(32, tile_size + 10)
        button_qsize = QSize(button_width, button_height)
        tile_chars = sorted(self.app_config.get("tiles", {}).keys())
        row, col = 0, 0
        container_width = self.button_container.width()
        est_button_w = button_width + self.button_grid_layout.spacing()
        max_cols = (
            max(1, container_width // est_button_w)
            if container_width > est_button_w
            else 8
        )

        for tile_char in tile_chars:
            tile_data = self.app_config["tiles"].get(
                tile_char,
                {
                    "color_qt": QColor(255, 0, 255),
                    "description": f"Undef '{tile_char}'",
                },
            )
            btn = QPushButton()
            btn.setFixedSize(button_qsize)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            icon = self._create_tile_icon(tile_char, tile_data, icon_qsize)
            btn.setIcon(icon)
            btn.setIconSize(icon_qsize)
            desc = tile_data.get("description", "No description")
            btn.setToolTip(f"'{tile_char}': {desc}")
            btn.clicked.connect(
                lambda checked, tc=tile_char: self.select_tile(tc))
            btn.setProperty("tile_char", tile_char)
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(self.show_context_menu)
            self.button_grid_layout.addWidget(btn, row, col)
            self.buttons[tile_char] = btn
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        while col < max_cols and col != 0:  # Fill last row
            self.button_grid_layout.addItem(
                QSpacerItem(
                    button_width,
                    button_height,
                    QSizePolicy.Policy.Fixed,
                    QSizePolicy.Policy.Fixed,
                ),
                row,
                col,
            )
            col += 1
        self.button_grid_layout.setRowStretch(row + 1, 1)  # Push up

        self._filter_palette()
        self.update_selection_visuals()
        self.update_selected_tile_info()
        self.palette_updated.emit()

    def select_tile(self, tile_char):
        """Slot for button clicks."""
        self.editor_widget.selected_tile = tile_char

    def update_selection_visuals(self, tile_char: str | None = None):
        """Updates button check state."""
        # ... (Logic unchanged) ...
        current_tile = (
            tile_char if tile_char is not None else self.editor_widget.selected_tile
        )
        button_to_check = self.buttons.get(current_tile)
        if button_to_check:
            if not button_to_check.isChecked():
                button_to_check.setChecked(True)
        else:  # Fallback
            default_button = self.buttons.get(
                self.app_config.get("default_tile", "."))
            if default_button and not default_button.isChecked():
                default_button.setChecked(True)

    def update_selected_tile_info(self, tile_char: str | None = None):
        """Updates info label."""
        # ... (Logic unchanged, uses self.app_config) ...
        current_tile = (
            tile_char if tile_char is not None else self.editor_widget.selected_tile
        )
        tile_data = self.app_config.get("tiles", {}).get(current_tile)
        if tile_data:
            desc = tile_data.get("description", "No description")
            self.selected_tile_label.setText(f"'{current_tile}': {desc}")
            self.selected_tile_label.setToolTip(
                f"Char: {current_tile}\nDesc: {desc}")
        else:
            self.selected_tile_label.setText(f"'{current_tile}': (Unknown)")
            self.selected_tile_label.setToolTip(
                f"Char: {current_tile}\nDesc: Unknown tile"
            )

    def show_context_menu(self, pos):
        """Shows context menu for tile buttons."""
        # ... (Logic unchanged, uses self.app_config for default check) ...
        button = self.sender()
        if not isinstance(button, QPushButton):
            return
        tile_char = button.property("tile_char")
        if not tile_char:
            return
        menu = QMenu(self)
        edit_action = menu.addAction(f"Edit '{tile_char}'...")
        delete_action = menu.addAction(f"Delete '{tile_char}'")
        if tile_char == self.app_config.get("default_tile"):
            delete_action.setText(f"Delete Default '{tile_char}'...")
        action = menu.exec(button.mapToGlobal(pos))
        if action == edit_action:
            self.edit_tile(tile_char)
        elif action == delete_action:
            self.delete_tile(tile_char)

    # Edit/Delete/Add methods now need to handle config update and trigger rebuild
    # They will need access to save_config and load_config (passed via MainWindow or imported)

    def edit_tile(self, tile_char):
        """Opens EditTileDialog, updates config, triggers rebuild."""
        # Needs access to save_config, load_config
        from tile_mapper_config import CONFIG_FILE  # Import locally
        from tile_mapper_config import load_config, save_config

        if tile_char not in self.app_config.get("tiles", {}):
            return
        tile_data = self.app_config["tiles"][tile_char]
        existing_chars = set(self.app_config.get("tiles", {}).keys())

        # Pass app_config to dialog
        dialog = EditTileDialog(
            tile_char, tile_data, existing_chars, self, app_config=self.app_config
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_tile_data()
            new_char, new_data = result["char"], result["data"]
            original_selected = self.editor_widget.selected_tile
            is_default = tile_char == self.app_config.get("default_tile")
            config_changed = False

            if new_char != tile_char:  # Renaming
                if new_char in self.app_config["tiles"]:
                    QMessageBox.warning(
                        self, "Rename Error", f"Char '{new_char}' exists."
                    )
                    return
                self.app_config["tiles"][new_char] = self.app_config["tiles"][
                    tile_char
                ].copy()
                self.app_config["tiles"][new_char].update(new_data)
                del self.app_config["tiles"][tile_char]
                if is_default:
                    self.app_config["default_tile"] = new_char
                if original_selected == tile_char:
                    self.editor_widget.selected_tile = new_char
                config_changed = True
            else:  # Just updating data
                if self.app_config["tiles"][tile_char].get(
                    "description"
                ) != new_data.get("description") or self.app_config["tiles"][
                    tile_char
                ].get(
                    "color"
                ) != new_data.get(
                    "color"
                ):
                    self.app_config["tiles"][tile_char].update(new_data)
                    config_changed = True

            if config_changed:
                if save_config(CONFIG_FILE, self.app_config):
                    self.rebuild_palette()
                    self.editor_widget.update()
                    QApplication.instance().statusBar().showMessage(
                        "Tile updated & saved.", 3000
                    )
                else:
                    QMessageBox.critical(
                        self, "Error", f"Failed save: {CONFIG_FILE}")
                    # Reload config to revert in-memory changes
                    self.app_config.clear()  # Clear current dict
                    self.app_config.update(load_config(CONFIG_FILE))  # Reload
                    self.rebuild_palette()  # Rebuild with reloaded state

    def delete_tile(self, tile_char):
        """Deletes tile, updates config, triggers rebuild."""
        # Needs access to save_config, load_config
        from tile_mapper_config import CONFIG_FILE  # Import locally
        from tile_mapper_config import load_config, save_config

        # ... (Confirmation logic unchanged) ...
        if tile_char not in self.app_config.get("tiles", {}):
            return
        is_default = tile_char == self.app_config.get("default_tile")
        if is_default and len(self.app_config.get("tiles", {})) <= 1:
            QMessageBox.critical(self, "Cannot Delete",
                                 "Cannot delete last tile.")
            return
        warning = "\n\nWARNING: Deleting default!" if is_default else ""
        reply = QMessageBox.question(
            self,
            "Confirm",
            f"Delete '{tile_char}'?{warning}\n(Affects maps)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            original_selected = self.editor_widget.selected_tile
            del self.app_config["tiles"][tile_char]
            new_default_selected = None
            if is_default:
                remaining = list(self.app_config.get("tiles", {}).keys())
                new_default = remaining[0] if remaining else "."
                self.app_config["default_tile"] = new_default
                log.info("New default tile set", new_default=new_default)
                new_default_selected = new_default

            if save_config(CONFIG_FILE, self.app_config):
                if original_selected == tile_char or new_default_selected:
                    select_tile = (
                        new_default_selected
                        if new_default_selected
                        else self.app_config.get("default_tile", ".")
                    )
                    # Check if target tile still exists before selecting
                    if select_tile not in self.app_config.get("tiles", {}):
                        select_tile = (
                            self.editor_widget._get_initial_selected_tile()
                        )  # Ultimate fallback
                    self.editor_widget.selected_tile = select_tile

                self.rebuild_palette()
                self.editor_widget.update()
                QApplication.instance().statusBar().showMessage(
                    f"Tile '{tile_char}' deleted.", 3000
                )
            else:
                QMessageBox.critical(
                    self, "Error", f"Failed save: {CONFIG_FILE}")
                self.app_config.clear()
                self.app_config.update(load_config(CONFIG_FILE))
                self.rebuild_palette()

    def add_new_tile(self):
        """Opens dialog to add tile, updates config, triggers rebuild."""
        # Needs access to save_config, load_config
        from tile_mapper_config import CONFIG_FILE  # Import locally
        from tile_mapper_config import load_config, save_config

        existing_chars = set(self.app_config.get("tiles", {}).keys())
        dialog = EditTileDialog(
            existing_chars=existing_chars, parent=self, app_config=self.app_config
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_tile_data()
            new_char, new_data = result["char"], result["data"]
            self.app_config["tiles"][new_char] = new_data
            if save_config(CONFIG_FILE, self.app_config):
                self.rebuild_palette()
                QApplication.instance().statusBar().showMessage("New tile added.", 3000)
            else:
                QMessageBox.critical(
                    self, "Error", f"Failed save: {CONFIG_FILE}")
                self.app_config.clear()
                self.app_config.update(load_config(CONFIG_FILE))
                self.rebuild_palette()

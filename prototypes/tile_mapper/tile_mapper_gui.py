# tile_mapper_gui.py
# v6: Rigorous import check, including QDialog and others.

import math
import os

import structlog

from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPalette

# --- PySide6 Imports - Thoroughly Checked ---
# Added QColorDialog explicitly where used
from PySide6.QtWidgets import QColorDialog
from PySide6.QtWidgets import QDialog  # Added QDialog
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# --- Project Module Imports ---
from tile_mapper_config import (
    CONFIG_FILE,
    FILE_READ_MODE,
    JSON_HANDLER,
    JSON_DecodeError,
    load_config,
    save_config,
)
from tile_mapper_core import TileMap, extract_map_region, save_extracted_map
from tile_mapper_dialogs import (
    MapSelectionDialog,
)  # EditTileDialog used in palette, MapSelectionDialog used here
from tile_mapper_widgets import TileEditorWidget, TilePaletteWidget

log = structlog.get_logger(__name__)


# --- Main Application Window ---
class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.app_config = load_config(CONFIG_FILE)
        self.setWindowTitle(self.app_config.get("window_title", "Tile Editor"))
        self.tilemap = TileMap(
            self.app_config.get("grid_width", 40),
            self.app_config.get("grid_height", 40),
            self.app_config.get("default_tile", "."),
        )
        self.current_filepath = None

        # --- UI Setup ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.ColorRole.Dark)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.editor_widget = TileEditorWidget(
            self.tilemap, self.scroll_area, self.app_config, self
        )
        self.palette_widget = TilePaletteWidget(
            self.editor_widget, self.app_config, self
        )
        self.scroll_area.setWidget(self.editor_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.palette_widget)
        main_layout.addWidget(self.scroll_area, 1)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.create_actions()
        self.create_menu_bar()
        self.create_tool_bar()
        self.statusBar()  # Ensure status bar exists

        self.resize(800, 650)

        # --- Signal Connections ---
        self.editor_widget.map_changed.connect(self.update_window_title)
        self.editor_widget.map_changed.connect(self.center_map_view_if_needed)
        self.palette_widget.palette_updated.connect(
            self.editor_widget.update_widget_size
        )
        self.editor_widget.selection_rect_changed.connect(
            self._update_selection_actions
        )

        # --- Final Init Steps ---
        self.editor_widget.setFocus()
        self.update_window_title()
        self._update_selection_actions(QRect())  # Init action state
        QTimer.singleShot(0, self.center_map_view)  # Initial center

    # --- Window Title ---
    def update_window_title(self):
        """Updates the window title with map dimensions and filename."""
        base_title = self.app_config.get("window_title", "Tile Editor")
        filename_part = (
            f" - {os.path.basename(self.current_filepath)}"
            if self.current_filepath
            else ""
        )
        dims_part = f" ({self.tilemap.width}x{self.tilemap.height})"
        self.setWindowTitle(f"{base_title}{filename_part}{dims_part}")

    # --- Actions, Menus, Toolbar ---
    def create_actions(self):
        """Creates QAction objects for menu items and toolbar."""
        # File Actions
        self.load_action = QAction(
            QIcon.fromTheme("document-open"), "&Load Map...", self
        )
        self.load_action.setShortcut(QKeySequence.StandardKey.Open)
        self.load_action.setStatusTip("Load maps from JSON")
        self.load_action.triggered.connect(self.load_map)

        self.load_text_action = QAction(
            QIcon.fromTheme("document-open"), "Load &Text Layout...", self
        )
        self.load_text_action.setStatusTip("Import map from text")
        self.load_text_action.triggered.connect(self.load_text_layout)

        self.save_action = QAction(
            QIcon.fromTheme("document-save"), "&Save Map...", self
        )
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setStatusTip(
            "Save current map to named entry in JSON")
        self.save_action.triggered.connect(self.save_map)

        self.save_selection_action = QAction(
            QIcon.fromTheme(
                "document-save-as"), "Save Selection As Map...", self
        )
        self.save_selection_action.setStatusTip(
            "Save selected region as new named map entry"
        )
        self.save_selection_action.triggered.connect(
            self.save_selection_as_map)
        self.save_selection_action.setEnabled(False)

        self.exit_action = QAction(QIcon.fromTheme(
            "application-exit"), "E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.setStatusTip("Exit")
        self.exit_action.triggered.connect(self.close)

        # Edit Actions
        self.select_tool_action = QAction(
            QIcon.fromTheme("edit-select"), "Select Region Tool", self
        )
        self.select_tool_action.setStatusTip(
            "Activate region selection (click-drag)")
        self.select_tool_action.setCheckable(True)
        self.select_tool_action.triggered.connect(self.toggle_selection_tool)

        # Help Action
        help_ctrl = self._find_control_by_action("show_help")
        help_key = help_ctrl.get("key", "F1") if help_ctrl else "F1"
        self.help_action = QAction(QIcon.fromTheme(
            "help-contents"), "&Help...", self)
        self.help_action.setStatusTip("Show controls")
        try:
            std_key = getattr(QKeySequence.StandardKey, help_key, None)
            self.help_action.setShortcut(std_key or QKeySequence(help_key))
        except Exception:
            self.help_action.setShortcut(QKeySequence("F1"))
        self.help_action.triggered.connect(self.show_help_dialog)

    def create_menu_bar(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.load_text_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_selection_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.select_tool_action)
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.help_action)

    def create_tool_bar(self):
        """Creates the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        toolbar.addAction(self.load_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.select_tool_action)
        toolbar.addAction(self.save_selection_action)

    # --- Action Handlers / Slots ---
    def toggle_selection_tool(self, checked):
        """Activates/deactivates the selection tool in the editor widget."""
        if checked:
            self.editor_widget.start_selection()
        else:
            self.editor_widget.stop_selection()

    def _update_selection_actions(self, selection_rect: QRect):
        """Updates action states based on selection validity."""
        self.save_selection_action.setEnabled(selection_rect.isValid())

    def _find_control_by_action(self, action_name_to_find):
        """Helper using self.app_config."""
        return self.app_config.get("controls", {}).get(action_name_to_find)

    def _format_control_string(self, control_data):
        """Helper using self.app_config."""
        mod = control_data.get("modifier", "None")
        trig = control_data.get("trigger", "?")
        key = control_data.get("key")
        desc = control_data.get("description", "-")
        parts = []
        if mod and mod.lower() != "none":
            parts.append(mod)
        trig_map = {
            "LeftClick": "L Click",
            "RightClick": "R Click",
            "LeftDrag": "L Drag",
            "ScrollUp": "Scroll Up",
            "ScrollDown": "Scroll Down",
        }
        if trig == "KeyPress" and key:
            parts.append(f"'{key}' Key")
        else:
            parts.append(trig_map.get(trig, trig))
        return f"{' + '.join(parts)}: {desc}"

    def show_help_dialog(self):
        """Displays controls help."""
        help_text = "<h2>Controls Help</h2>"
        controls = self.app_config.get("controls", {})
        groups = {"Mouse": [], "Scroll": [], "Keyboard": []}
        sorted_ctrls = sorted(
            controls.items(), key=lambda item: item[1].get("description", "")
        )
        for name, data in sorted_ctrls:
            fmt = self._format_control_string(data)
            trig = data.get("trigger", "").lower()
            if "click" in trig or "drag" in trig:
                groups["Mouse"].append(fmt)
            elif "scroll" in trig:
                groups["Scroll"].append(fmt)
            elif "key" in trig:
                groups["Keyboard"].append(fmt)
        for group, items in groups.items():
            if items:
                help_text += (
                    f"<b>{
                        group}:</b><ul>{''.join(f'<li>{c}</li>' for c in items)}</ul>"
                )
        # Use QMessageBox directly
        msg = QMessageBox(self)
        msg.setWindowTitle("Help")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def save_map(self):
        """Saves the *entire current* map to a named entry in a JSON file."""
        default_dir = os.path.dirname(self.current_filepath or "")
        default_fname = os.path.basename(self.current_filepath or "map.json")
        # Use QFileDialog directly
        fpath, sel_filter = QFileDialog.getSaveFileName(
            self, "Save Map", os.path.join(
                default_dir, default_fname), "*.json"
        )
        if not fpath:
            return
        if "*.json" in sel_filter and not fpath.lower().endswith(".json"):
            fpath += ".json"

        # Use QInputDialog directly
        key, ok = QInputDialog.getText(
            self, "Map Key", "Enter name/key for this map:")
        if not ok or not key.strip():
            QMessageBox.warning(self, "Save", "Key empty.")
            return
            key = key.strip()

        map_data = self.tilemap.export_map_data()
        if save_extracted_map(fpath, key, map_data):  # Use core save function
            self.current_filepath = fpath
            self.update_window_title()
            self.statusBar().showMessage(f"Map '{key}' saved to {fpath}", 3000)
        else:
            QMessageBox.critical(self, "Error", f"Could not save to {fpath}")
            self.statusBar().showMessage("Save error.", 5000)

    def save_selection_as_map(self):
        """Saves the selected region as a new map entry."""
        selection = self.editor_widget.get_current_selection()
        if not selection:
            QMessageBox.warning(self, "Select", "Select region first.")
            return

        extracted = extract_map_region(self.tilemap, selection)
        if not extracted:
            QMessageBox.critical(self, "Error", "Could not extract region.")
            return

        default_dir = os.path.dirname(self.current_filepath or "")
        default_fname = os.path.basename(self.current_filepath or "map.json")
        # Use QFileDialog directly
        fpath, sel_filter = QFileDialog.getSaveFileName(
            self,
            "Save Selection To",
            os.path.join(default_dir, default_fname),
            "*.json",
        )
        if not fpath:
            return
        if "*.json" in sel_filter and not fpath.lower().endswith(".json"):
            fpath += ".json"

        # Use QInputDialog directly
        new_key, ok = QInputDialog.getText(
            self, "New Map Key", "Enter unique key for extracted map:"
        )
        if not ok or not new_key.strip():
            QMessageBox.warning(self, "Save", "Key empty.")
            return
            new_key = new_key.strip()

        if save_extracted_map(fpath, new_key, extracted):  # Use core save function
            self.statusBar().showMessage(
                f"Selection saved as '{new_key}' to {fpath}", 3000
            )
        else:
            QMessageBox.critical(
                self, "Error", f"Could not save extracted map to {fpath}"
            )
            self.statusBar().showMessage("Save error.", 5000)

    def load_map(self):
        """Loads map(s) from JSON."""
        # Use QFileDialog directly
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Load Map(s)", "", "*.json")
        if not fpath:
            return

        try:
            # Use config module imports directly now
            with open(fpath, FILE_READ_MODE) as f:
                content = f.read()
            if not content:
                QMessageBox.critical(self, "Error", f"File empty: {fpath}")
                return
            file_content = JSON_HANDLER.loads(content)

            maps_to_load = {}
            mode, rows, cols = "Single", 1, 1
            sel_keys = []
            is_new = isinstance(file_content, dict) and isinstance(
                file_content.get("maps"), dict
            )

            if is_new:
                avail = file_content.get("maps", {})
                if not avail:
                    QMessageBox.information(self, "Empty", "No maps in file.")
                    return
                # Use MapSelectionDialog directly
                dialog = MapSelectionDialog(list(avail.keys()), self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    sel_keys, mode, rows, cols = dialog.get_selection()
                    if not sel_keys:
                        return
                    maps_to_load = {
                        key: avail.get(key) for key in sel_keys if key in avail
                    }
                else:
                    return
            else:  # Old format
                if isinstance(file_content, dict) and "rows" in file_content:
                    maps_to_load["map_1"] = file_content
                    sel_keys = ["map_1"]
                else:
                    QMessageBox.critical(self, "Error", "Invalid map format.")
                    return

            if not maps_to_load:
                QMessageBox.warning(
                    self, "Load", "No valid map data selected/found.")
                return

            # Validate maps & check tiles
            all_chars, temp_maps, valid_keys = set(), {}, []
            keys_proc = sel_keys if sel_keys else list(maps_to_load.keys())
            for k in keys_proc:
                data = maps_to_load.get(k)
                if not data:
                    continue
                temp = TileMap(0, 0, ".")
                if temp.load_from_data(data, self.app_config):  # Pass config
                    all_chars.update(temp.get_unique_tiles())
                    temp_maps[k] = temp
                    valid_keys.append(k)
                else:
                    QMessageBox.warning(self, "Warn", f"Skipping '{
                                        k}' (load error).")
            if not temp_maps:
                QMessageBox.critical(self, "Error", "No maps loaded.")
                return

            # Handle undefined tiles
            defined = set(self.app_config.get("tiles", {}).keys())
            undefined = all_chars - defined
            if undefined:
                modified = False
                added = []
                fb_color = [255, 0, 255]
                fb_q = QColor(*fb_color)  # Use QColor import
                for c in sorted(list(undefined)):
                    if c not in self.app_config.get("tiles", {}):
                        if "tiles" not in self.app_config:
                            self.app_config["tiles"] = {}
                        self.app_config["tiles"][c] = {
                            "color": fb_color,
                            "color_qt": fb_q,
                            "description": f"Undef '{c}'",
                        }
                        modified = True
                        added.append(c)
                if modified:
                    if save_config(
                        CONFIG_FILE, self.app_config
                    ):  # save_config imported
                        self.palette_widget.rebuild_palette()
                        QMessageBox.warning(
                            self,
                            "Tiles",
                            f"Added defaults: {
                                ', '.join(added)}\nConfig saved.",
                        )
                    else:
                        QMessageBox.critical(
                            self, "Error", "Failed saving config.")

            # Combine maps
            valid_maps = {k: temp_maps[k] for k in valid_keys}
            combined = self._combine_maps(valid_maps, mode, rows, cols)

            # Load into editor
            self.editor_widget.selection_rect = None
            self.editor_widget.selection_rect_changed.emit(QRect())
            if self.select_tool_action.isChecked():
                self.select_tool_action.setChecked(False)
            self.editor_widget.resize_map(
                combined["width"],
                combined["height"],
                combined["default_tile"],
                combined["tiles"],
            )

            self.current_filepath = fpath
            self.update_window_title()
            self.statusBar().showMessage(f"Map(s) loaded from {fpath}", 3000)
            # Centering via signal

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"File not found:\n{fpath}")
        except JSON_DecodeError as e:
            QMessageBox.critical(
                self, "Error", f"Invalid JSON:\n{fpath}\n{e}"
            )  # Use imported alias
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load error:\n{e}")
            log.exception("Load error")

    def _combine_maps(self, loaded_maps_dict, tiling_mode, grid_rows=1, grid_cols=1):
        """Combines multiple maps. Uses self.app_config for default tile."""
        map_keys = list(loaded_maps_dict.keys())
        if not map_keys:
            return {
                "width": 0,
                "height": 0,
                "default_tile": self.app_config.get("default_tile", "."),
                "tiles": [],
            }

        combined_default = self.app_config.get("default_tile", ".")
        total_w, total_h = 0, 0
        map_dims = {k: (m.width, m.height)
                    for k, m in loaded_maps_dict.items()}
        cs, rs = [], []
        actual_cols = max(1, grid_cols)
        actual_rows = max(1, grid_rows)

        if tiling_mode == "Single":
            k = map_keys[0]
            m = loaded_maps_dict[k]
            return {
                "width": m.width,
                "height": m.height,
                "default_tile": m.default_tile,
                "tiles": [r[:] for r in m.tiles],
            }
        elif tiling_mode == "Vertical":
            total_w = max((d[0] for d in map_dims.values()), default=0)
            total_h = sum(d[1] for d in map_dims.values())
        elif tiling_mode == "Horizontal":
            total_w = sum(d[0] for d in map_dims.values())
            total_h = max((d[1] for d in map_dims.values()), default=0)
        elif tiling_mode == "Grid":
            n = len(map_keys)
            actual_rows = max(1, math.ceil(n / actual_cols))
            cw, rh = [0] * actual_cols, [0] * actual_rows
            for i, key in enumerate(map_keys):
                row, col = i // actual_cols, i % actual_cols
                if row < actual_rows:
                    w, h = map_dims[key]
                    if col < len(cw):
                        cw[col] = max(cw[col], w)
                    if row < len(rh):
                        rh[row] = max(rh[row], h)
            total_w, total_h = sum(cw), sum(rh)
            cs, rs = [sum(cw[:i]) for i in range(actual_cols)], [
                sum(rh[:i]) for i in range(actual_rows)
            ]

        total_w, total_h = max(0, total_w), max(0, total_h)
        combined = [[combined_default] * total_w for _ in range(total_h)]
        off_x, off_y = 0, 0
        for i, k in enumerate(map_keys):
            m = loaded_maps_dict[k]
            w, h = map_dims[k]
            px, py = 0, 0
            if tiling_mode == "Vertical":
                py = off_y
                off_y += h
            elif tiling_mode == "Horizontal":
                px = off_x
                off_x += w
            elif tiling_mode == "Grid":
                r, c = i // actual_cols, i % actual_cols
                if r < len(rs) and c < len(cs):
                    px, py = cs[c], rs[r]
                else:
                    log.warning(
                        "Skipping map due to grid index error", map=k
                    )
                    continue

            # Efficient copy
            for r_idx in range(h):
                tr = py + r_idx
                if 0 <= tr < total_h:
                    src_s, src_e = 0, w
                    tgt_s, tgt_e = px, px + w
                    if tgt_s < 0:
                        src_s -= tgt_s
                        tgt_s = 0
                    if tgt_e > total_w:
                        src_e -= tgt_e - total_w
                        tgt_e = total_w
                    if tgt_s < tgt_e and 0 <= src_s < src_e <= w:
                        combined[tr][tgt_s:tgt_e] = m.tiles[r_idx][src_s:src_e]
        return {
            "width": total_w,
            "height": total_h,
            "default_tile": combined_default,
            "tiles": combined,
        }

    def load_text_layout(self):
        """Loads map from text file."""
        # Use QFileDialog directly
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Load Text Layout", "", "*.txt")
        if not fpath:
            return

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f]
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if not lines:
                QMessageBox.warning(self, "Empty", "Text file empty.")
                return

            width = max(len(ln) for ln in lines) if lines else 0
            height = len(lines)
            default = self.app_config.get("default_tile", ".")
            norm_rows = [ln.ljust(width, default) for ln in lines]
            unique = set("".join(norm_rows))
            known = set(self.app_config.get("tiles", {}).keys())
            unknown = sorted(list(unique - known))
            modified = False
            replace = {}
            added = []

            for tile in unknown:
                if tile == default:
                    continue
                # Use QMessageBox directly
                reply = QMessageBox.question(
                    self,
                    "New",
                    f"Undef tile:'{tile}'.Define?",
                    QMessageBox.Ok | QMessageBox.Cancel,
                    QMessageBox.Ok,
                )
                if reply == QMessageBox.Ok:
                    # Use QInputDialog and QColorDialog directly
                    desc, ok = QInputDialog.getText(
                        self, "Desc", f"Desc for '{tile}':", text=f"Tile '{tile}'"
                    )
                    desc = desc if ok else f"Tile '{tile}'"
                    color = QColorDialog.getColor(
                        QColor(200, 200, 200), self, f"Color '{tile}'"
                    )
                    color = color if color.isValid() else QColor(200, 200, 200)
                    if "tiles" not in self.app_config:
                        self.app_config["tiles"] = {}
                    self.app_config["tiles"][tile] = {
                        "description": desc,
                        "color_qt": color,
                        "color": [color.red(), color.green(), color.blue()],
                    }
                    modified = True
                    added.append(tile)
                else:
                    replace[tile] = default

            final_rows = [
                [replace.get(c, c) for c in list(row_s)] for row_s in norm_rows
            ]

            if modified:
                if save_config(CONFIG_FILE, self.app_config):  # save_config imported
                    self.palette_widget.rebuild_palette()
                    self.statusBar().showMessage("Config saved.", 3000)
                else:
                    QMessageBox.critical(self, "Error", "Failed save config.")

            # Load into editor
            self.editor_widget.selection_rect = None
            self.editor_widget.selection_rect_changed.emit(QRect())
            if self.select_tool_action.isChecked():
                self.select_tool_action.setChecked(False)
            self.editor_widget.resize_map(
                width, height, default, new_tiles_data=final_rows
            )

            self.current_filepath = fpath
            self.update_window_title()
            self.statusBar().showMessage(f"Text loaded from {fpath}", 3000)
            # Centering via signal

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Not found: {fpath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load text error: {e}")
            log.exception("Load text error")

    def center_map_view(self):
        """Centers the map view."""
        try:
            QApplication.processEvents()
            ws, vps = self.editor_widget.size(), self.scroll_area.viewport().size()
            h_bar, v_bar = (
                self.scroll_area.horizontalScrollBar(),
                self.scroll_area.verticalScrollBar(),
            )
            cx = max(0, (ws.width() - vps.width()) / 2)
            cy = max(0, (ws.height() - vps.height()) / 2)
            h_bar.setValue(min(int(cx), h_bar.maximum()))
            v_bar.setValue(min(int(cy), v_bar.maximum()))
        except Exception as e:
            log.error("Error centering", error=str(e))

    def center_map_view_if_needed(self):
        """Calls center_map_view via QTimer."""
        QTimer.singleShot(0, self.center_map_view)

    def update_palette_selection(self):
        """Triggers palette visual update."""
        if hasattr(self, "palette_widget") and self.palette_widget:
            self.palette_widget.update_selection_visuals()
            self.palette_widget.update_selected_tile_info()

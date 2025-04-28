import sys
import json
import os
import math # For zoom centering
import re # For parsing map selection ranges
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSizePolicy, QColorDialog,
    QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox,
    QSpacerItem, QMenu, QInputDialog, QScrollArea, QScrollBar,
    QListWidget, QListWidgetItem, QAbstractItemView,  # For map selection dialog
    QComboBox, QSpinBox # For tiling options dialog
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QAction, QKeySequence, QMouseEvent,
    QPaintEvent, QPixmap, QIcon, QContextMenuEvent, QFont,
    QWheelEvent, QPalette
)
from PySide6.QtCore import Qt, QPoint, QRect, QSize, Signal, QTimer

# --- Configuration File ---
CONFIG_FILE = "tile_editor_config.json"
CURRENT_FORMAT_VERSION = 2 # For multi-map format

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "format_version": CURRENT_FORMAT_VERSION, # Add format version
    "grid_width": 40,
    "grid_height": 40,
    "tile_size": 16,
    "min_tile_size": 4,
    "max_tile_size": 64,
    "zoom_step": 1.2,
    "pan_step": 50,
    "default_tile": ".",
    "wall_tile": "#",
    "door_tile": "+", # Added door tile config
    "window_title": "PySide6 Tile Map Editor (Configurable)",
    "preview_line_color": [255, 0, 0],
    "preview_line_thickness": 1,
    "tiles": {
        ".": {"color": [220, 220, 220], "description": "Empty Floor"},
        "#": {"color": [100, 100, 100], "description": "Wall"},
        "+": {"color": [200, 150, 100], "description": "Door"},
        "k": {"color": [255, 215, 0], "description": "Kitchen"},
        "b": {"color": [0, 0, 200], "description": "Bedroom"},
        "l": {"color": [0, 150, 0], "description": "Living Space"},
        "t": {"color": [150, 150, 255], "description": "Toilet"},
    },
    "controls": {
        "place_tile_click": {"modifier": "None", "trigger": "LeftClick", "description": "Place selected tile (single click)"},
        "draw_line": {"modifier": "None", "trigger": "LeftDrag", "description": "Draw line of selected tile"},
        "erase_tile": {"modifier": "None", "trigger": "RightClick", "description": "Erase tile (set to default)"},
        "draw_rect": {"modifier": "Shift", "trigger": "LeftDrag", "description": "Draw rectangle of selected tile"},
        "fill_perimeter": {"modifier": "Ctrl", "trigger": "LeftClick", "description": "Fill perimeter of non-default area"},
        "wall_perimeter": {"modifier": "Ctrl+Shift", "trigger": "LeftClick", "description": "Wall perimeter of same-tile area (respects walls/doors)"}, # Updated desc
        "flood_fill": {"modifier": "Alt", "trigger": "LeftClick", "description": "Flood fill area with selected tile"}, # New Fill Tool
        "zoom_in": {"modifier": "Ctrl", "trigger": "ScrollUp", "description": "Zoom In"},
        "zoom_out": {"modifier": "Ctrl", "trigger": "ScrollDown", "description": "Zoom Out"},
        "pan_right": {"modifier": "Shift", "trigger": "ScrollUp", "description": "Pan Right"},
        "pan_left": {"modifier": "Shift", "trigger": "ScrollDown", "description": "Pan Left"},
        "show_help": {"modifier": "None", "trigger": "KeyPress", "key": "F1", "description": "Show this help message"},
        "select_tile_1": {"modifier": "None", "trigger": "KeyPress", "key": "1", "description": "Select Tile 1"},
        "select_tile_2": {"modifier": "None", "trigger": "KeyPress", "key": "2", "description": "Select Tile 2"},
        "select_tile_3": {"modifier": "None", "trigger": "KeyPress", "key": "3", "description": "Select Tile 3"},
        "select_tile_4": {"modifier": "None", "trigger": "KeyPress", "key": "4", "description": "Select Tile 4"},
        "select_tile_5": {"modifier": "None", "trigger": "KeyPress", "key": "5", "description": "Select Tile 5"},
        "select_tile_6": {"modifier": "None", "trigger": "KeyPress", "key": "6", "description": "Select Tile 6"},
        "select_tile_7": {"modifier": "None", "trigger": "KeyPress", "key": "7", "description": "Select Tile 7"},
        "select_tile_8": {"modifier": "None", "trigger": "KeyPress", "key": "8", "description": "Select Tile 8"},
        "select_tile_9": {"modifier": "None", "trigger": "KeyPress", "key": "9", "description": "Select Tile 9"},
    }
}

# --- Global Configuration Variable ---
AppConfig = {}

# --- Helper to parse modifier strings ---
def parse_modifier(modifier_str):
    """Parses a modifier string (e.g., 'Ctrl+Shift') into Qt.KeyboardModifiers."""
    modifier = Qt.KeyboardModifier.NoModifier
    if not modifier_str or modifier_str.lower() == "none":
        return modifier
    parts = [part.strip().lower() for part in modifier_str.split('+')]
    if "ctrl" in parts:
        modifier |= Qt.KeyboardModifier.ControlModifier
    if "shift" in parts:
        modifier |= Qt.KeyboardModifier.ShiftModifier
    if "alt" in parts:
        modifier |= Qt.KeyboardModifier.AltModifier
    return modifier

# --- Helper to get Qt.Key from string ---
def parse_key(key_str):
    """Parses a key string (e.g., 'F1', '1', 'A') into Qt.Key enum."""
    if not key_str:
        return None
    key_str_upper = key_str.upper()
    if key_str_upper.startswith('F') and key_str_upper[1:].isdigit():
        f_num = int(key_str_upper[1:])
        if 1 <= f_num <= 35:
             return getattr(Qt.Key, f"Key_F{f_num}", None)
    if key_str.isdigit() and len(key_str) == 1:
        return getattr(Qt.Key, f"Key_{key_str}", None)
    if len(key_str) == 1 and key_str.isalpha():
         return getattr(Qt.Key, f"Key_{key_str_upper}", None)
    key_map = {
        'esc': Qt.Key.Key_Escape,'escape': Qt.Key.Key_Escape,'tab': Qt.Key.Key_Tab,
        'enter': Qt.Key.Key_Enter,'return': Qt.Key.Key_Return,'space': Qt.Key.Key_Space,
        'backspace': Qt.Key.Key_Backspace,'delete': Qt.Key.Key_Delete,'insert': Qt.Key.Key_Insert,
        'home': Qt.Key.Key_Home,'end': Qt.Key.Key_End,'pageup': Qt.Key.Key_PageUp,
        'pagedown': Qt.Key.Key_PageDown,'up': Qt.Key.Key_Up,'down': Qt.Key.Key_Down,
        'left': Qt.Key.Key_Left,'right': Qt.Key.Key_Right,
    }
    return key_map.get(key_str.lower())


# --- Configuration Loading/Saving ---
def load_config(filepath):
    """Loads configuration from JSON file, returns defaults on failure."""
    global AppConfig
    loaded_config = None
    config_was_modified = False

    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
                # --- Validation and Defaulting ---
                if isinstance(loaded_config, dict):
                    # Check format version first (optional, but good practice)
                    if loaded_config.get("format_version") != CURRENT_FORMAT_VERSION:
                         print(f"Warning: Config file format mismatch or missing. Applying defaults.")
                         # Force applying all defaults if format is wrong/missing
                         base_config = DEFAULT_CONFIG.copy()
                         base_config.update(loaded_config) # Overwrite defaults with loaded values where possible
                         loaded_config = base_config
                         config_was_modified = True

                    # Ensure essential top-level keys exist
                    for key, value in DEFAULT_CONFIG.items():
                        if key not in loaded_config:
                            print(f"Config Warning: Missing top-level key '{key}', adding from default.")
                            loaded_config[key] = value
                            config_was_modified = True

                    # Ensure controls exist and have all sub-keys
                    default_controls = DEFAULT_CONFIG.get("controls", {})
                    loaded_controls = loaded_config.get("controls", {})
                    if not isinstance(loaded_controls, dict): # Handle case where controls is not a dict
                        print("Config Warning: 'controls' key is not a dictionary. Resetting to default.")
                        loaded_controls = default_controls
                        loaded_config["controls"] = loaded_controls
                        config_was_modified = True

                    for control_key, default_control_data in default_controls.items():
                        if control_key not in loaded_controls:
                            print(f"Config Warning: Missing control '{control_key}', adding from default.")
                            loaded_controls[control_key] = default_control_data
                            config_was_modified = True
                        elif isinstance(loaded_controls[control_key], dict): # Check if existing entry is a dict
                            loaded_control_data = loaded_controls[control_key]
                            for sub_key, default_sub_value in default_control_data.items():
                                if sub_key not in loaded_control_data:
                                     print(f"Config Warning: Missing sub-key '{sub_key}' in control '{control_key}', adding from default.")
                                     loaded_control_data[sub_key] = default_sub_value
                                     config_was_modified = True
                        else: # If existing entry is not a dict, overwrite with default
                             print(f"Config Warning: Control '{control_key}' is not a dictionary. Resetting to default.")
                             loaded_controls[control_key] = default_control_data
                             config_was_modified = True


                    # Convert/Validate Tile Colors
                    loaded_tiles = loaded_config.get("tiles", {})
                    if not isinstance(loaded_tiles, dict):
                        print("Config Warning: 'tiles' key is not a dictionary. Resetting to default.")
                        loaded_tiles = DEFAULT_CONFIG["tiles"]
                        loaded_config["tiles"] = loaded_tiles
                        config_was_modified = True

                    for tile_char, tile_data in loaded_tiles.items():
                         if not isinstance(tile_data, dict): # Ensure tile data is a dict
                              print(f"Config Warning: Data for tile '{tile_char}' is not a dictionary. Using fallback.")
                              loaded_tiles[tile_char] = {"color": [255,0,255], "description": f"Invalid Tile '{tile_char}'"}
                              tile_data = loaded_tiles[tile_char]
                              config_was_modified = True

                         if isinstance(tile_data.get("color"), list) and len(tile_data["color"]) == 3:
                             rgb = tile_data["color"]
                             tile_data["color_qt"] = QColor(rgb[0], rgb[1], rgb[2])
                         else:
                             print(f"Config Warning: Invalid color format for tile '{tile_char}'. Using fallback.")
                             tile_data["color_qt"] = QColor(255, 0, 255) # Magenta fallback
                             tile_data["color"] = [255, 0, 255] # Ensure raw color is also fallback
                             config_was_modified = True

                    # Convert/Validate Preview Color
                    preview_rgb = loaded_config.get("preview_line_color", DEFAULT_CONFIG["preview_line_color"])
                    if isinstance(preview_rgb, list) and len(preview_rgb) == 3:
                        loaded_config["preview_line_color_qt"] = QColor(preview_rgb[0], preview_rgb[1], preview_rgb[2])
                    else:
                        print("Config Warning: Invalid preview_line_color format. Using default.")
                        default_rgb = DEFAULT_CONFIG["preview_line_color"]
                        loaded_config["preview_line_color_qt"] = QColor(default_rgb[0], default_rgb[1], default_rgb[2])
                        loaded_config["preview_line_color"] = default_rgb
                        config_was_modified = True

                    # Validate numeric settings
                    for key in ["preview_line_thickness", "tile_size", "min_tile_size", "max_tile_size", "pan_step"]:
                         val = loaded_config.get(key, DEFAULT_CONFIG[key]) # Get value or default
                         if not isinstance(val, int) or val < 1:
                             print(f"Config Warning: Invalid value for '{key}'. Using default.")
                             loaded_config[key] = DEFAULT_CONFIG[key]
                             config_was_modified = True # May have been added above, but ensure flag is set if value invalid
                    zoom_val = loaded_config.get("zoom_step", DEFAULT_CONFIG["zoom_step"])
                    if not isinstance(zoom_val, (float, int)) or zoom_val <= 1.0:
                         print(f"Config Warning: Invalid value for 'zoom_step'. Using default.")
                         loaded_config["zoom_step"] = DEFAULT_CONFIG["zoom_step"]
                         config_was_modified = True

                    # Validate min/max tile size
                    if loaded_config.get("min_tile_size", 1) >= loaded_config.get("max_tile_size", 64):
                        print("Config Warning: min_tile_size >= max_tile_size. Resetting to defaults.")
                        loaded_config["min_tile_size"] = DEFAULT_CONFIG["min_tile_size"]
                        loaded_config["max_tile_size"] = DEFAULT_CONFIG["max_tile_size"]
                        config_was_modified = True
                    # Clamp current tile_size
                    loaded_config["tile_size"] = max(loaded_config["min_tile_size"], min(loaded_config["tile_size"], loaded_config["max_tile_size"]))

                    # --- End Validation ---

                    print(f"Configuration loaded from {filepath}")
                    AppConfig = loaded_config

                    if config_was_modified:
                        print("Saving configuration file with added/corrected default values...")
                        save_config(filepath, AppConfig)

                    return # Success
                else:
                    print(f"Warning: Invalid configuration format in {filepath}. Using defaults.")
                    loaded_config = None
        else:
            print(f"Warning: Configuration file {filepath} not found. Using defaults and creating file.")
            loaded_config = DEFAULT_CONFIG.copy()
            config_was_modified = True
            # Convert default colors to QColor immediately
            for tile_data in loaded_config.get("tiles", {}).values():
                 if isinstance(tile_data.get("color"), list) and len(tile_data["color"]) == 3:
                     rgb = tile_data["color"]
                     tile_data["color_qt"] = QColor(rgb[0], rgb[1], rgb[2])
                 else:
                     tile_data["color_qt"] = QColor(255, 0, 255)
            default_preview_rgb = loaded_config["preview_line_color"]
            loaded_config["preview_line_color_qt"] = QColor(default_preview_rgb[0], default_preview_rgb[1], default_preview_rgb[2])

            if save_config(filepath, loaded_config):
                 AppConfig = loaded_config
                 return
            else:
                 print("Error saving default config. Using in-memory defaults.")
                 loaded_config = None

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. Using defaults.")
        loaded_config = None
    except Exception as e:
        print(f"Error loading configuration: {e}. Using defaults.")
        loaded_config = None

    # Fallback to hardcoded defaults if any error occurred and wasn't recovered
    if AppConfig is None or not AppConfig:
        print("Using hardcoded default configuration.")
        AppConfig = DEFAULT_CONFIG.copy()
        # Convert default colors to QColor
        for tile_data in AppConfig.get("tiles", {}).values():
            if isinstance(tile_data.get("color"), list) and len(tile_data["color"]) == 3:
                rgb = tile_data["color"]
                tile_data["color_qt"] = QColor(rgb[0], rgb[1], rgb[2])
            else:
                tile_data["color_qt"] = QColor(255, 0, 255)
        default_preview_rgb = AppConfig["preview_line_color"]
        AppConfig["preview_line_color_qt"] = QColor(default_preview_rgb[0], default_preview_rgb[1], default_preview_rgb[2])
        # Ensure numeric defaults are present
        for key in ["preview_line_thickness", "tile_size", "min_tile_size", "max_tile_size", "pan_step", "zoom_step"]:
             AppConfig[key] = DEFAULT_CONFIG[key]


def save_config(filepath, config_data):
    """Saves the configuration dictionary to a JSON file."""
    try:
        save_data = {}
        qcolor_keys = [k for k in config_data if k.endswith("_qt")]
        keys_to_skip = ["tiles"] + qcolor_keys

        for key, value in config_data.items():
             if key not in keys_to_skip:
                 save_data[key] = value

        # Process 'tiles' separately
        save_data["tiles"] = {}
        for char, tile_info in config_data.get("tiles", {}).items():
            color_qt = tile_info.get("color_qt")
            if isinstance(color_qt, QColor):
                 color_list = [color_qt.red(), color_qt.green(), color_qt.blue()]
            elif isinstance(tile_info.get("color"), list) and len(tile_info["color"]) == 3:
                 color_list = tile_info["color"]
            else:
                 color_list = [255, 0, 255] # Fallback Magenta

            save_data["tiles"][char] = {
                 "color": color_list,
                 "description": tile_info.get("description", "")
             }

        # Ensure specific keys are present using defaults if needed
        for key in ["format_version", "preview_line_color", "preview_line_thickness", "tile_size", "min_tile_size", "max_tile_size", "zoom_step", "pan_step", "default_tile", "wall_tile", "door_tile", "window_title", "controls"]:
             if key not in save_data:
                 if key == "preview_line_color" and "preview_line_color_qt" in config_data:
                     preview_color_qt = config_data["preview_line_color_qt"]
                     save_data[key] = [preview_color_qt.red(), preview_color_qt.green(), preview_color_qt.blue()]
                 else:
                     save_data[key] = config_data.get(key, DEFAULT_CONFIG.get(key))

        # Ensure format version is current
        save_data["format_version"] = CURRENT_FORMAT_VERSION

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"Configuration saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


# --- TileMap Data Structure ---
class TileMap:
    """Stores and manages the grid data."""
    def __init__(self, width, height, default_tile):
        # Allow creation of 0x0 maps initially for loading logic
        width = max(0, width)
        height = max(0, height)
        self.width = width
        self.height = height
        self.default_tile = default_tile
        # Initialize grid with the default tile
        self.tiles = [[self.default_tile for _ in range(width)] for _ in range(height)]

    def set_tile(self, x, y, tile):
        """Sets the tile at the given grid coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.tiles[y][x] = tile
            return True
        return False # Indicate failure if out of bounds

    def get_tile(self, x, y):
        """Gets the tile at the given grid coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None # Return None if out of bounds

    def get_unique_tiles(self):
        """Returns a set of unique tile characters used in the map."""
        unique_chars = set()
        for row in self.tiles:
            unique_chars.update(row)
        return unique_chars

    def export_map_data(self):
         """Exports just the map data part for multi-map saving."""
         return {
             "grid_width": self.width,
             "grid_height": self.height,
             "default_tile": self.default_tile,
             "rows": ["".join(row) for row in self.tiles]
         }

    def load_from_data(self, data):
        """Loads tile data from a dictionary (e.g., from JSON map data)."""
        try:
            rows = data.get("rows", [])
            if not isinstance(rows, list):
                 print("Error: Invalid map format - 'rows' key is not a list.")
                 return False

            # --- Handle null/missing dimension and default tile ---
            raw_height = data.get("grid_height")
            raw_width = data.get("grid_width")
            raw_default = data.get("default_tile")

            new_height = len(rows) if raw_height is None else int(raw_height)
            map_default_tile = AppConfig.get("default_tile", ".") if raw_default is None else str(raw_default)

            if new_height < 0: new_height = 0 # Ensure non-negative

            # Infer width if null or missing
            new_width = 0
            if raw_width is not None:
                new_width = int(raw_width)
            elif new_height > 0 and rows and isinstance(rows[0], str):
                new_width = len(rows[0])

            if new_width < 0: new_width = 0 # Ensure non-negative

            # Validation
            if not all(isinstance(row, str) for row in rows):
                 print("Error: Invalid map format - not all rows are strings.")
                 return False
            # Ensure consistent row lengths AFTER inferring width (only if width > 0)
            if new_width > 0 and not all(len(row) == new_width for row in rows):
                 # Allow loading slightly inconsistent rows, but print warning
                 print(f"Warning: Rows in loaded map data have inconsistent lengths. Using max width {new_width}.")
                 # Optionally pad rows here, or let the assignment handle it (might error later)
                 # For simplicity, we'll proceed but map might be visually strange
                 pass # rows = [row.ljust(new_width, map_default_tile) for row in rows] # Example padding
            if len(rows) != new_height:
                 print(f"Warning: Map data height mismatch ('grid_height'={raw_height} vs actual rows={len(rows)}). Using actual row count.")
                 new_height = len(rows)

            # Update dimensions and tiles
            self.height = new_height
            self.width = new_width
            self.default_tile = map_default_tile
            # Ensure self.tiles is correct size even if rows were inconsistent/empty
            self.tiles = [[map_default_tile for _ in range(new_width)] for _ in range(new_height)]
            # Copy loaded row data into the correctly sized grid
            copy_h = min(len(rows), new_height)
            for y in range(copy_h):
                row_list = list(rows[y])
                copy_w = min(len(row_list), new_width)
                for x in range(copy_w):
                    self.tiles[y][x] = row_list[x]

            print(f"Loaded map data with dimensions: {self.width}x{self.height}, Default Tile: '{self.default_tile}'")
            return True

        except (ValueError, TypeError) as e: # Catch potential errors from int() or str() if data malformed
             print(f"Error processing map data values: {e}")
             return False
        except Exception as e:
            print(f"Error loading map data: {e}")
            return False

# --- Helper Functions ---
def get_neighbors(x, y):
    """Returns 4-directional neighbors (up, down, left, right)."""
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

def draw_line(tilemap, start_pos, end_pos, tile):
    """Draws a line of tiles using Bresenham's algorithm."""
    x1, y1 = start_pos.x(), start_pos.y()
    x2, y2 = end_pos.x(), end_pos.y()
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    while True:
        tilemap.set_tile(x1, y1, tile)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy

def fill_rectangle(tilemap, start_pos, end_pos, tile):
    """Fills a rectangle defined by start and end points."""
    x1, y1 = start_pos.x(), start_pos.y()
    x2, y2 = end_pos.x(), end_pos.y()
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            tilemap.set_tile(x, y, tile)

def flood_fill(tilemap, start_x, start_y, match_tile_type=None):
    """
    Performs a flood fill to find contiguous area.
    If match_tile_type is None, finds contiguous non-default tiles.
    If match_tile_type is specified, finds contiguous tiles of that specific type.
    Returns a set of (x, y) coordinates of the filled area.
    """
    start_tile = tilemap.get_tile(start_x, start_y)
    if start_tile is None: return set() # Clicked outside map

    if match_tile_type is None:
        # Mode 1: Fill non-default area (like original ctrl_click_fill)
        if start_tile == tilemap.default_tile:
            return set() # Cannot fill starting from default tile in this mode
        target_type = start_tile # The type we are looking for is the one we clicked on
    else:
        # Mode 2: Fill specific type (for ctrl_shift_click_wall & flood_fill_replace)
        if start_tile != match_tile_type:
            return set() # Clicked tile doesn't match the type we want to fill
        target_type = match_tile_type

    seen = set()
    frontier = [(start_x, start_y)]
    connected_area = set()

    while frontier:
        cx, cy = frontier.pop()

        # Boundary check
        if not (0 <= cx < tilemap.width and 0 <= cy < tilemap.height):
            continue

        # Visited check
        if (cx, cy) in seen:
            continue

        current_tile = tilemap.get_tile(cx, cy)

        # Match check (only add if it matches the target type)
        if current_tile == target_type:
            seen.add((cx, cy))
            connected_area.add((cx, cy))
            # Add neighbors to the frontier
            for nx, ny in get_neighbors(cx, cy):
                if (nx, ny) not in seen: # Optimization: don't add if already seen
                    frontier.append((nx, ny))
        else:
            # If it doesn't match, mark as seen so we don't process its neighbors from here
            seen.add((cx,cy))

    return connected_area


def flood_fill_replace(tilemap, start_x, start_y, fill_tile):
    """
    Performs a classic flood fill replace. Replaces the contiguous area
    of the same tile type as the starting tile with the fill_tile.
    """
    original_tile = tilemap.get_tile(start_x, start_y)

    # Don't fill if clicking outside, or if start tile is already the fill tile
    if original_tile is None or original_tile == fill_tile:
        return 0 # Return 0 tiles filled

    q = [(start_x, start_y)] # Queue for BFS
    visited = set()
    filled_count = 0

    while q:
        x, y = q.pop(0)

        # Check boundaries
        if not (0 <= x < tilemap.width and 0 <= y < tilemap.height):
            continue

        # Check if visited
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Check if tile matches the original type
        current_tile = tilemap.get_tile(x, y)
        if current_tile == original_tile:
            # Replace tile and add neighbors to queue
            if tilemap.set_tile(x, y, fill_tile):
                filled_count += 1
                for nx, ny in get_neighbors(x, y):
                    if (nx, ny) not in visited:
                        q.append((nx, ny))

    print(f"Flood Fill: Replaced {filled_count} tiles of type '{original_tile}' with '{fill_tile}'.")
    return filled_count


def ctrl_click_fill(tilemap, x, y, fill_tile):
    """
    Finds a contiguous area of non-default tiles starting from (x,y)
    and fills its perimeter (adjacent default tiles) with the fill_tile.
    Creates square corners.
    """
    connected_area = flood_fill(tilemap, x, y, match_tile_type=None)
    if not connected_area:
        print(f"Ctrl+Click: No area to fill (clicked on default '{tilemap.default_tile}' or outside).")
        return

    perimeter_to_fill = set()
    for cx, cy in connected_area:
        for nx, ny in get_neighbors(cx, cy):
            if 0 <= nx < tilemap.width and 0 <= ny < tilemap.height:
                neighbor_tile = tilemap.get_tile(nx, ny)
                if neighbor_tile == tilemap.default_tile:
                    perimeter_to_fill.add((nx, ny))

    if not perimeter_to_fill:
         print("Ctrl+Click: Area found, but no default tile perimeter detected.")

    painted_count = 0
    for px, py in perimeter_to_fill:
        if tilemap.set_tile(px, py, fill_tile):
            painted_count += 1

    print(f"Ctrl+Click: Painted {painted_count} perimeter tiles with '{fill_tile}'.")


def ctrl_shift_click_wall(tilemap, x, y, wall_fill_tile):
    """
    Finds a contiguous area of the *same type* as the tile at (x,y).
    Fills the perimeter of this area with wall_fill_tile, but *respects*
    existing wall tiles and door tiles. Creates square corners.
    """
    clicked_tile_type = tilemap.get_tile(x, y)
    wall_char = AppConfig.get("wall_tile", "#")
    door_char = AppConfig.get("door_tile", "+") # Get door tile from config
    protected_chars = {wall_char, door_char} # Tiles to not overwrite

    if clicked_tile_type is None or clicked_tile_type == tilemap.default_tile:
        print(f"Ctrl+Shift+Click: Cannot start walling from default tile ('{tilemap.default_tile}') or outside map.")
        return

    connected_area = flood_fill(tilemap, x, y, match_tile_type=clicked_tile_type)
    if not connected_area:
        print(f"Ctrl+Shift+Click: No contiguous area of type '{clicked_tile_type}' found at ({x},{y}).")
        return

    perimeter_to_wall = set()
    for cx, cy in connected_area:
        for nx, ny in get_neighbors(cx, cy):
            if 0 <= nx < tilemap.width and 0 <= ny < tilemap.height:
                if (nx, ny) not in connected_area: # Add if neighbor is outside the area
                    perimeter_to_wall.add((nx, ny))

    if not perimeter_to_wall:
         print(f"Ctrl+Shift+Click: Area of '{clicked_tile_type}' found, but no perimeter tiles detected.")

    painted_count = 0
    skipped_count = 0
    for px, py in perimeter_to_wall:
        existing_tile = tilemap.get_tile(px, py)
        # Only paint if the existing tile is NOT a protected character
        if existing_tile not in protected_chars:
            if tilemap.set_tile(px, py, wall_fill_tile):
                painted_count += 1
        else:
            skipped_count += 1
            print(f"Debug: Skipped protected tile '{existing_tile}' at ({px},{py})")


    print(f"Ctrl+Shift+Click: Painted {painted_count} perimeter tiles with '{wall_fill_tile}'. Skipped {skipped_count} protected tiles ({protected_chars}).")


# --- Edit Tile Dialog ---
class EditTileDialog(QDialog):
    """Dialog for adding or editing a tile type."""
    def __init__(self, tile_char=None, tile_data=None, existing_chars=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Tile" if tile_char else "Add New Tile")
        self.existing_chars = existing_chars or set()
        self.original_char = tile_char # Keep track for editing validation
        self.is_editing = tile_char is not None

        # --- Widgets ---
        self.char_edit = QLineEdit(tile_char if tile_char else "")
        self.char_edit.setMaxLength(1) # Only single character allowed

        self.desc_edit = QLineEdit(tile_data.get("description", "") if tile_data else "")
        self.color_button = QPushButton()
        self.color_button.setFixedSize(40, 25)
        self.color_button.setFlat(False)

        initial_color = tile_data.get("color_qt", QColor(200, 200, 200)) if tile_data else QColor(200, 200, 200)
        self.set_button_color(initial_color)
        self.color_button.clicked.connect(self.pick_color)

        # --- Layout ---
        form_layout = QFormLayout()
        form_layout.addRow("Character:", self.char_edit)
        form_layout.addRow("Description:", self.desc_edit)
        form_layout.addRow("Color:", self.color_button)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)

        # --- Main Layout ---
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        self.setMinimumWidth(300)

    def set_button_color(self, color):
        """Sets the background color and stores the QColor."""
        self.current_color = color
        self.color_button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

    def pick_color(self):
        """Opens a color dialog to choose a new color."""
        new_color = QColorDialog.getColor(self.current_color, self, "Select Tile Color")
        if new_color.isValid():
            self.set_button_color(new_color)

    def validate_and_accept(self):
        """Validates input before accepting the dialog."""
        char = self.char_edit.text()

        if not char:
            QMessageBox.warning(self, "Validation Error", "Tile character cannot be empty.")
            return
        if len(char) > 1:
             QMessageBox.warning(self, "Validation Error", "Tile character must be a single character.")
             return
        # Check if character is already used (unless it's the original character being edited)
        if char != self.original_char and char in self.existing_chars:
            QMessageBox.warning(self, "Validation Error", f"The character '{char}' is already used by another tile.")
            return

        # Add warning if changing the character *from* the current default tile
        current_default = AppConfig.get("default_tile")
        if self.is_editing and self.original_char == current_default and char != current_default:
             reply = QMessageBox.warning(self, "Confirm Default Tile Change",
                                        f"You are changing the character ('{self.original_char}') currently designated as the 'default_tile' in the configuration.\n\n"
                                        f"This is allowed, but you should update the 'default_tile' setting in '{CONFIG_FILE}' manually if you want '{char}' to be the new default.\n\n"
                                        f"Proceed with editing this tile character?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                 return # Abort accept if user cancels

        self.accept()

    def get_tile_data(self):
        """Returns the entered tile data."""
        return {
            "char": self.char_edit.text(),
            "data": {
                 "description": self.desc_edit.text(),
                 "color_qt": self.current_color,
            }
        }


# --- Tile Editor Widget ---
class TileEditorWidget(QWidget):
    """The main widget for drawing and interacting with the tile map."""
    map_changed = Signal()
    zoom_changed = Signal() # Signal emitted when zoom level changes

    def __init__(self, tilemap, scroll_area, parent=None): # Accept scroll_area
        super().__init__(parent)
        self.tilemap = tilemap # Reference to the shared TileMap
        self.scroll_area = scroll_area # Store reference to the scroll area
        self._selected_tile = self._get_initial_selected_tile()
        self.start_drag_pos = None
        self.current_mouse_pos = QPoint(0, 0) # Initialize mouse position
        self.drawing_line = False
        self.drawing_rect = False
        self._tile_size = AppConfig.get("tile_size", 16)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.update_widget_size()
        # Use Preferred size policy so scroll area can manage it
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # Ensure the widget has a background for proper rendering in scroll area
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(Qt.GlobalColor.darkGray)) # Example dark background
        self.setPalette(pal)


    def _get_initial_selected_tile(self):
        """Determines a valid initial tile selection."""
        tiles = AppConfig.get("tiles", {})
        default_tile = AppConfig.get("default_tile", ".")
        wall_tile = AppConfig.get("wall_tile", "#")
        first_choice = None
        # Try to find a tile that is neither default nor wall first
        for char in tiles.keys():
            if char != default_tile and char != wall_tile:
                return char
        # If none found, try first non-default
        for char in tiles.keys():
             if char != default_tile:
                 return char
        # Fallback to default if only default exists
        return default_tile


    def update_widget_size(self):
         """Updates minimum and preferred size based on map and tile size."""
         self._tile_size = AppConfig.get("tile_size", 16) # Get current tile size
         width = self.tilemap.width * self._tile_size
         height = self.tilemap.height * self._tile_size
         # Ensure minimum size is at least 1x1 to avoid layout issues
         self.setMinimumSize(max(1, width), max(1, height))
         self.updateGeometry() # Inform layout/scroll area
         self.update() # Trigger repaint

    def sizeHint(self):
        """Provide a preferred size based on current tile size."""
        current_tile_size = AppConfig.get("tile_size", 16)
        width = self.tilemap.width * current_tile_size
        height = self.tilemap.height * current_tile_size
        # Return at least 1x1
        return QSize(max(1, width), max(1, height))

    def resize_map(self, new_width, new_height, new_default_tile, new_tiles_data=None):
        """
        Resizes the *existing* underlying tilemap instance and redraws.
        If new_tiles_data is provided (list of lists), it replaces the current tiles.
        Otherwise, it preserves existing tiles within the new bounds.
        """
        # Allow 0 dimensions
        new_width = max(0, new_width)
        new_height = max(0, new_height)
        # if new_width <= 0 or new_height <= 0: return # Allow 0 dimensions

        old_tiles = None
        if new_tiles_data is None:
            old_tiles = [row[:] for row in self.tilemap.tiles]
            old_width = self.tilemap.width
            old_height = self.tilemap.height

        self.tilemap.width = new_width
        self.tilemap.height = new_height
        self.tilemap.default_tile = new_default_tile

        if new_tiles_data is not None:
            # Check dimensions carefully, allowing empty rows for 0 height
            if len(new_tiles_data) == new_height and \
               (new_height == 0 or all(len(row) == new_width for row in new_tiles_data)):
                self.tilemap.tiles = new_tiles_data
            else:
                print(f"Error: Provided new_tiles_data dimensions ({len(new_tiles_data)}x{len(new_tiles_data[0]) if new_height>0 else 0}) mismatch new ({new_height}x{new_width}). Re-initializing.")
                self.tilemap.tiles = [[new_default_tile for _ in range(new_width)] for _ in range(new_height)]
        else:
            self.tilemap.tiles = [[new_default_tile for _ in range(new_width)] for _ in range(new_height)]
            if old_tiles:
                copy_height = min(old_height, new_height)
                copy_width = min(old_width, new_width)
                for y in range(copy_height):
                    # Ensure old row exists before trying to copy from it
                    if y < len(old_tiles):
                        for x in range(copy_width):
                             # Ensure old column exists
                             if x < len(old_tiles[y]):
                                 self.tilemap.tiles[y][x] = old_tiles[y][x]


        self.update_widget_size()
        self.map_changed.emit()


    @property
    def selected_tile(self):
        return self._selected_tile

    @selected_tile.setter
    def selected_tile(self, tile_char):
        if tile_char in AppConfig.get("tiles", {}):
            if self._selected_tile != tile_char:
                 self._selected_tile = tile_char
                 print(f"Selected tile: {self._selected_tile}")
        else:
            print(f"Warning: Attempted to select unknown tile '{tile_char}'. Reverting selection.")
            safe_tile = self._get_initial_selected_tile()
            if self._selected_tile != safe_tile:
                self._selected_tile = safe_tile
                print(f"Selected tile reverted to: {self._selected_tile}")
            # Access MainWindow correctly to update palette
            main_window = self.window()
            if isinstance(main_window, MainWindow):
                main_window.update_palette_selection()


    def pixel_to_grid(self, pos: QPoint):
        """Converts pixel coordinates (QWidget) to grid coordinates."""
        current_tile_size = AppConfig.get("tile_size", 16)
        if current_tile_size <= 0: return QPoint(0,0)
        x = pos.x() // current_tile_size
        y = pos.y() // current_tile_size
        x = max(0, min(x, self.tilemap.width - 1)) if self.tilemap.width > 0 else 0
        y = max(0, min(y, self.tilemap.height - 1)) if self.tilemap.height > 0 else 0
        return QPoint(x, y)

    def grid_to_pixel(self, grid_pos: QPoint):
        """Converts grid coordinates to top-left pixel coordinates."""
        current_tile_size = AppConfig.get("tile_size", 16)
        return QPoint(grid_pos.x() * current_tile_size, grid_pos.y() * current_tile_size)


    def paintEvent(self, event: QPaintEvent):
        """Handles painting the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        tile_configs = AppConfig.get("tiles", {})
        unknown_color = QColor(255, 0, 255) # Magenta for truly unknown/error
        current_tile_size = AppConfig.get("tile_size", 16) # Use current tile size

        # Draw tiles
        for y in range(self.tilemap.height):
            for x in range(self.tilemap.width):
                tile_char = self.tilemap.get_tile(x, y)
                tile_data = tile_configs.get(tile_char)
                # Use magenta only if tile_data is completely missing from config
                color = tile_data.get("color_qt", unknown_color) if tile_data else unknown_color
                rect = QRect(x * current_tile_size, y * current_tile_size, current_tile_size, current_tile_size)
                painter.fillRect(rect, color)

        # Draw grid lines only if tile size is large enough
        if current_tile_size > 4:
            pen = QPen(QColor(180, 180, 180), 1, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            grid_pixel_width = self.tilemap.width * current_tile_size
            grid_pixel_height = self.tilemap.height * current_tile_size

            for x_coord in range(self.tilemap.width + 1):
                px = x_coord * current_tile_size
                painter.drawLine(px, 0, px, grid_pixel_height) # Draw full height/width
            for y_coord in range(self.tilemap.height + 1):
                py = y_coord * current_tile_size
                painter.drawLine(0, py, grid_pixel_width, py) # Draw full height/width

        # Draw Preview (Line/Rectangle)
        if self.start_drag_pos and self.current_mouse_pos:
            end_pos = self.pixel_to_grid(self.current_mouse_pos)

            # Get preview style from config
            preview_color = AppConfig.get("preview_line_color_qt", QColor(255, 0, 0)) # Default Red
            preview_thickness = AppConfig.get("preview_line_thickness", 1)

            # Create the preview pen
            preview_pen = QPen(preview_color, preview_thickness, Qt.PenStyle.DotLine)
            painter.setPen(preview_pen)

            if self.drawing_rect:
                 x1, y1 = self.start_drag_pos.x(), self.start_drag_pos.y()
                 x2, y2 = end_pos.x(), end_pos.y()
                 min_x, max_x = min(x1, x2), max(x1, x2)
                 min_y, max_y = min(y1, y2), max(y1, y2)
                 # Adjust rect drawing to be pixel-perfect for the grid lines
                 preview_rect = QRect(min_x * current_tile_size, min_y * current_tile_size,
                                     (max_x - min_x + 1) * current_tile_size,
                                     (max_y - min_y + 1) * current_tile_size)
                 # Draw rect slightly inside to align with grid visually if thickness > 1
                 if preview_thickness > 1:
                      offset = preview_thickness / 2.0
                      preview_rect.adjust(offset, offset, -offset, -offset)
                 painter.drawRect(preview_rect)

            elif self.drawing_line:
                 # Draw line centered within start/end tiles
                 start_pixel = QPoint(self.start_drag_pos.x() * current_tile_size + current_tile_size // 2,
                                      self.start_drag_pos.y() * current_tile_size + current_tile_size // 2)
                 # Use current mouse pos for smoother preview line end
                 end_pixel = self.current_mouse_pos
                 painter.drawLine(start_pixel, end_pixel)


    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse button presses based on configured controls."""
        self.current_mouse_pos = event.position().toPoint()
        grid_pos = self.pixel_to_grid(self.current_mouse_pos)
        current_modifiers = event.modifiers()
        controls = AppConfig.get("controls", {})

        # Check Left Click actions
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_drag_pos = grid_pos
            self.drawing_line = False # Reset flags
            self.drawing_rect = False

            # Check configured actions for LeftClick trigger
            action_found = False
            for action_name, control in controls.items():
                if control.get("trigger") == "LeftClick":
                    required_modifier = parse_modifier(control.get("modifier", "None"))
                    # Use exact modifier match
                    if current_modifiers == required_modifier:
                        print(f"Debug: Matched LeftClick action '{action_name}'")
                        # Set flags based on associated drag action (heuristic based on name/modifier)
                        if required_modifier == parse_modifier("Shift"):
                             self.drawing_rect = True # Assume Shift+Click starts rect drag
                        elif required_modifier == Qt.KeyboardModifier.NoModifier:
                             # Check if it's the specific place_tile action or a potential line drag start
                             if action_name == "place_tile_click":
                                 pass # Handled on release
                             else:
                                 self.drawing_line = True # Assume other NoModifier+Click starts line drag
                        # Ctrl, Alt, Ctrl+Shift clicks don't start drags
                        action_found = True
                        # Special case: If flood fill is immediate (on press), trigger here
                        if action_name == "flood_fill":
                             print(f"Executing flood_fill at {grid_pos.x()},{grid_pos.y()} (on press)")
                             filled = flood_fill_replace(self.tilemap, grid_pos.x(), grid_pos.y(), self.selected_tile)
                             if filled > 0: self.update() # Redraw if something changed
                             self.start_drag_pos = None # Don't treat flood fill as start of drag
                             action_found = True # Ensure other logic knows it was handled

                        break # Stop checking once an action is matched

            # If no specific LeftClick action matched for the *exact* modifier combo,
            # but it's a plain left click, assume it starts a line draw.
            if not action_found and current_modifiers == Qt.KeyboardModifier.NoModifier:
                 print("Debug: Defaulting to draw_line start for unmodified LeftClick.")
                 self.drawing_line = True

            # self.update() # Update only needed if visual state changes on press

        # Check Right Click actions
        elif event.button() == Qt.MouseButton.RightButton:
            for action_name, control in controls.items():
                if control.get("trigger") == "RightClick":
                    required_modifier = parse_modifier(control.get("modifier", "None"))
                    if current_modifiers == required_modifier:
                        print(f"Debug: Matched RightClick action '{action_name}'")
                        if action_name == "erase_tile":
                            current_default_tile = self.tilemap.default_tile
                            self.tilemap.set_tile(grid_pos.x(), grid_pos.y(), current_default_tile)
                            self.update()
                        # Add other potential RightClick actions here
                        break # Assume only one action per button/modifier combo


    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles mouse movement for drag previews and updates cursor position."""
        self.current_mouse_pos = event.position().toPoint() # Update position continuously
        # Update preview only if actively dragging a line or rectangle
        if event.buttons() & Qt.MouseButton.LeftButton and self.start_drag_pos:
            if self.drawing_line or self.drawing_rect:
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handles mouse button releases based on configured controls."""
        if not self.current_mouse_pos:
             self.current_mouse_pos = event.position().toPoint()

        end_pos = self.pixel_to_grid(self.current_mouse_pos)
        # Use modifiers active *at the time of release*
        current_modifiers = QApplication.keyboardModifiers() # Get current modifiers state
        controls = AppConfig.get("controls", {})

        if event.button() == Qt.MouseButton.LeftButton and self.start_drag_pos:
            is_drag = (self.start_drag_pos != end_pos)
            start_x, start_y = self.start_drag_pos.x(), self.start_drag_pos.y()
            end_x, end_y = end_pos.x(), end_pos.y()

            action_to_perform = None

            # Determine action based on drag state and modifiers
            trigger_type = "LeftDrag" if is_drag else "LeftClick"

            for action_name, control in controls.items():
                if control.get("trigger") == trigger_type:
                    required_modifier = parse_modifier(control.get("modifier", "None"))
                    if current_modifiers == required_modifier:
                        action_to_perform = action_name
                        break # Found the primary action

            # Fallback for simple click if no specific click action defined for modifier
            place_tile_control = self._find_control_by_action("place_tile_click")
            if not is_drag and action_to_perform is None and \
               place_tile_control and parse_modifier(place_tile_control.get("modifier", "None")) == Qt.KeyboardModifier.NoModifier and \
               current_modifiers == Qt.KeyboardModifier.NoModifier:
                 action_to_perform = "place_tile_click"

            # --- Perform the determined action ---
            print(f"Debug: MouseRelease - Action: {action_to_perform}, is_drag: {is_drag}, mods: {current_modifiers}")
            redraw_needed = False

            if action_to_perform == "wall_perimeter":
                 print(f"Executing wall_perimeter at {start_x},{start_y}")
                 ctrl_shift_click_wall(self.tilemap, start_x, start_y, self.selected_tile)
                 redraw_needed = True
            elif action_to_perform == "fill_perimeter":
                 print(f"Executing fill_perimeter at {start_x},{start_y}")
                 ctrl_click_fill(self.tilemap, start_x, start_y, self.selected_tile)
                 redraw_needed = True
            elif action_to_perform == "flood_fill":
                 # Flood fill might be triggered on press, check if it needs triggering on release
                 # If triggered on Press, action_to_perform would likely be None here unless drag happened
                 # Let's assume for now it's on Press, so this block might not be hit for typical flood fill.
                 # If you want it on release, remove the trigger from mousePressEvent.
                 print(f"Executing flood_fill at {start_x},{start_y} (on release)")
                 filled = flood_fill_replace(self.tilemap, start_x, start_y, self.selected_tile)
                 if filled > 0: redraw_needed = True
            elif action_to_perform == "draw_rect":
                 print(f"Executing draw_rect from {start_x},{start_y} to {end_x},{end_y}")
                 fill_rectangle(self.tilemap, self.start_drag_pos, end_pos, self.selected_tile)
                 redraw_needed = True
            elif action_to_perform == "draw_line":
                 print(f"Executing draw_line from {start_x},{start_y} to {end_x},{end_y}")
                 draw_line(self.tilemap, self.start_drag_pos, end_pos, self.selected_tile)
                 redraw_needed = True
            elif action_to_perform == "place_tile_click":
                 print(f"Executing place_tile_click at {end_x},{end_y}")
                 self.tilemap.set_tile(end_x, end_y, self.selected_tile)
                 redraw_needed = True
            # Add elif for other actions as needed

            # --- Cleanup ---
            self.start_drag_pos = None
            self.drawing_line = False
            self.drawing_rect = False
            if redraw_needed:
                self.update() # Redraw the final result if an action occurred

    def _find_control_by_action(self, action_name_to_find):
        """Helper to find a control definition by its name."""
        return AppConfig.get("controls", {}).get(action_name_to_find)


    def wheelEvent(self, event: QWheelEvent):
        """Handles mouse wheel events for zooming and panning based on config."""
        # Update mouse position from wheel event
        self.current_mouse_pos = event.position().toPoint()

        current_modifiers = event.modifiers()
        controls = AppConfig.get("controls", {})
        angle = event.angleDelta().y() # Vertical scroll amount

        action_to_perform = None
        trigger_type = "ScrollUp" if angle > 0 else "ScrollDown"

        for action_name, control in controls.items():
            if control.get("trigger") == trigger_type:
                required_modifier = parse_modifier(control.get("modifier", "None"))
                if current_modifiers == required_modifier:
                    action_to_perform = action_name
                    break

        print(f"Debug: WheelEvent - Action: {action_to_perform}, mods: {current_modifiers}, angle: {angle}")

        # --- Use stored reference to scroll area ---
        if not self.scroll_area or not isinstance(self.scroll_area, QScrollArea):
             print("Error: Scroll area reference missing or invalid.")
             event.ignore()
             return

        if action_to_perform in ["zoom_in", "zoom_out"]:
            zoom_factor = AppConfig.get("zoom_step", 1.2)
            min_size = AppConfig.get("min_tile_size", 4)
            max_size = AppConfig.get("max_tile_size", 64)
            old_tile_size = AppConfig.get("tile_size", 16)

            # --- Zoom Calculation ---
            if action_to_perform == "zoom_in":
                new_tile_size = math.ceil(old_tile_size * zoom_factor)
            else: # zoom_out
                new_tile_size = math.floor(old_tile_size / zoom_factor)

            new_tile_size = max(min_size, min(new_tile_size, max_size))

            if new_tile_size != old_tile_size:
                # --- Calculate zoom centering ---
                # Mouse position relative to the widget's top-left corner
                widget_point = self.current_mouse_pos

                # Calculate relative position before zoom
                rel_x = widget_point.x() / self.width() if self.width() > 0 else 0
                rel_y = widget_point.y() / self.height() if self.height() > 0 else 0

                # Update tile size in config and trigger widget resize
                AppConfig["tile_size"] = new_tile_size
                print(f"Zooming: New tile size = {new_tile_size}")
                self.update_widget_size() # This resizes the widget via setMinimumSize/updateGeometry

                # --- Defer scrollbar adjustment ---
                # Allows the scroll area to update its ranges based on the new widget size
                def adjust_scrollbars():
                    new_widget_width = self.width()
                    new_widget_height = self.height()

                    # Target pixel on the *newly sized* widget corresponding to the relative mouse pos
                    target_widget_x = rel_x * new_widget_width
                    target_widget_y = rel_y * new_widget_height

                    # Mouse position relative to the scroll area's viewport
                    # Use globalPosition and mapFromGlobal for better accuracy across widgets
                    global_mouse_pos = event.globalPosition()
                    viewport_mouse_pos = self.scroll_area.viewport().mapFromGlobal(global_mouse_pos)

                    # Calculate new scrollbar values
                    new_scroll_x = int(target_widget_x - viewport_mouse_pos.x())
                    new_scroll_y = int(target_widget_y - viewport_mouse_pos.y())

                    # Clamp scroll values to valid range
                    h_bar = self.scroll_area.horizontalScrollBar()
                    v_bar = self.scroll_area.verticalScrollBar()
                    new_scroll_x = max(h_bar.minimum(), min(new_scroll_x, h_bar.maximum()))
                    new_scroll_y = max(v_bar.minimum(), min(new_scroll_y, v_bar.maximum()))

                    h_bar.setValue(new_scroll_x)
                    v_bar.setValue(new_scroll_y)
                    print(f"Debug: Adjusted scrollbars to X={new_scroll_x}, Y={new_scroll_y}")

                QTimer.singleShot(0, adjust_scrollbars)
                # --- End defer ---

                self.zoom_changed.emit() # Notify palette

            event.accept() # Consume the event

        elif action_to_perform in ["pan_left", "pan_right"]:
            h_scrollbar = self.scroll_area.horizontalScrollBar()
            pan_amount = AppConfig.get("pan_step", 50)
            current_val = h_scrollbar.value()
            if action_to_perform == "pan_left":
                 h_scrollbar.setValue(current_val - pan_amount)
            else: # pan_right
                 h_scrollbar.setValue(current_val + pan_amount)
            print(f"Debug: Panning HScroll from {current_val} to {h_scrollbar.value()}")
            event.accept() # Consume the event
        else:
            # If no specific action matched, let the QScrollArea handle default scrolling
            print("Debug: Ignoring wheel event for default scroll.")
            event.ignore()


    def keyPressEvent(self, event):
        """Handles key presses for changing tiles and other actions based on config."""
        current_modifiers = event.modifiers()
        current_key = event.key()
        controls = AppConfig.get("controls", {})
        available_tiles = sorted(AppConfig.get("tiles", {}).keys())

        action_found = False
        for action_name, control in controls.items():
            if control.get("trigger") == "KeyPress":
                required_modifier = parse_modifier(control.get("modifier", "None"))
                required_key_str = control.get("key")
                required_key_enum = parse_key(required_key_str)

                if required_key_enum is not None and current_key == required_key_enum and current_modifiers == required_modifier:
                    print(f"Debug: Matched KeyPress action '{action_name}'")
                    # Handle tile selection actions
                    if action_name.startswith("select_tile_"):
                         try:
                             tile_index_str = action_name.split("_")[-1]
                             tile_index = int(tile_index_str) - 1 # 1-based index in config name
                             if 0 <= tile_index < len(available_tiles):
                                 self.selected_tile = available_tiles[tile_index]
                                 # Notify palette to update visually
                                 main_window = self.window() # Get top-level window
                                 if isinstance(main_window, MainWindow):
                                     main_window.update_palette_selection()
                                 action_found = True
                         except (ValueError, IndexError):
                             print(f"Warning: Invalid tile index in control name '{action_name}'")
                    # Note: show_help is handled by MainWindow's QAction shortcut now

                    if action_found:
                        event.accept() # Consume event if handled
                        return # Stop processing keys

        if not action_found:
            print(f"Debug: Ignoring KeyPress event: Key={current_key}, Modifiers={current_modifiers}")
            event.ignore() # Let parent handle if not consumed


# --- Tile Palette Widget ---
class TilePaletteWidget(QWidget):
    """Widget to display tile selection buttons, supporting add/edit."""
    palette_updated = Signal()

    def __init__(self, editor_widget, parent=None):
        super().__init__(parent)
        self.editor_widget = editor_widget
        self.button_group_layout = QHBoxLayout()
        self.buttons = {}
        self.init_ui()
        self.rebuild_palette()
        # Connect editor's zoom signal to rebuild palette with new icon sizes
        self.editor_widget.zoom_changed.connect(self.rebuild_palette)


    def init_ui(self):
        """Sets up the static parts of the palette UI."""
        main_layout = QHBoxLayout()
        main_layout.addWidget(QLabel("Tiles:"))
        main_layout.addLayout(self.button_group_layout)

        self.add_button = QPushButton("+ Add")
        self.add_button.setToolTip("Add a new tile type")
        self.add_button.clicked.connect(self.add_new_tile)

        main_layout.addWidget(self.add_button)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def _create_tile_icon(self, tile_char, tile_data, icon_size):
        """Creates a QPixmap icon with color background and text character."""
        # Ensure icon_size is valid QSize
        if not isinstance(icon_size, QSize) or not icon_size.isValid() or icon_size.width() <= 0 or icon_size.height() <= 0:
             print(f"Warning: Invalid icon_size provided for tile '{tile_char}'. Using default 16x16.")
             icon_size = QSize(16, 16) # Fallback size

        pixmap = QPixmap(icon_size)
        # Use magenta as fallback color directly here if color_qt is missing
        color = tile_data.get("color_qt", QColor(255, 0, 255))
        pixmap.fill(color)

        painter = QPainter(pixmap)
        luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())
        text_color = Qt.GlobalColor.black if luminance > 128 else Qt.GlobalColor.white
        painter.setPen(text_color)

        # Adjust font size based on icon size (heuristic)
        font_size = max(6, int(icon_size.height() * 0.6))
        font = QFont()
        font.setPointSize(font_size)
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, tile_char)
        painter.end()
        return QIcon(pixmap)


    def rebuild_palette(self):
        """Clears and rebuilds the tile buttons based on AppConfig."""
        print("Debug: Rebuilding palette...") # Add debug print
        for i in reversed(range(self.button_group_layout.count())):
            widget = self.button_group_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.buttons.clear()

        # Use current tile size from config for icons
        tile_size = AppConfig.get("tile_size", 16)
        icon_qsize = QSize(tile_size, tile_size)
        # Ensure minimum button size for usability, but scale with icon
        button_width = max(32, tile_size + 10)
        button_height = max(32, tile_size + 10)
        button_qsize = QSize(button_width, button_height)

        tile_chars = sorted(AppConfig.get("tiles", {}).keys())

        for tile_char in tile_chars:
            tile_data = AppConfig["tiles"].get(tile_char, {
                "color_qt": QColor(255, 0, 255),
                "description": f"Undefined '{tile_char}'"
            })

            btn = QPushButton()
            btn.setFixedSize(button_qsize)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)

            icon = self._create_tile_icon(tile_char, tile_data, icon_qsize)
            btn.setIcon(icon)
            btn.setIconSize(icon_qsize) # Set icon size hint
            btn.setToolTip(f"{tile_char}: {tile_data.get('description', 'No description')}")

            btn.clicked.connect(lambda checked, tc=tile_char: self.select_tile(tc))
            btn.setProperty("tile_char", tile_char)
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(self.show_context_menu)

            self.button_group_layout.addWidget(btn)
            self.buttons[tile_char] = btn

        self.update_selection_visuals()
        self.palette_updated.emit() # Signal that palette structure/icons might have changed


    def select_tile(self, tile_char):
        """Slot to handle button clicks and update the editor's selected tile."""
        self.editor_widget.selected_tile = tile_char

    def update_selection_visuals(self):
        """Updates which button in the palette is visually checked."""
        current_tile = self.editor_widget.selected_tile
        button_to_check = self.buttons.get(current_tile)
        if button_to_check:
             if not button_to_check.isChecked():
                 button_to_check.setChecked(True)
        else:
             # If current selection somehow invalid, select default
             default_tile = AppConfig.get("default_tile", ".")
             if default_tile in self.buttons:
                  if not self.buttons[default_tile].isChecked():
                      self.buttons[default_tile].setChecked(True)
                  # Force editor selection to match if it was invalid
                  if self.editor_widget.selected_tile != default_tile:
                       self.editor_widget.selected_tile = default_tile


    def show_context_menu(self, pos):
        """Shows the context menu (Edit/Delete) for a tile button."""
        button = self.sender()
        if not isinstance(button, QPushButton): return
        tile_char = button.property("tile_char")
        if not tile_char: return

        menu = QMenu(self)
        edit_action = menu.addAction(f"Edit Tile '{tile_char}'...")

        # Allow deletion, but add a stronger warning if it's the default
        delete_action = menu.addAction(f"Delete Tile '{tile_char}'")
        if tile_char == AppConfig.get("default_tile"):
            delete_action.setText(f"Delete Default Tile '{tile_char}'...") # Indicate special case

        action = menu.exec(button.mapToGlobal(pos))

        if action == edit_action:
            self.edit_tile(tile_char)
        elif action == delete_action:
            self.delete_tile(tile_char) # Pass to delete handler

    def edit_tile(self, tile_char):
        """Opens the EditTileDialog to modify an existing tile."""
        if tile_char not in AppConfig.get("tiles", {}): return

        tile_data = AppConfig["tiles"][tile_char]
        existing_chars = set(AppConfig.get("tiles", {}).keys())

        dialog = EditTileDialog(tile_char, tile_data, existing_chars, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_tile_data()
            new_char = result["char"]
            new_data = result["data"]
            original_selected = self.editor_widget.selected_tile
            is_default_tile_being_edited = (tile_char == AppConfig.get("default_tile"))

            # If changing the character of the default tile, update config
            if is_default_tile_being_edited and new_char != tile_char:
                 print(f"Warning: Default tile character changed from '{tile_char}' to '{new_char}'. Updating config.")
                 AppConfig["default_tile"] = new_char # Update default tile in config

            # Update AppConfig tile dictionary
            if new_char != tile_char:
                # Need to handle potential key collision if renaming to an existing char
                if new_char in AppConfig["tiles"]:
                     QMessageBox.warning(self, "Rename Error", f"Cannot rename tile '{tile_char}' to '{new_char}' as character already exists.")
                     return # Abort rename

                # Proceed with rename: update data under new key, remove old key
                AppConfig["tiles"][new_char] = AppConfig["tiles"][tile_char] # Copy data first
                AppConfig["tiles"][new_char].update(new_data) # Apply changes
                del AppConfig["tiles"][tile_char] # Remove old entry

                if original_selected == tile_char:
                    self.editor_widget.selected_tile = new_char # Update selection if it was the edited tile
            else:
                AppConfig["tiles"][tile_char].update(new_data) # Just update data if char is same

            # Save config and update UI
            if save_config(CONFIG_FILE, AppConfig):
                 self.rebuild_palette()
                 self.update_selection_visuals()
                 self.editor_widget.update() # Redraw editor
                 QApplication.instance().statusBar().showMessage("Tile updated and configuration saved.", 3000)
            else:
                 QMessageBox.critical(self, "Error", f"Failed to save configuration file: {CONFIG_FILE}")
                 load_config(CONFIG_FILE) # Reload to revert
                 self.rebuild_palette()


    def delete_tile(self, tile_char):
        """Deletes a tile type after confirmation."""
        if tile_char not in AppConfig.get("tiles", {}): return

        is_default_tile = (tile_char == AppConfig.get("default_tile"))
        warning_msg = ""
        if is_default_tile:
             # Prevent deletion if it's the only tile left
             if len(AppConfig.get("tiles", {})) <= 1:
                  QMessageBox.critical(self, "Cannot Delete", "Cannot delete the last remaining tile, especially if it's the default.")
                  return
             warning_msg = "\n\nWARNING: This is the current default tile! Deleting it will force the application to choose a new default from the remaining tiles."

        reply = QMessageBox.question(self, "Confirm Delete",
                                     f"Are you sure you want to delete the tile '{tile_char}'?"
                                     f"{warning_msg}\n\n"
                                     f"(This cannot be undone easily and might affect saved maps that use this tile)",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            original_selected = self.editor_widget.selected_tile
            del AppConfig["tiles"][tile_char]

            # If we deleted the default, find a new default
            if is_default_tile:
                 remaining_tiles = list(AppConfig.get("tiles", {}).keys())
                 # Should always have remaining tiles based on check above
                 new_default = remaining_tiles[0] if remaining_tiles else "." # Absolute fallback
                 AppConfig["default_tile"] = new_default
                 print(f"Warning: Deleted default tile '{tile_char}'. Setting new default to '{new_default}'.")


            if save_config(CONFIG_FILE, AppConfig):
                 # If the deleted tile was selected, select the (potentially new) default tile
                 current_default_tile = AppConfig.get("default_tile", ".")
                 if original_selected == tile_char or is_default_tile:
                      self.editor_widget.selected_tile = current_default_tile
                 self.rebuild_palette()
                 self.editor_widget.update()
                 QApplication.instance().statusBar().showMessage(f"Tile '{tile_char}' deleted and configuration saved.", 3000)
            else:
                 QMessageBox.critical(self, "Error", f"Failed to save configuration file: {CONFIG_FILE}")
                 load_config(CONFIG_FILE) # Reload to revert
                 self.rebuild_palette()


    def add_new_tile(self):
        """Opens the EditTileDialog to add a new tile."""
        existing_chars = set(AppConfig.get("tiles", {}).keys())
        dialog = EditTileDialog(existing_chars=existing_chars, parent=self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_tile_data()
            new_char = result["char"]
            new_data = result["data"]
            AppConfig["tiles"][new_char] = new_data

            if save_config(CONFIG_FILE, AppConfig):
                 self.rebuild_palette()
                 QApplication.instance().statusBar().showMessage("New tile added and configuration saved.", 3000)
            else:
                 QMessageBox.critical(self, "Error", f"Failed to save configuration file: {CONFIG_FILE}")
                 load_config(CONFIG_FILE)
                 self.rebuild_palette()


# --- Map Selection Dialog ---
# (Content unchanged from previous version)
class MapSelectionDialog(QDialog):
    def __init__(self, map_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Maps to Load")
        self.map_keys = map_keys
        self.selected_keys = []

        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection) # Allow multi-select
        self.list_widget.addItems(self.map_keys)
        layout.addWidget(QLabel("Available Maps:"))
        layout.addWidget(self.list_widget)

        self.selection_input = QLineEdit()
        self.selection_input.setPlaceholderText("Or enter selection (e.g., 1, 3-5, lobby)")
        layout.addWidget(QLabel("Selection String:"))
        layout.addWidget(self.selection_input)

        # Tiling Options
        tiling_layout = QHBoxLayout()
        tiling_layout.addWidget(QLabel("Tiling:"))
        self.tiling_combo = QComboBox()
        self.tiling_combo.addItems(["Single", "Vertical", "Horizontal", "Grid"])
        self.tiling_combo.currentTextChanged.connect(self.update_grid_options)
        tiling_layout.addWidget(self.tiling_combo)

        self.rows_label = QLabel("Rows:")
        self.rows_spin = QSpinBox()
        self.rows_spin.setMinimum(1)
        self.rows_spin.setValue(2)
        self.cols_label = QLabel("Cols:")
        self.cols_spin = QSpinBox()
        self.cols_spin.setMinimum(1)
        self.cols_spin.setValue(2)

        tiling_layout.addWidget(self.rows_label)
        tiling_layout.addWidget(self.rows_spin)
        tiling_layout.addWidget(self.cols_label)
        tiling_layout.addWidget(self.cols_spin)
        tiling_layout.addStretch()
        layout.addLayout(tiling_layout)
        self.update_grid_options("Single") # Initial state

        # Dialog Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.process_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setMinimumWidth(400)

    def update_grid_options(self, tiling_mode):
        is_grid = (tiling_mode == "Grid")
        self.rows_label.setVisible(is_grid)
        self.rows_spin.setVisible(is_grid)
        self.cols_label.setVisible(is_grid)
        self.cols_spin.setVisible(is_grid)

    def parse_selection_string(self, selection_str):
        """Parses a selection string like '1, 3-5, lobby'."""
        selected = set()
        parts = selection_str.split(',')
        all_keys_set = set(self.map_keys) # Use the actual keys passed to the dialog

        for part in parts:
            part = part.strip()
            if not part: continue

            # Check for range (numeric only for now)
            range_match = re.match(r'^(\d+)\s*-\s*(\d+)$', part)
            if range_match:
                try:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    if start <= end:
                         # Find numeric keys within range (convert keys to int for comparison)
                         for key in self.map_keys:
                             try:
                                 key_int = int(key)
                                 if start <= key_int <= end:
                                     selected.add(key)
                             except ValueError:
                                 continue # Ignore non-numeric keys for numeric range
                    else:
                         print(f"Warning: Invalid range '{part}' (start > end).")
                except ValueError:
                    print(f"Warning: Could not parse range '{part}' as numbers.")
            # Check if it's a direct key
            elif part in all_keys_set:
                selected.add(part)
            else:
                print(f"Warning: Unknown map key or invalid format '{part}'.")

        # Return sorted list, maintaining original order if possible
        return sorted(list(selected), key=lambda k: self.map_keys.index(k) if k in self.map_keys else float('inf'))


    def process_selection(self):
        """Processes list selection and input string."""
        input_str = self.selection_input.text().strip()
        list_selected_items = self.list_widget.selectedItems()

        self.selected_keys = [] # Reset selection

        if input_str:
            # Prioritize input string if provided
            if input_str.lower() == "all":
                self.selected_keys = self.map_keys
            else:
                self.selected_keys = self.parse_selection_string(input_str)
        elif list_selected_items:
            # Use list selection if input string is empty
            self.selected_keys = [item.text() for item in list_selected_items]
        else:
            # No selection made
             QMessageBox.warning(self, "No Selection", "Please select maps from the list or enter a selection string.")
             return # Keep dialog open

        if not self.selected_keys:
             QMessageBox.warning(self, "Invalid Selection", "No valid maps found for the given selection.")
             return

        # Validate tiling mode for selection
        self.tiling_mode = self.tiling_combo.currentText()
        if len(self.selected_keys) <= 1 and self.tiling_mode != "Single":
            print("Info: Tiling mode reset to 'Single' for single map selection.")
            self.tiling_mode = "Single"
            self.tiling_combo.setCurrentText("Single") # Update combo box display
        elif len(self.selected_keys) > 1 and self.tiling_mode == "Single":
            QMessageBox.warning(self, "Invalid Tiling", "Please choose a tiling mode (Vertical, Horizontal, Grid) when selecting multiple maps.")
            return

        self.grid_rows = self.rows_spin.value() if self.tiling_mode == "Grid" else 1
        self.grid_cols = self.cols_spin.value() if self.tiling_mode == "Grid" else 1

        if self.tiling_mode == "Grid" and self.grid_rows * self.grid_cols < len(self.selected_keys):
             QMessageBox.warning(self, "Grid Too Small", f"Grid size ({self.grid_rows}x{self.grid_cols}) is too small for {len(self.selected_keys)} selected maps.")
             return


        self.accept() # Close dialog if selection is valid

    def get_selection(self):
        """Returns the selected map keys and tiling options."""
        return self.selected_keys, self.tiling_mode, self.grid_rows, self.grid_cols


# --- Main Application Window ---
class MainWindow(QMainWindow):
    """The main application window."""
    def __init__(self):
        super().__init__()

        load_config(CONFIG_FILE) # Load config before creating widgets that use it
        self.setWindowTitle(AppConfig.get("window_title", "Tile Editor"))

        # Create the single TileMap instance that will be shared
        self.tilemap = TileMap(
            AppConfig.get("grid_width", 40),
            AppConfig.get("grid_height", 40),
            AppConfig.get("default_tile", ".")
        )

        # --- Setup Scroll Area FIRST ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.ColorRole.Dark) # Match background
        self.scroll_area.setWidgetResizable(False) # Widget manages its own size via sizeHint/minimumSize

        # Create editor and pass the scroll area reference
        self.editor_widget = TileEditorWidget(self.tilemap, self.scroll_area, self) # Pass scroll_area
        self.palette_widget = TilePaletteWidget(self.editor_widget, self)

        # --- Set editor widget AFTER it's created ---
        self.scroll_area.setWidget(self.editor_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.palette_widget)
        main_layout.addWidget(self.scroll_area, 1) # Scroll area takes expanding space

        # Central widget setup
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.create_actions()
        self.create_menu_bar()
        self.statusBar().showMessage("Ready. Config loaded.", 2000)

        self.resize(800, 650)

        # Connect signals
        self.editor_widget.map_changed.connect(self.update_window_title)
        # Palette update signal triggers editor resize/repaint if tile_size changed
        self.palette_widget.palette_updated.connect(self.editor_widget.update_widget_size)
        # self.palette_widget.palette_updated.connect(self.editor_widget.update) # update_widget_size calls update

        self.editor_widget.setFocus() # Set focus to editor for key/wheel events
        self.update_window_title()

    def update_window_title(self):
        """Updates the window title with map dimensions."""
        base_title = AppConfig.get("window_title", "Tile Editor")
        # Access dimensions directly from the shared self.tilemap instance
        self.setWindowTitle(f"{base_title} - {self.tilemap.width}x{self.tilemap.height}")

    def create_actions(self):
        """Creates QAction objects for menu items."""
        self.save_action = QAction("&Save Map...", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_map)

        self.load_action = QAction("&Load Map...", self)
        self.load_action.setShortcut(QKeySequence.StandardKey.Open)
        self.load_action.triggered.connect(self.load_map)

        self.load_text_action = QAction("Load &Text Layout...", self)
        self.load_text_action.triggered.connect(self.load_text_layout)

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)

        # --- Help Action ---
        help_control = self._find_control_by_action("show_help")
        help_shortcut_str = "F1" # Default shortcut
        if help_control and help_control.get("key"):
             help_shortcut_str = help_control.get("key")
        self.help_action = QAction("&Help...", self)
        # Use QKeySequence.fromString for more flexibility if needed later
        self.help_action.setShortcut(QKeySequence(help_shortcut_str))
        self.help_action.triggered.connect(self.show_help_dialog)
        # --- End Help Action ---

    def create_menu_bar(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        # File Menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.load_text_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.help_action)


    def _find_control_by_action(self, action_name_to_find):
        """Helper to find a control definition by its name."""
        return AppConfig.get("controls", {}).get(action_name_to_find)

    def _format_control_string(self, control_data):
        """Formats a control definition into a user-readable string."""
        mod = control_data.get("modifier", "None")
        trig = control_data.get("trigger", "Unknown")
        key = control_data.get("key")
        desc = control_data.get("description", "No description")

        parts = []
        if mod and mod.lower() != "none":
            parts.append(mod)

        if trig == "KeyPress" and key:
            parts.append(f"'{key}' Key")
        elif trig == "LeftClick":
            parts.append("Left Click")
        elif trig == "RightClick":
            parts.append("Right Click")
        elif trig == "LeftDrag":
            parts.append("Left Drag")
        elif trig == "ScrollUp":
            parts.append("Scroll Up")
        elif trig == "ScrollDown":
            parts.append("Scroll Down")
        else:
             parts.append(trig) # Fallback for unknown triggers

        return f"{' + '.join(parts)}: {desc}"


    def show_help_dialog(self):
        """Displays a message box with configured controls."""
        help_text = "<h2>Controls Help</h2>" # Use HTML for better formatting
        controls = AppConfig.get("controls", {})
        # Group controls for better readability
        mouse_controls = []
        key_controls = []
        scroll_controls = []

        # Sort controls by description for consistent order
        sorted_controls = sorted(controls.items(), key=lambda item: item[1].get("description", ""))

        for name, data in sorted_controls:
            formatted = self._format_control_string(data)
            trigger = data.get("trigger", "").lower()
            if "click" in trigger or "drag" in trigger:
                 mouse_controls.append(formatted)
            elif "key" in trigger:
                 key_controls.append(formatted)
            elif "scroll" in trigger:
                 scroll_controls.append(formatted)

        if mouse_controls:
            help_text += "<b>Mouse:</b><ul>" + "".join(f"<li>{c}</li>" for c in mouse_controls) + "</ul>"
        if scroll_controls:
            help_text += "<b>Scroll Wheel:</b><ul>" + "".join(f"<li>{c}</li>" for c in scroll_controls) + "</ul>"
        if key_controls:
            help_text += "<b>Keyboard:</b><ul>" + "".join(f"<li>{c}</li>" for c in key_controls) + "</ul>"

        # Use QMessageBox with RichText format
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Controls Help")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(help_text)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()


    def save_map(self):
        """Opens a file dialog to save the current map to a JSON file (potentially multi-map)."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Map File", "", "TileMap JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        if "JSON Files" in selected_filter and not file_path.lower().endswith(".json"):
            file_path += ".json"

        # Prompt for map key/name
        map_key, ok = QInputDialog.getText(self, "Enter Map Key", "Enter a name/key for this map (e.g., level_1, main_hall):")
        if not ok or not map_key.strip():
            QMessageBox.warning(self, "Save Cancelled", "Map key cannot be empty.")
            return
        map_key = map_key.strip() # Use entered key

        # --- Read existing file (if any) to preserve other maps ---
        file_data = {"format_version": CURRENT_FORMAT_VERSION, "maps": {}}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                    # Check if it's the new format, otherwise start fresh
                    if isinstance(existing_data, dict) and "maps" in existing_data and isinstance(existing_data["maps"], dict):
                         file_data = existing_data
                         # Ensure format version is present/updated
                         file_data["format_version"] = CURRENT_FORMAT_VERSION
                    else:
                         print(f"Warning: Existing file '{file_path}' is not in multi-map format or is invalid. Overwriting with new format.")
            except (json.JSONDecodeError, IOError, TypeError) as e:
                print(f"Warning: Could not read or parse existing file '{file_path}': {e}. Will create a new file.")

        # Add/overwrite the current map data under the specified key
        file_data["maps"][map_key] = self.tilemap.export_map_data()

        # --- Save the combined data ---
        try:
            with open(file_path, 'w') as f:
                json.dump(file_data, f, indent=4) # Use indent=4 for readability
            print(f"Map '{map_key}' saved successfully to {file_path}")
            self.statusBar().showMessage(f"Map '{map_key}' saved to {file_path}", 3000)
        except Exception as e:
            print(f"Error saving map: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save map to file:\n{e}")
            self.statusBar().showMessage(f"Error saving map: {e}", 5000)


    def load_map(self):
        """Handles loading maps from JSON (old or new format) with selection and tiling."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Map(s)", "", "TileMap JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                file_content = json.load(f)

            maps_to_load_data = {} # Store {key: map_data_dict}
            tiling_mode = "Single"
            grid_rows = 1
            grid_cols = 1

            # --- Detect format and select maps ---
            is_new_format = isinstance(file_content, dict) and "maps" in file_content and isinstance(file_content["maps"], dict)

            if is_new_format:
                available_maps_data = file_content.get("maps", {})
                if not available_maps_data:
                     QMessageBox.information(self, "Empty File", "The selected multi-map file contains no maps.")
                     return

                map_keys = list(available_maps_data.keys())
                dialog = MapSelectionDialog(map_keys, self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    selected_keys, tiling_mode, grid_rows, grid_cols = dialog.get_selection()
                    if not selected_keys: return # Should be caught by dialog

                    for key in selected_keys:
                        if key in available_maps_data:
                            maps_to_load_data[key] = available_maps_data[key]
                        else:
                            print(f"Warning: Selected map key '{key}' not found in file.")
                else:
                    return # User cancelled selection

            else: # Assume old format
                print("Loading map in old (single map) format.")
                # Validate old format structure before proceeding
                if isinstance(file_content, dict) and "rows" in file_content:
                     maps_to_load_data["map_1"] = file_content # Load under a default key
                     tiling_mode = "Single"
                else:
                     QMessageBox.critical(self, "Load Error", "Invalid old map format detected.")
                     return


            if not maps_to_load_data:
                QMessageBox.warning(self, "Load Error", "No valid map data could be selected or loaded.")
                return

            # --- Validate selected maps and check for undefined tiles ---
            all_map_chars = set()
            temp_maps = {} # Store temporary TileMap objects for validation
            valid_map_keys_in_order = [] # Keep track of valid maps in selected order

            # Iterate in the order the user selected (or map keys were listed)
            keys_to_process = maps_to_load_data.keys()
            if is_new_format: # Use the potentially re-ordered list from selection dialog
                 keys_to_process = selected_keys

            for key in keys_to_process:
                if key not in maps_to_load_data: continue # Skip if key wasn't valid initially
                map_data_dict = maps_to_load_data[key]
                temp_map = TileMap(0, 0, '.') # Create dummy map
                if temp_map.load_from_data(map_data_dict): # Validate structure and content
                    all_map_chars.update(temp_map.get_unique_tiles())
                    temp_maps[key] = temp_map # Store valid temp map
                    valid_map_keys_in_order.append(key)
                else:
                    print(f"Warning: Skipping map '{key}' due to load error.")
                    QMessageBox.warning(self, "Map Load Warning", f"Could not load map data for '{key}'. It will be skipped.")

            if not temp_maps: # Check if any maps were successfully loaded
                 QMessageBox.critical(self, "Load Error", "None of the selected maps could be loaded successfully.")
                 return

            # Now check combined characters against config
            defined_chars = set(AppConfig.get("tiles", {}).keys())
            undefined_chars = all_map_chars - defined_chars
            config_modified_by_load = False
            newly_defined_chars = []

            if undefined_chars:
                print(f"Warning: Maps contain undefined tile characters: {undefined_chars}")
                fallback_color_rgb = [255, 0, 255] # Magenta
                fallback_color_qt = QColor(*fallback_color_rgb)
                for char in sorted(list(undefined_chars)): # Process alphabetically
                    print(f"Adding fallback definition for undefined tile '{char}'.")
                    AppConfig["tiles"][char] = {
                        "color": fallback_color_rgb, "color_qt": fallback_color_qt,
                        "description": f"Undefined '{char}' (from loaded map)"
                    }
                    newly_defined_chars.append(char)
                    config_modified_by_load = True

                if config_modified_by_load:
                    if save_config(CONFIG_FILE, AppConfig):
                        self.palette_widget.rebuild_palette()
                        undefined_list_str = ", ".join(f"'{c}'" for c in sorted(newly_defined_chars))
                        warning_message = (
                            f"The loaded map(s) contained undefined tile characters:\n\n{undefined_list_str}\n\n"
                            f"They have been added with a default appearance and the configuration file updated."
                        )
                        QMessageBox.warning(self, "Undefined Tiles Found", warning_message)
                    else:
                        QMessageBox.critical(self, "Config Error", "Failed to save updated configuration after adding undefined tiles.")

            # --- Combine maps based on tiling using only the valid maps in order ---
            valid_maps_to_combine = {key: temp_maps[key] for key in valid_map_keys_in_order}
            combined_map_data = self._combine_maps(valid_maps_to_combine, tiling_mode, grid_rows, grid_cols)

            # --- Load the final combined map into the editor ---
            self.editor_widget.resize_map(
                combined_map_data["width"],
                combined_map_data["height"],
                combined_map_data["default_tile"],
                new_tiles_data=combined_map_data["tiles"]
            )

            self.statusBar().showMessage(f"Map(s) loaded from {file_path}", 3000)
            print(f"Map(s) loaded successfully from {file_path}")

        except FileNotFoundError:
             self.statusBar().showMessage(f"Error: File not found at {file_path}", 5000)
        except json.JSONDecodeError:
             self.statusBar().showMessage("Error: Invalid JSON file.", 5000)
        except Exception as e:
             self.statusBar().showMessage(f"Error loading map: {e}", 5000)
             print(f"An unexpected error occurred during loading: {e}")


    def _combine_maps(self, loaded_maps_dict, tiling_mode, grid_rows=1, grid_cols=1):
        """Combines multiple TileMap objects into a single grid based on tiling."""
        map_keys = list(loaded_maps_dict.keys()) # These are already validated and ordered
        if not map_keys:
            return {"width": 0, "height": 0, "default_tile": AppConfig.get("default_tile", "."), "tiles": []}

        # Use global default tile for the combined map for consistency
        combined_default_tile = AppConfig.get("default_tile", ".")

        total_width = 0
        total_height = 0
        map_dims = {key: (m.width, m.height) for key, m in loaded_maps_dict.items()}

        if tiling_mode == "Single":
            # This case should ideally be handled before calling _combine_maps if only one map is valid
            key = map_keys[0]
            map_to_load = loaded_maps_dict[key]
            # Return data structure compatible with resize_map
            return {
                "width": map_to_load.width, "height": map_to_load.height,
                "default_tile": map_to_load.default_tile, # Use map's own default here
                "tiles": [row[:] for row in map_to_load.tiles] # Return copy
            }
        elif tiling_mode == "Vertical":
            total_width = max((w for w, h in map_dims.values()), default=0)
            total_height = sum(h for w, h in map_dims.values())
        elif tiling_mode == "Horizontal":
            total_width = sum(w for w, h in map_dims.values())
            total_height = max((h for w, h in map_dims.values()), default=0)
        elif tiling_mode == "Grid":
            # Ensure grid dimensions are sufficient
            num_maps = len(map_keys)
            actual_cols = max(1, grid_cols)
            actual_rows = max(1, math.ceil(num_maps / actual_cols))

            col_widths = [0] * actual_cols
            row_heights = [0] * actual_rows

            for i, key in enumerate(map_keys):
                 row = i // actual_cols
                 col = i % actual_cols
                 if row < actual_rows: # Should always be true with ceil calculation
                     w, h = map_dims[key]
                     col_widths[col] = max(col_widths[col], w)
                     row_heights[row] = max(row_heights[row], h)

            total_width = sum(col_widths)
            total_height = sum(row_heights)
            col_starts = [sum(col_widths[:i]) for i in range(actual_cols)]
            row_starts = [sum(row_heights[:i]) for i in range(actual_rows)]


        print(f"Combining maps ({tiling_mode}): Total Size {total_width}x{total_height}")
        # Ensure non-negative dimensions before creating grid
        total_width = max(0, total_width)
        total_height = max(0, total_height)
        combined_tiles = [[combined_default_tile for _ in range(total_width)] for _ in range(total_height)]

        current_x_offset = 0
        current_y_offset = 0

        for i, key in enumerate(map_keys):
            map_obj = loaded_maps_dict[key]
            w, h = map_dims[key]

            # Calculate placement offset based on mode
            place_x, place_y = 0, 0
            if tiling_mode == "Vertical":
                place_x = 0 # Align left
                place_y = current_y_offset
                current_y_offset += h # Move down for next map based on its height
            elif tiling_mode == "Horizontal":
                place_x = current_x_offset
                place_y = 0 # Align top
                current_x_offset += w # Move right for next map based on its width
            elif tiling_mode == "Grid":
                 actual_cols = max(1, grid_cols) # Use validated cols
                 row = i // actual_cols
                 col = i % actual_cols
                 if row < len(row_starts) and col < len(col_starts): # Check bounds
                     place_x = col_starts[col]
                     place_y = row_starts[row]
                 else:
                     print(f"Warning: Map '{key}' index {i} exceeds grid capacity. Skipping.")
                     continue # Skip map if grid calculation is wrong

            # Copy tiles
            for r in range(h):
                for c in range(w):
                    target_r, target_c = place_y + r, place_x + c
                    # Ensure we don't write out of bounds of the combined grid
                    if 0 <= target_r < total_height and 0 <= target_c < total_width:
                         # Use get_tile which handles map's internal bounds check
                         tile_to_copy = map_obj.get_tile(c, r)
                         if tile_to_copy is not None: # Should not be None if c,r are valid
                            combined_tiles[target_r][target_c] = tile_to_copy
                         else: # Fallback if get_tile returns None unexpectedly
                            combined_tiles[target_r][target_c] = combined_default_tile


        return {
            "width": total_width, "height": total_height,
            "default_tile": combined_default_tile, "tiles": combined_tiles
        }


    def load_text_layout(self):
        """Loads a text-based map layout and builds the map."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Text Layout", "", "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                lines = [line.rstrip('\n') for line in f if line.strip('\n\r')]

            if not lines:
                QMessageBox.warning(self, "Warning", "Text file appears empty.")
                return

            width = 0
            if lines:
                 width = max(len(line) for line in lines)
            height = len(lines)

            default_tile = AppConfig.get("default_tile", ".")
            normalized_rows = [line.ljust(width, default_tile) for line in lines]
            unique_tiles_in_text = set(''.join(normalized_rows))
            known_tiles = set(AppConfig.get("tiles", {}).keys())
            unknown_tiles = sorted(list(unique_tiles_in_text - known_tiles - {default_tile}))

            config_modified = False
            final_rows_to_load = [list(row) for row in normalized_rows] # Convert to list of lists early

            for tile_idx, tile in enumerate(unknown_tiles): # Use enumerate for index if needed later
                reply = QMessageBox.question(
                    self, "New Tile Found",
                    f"The layout contains an undefined tile character: '{tile}'\n\n"
                    f"Would you like to define it now (OK) or replace it with the default tile '{default_tile}' (Cancel)?",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Ok
                )

                if reply == QMessageBox.StandardButton.Ok:
                    desc, ok = QInputDialog.getText(self, "Tile Description", f"Enter description for tile '{tile}':")
                    if not ok: desc = ""
                    if not desc: desc = f"Tile '{tile}' (from text)"

                    color = QColorDialog.getColor(QColor(200, 200, 200), self, f"Choose color for tile '{tile}'")
                    if not color.isValid():
                        color = QColor(200, 200, 200)

                    print(f"Defining new tile '{tile}' from text layout.")
                    AppConfig["tiles"][tile] = {
                        "description": desc,
                        "color_qt": color,
                        "color": [color.red(), color.green(), color.blue()],
                    }
                    config_modified = True
                else:
                    print(f"Replacing undefined tile '{tile}' with default '{default_tile}'.")
                    # Replace directly in the list of lists
                    for r in range(height):
                        for c in range(width):
                            if final_rows_to_load[r][c] == tile:
                                final_rows_to_load[r][c] = default_tile

            if config_modified:
                if save_config(CONFIG_FILE, AppConfig):
                    self.palette_widget.rebuild_palette()
                    self.statusBar().showMessage("Configuration updated with new tiles.", 3000)
                else:
                     QMessageBox.critical(self, "Config Error",
                                          f"Failed to save updated configuration file '{CONFIG_FILE}' "
                                          f"after defining new tiles. New tiles may not persist.")

            # --- Load into the editor by resizing the *shared* tilemap ---
            self.editor_widget.resize_map(
                width,
                height,
                default_tile,
                new_tiles_data=final_rows_to_load # Pass the processed tile data
            )

            self.statusBar().showMessage(f"Text layout loaded from {file_path}", 3000)
            print(f"Text layout loaded successfully from {file_path}")

        except FileNotFoundError:
             self.statusBar().showMessage(f"Error: File not found at {file_path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Text Layout", f"An unexpected error occurred: {e}")
            print(f"Error loading text layout: {e}")


    def update_palette_selection(self):
        """Called by editor (e.g., on keypress) to update palette visuals."""
        self.palette_widget.update_selection_visuals()


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_config(CONFIG_FILE) # Load config before creating MainWindow
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

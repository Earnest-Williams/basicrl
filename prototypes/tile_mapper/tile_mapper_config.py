# tile_mapper_config.py

from PySide6.QtGui import QColor  # Needed for default color conversion
import copy  # For deep copying defaults
import io
import os

import structlog

log = structlog.get_logger(__name__)


# --- Orjson/JSON Handling ---
try:
    import orjson

    JSON_HANDLER = orjson
    JSON_LOADS_KWARGS = {}
    JSON_DUMPS_KWARGS = {
        "option": orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS
    }
    JSON_DecodeError = orjson.JSONDecodeError
    FILE_READ_MODE = "rb"
    FILE_WRITE_MODE = "wb"
    USE_ORJSON = True
    log.info("Using orjson for JSON operations")
except ImportError:
    import json

    JSON_HANDLER = json
    JSON_LOADS_KWARGS = {}
    JSON_DUMPS_KWARGS = {"indent": 4}
    JSON_DecodeError = json.JSONDecodeError
    FILE_READ_MODE = "r"
    FILE_WRITE_MODE = "w"
    USE_ORJSON = False
    log.warning("orjson not found. Falling back to standard json library (slower)")

# --- Configuration File Constants ---
CONFIG_FILE = "tile_editor_config.json"
CURRENT_FORMAT_VERSION = 2

# --- Default Configuration Structure ---
DEFAULT_CONFIG = {
    "format_version": CURRENT_FORMAT_VERSION,
    "grid_width": 40,
    "grid_height": 40,
    "tile_size": 16,
    "min_tile_size": 4,
    "max_tile_size": 64,
    "zoom_step": 1.2,
    "pan_step": 50,
    "default_tile": ".",
    "wall_tile": "#",
    "door_tile": "+",
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
        "place_tile_click": {
            "modifier": "None",
            "trigger": "LeftClick",
            "description": "Place selected tile (single click)",
        },
        "draw_line": {
            "modifier": "None",
            "trigger": "LeftDrag",
            "description": "Draw line of selected tile",
        },
        "erase_tile": {
            "modifier": "None",
            "trigger": "RightClick",
            "description": "Erase tile (set to default)",
        },
        "draw_rect": {
            "modifier": "Shift",
            "trigger": "LeftDrag",
            "description": "Draw rectangle of selected tile",
        },
        "fill_perimeter": {
            "modifier": "Ctrl",
            "trigger": "LeftClick",
            "description": "Fill perimeter of non-default area",
        },
        "wall_perimeter": {
            "modifier": "Ctrl+Shift",
            "trigger": "LeftClick",
            "description": "Wall perimeter of same-tile area (respects walls/doors)",
        },
        "flood_fill": {
            "modifier": "Alt",
            "trigger": "LeftClick",
            "description": "Flood fill area with selected tile",
        },
        "zoom_in": {
            "modifier": "Ctrl",
            "trigger": "ScrollUp",
            "description": "Zoom In",
        },
        "zoom_out": {
            "modifier": "Ctrl",
            "trigger": "ScrollDown",
            "description": "Zoom Out",
        },
        "pan_right": {
            "modifier": "Shift",
            "trigger": "ScrollUp",
            "description": "Pan Right",
        },
        "pan_left": {
            "modifier": "Shift",
            "trigger": "ScrollDown",
            "description": "Pan Left",
        },
        "show_help": {
            "modifier": "None",
            "trigger": "KeyPress",
            "key": "F1",
            "description": "Show this help message",
        },
        "select_tile_1": {
            "modifier": "None",
            "trigger": "KeyPress",
            "key": "1",
            "description": "Select Tile 1",
        },
        # ... (add other select_tile_N entries if desired) ...
        "select_tile_9": {
            "modifier": "None",
            "trigger": "KeyPress",
            "key": "9",
            "description": "Select Tile 9",
        },
    },
}


def _validate_and_populate_config(loaded_config_dict: dict) -> tuple[dict, bool]:
    """Validates loaded config, populates defaults, converts colors. Returns (validated_config, was_modified)."""
    validated_config = loaded_config_dict
    config_was_modified = False

    # Use deepcopy of defaults to avoid modifying the original DEFAULT_CONFIG
    defaults = copy.deepcopy(DEFAULT_CONFIG)

    # Check format version first
    if validated_config.get("format_version") != CURRENT_FORMAT_VERSION:
        log.warning("Config file format mismatch or missing. Applying defaults")
        base_config = defaults  # Start with defaults
        # Overwrite defaults with loaded values
        base_config.update(validated_config)
        validated_config = base_config
        config_was_modified = True

    # Ensure essential top-level keys exist
    for key, value in defaults.items():
        if key not in validated_config:
            log.warning("Missing top-level key, adding default", key=key)
            validated_config[key] = value  # Add default value (already copied)
            config_was_modified = True

    # Ensure controls exist and have all sub-keys
    default_controls = defaults.get("controls", {})
    loaded_controls = validated_config.get("controls", {})
    if not isinstance(loaded_controls, dict):
        log.warning("'controls' key is not a dictionary. Resetting to default")
        loaded_controls = default_controls  # Use copied defaults
        validated_config["controls"] = loaded_controls
        config_was_modified = True

    for control_key, default_control_data in default_controls.items():
        if control_key not in loaded_controls:
            log.warning("Missing control, adding default", control=control_key)
            # Use copied defaults
            loaded_controls[control_key] = default_control_data
            config_was_modified = True
        elif isinstance(loaded_controls[control_key], dict):
            loaded_control_data = loaded_controls[control_key]
            for sub_key, default_sub_value in default_control_data.items():
                if sub_key not in loaded_control_data:
                    log.warning(
                        "Missing sub-key in control, adding",
                        control=control_key,
                        sub_key=sub_key,
                    )
                    loaded_control_data[sub_key] = default_sub_value
                    config_was_modified = True
        else:
            log.warning("Control is not a dictionary. Resetting", control=control_key)
            # Use copied defaults
            loaded_controls[control_key] = default_control_data
            config_was_modified = True

    # Convert/Validate Tile Colors and add QColor objects
    loaded_tiles = validated_config.get("tiles", {})
    if not isinstance(loaded_tiles, dict):
        log.warning("'tiles' key is not a dictionary. Resetting")
        loaded_tiles = defaults["tiles"]  # Use copied defaults
        validated_config["tiles"] = loaded_tiles
        config_was_modified = True

    for tile_char, tile_data in loaded_tiles.items():
        if not isinstance(tile_data, dict):
            log.warning("Tile data is not dict. Using fallback", tile=tile_char)
            loaded_tiles[tile_char] = {
                "color": [255, 0, 255],
                "description": f"Invalid Tile '{tile_char}'",
            }
            tile_data = loaded_tiles[tile_char]
            config_was_modified = True

        # Ensure color_qt is created if missing or invalid
        create_color = False
        if "color_qt" not in tile_data or not isinstance(tile_data["color_qt"], QColor):
            create_color = True

        if create_color:
            if (
                isinstance(tile_data.get("color"), list)
                and len(tile_data["color"]) == 3
            ):
                rgb = tile_data["color"]
                try:
                    tile_data["color_qt"] = QColor(rgb[0], rgb[1], rgb[2])
                except (TypeError, ValueError):
                    log.warning(
                        "Invalid color values for tile. Using fallback",
                        tile=tile_char,
                    )
                    tile_data["color_qt"] = QColor(255, 0, 255)
                    if tile_data.get("color") != [255, 0, 255]:
                        config_was_modified = True
                    tile_data["color"] = [255, 0, 255]
            else:
                log.warning(
                    "Invalid or missing color for tile. Using fallback",
                    tile=tile_char,
                )
                tile_data["color_qt"] = QColor(255, 0, 255)
                if tile_data.get("color") != [255, 0, 255]:
                    config_was_modified = True
                tile_data["color"] = [255, 0, 255]

    # Convert/Validate Preview Color and add QColor
    preview_rgb = validated_config.get(
        "preview_line_color", defaults["preview_line_color"]
    )
    create_preview_color = False
    if "preview_line_color_qt" not in validated_config or not isinstance(
        validated_config["preview_line_color_qt"], QColor
    ):
        create_preview_color = True

    if create_preview_color:
        if isinstance(preview_rgb, list) and len(preview_rgb) == 3:
            try:
                validated_config["preview_line_color_qt"] = QColor(
                    preview_rgb[0], preview_rgb[1], preview_rgb[2]
                )
            except (TypeError, ValueError):
                log.warning("Invalid preview_line_color value. Using default")
                default_rgb = defaults["preview_line_color"]
                validated_config["preview_line_color_qt"] = QColor(*default_rgb)
            if validated_config.get("preview_line_color") != default_rgb:
                config_was_modified = True
            validated_config["preview_line_color"] = default_rgb
        else:
            log.warning("Invalid preview_line_color format. Using default")
            default_rgb = defaults["preview_line_color"]
            validated_config["preview_line_color_qt"] = QColor(*default_rgb)
            if validated_config.get("preview_line_color") != default_rgb:
                config_was_modified = True
            validated_config["preview_line_color"] = default_rgb

    # Validate numeric settings
    for key in [
        "preview_line_thickness",
        "tile_size",
        "min_tile_size",
        "max_tile_size",
        "pan_step",
    ]:
        val = validated_config.get(key, defaults[key])
        if not isinstance(val, int) or val < 1:
            log.warning("Invalid numeric config value, using default", key=key)
            validated_config[key] = defaults[key]
            config_was_modified = True
    zoom_val = validated_config.get("zoom_step", defaults["zoom_step"])
    if not isinstance(zoom_val, (float, int)) or zoom_val <= 1.0:
        log.warning("Invalid value for 'zoom_step'. Using default")
        validated_config["zoom_step"] = defaults["zoom_step"]
        config_was_modified = True

    # Validate min/max tile size relation and clamp current size
    if validated_config.get("min_tile_size", 1) >= validated_config.get(
        "max_tile_size", 64
    ):
        log.warning("min_tile_size >= max_tile_size. Resetting")
        validated_config["min_tile_size"] = defaults["min_tile_size"]
        validated_config["max_tile_size"] = defaults["max_tile_size"]
        config_was_modified = True
    validated_config["tile_size"] = max(
        validated_config["min_tile_size"],
        min(validated_config["tile_size"], validated_config["max_tile_size"]),
    )

    return validated_config, config_was_modified


def load_config(filepath: str = CONFIG_FILE) -> dict:
    """
    Loads configuration from JSON file.
    Returns a validated config dictionary (using defaults on failure/missing).
    """
    loaded_config_dict = None
    config_to_use = None
    config_was_modified = False

    try:
        if os.path.exists(filepath):
            with open(filepath, FILE_READ_MODE) as f:
                content = f.read()
                if not content:
                    log.warning(
                        "Config file is empty. Using defaults", filepath=filepath
                    )
                    # Proceed as if file doesn't exist
                elif USE_ORJSON:
                    loaded_config_dict = JSON_HANDLER.loads(
                        content, **JSON_LOADS_KWARGS
                    )
                else:
                    # Use StringIO for standard json from string
                    config_io = (
                        io.StringIO(content)
                        if isinstance(content, str)
                        else io.BytesIO(content)
                    )
                    loaded_config_dict = JSON_HANDLER.load(
                        config_io, **JSON_LOADS_KWARGS
                    )

                if isinstance(loaded_config_dict, dict):
                    config_to_use, config_was_modified = _validate_and_populate_config(
                        loaded_config_dict
                    )
                    log.info("Configuration loaded and validated", filepath=filepath)
                else:
                    log.warning(
                        "Invalid config format. Using defaults",
                        filepath=filepath,
                    )
                    # Fall through to default handling

        else:  # File does not exist
            log.warning(
                "Config file not found. Using defaults and creating file",
                filepath=filepath,
            )
            config_was_modified = True
            # Fall through to default handling

    except JSON_DecodeError as e:
        log.error(
            "Could not decode JSON. Using defaults",
            filepath=filepath,
            error=str(e),
        )
    except Exception as e:
        log.exception("Error loading configuration. Using defaults", filepath=filepath)

    # --- Default Handling ---
    # If config_to_use is still None, it means loading failed or file didn't exist
    if config_to_use is None:
        log.info("Using default configuration")
        # Deepcopy ensures defaults are fresh each time
        config_to_use = copy.deepcopy(DEFAULT_CONFIG)
        # Populate QColor objects for the defaults
        config_to_use, _ = _validate_and_populate_config(config_to_use)
        # Mark as modified if we had to generate defaults (might trigger save)
        config_was_modified = True  # True if file didn't exist or load failed

    # Save if needed (e.g., file missing, format updated, defaults added)
    if config_was_modified:
        log.info("Attempting to save updated/default configuration")
        if save_config(filepath, config_to_use):
            log.info("Configuration saved successfully")
        else:
            log.error("Failed to save configuration file")

    return config_to_use


def save_config(filepath: str, config_data: dict) -> bool:
    """
    Saves the configuration dictionary to a JSON file.
    Removes temporary QColor objects before saving.
    Returns True on success, False on failure.
    """
    try:
        # Prepare data for serialization (remove non-serializable QColor)
        save_data = {}
        keys_to_skip = ["tiles"]  # Handle separately

        for key, value in config_data.items():
            if key.endswith("_qt") or key in keys_to_skip:
                continue
            save_data[key] = value  # Copy serializable data

        # Process 'tiles' to store only RGB lists
        save_data["tiles"] = {}
        for char, tile_info in config_data.get("tiles", {}).items():
            color_list = [255, 0, 255]  # Default fallback
            # Try getting color from list first, then QColor
            if (
                isinstance(tile_info.get("color"), list)
                and len(tile_info["color"]) == 3
            ):
                color_list = tile_info["color"]
            elif isinstance(tile_info.get("color_qt"), QColor):
                cq = tile_info["color_qt"]
                color_list = [cq.red(), cq.green(), cq.blue()]

            save_data["tiles"][char] = {
                "color": color_list,
                "description": tile_info.get("description", ""),
            }

        # Ensure essential keys are present (use DEFAULT_CONFIG as ultimate fallback)
        essential_keys = [
            "format_version",
            "grid_width",
            "grid_height",
            "tile_size",
            "min_tile_size",
            "max_tile_size",
            "zoom_step",
            "pan_step",
            "default_tile",
            "wall_tile",
            "door_tile",
            "window_title",
            "preview_line_color",
            "preview_line_thickness",
            "controls",
        ]
        for key in essential_keys:
            if key not in save_data:
                # Get value from current config or default config
                save_data[key] = config_data.get(key, DEFAULT_CONFIG.get(key))
                # Special case: ensure preview color list is correct
                if key == "preview_line_color" and isinstance(
                    config_data.get("preview_line_color_qt"), QColor
                ):
                    pvc = config_data["preview_line_color_qt"]
                    save_data[key] = [pvc.red(), pvc.green(), pvc.blue()]

        # Ensure format version is current
        save_data["format_version"] = CURRENT_FORMAT_VERSION

        # Write using the selected handler and mode
        with open(filepath, FILE_WRITE_MODE) as f:
            if USE_ORJSON:
                f.write(JSON_HANDLER.dumps(save_data, **JSON_DUMPS_KWARGS))
            else:
                JSON_HANDLER.dump(save_data, f, **JSON_DUMPS_KWARGS)

        return True
    except Exception as e:
        log.exception("Error saving configuration", filepath=filepath)
        return False

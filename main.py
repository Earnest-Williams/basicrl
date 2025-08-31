# main.py
import logging
import sys
import time
import tomllib # Standard in Python 3.11+
from pathlib import Path
from typing import Any
from typing import Dict as PyDict # Use PyDict for Dict type alias

import numpy as np # Added for map printing
import structlog
import yaml
from PySide6.QtWidgets import QApplication
from structlog.stdlib import add_log_level, add_logger_name # Ensure these are imported

# Use absolute imports relative to project root (basicrl)
from engine.main_loop import MainLoop
from engine.window_manager import WindowManager
from game.game_state import GameState

# Import core world components; fail fast if dependencies are missing
try:
    from game.world.game_map import TILE_ID_FLOOR, TILE_ID_WALL, GameMap
except ImportError as e:
    structlog.get_logger().error(
        "CRITICAL: Failed to import GameMap and TILE IDs.", error=str(e)
    )
    raise

try:
    from game.world.procgen import generate_dungeon
except ImportError as e:
    structlog.get_logger().error(
        "CRITICAL: Failed to import generate_dungeon.", error=str(e)
    )
    raise


# --- Paths relative to this script's location ---
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = SCRIPT_DIR / "config"
FONTS_DIR = SCRIPT_DIR / "fonts"

CONFIG_FILE = CONFIG_DIR / "config.yaml"
ITEMS_CONFIG_FILE = CONFIG_DIR / "items.yaml"
EFFECTS_CONFIG_FILE = CONFIG_DIR / "effects.yaml"
KEYBINDINGS_FILE = CONFIG_DIR / "keybindings.toml"
SETTINGS_FILE = CONFIG_DIR / "settings.toml"
# --- End Paths ---


# --- Structlog Setup ---
def setup_logging():
    """Configures structlog for console output."""
    # Ensure standard logging is configured first if not done elsewhere
    logging.basicConfig(level=logging.DEBUG, format="%(message)s") # Set base level for stdlib

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_logger_name,
            add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        # *** MODIFIED: Changed level to DEBUG ***
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        # *** END MODIFICATION ***
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

log = structlog.get_logger() # module-level logger
# --- End Structlog Setup ---


# --- Config Loading Helpers ---
def load_toml_config(config_path: Path, config_name: str) -> PyDict[str, Any]:
    """Loads a TOML configuration file."""
    # (Implementation unchanged)
    if not config_path.is_file():
        log.error(f"{config_name} config file not found", path=str(config_path))
        return {}
    try:
        with config_path.open("rb") as f: # tomllib requires bytes mode
            config_data = tomllib.load(f)
        log.info(f"{config_name} config loaded", path=str(config_path))
        return config_data
    except tomllib.TOMLDecodeError as e:
        log.error( f"Error parsing TOML for {config_name}", path=str(config_path), error=str(e), exc_info=True, )
        return {}
    except Exception as e:
        log.error( f"Failed to load {config_name} config", path=str(config_path), error=str(e), exc_info=True, )
        return {}

def load_yaml_config(config_path: Path, config_name: str) -> PyDict[str, Any]:
    """Loads a generic YAML configuration file."""
    # (Implementation unchanged)
    if not config_path.is_file():
        log.error(f"{config_name} config file not found", path=str(config_path))
        raise FileNotFoundError( f"{config_name} configuration file not found: {config_path}" )
    try:
        with config_path.open("r") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            log.warning(f"{config_name} config file is empty.", path=str(config_path))
            return {}
        log.info(f"{config_name} config loaded", path=str(config_path))
        return config_data
    except yaml.YAMLError as e:
        log.error( f"Error parsing YAML for {config_name}", path=str(config_path), error=str(e), exc_info=True, )
        raise # Re-raise parsing errors
    except Exception as e:
        log.error( f"Failed to load {config_name} config", path=str(config_path), error=str(e), exc_info=True, )
        raise # Re-raise other load errors

# --- End Config Loading ---


# --- Debug Map Printing Function ---
def print_map_section(game_map: GameMap, center_x: int, center_y: int, radius: int = 5):
    """Prints a section of the map centered around (x, y) to the console."""
    # (Implementation unchanged)
    if not isinstance(game_map, GameMap): log.warning("Cannot print map section: Invalid GameMap object."); return
    y_min = max(0, center_y - radius); y_max = min(game_map.height, center_y + radius + 1)
    x_min = max(0, center_x - radius); x_max = min(game_map.width, center_x + radius + 1)
    print(f"\n--- Map Section around ({center_x},{center_y}) ---")
    header = "   " + "".join([f"{x:<3}" for x in range(x_min, x_max)])
    print(header); print("  " + "-" * (len(header)-2))
    for y in range(y_min, y_max):
        row_str = f"{y:<2}|"
        for x in range(x_min, x_max):
            char = "?"; tile_id = game_map.tiles[y, x]
            if tile_id == TILE_ID_FLOOR: char = "."
            elif tile_id == TILE_ID_WALL: char = "#"
            else: char = str(tile_id)
            if x == center_x and y == center_y: row_str += f"[{char}]"
            else: row_str += f" {char} "
        print(row_str)
    print("------------------------------------\n")
# --- End Debug Map Printing ---


def main() -> None:
    """Main entry point for the application."""
    setup_logging()
    log.info("Application starting...")
    log.info(f"Script directory: {SCRIPT_DIR}")
    log.info(f"Config directory: {CONFIG_DIR}")

    app = QApplication(sys.argv)

    try:
        # --- Load Configurations ---
        config = load_yaml_config(CONFIG_FILE, "Main")
        item_templates = load_yaml_config(ITEMS_CONFIG_FILE, "Items").get( "templates", {} )
        effect_definitions = load_yaml_config(EFFECTS_CONFIG_FILE, "Effects").get( "effects", {} )
        keybindings_config = load_toml_config(KEYBINDINGS_FILE, "Keybindings")
        settings_config = load_toml_config(SETTINGS_FILE, "Settings")
        log.info( "Configurations loaded", items=len(item_templates), effects=len(effect_definitions),
                  keybindings=len(keybindings_config.get("bindings", {})), settings=len(settings_config) )

        # --- Extract Config Values ---
        # (Extraction logic unchanged)
        initial_tileset_folder_rel: str = config.get( "initial_tileset_folder", "fonts/classic_roguelike_sliced_svgs" )
        initial_tileset_folder_abs = SCRIPT_DIR / initial_tileset_folder_rel
        log.debug( "Resolved initial tileset path", relative=initial_tileset_folder_rel, absolute=str(initial_tileset_folder_abs), )
        initial_tile_width: int = config.get("initial_tile_width", 16)
        initial_tile_height: int = config.get("initial_tile_height", 16)
        min_tile_size: int = config.get("minimum_tile_size", 4)
        scroll_debounce_ms: int = config.get("scroll_scale_debounce_ms", 200)
        resize_debounce_ms: int = config.get("resize_debounce_ms", 100)
        map_width: int = config.get("map_width", 80)
        map_height: int = config.get("map_height", 50)
        dungeon_seed_cfg = config.get("dungeon_seed")
        player_glyph: int = config.get("player_glyph", 113)
        player_start_hp: int = config.get("player_start_hp", 30)
        player_fov_radius: int = config.get("player_fov_radius", 8)
        lighting_config = config.get("lighting", {})
        lighting_ambient: float = lighting_config.get("ambient_level", 0.15)
        lighting_min_fov: float = lighting_config.get("min_fov_level", 0.25)
        lighting_falloff: float = lighting_config.get("falloff_power", 1.5)
        hv_config = config.get("height_visualization", {})
        vis_enabled_default: bool = hv_config.get("enabled_by_default", False)
        vis_max_diff: int = hv_config.get("max_relative_difference", 10)
        vis_color_high: list = hv_config.get("color_high", [255, 255, 0])
        vis_color_mid: list = hv_config.get("color_mid", [0, 255, 0])
        vis_color_low: list = hv_config.get("color_low", [0, 128, 255])
        vis_blend_factor: float = hv_config.get("blend_factor", 0.3)
        gameplay_rules = config.get("gameplay_rules", {})
        max_traversable_step: int = gameplay_rules.get("max_traversable_step", 2)
        log.info("Main configuration values extracted.")

        # --- Game Initialization ---
        log.info("Creating game map", width=map_width, height=map_height)
        game_map = GameMap(width=map_width, height=map_height)

        log.info("Generating dungeon layout...")
        dungeon_seed = ( int(time.time() * 1000) if dungeon_seed_cfg is None else int(dungeon_seed_cfg) )
        rng_seed_to_pass = dungeon_seed
        log.info("Using dungeon seed", seed=dungeon_seed)
        player_start_pos = generate_dungeon( game_map, map_width, map_height, seed=dungeon_seed )
        log.info("Dungeon generated", player_start=player_start_pos)

        # Print map section after generation
        print_map_section(game_map, player_start_pos[0], player_start_pos[1], radius=10)

        log.info("Initializing game state...")
        game_state = GameState(
            existing_map=game_map, player_start_pos=player_start_pos,
            player_glyph=player_glyph, player_start_hp=player_start_hp,
            player_fov_radius=player_fov_radius, item_templates=item_templates,
            effect_definitions=effect_definitions, rng_seed=rng_seed_to_pass,
        )

        # Spawn initial items
        if player_start_pos:
            spawn_x, spawn_y = player_start_pos
            if hasattr(game_state, "item_registry") and game_state.item_registry:
                log.info("Spawning initial items near player", pos=(spawn_x, spawn_y))
                game_state.item_registry.create_item( "simple_dagger", "ground", x=spawn_x + 1, y=spawn_y )
                game_state.item_registry.create_item( "cookies", "ground", x=spawn_x, y=spawn_y + 1 )
                game_state.item_registry.create_item( "torch", "ground", x=spawn_x - 1, y=spawn_y )
            else: log.warning( "ItemRegistry not found in GameState, skipping initial item spawn." )

        log.info("Creating main window...")
        # WindowManager Instantiation
        window = WindowManager(
            app_config=config, keybindings_config=keybindings_config,
            initial_tileset_path=str(initial_tileset_folder_abs),
            initial_tile_width=initial_tile_width, initial_tile_height=initial_tile_height,
            map_width=game_state.map_width, map_height=game_state.map_height,
            min_tile_size_cfg=min_tile_size, scroll_debounce_cfg=scroll_debounce_ms,
            resize_debounce_cfg=resize_debounce_ms,
        )

        log.info("Initializing main loop...")
        main_loop = MainLoop(
            game_state=game_state, window=window,
            vis_enabled_default=vis_enabled_default, vis_max_diff=vis_max_diff,
            vis_color_high=vis_color_high, vis_color_mid=vis_color_mid,
            vis_color_low=vis_color_low, vis_blend_factor=vis_blend_factor,
            max_traversable_step=max_traversable_step, lighting_ambient=lighting_ambient,
            lighting_min_fov=lighting_min_fov, lighting_falloff=lighting_falloff,
        )
        window.set_main_loop(main_loop)

    # --- Exception Handling ---
    except FileNotFoundError as e:
        log.critical("Required file not found during init", error=str(e), exc_info=True)
        sys.exit(f"Initialization failed: File not found - {e}")
    except KeyError as e:
        log.critical( "Missing required key, possibly in config", key=str(e), exc_info=True )
        sys.exit(f"Configuration failed: Missing key {e}")
    except TypeError as e:
        log.critical("Fatal Type Error during init", error=str(e), exc_info=True)
        sys.exit(f"Initialization failed (TypeError): {e}")
    except Exception as e:
        log.critical("Fatal initialization error", error=str(e), exc_info=True)
        sys.exit(f"Initialization failed: {e}")
    # --- End Exception Handling ---

    log.info("Showing window and starting application loop...")
    window.show()
    sys.exit(app.exec()) # Start the Qt event loop


if __name__ == "__main__":
    main()

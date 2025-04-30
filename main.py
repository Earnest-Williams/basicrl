# main.py
import sys
import time
import yaml
import logging
import tomllib  # Ensure this is imported (standard in Python 3.11+)

# --- MODIFIED: Add pathlib ---
from pathlib import Path

# --- End MODIFIED ---
from typing import Any

import structlog
from structlog.stdlib import add_logger_name, add_log_level
from PySide6.QtWidgets import QApplication

# Absolute imports are generally preferred when the module execution context is clear
from engine.tileset_loader import load_tiles
from engine.main_loop import MainLoop
from engine.window_manager import WindowManager
from game.world.game_map import GameMap
from game.game_state import GameState
from game.world.procgen import generate_dungeon

# --- MODIFIED: Define paths relative to this script's location ---
SCRIPT_DIR = Path(__file__).parent.resolve()  # Gets the directory containing main.py
CONFIG_DIR = SCRIPT_DIR / "config"
FONTS_DIR = SCRIPT_DIR / "fonts"  # Example for assets

CONFIG_FILE = CONFIG_DIR / "config.yaml"
ITEMS_CONFIG_FILE = CONFIG_DIR / "items.yaml"
EFFECTS_CONFIG_FILE = CONFIG_DIR / "effects.yaml"
KEYBINDINGS_FILE = CONFIG_DIR / "keybindings.toml"  # Define path for keybindings
SETTINGS_FILE = CONFIG_DIR / "settings.toml"  # Define path for settings
# --- End MODIFIED ---


# --- Structlog Setup ---
def setup_logging():
    # ... (logging setup unchanged) ...
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_logger_name,
            add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()
# --- End Structlog Setup ---


# --- ADDED HELPER FUNCTION FOR TOML ---
def load_toml_config(config_path: Path, config_name: str) -> dict[str, Any]:
    """Loads a TOML configuration file."""
    if not config_path.is_file():
        log.error(f"{config_name} config file not found", path=str(config_path))
        # Return empty dict or raise error? Empty dict seems safer for optional configs.
        return {}
    try:
        with config_path.open("rb") as f:  # tomllib reads bytes
            config_data = tomllib.load(f)
        log.info(
            f"{config_name} configuration loaded successfully", path=str(config_path)
        )
        return config_data
    except tomllib.TOMLDecodeError as e:
        log.error(
            f"Error parsing TOML for {config_name}",
            path=str(config_path),
            error=str(e),
            exc_info=True,
        )
        return {}  # Return empty on error
    except Exception as e:
        log.error(
            f"Failed to load {config_name} configuration",
            path=str(config_path),
            error=str(e),
            exc_info=True,
        )
        return {}  # Return empty on error


# --- END ADDED HELPER FUNCTION ---


def load_yaml_config(config_path: Path, config_name: str) -> dict[str, Any]:
    """Loads a generic YAML configuration file."""
    # Now accepts Path object
    if not config_path.is_file():
        log.error(f"{config_name} config file not found", path=str(config_path))
        raise FileNotFoundError(
            f"{config_name} configuration file not found: {config_path}"
        )
    try:
        # Use Path object directly
        with config_path.open("r") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            log.warning(f"{config_name} config file is empty.", path=str(config_path))
            return {}
        log.info(
            f"{config_name} configuration loaded successfully", path=str(config_path)
        )
        return config_data
    except yaml.YAMLError as e:
        log.error(
            f"Error parsing YAML for {config_name}",
            path=str(config_path),
            error=str(e),
            exc_info=True,
        )
        raise
    except Exception as e:
        log.error(
            f"Failed to load {config_name} configuration",
            path=str(config_path),
            error=str(e),
            exc_info=True,
        )
        raise


def main() -> None:
    """Main entry point for the application."""
    setup_logging()
    log.info("Application starting...")
    log.info(f"Script directory: {SCRIPT_DIR}")
    log.info(f"Config directory: {CONFIG_DIR}")
    log.info(f"Fonts directory: {FONTS_DIR}")

    app = QApplication(sys.argv)

    try:
        # --- Load Configurations ---
        config = load_yaml_config(CONFIG_FILE, "Main")
        item_templates = load_yaml_config(ITEMS_CONFIG_FILE, "Items").get(
            "templates", {}
        )
        effect_definitions = load_yaml_config(EFFECTS_CONFIG_FILE, "Effects").get(
            "effects", {}
        )
        # --- LOAD KEYBINDINGS ---
        keybindings_config = load_toml_config(KEYBINDINGS_FILE, "Keybindings")
        # --- LOAD SETTINGS (Example, not used yet) ---
        settings_config = load_toml_config(SETTINGS_FILE, "Settings")  # Load settings
        log.info("Item templates loaded", count=len(item_templates))
        log.info("Effect definitions loaded", count=len(effect_definitions))
        log.info(
            "Keybindings loaded", count=len(keybindings_config.get("bindings", {}))
        )
        log.info("Settings loaded", count=len(settings_config))

        # --- Extract values from main config safely ---
        initial_tileset_folder_rel: str = config.get(
            "initial_tileset_folder", "fonts/classic_roguelike_sliced_svgs"
        )
        initial_tileset_folder_abs = SCRIPT_DIR / initial_tileset_folder_rel
        log.debug(
            "Resolved initial tileset path",
            relative=initial_tileset_folder_rel,
            absolute=str(initial_tileset_folder_abs),
        )

        initial_tile_width: int = config.get("initial_tile_width", 16)
        initial_tile_height: int = config.get("initial_tile_height", 16)
        min_tile_size: int = config.get("minimum_tile_size", 4)
        scroll_debounce_ms: int = config.get("scroll_scale_debounce_ms", 200)
        resize_debounce_ms: int = config.get("resize_debounce_ms", 100)
        map_width: int = config.get("map_width", 80)
        map_height: int = config.get("map_height", 50)
        dungeon_seed_cfg = config.get("dungeon_seed")
        player_glyph: int = config.get("player_glyph", 113)  # '@'
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

        log.info("Main configuration values loaded.")

        # --- Game Initialization ---
        log.info("Loading initial tileset", path=str(initial_tileset_folder_abs))
        clamped_width = max(min_tile_size, initial_tile_width)
        clamped_height = max(min_tile_size, initial_tile_height)
        initial_tiles, _ = load_tiles(
            str(initial_tileset_folder_abs), clamped_width, clamped_height
        )

        log.info("Creating game map", width=map_width, height=map_height)
        game_map = GameMap(width=map_width, height=map_height)

        log.info("Generating dungeon layout...")
        dungeon_seed = (
            int(time.time() * 1000)
            if dungeon_seed_cfg is None
            else int(dungeon_seed_cfg)
        )
        rng_seed_to_pass = dungeon_seed  # Use same seed for RNG init
        log.info("Using dungeon seed", seed=dungeon_seed)
        player_start_pos = generate_dungeon(
            game_map, map_width, map_height, seed=dungeon_seed
        )
        log.info("Dungeon generated", player_start=player_start_pos)

        log.info("Initializing game state...")
        game_state = GameState(
            existing_map=game_map,
            player_start_pos=player_start_pos,
            player_glyph=player_glyph,
            player_start_hp=player_start_hp,
            player_fov_radius=player_fov_radius,
            item_templates=item_templates,
            effect_definitions=effect_definitions,
            rng_seed=rng_seed_to_pass,  # Pass seed to GameState for its RNG
        )

        # Spawn initial items (Example)
        if player_start_pos:
            spawn_x, spawn_y = player_start_pos
            if hasattr(game_state, "item_registry") and game_state.item_registry:
                log.info("Spawning initial items near player", pos=(spawn_x, spawn_y))
                # Add checks for item creation success if needed
                game_state.item_registry.create_item(
                    "simple_dagger", "ground", x=spawn_x + 1, y=spawn_y
                )
                game_state.item_registry.create_item(
                    "cookies", "ground", x=spawn_x, y=spawn_y + 1
                )
                game_state.item_registry.create_item(
                    "torch", "ground", x=spawn_x - 1, y=spawn_y
                )
            else:
                log.warning(
                    "ItemRegistry not found in GameState, skipping initial item spawn."
                )

        log.info("Creating main window...")
        window = WindowManager(
            # Pass BOTH config dictionaries
            app_config=config,
            keybindings_config=keybindings_config,  # Pass the loaded keybindings
            initial_tileset_path=str(initial_tileset_folder_abs),
            initial_tiles=initial_tiles,
            initial_tile_width=clamped_width,
            initial_tile_height=clamped_height,
            map_width=game_state.map_width,
            map_height=game_state.map_height,
            min_tile_size_cfg=min_tile_size,
            scroll_debounce_cfg=scroll_debounce_ms,
            resize_debounce_cfg=resize_debounce_ms,
        )

        log.info("Initializing main loop...")
        main_loop = MainLoop(
            game_state=game_state,
            window=window,
            vis_enabled_default=vis_enabled_default,
            vis_max_diff=vis_max_diff,
            vis_color_high=vis_color_high,
            vis_color_mid=vis_color_mid,
            vis_color_low=vis_color_low,
            vis_blend_factor=vis_blend_factor,
            max_traversable_step=max_traversable_step,
            lighting_ambient=lighting_ambient,
            lighting_min_fov=lighting_min_fov,
            lighting_falloff=lighting_falloff,
        )
        window.set_main_loop(main_loop)

    except FileNotFoundError as e:
        log.critical("Required file not found", error=str(e), exc_info=True)
        sys.exit(f"Initialization failed: File not found - {e}")
    except KeyError as e:
        # This might catch errors if config structure is wrong after loading
        log.critical(
            "Missing required key, possibly in config",
            key=str(e),
            # path=str(CONFIG_FILE), # Path might not be accurate if error is elsewhere
            exc_info=True,
        )
        sys.exit(f"Configuration failed: Missing key {e}")
    except Exception as e:
        log.critical("Fatal initialization error", error=str(e), exc_info=True)
        sys.exit(f"Initialization failed: {e}")

    log.info("Showing window and starting application loop...")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# main.py
import sys
import time
import yaml
import logging
from pathlib import Path

import structlog
from structlog.stdlib import add_logger_name, add_log_level
from PySide6.QtWidgets import QApplication

from engine.tileset_loader import load_tiles
from engine.main_loop import MainLoop
from engine.window_manager import WindowManager
from game.world.game_map import GameMap
from game.game_state import GameState
from game.world.procgen import generate_dungeon

CONFIG_FILE = Path("config/config.yaml")


# --- Structlog Setup ---
def setup_logging():
    """Configures structlog for console output."""
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


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.is_file():
        log.error("Config file not found", path=str(config_path))
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None:
            log.warning("Config file is empty, using defaults.", path=str(config_path))
            return {}
        log.info("Configuration loaded successfully", path=str(config_path))
        return config
    except yaml.YAMLError as e:
        log.error(
            "Error parsing YAML configuration",
            path=str(config_path),
            error=str(e),
            exc_info=True,
        )
        raise
    except Exception as e:
        log.error(
            "Failed to load configuration",
            path=str(config_path),
            error=str(e),
            exc_info=True,
        )
        raise


def main() -> None:
    """Main entry point for the application."""
    setup_logging()
    log.info("Application starting...")

    app = QApplication(sys.argv)

    try:
        # --- Load Configuration ---
        config = load_config(CONFIG_FILE)

        # --- Extract values from config safely with defaults ---
        # Display & Tileset
        initial_tileset_folder: str = config.get(
            "initial_tileset_folder", "fonts/classic_roguelike_sliced_svgs"
        )
        initial_tile_width: int = config.get("initial_tile_width", 16)
        initial_tile_height: int = config.get("initial_tile_height", 16)
        min_tile_size: int = config.get("minimum_tile_size", 4)
        scroll_debounce_ms: int = config.get("scroll_scale_debounce_ms", 200)
        resize_debounce_ms: int = config.get("resize_debounce_ms", 100)

        # Map Dimensions
        map_width: int = config.get("map_width", 80)
        map_height: int = config.get("map_height", 50)

        # Procedural Generation
        dungeon_seed_cfg = config.get("dungeon_seed")

        # Player Settings
        player_glyph: int = config.get("player_glyph", 113)  # User specified
        player_start_hp: int = config.get("player_start_hp", 30)
        player_fov_radius: int = config.get("player_fov_radius", 8)

        # --- Added Lighting Config Loading ---
        lighting_config = config.get("lighting", {})
        lighting_ambient: float = lighting_config.get("ambient_level", 0.15)
        lighting_min_fov: float = lighting_config.get("min_fov_level", 0.25)
        lighting_falloff: float = lighting_config.get("falloff_power", 1.5)
        # --- End Lighting Config Loading ---

        # Height Visualization
        hv_config = config.get("height_visualization", {})
        vis_enabled_default: bool = hv_config.get("enabled_by_default", False)
        vis_max_diff: int = hv_config.get("max_relative_difference", 10)
        vis_color_high: list = hv_config.get("color_high", [255, 255, 0])
        vis_color_mid: list = hv_config.get("color_mid", [0, 255, 0])
        vis_color_low: list = hv_config.get("color_low", [0, 128, 255])
        vis_blend_factor: float = hv_config.get("blend_factor", 0.3)

        # Gameplay Rules
        gameplay_rules = config.get("gameplay_rules", {})
        max_traversable_step: int = gameplay_rules.get("max_traversable_step", 2)

        # --- Updated Logging for Config Values ---
        log.info(
            "Configuration values",
            tileset=initial_tileset_folder,
            tile_w=initial_tile_width,
            tile_h=initial_tile_height,
            map_w=map_width,
            map_h=map_height,
            seed_cfg=dungeon_seed_cfg,
            player_glyph=player_glyph,
            player_hp=player_start_hp,
            player_fov=player_fov_radius,
            light_ambient=lighting_ambient,
            light_min_fov=lighting_min_fov,
            light_falloff=lighting_falloff,  # Added lighting
            vis_enabled=vis_enabled_default,
            vis_max_diff=vis_max_diff,
            vis_blend=vis_blend_factor,
            max_step=max_traversable_step,
        )
        # --- End Updated Logging ---

        # --- Game Initialization using Config ---
        log.info("Loading initial tileset", path=initial_tileset_folder)
        clamped_width = max(min_tile_size, initial_tile_width)
        clamped_height = max(min_tile_size, initial_tile_height)
        if clamped_width != initial_tile_width or clamped_height != initial_tile_height:
            log.info(
                "Tile dimensions clamped",
                initial_w=initial_tile_width,
                initial_h=initial_tile_height,
                clamped_w=clamped_width,
                clamped_h=clamped_height,
                min_size=min_tile_size,
            )
        initial_tiles, _ = load_tiles(
            initial_tileset_folder, clamped_width, clamped_height
        )

        log.info("Creating game map", width=map_width, height=map_height)
        game_map = GameMap(width=map_width, height=map_height)

        log.info("Generating dungeon layout...")
        dungeon_seed = (
            int(time.time() * 1000)
            if dungeon_seed_cfg is None
            else int(dungeon_seed_cfg)
        )
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
        )

        log.info("Creating main window...")
        window = WindowManager(
            initial_tileset_path=initial_tileset_folder,
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
        # --- UPDATED MainLoop call with lighting params ---
        main_loop = MainLoop(
            game_state=game_state,
            window=window,
            # Visualization config
            vis_enabled_default=vis_enabled_default,
            vis_max_diff=vis_max_diff,
            vis_color_high=vis_color_high,
            vis_color_mid=vis_color_mid,
            vis_color_low=vis_color_low,
            vis_blend_factor=vis_blend_factor,
            # Gameplay config
            max_traversable_step=max_traversable_step,
            # Lighting config
            lighting_ambient=lighting_ambient,
            lighting_min_fov=lighting_min_fov,
            lighting_falloff=lighting_falloff,
        )
        # --- END UPDATED MainLoop call ---
        window.set_main_loop(main_loop)

    except FileNotFoundError as e:
        log.critical("Config file missing", error=str(e), exc_info=True)
        sys.exit(f"Configuration failed: {e}")
    except KeyError as e:
        log.critical(
            "Missing required key in config",
            key=str(e),
            path=str(CONFIG_FILE),
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

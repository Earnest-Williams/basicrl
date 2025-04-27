# main.py
import sys
import time
import yaml
from pathlib import Path

from PySide6.QtWidgets import QApplication

from engine.tileset_loader import load_tiles
from engine.main_loop import MainLoop

# Corrected import - only import the class
from engine.window_manager import WindowManager
from game.world.game_map import GameMap
from game.game_state import GameState
from game.world.procgen import generate_dungeon


CONFIG_FILE = Path("config/config.yaml")


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        print(f"Failed to load configuration {config_path}: {e}")
        raise


def main() -> None:
    """Main entry point for the application."""
    app = QApplication(sys.argv)

    try:
        # --- Load Configuration ---
        config = load_config(CONFIG_FILE)

        # --- Extract values from config ---
        initial_tileset_folder: str = config["initial_tileset_folder"]
        initial_tile_width: int = config["initial_tile_width"]
        initial_tile_height: int = config["initial_tile_height"]
        map_width: int = config["map_width"]
        map_height: int = config["map_height"]
        dungeon_seed_cfg = config.get("dungeon_seed")

        # --- Game Initialization using Config ---
        print(f"Loading initial tileset: {initial_tileset_folder}")
        # Use a literal default value (e.g., 4) if key is missing from config
        min_tile_size = config.get("minimum_tile_size", 4)  # FIXED: Literal default
        clamped_width = max(min_tile_size, initial_tile_width)
        clamped_height = max(min_tile_size, initial_tile_height)

        initial_tiles, _ = load_tiles(
            initial_tileset_folder, clamped_width, clamped_height
        )

        print("Creating game map...")
        game_map = GameMap(width=map_width, height=map_height)

        print("Generating dungeon layout...")
        dungeon_seed = (
            int(time.time()) if dungeon_seed_cfg is None else int(dungeon_seed_cfg)
        )
        print(f"Using dungeon seed: {dungeon_seed}")
        player_start_pos = generate_dungeon(
            game_map, map_width, map_height, seed=dungeon_seed
        )

        print("Initializing game state...")
        game_state = GameState(
            existing_map=game_map,
            player_start_pos=player_start_pos,
            player_glyph=config.get("player_glyph", 64),
            player_start_hp=config.get("player_start_hp", 30),
            player_fov_radius=config.get("player_fov_radius", 4),
        )

        print("Creating main window...")
        window = WindowManager(
            initial_tileset_path=initial_tileset_folder,
            initial_tiles=initial_tiles,
            initial_tile_width=clamped_width,
            initial_tile_height=clamped_height,
            map_width=game_state.map_width,
            map_height=game_state.map_height,
            # Pass relevant config values with literal defaults
            min_tile_size_cfg=min_tile_size,
            scroll_debounce_cfg=config.get(
                "scroll_scale_debounce_ms", 200  # FIXED: Literal default
            ),
            resize_debounce_cfg=config.get(
                "resize_debounce_ms", 100  # FIXED: Literal default
            ),
        )

        print("Initializing main loop...")
        main_loop = MainLoop(game_state=game_state, window=window)
        window.set_main_loop(main_loop)

    except FileNotFoundError as e:
        print(f"\n--- CONFIGURATION ERROR ---")
        print(f"Error: {e}")
        sys.exit(f"Configuration failed: {e}")
    except KeyError as e:
        print(f"\n--- CONFIGURATION ERROR ---")
        print(f"Error: Missing required key in {CONFIG_FILE}: {e}")
        sys.exit(f"Configuration failed: Missing key {e}")
    except Exception as e:
        print(f"\n--- FATAL INITIALIZATION ERROR ---")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(f"Initialization failed: {e}")

    print("Showing window and starting application loop...")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

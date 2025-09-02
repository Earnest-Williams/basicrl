# constants.py
"""
Global constants for the FOV/Light/Memory Simulation Game.
Includes base colors for light sources.
"""
from typing import Dict, Tuple

# --- Core Gameplay ---
MAX_LOS_DISTANCE = 500

# --- Tile IDs ---
WALL_ID = 0
FLOOR_ID = 1
PILLAR_ID = 2

# --- Rendering Characters ---
PLAYER_CHAR = "@"
LIGHT_CHAR = "*"
VISIBLE_WALL = "█"
VISIBLE_PILLAR = "▣"
VISIBLE_FLOOR = "·"
MEMORY_LIGHT = "+"
UNSEEN = " "

# --- Memory Fade ---
MEMORY_DURATION = 60.0
# Linear decay: intensity decreases from 1.0 to 0.0 over MEMORY_DURATION seconds.
MEMORY_DECAY_RATE = 1.0 / MEMORY_DURATION
MEMORY_LEVEL_COUNT = 5
MEMORY_WALL_LEVELS = ["▓", "▒", "░", "⋅", " "]
MEMORY_PILLAR_LEVELS = ["▤", "▥", "▫", "◦", " "]
MEMORY_FLOOR_LEVELS = [".", "·", "⋅", " ", " "]

# --- ANSI Colors (Basic) ---
COLOR = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BRIGHT_BLACK": "\033[90m",
    "BRIGHT_RED": "\033[91m",
    "BRIGHT_GREEN": "\033[92m",
    "BRIGHT_YELLOW": "\033[93m",
    "BRIGHT_BLUE": "\033[94m",
    "BRIGHT_MAGENTA": "\033[95m",
    "BRIGHT_CYAN": "\033[96m",
    "BRIGHT_WHITE": "\033[97m",
}

# --- True Color Definitions ---
# Base colors for different light source types (adjust as desired)
TORCH_COLOR_RGB: Tuple[int, int, int] = (255, 160, 60)  # Orangey-yellow
ORB_COLOR_RGB: Tuple[int, int, int] = (160, 200, 255)  # Bluish-white
# Ambient color for unlit areas or minimum light
AMBIENT_COLOR_RGB: Tuple[int, int, int] = (
    30, 30, 45)  # Dim, slightly blue/purple
# Max level used for mapping intensity -> level for visibility checks
MAX_LIGHT_LEVEL_FOR_VIS_CHECK = 6

# --- Memory Color ---
MEMORY_COLOR = COLOR["DIM"] + COLOR["BRIGHT_BLACK"]

# --- Light Falloff ---
# Note: The inverse square calculation is now done directly in the Numba func.
# This constant isn't directly used by the inverse square formula chosen, but keep for reference?
LIGHT_LEVEL_FALLOFF_RATE = 5

# --- Debugging ---
# Options: "normal", "level" (shows brightness), "intensity" (memory), "level_color" (shows blended color)
DEBUG_RENDER_MODE = "normal"

# --- Light Level Data --- (Used for visibility checks based on mapped level)
LIGHT_LEVEL_DATA: Dict[str, Dict[str, Dict[str, int]]] = {
    "0": {
        "description": "None",
        "small": {"id_range": 0, "noticeable_range": 0},
        "medium": {"id_range": 0, "noticeable_range": 0},
        "large": {"id_range": 0, "noticeable_range": 0},
        "giant": {"id_range": 0, "noticeable_range": 0},
        "floor": {"id_range": 0, "noticeable_range": 0},
        "wall": {"id_range": 0, "noticeable_range": 0},
    },
    "1": {
        "description": "Ember",
        "small": {"id_range": 2, "noticeable_range": 5},
        "medium": {"id_range": 5, "noticeable_range": 10},
        "large": {"id_range": 8, "noticeable_range": 15},
        "giant": {"id_range": 10, "noticeable_range": 20},
        "floor": {"id_range": 3, "noticeable_range": 6},
        "wall": {"id_range": 2, "noticeable_range": 5},
    },
    "2": {
        "description": "Candle",
        "small": {"id_range": 5, "noticeable_range": 10},
        "medium": {"id_range": 10, "noticeable_range": 20},
        "large": {"id_range": 15, "noticeable_range": 30},
        "giant": {"id_range": 20, "noticeable_range": 40},
        "floor": {"id_range": 6, "noticeable_range": 12},
        "wall": {"id_range": 5, "noticeable_range": 10},
    },
    "3": {
        "description": "Torch",
        "small": {"id_range": 10, "noticeable_range": 20},
        "medium": {"id_range": 20, "noticeable_range": 40},
        "large": {"id_range": 30, "noticeable_range": 60},
        "giant": {"id_range": 40, "noticeable_range": 80},
        "floor": {"id_range": 10, "noticeable_range": 20},
        "wall": {"id_range": 8, "noticeable_range": 16},
    },
    "4": {
        "description": "Lantern",
        "small": {"id_range": 15, "noticeable_range": 30},
        "medium": {"id_range": 25, "noticeable_range": 50},
        "large": {"id_range": 50, "noticeable_range": 100},
        "giant": {"id_range": 60, "noticeable_range": 120},
        "floor": {"id_range": 15, "noticeable_range": 30},
        "wall": {"id_range": 12, "noticeable_range": 25},
    },
    "5": {
        "description": "Orb",
        "small": {"id_range": 25, "noticeable_range": 50},
        "medium": {"id_range": 50, "noticeable_range": 75},
        "large": {"id_range": 75, "noticeable_range": 125},
        "giant": {"id_range": 100, "noticeable_range": 150},
        "floor": {"id_range": 25, "noticeable_range": 40},
        "wall": {"id_range": 20, "noticeable_range": 35},
    },
    "6": {
        "description": "Radiance",
        "small": {"id_range": 40, "noticeable_range": 75},
        "medium": {"id_range": 75, "noticeable_range": 100},
        "large": {"id_range": 100, "noticeable_range": 150},
        "giant": {"id_range": 150, "noticeable_range": 200},
        "floor": {"id_range": 40, "noticeable_range": 60},
        "wall": {"id_range": 30, "noticeable_range": 50},
    },
}

# --- Tile to Size Category Mapping ---
TILE_ID_TO_CATEGORY: Dict[int, str] = {
    WALL_ID: "wall",
    FLOOR_ID: "floor",
    PILLAR_ID: "medium",
}
DEFAULT_ENTITY_CATEGORY = "medium"

# --- Helper: Convert RGB tuple to Numba-compatible array ---
# Numba struggles with tuples directly sometimes, especially in jitclass or complex signatures.
# Using a simple function to convert can help ensure compatibility.
# (We'll actually pass individual r,g,b components to Numba function to avoid issues)
# def rgb_to_numba_array(rgb: Tuple[int, int, int]) -> np.ndarray:
#      return np.array(rgb, dtype=np.uint8) # Or float if needed for Numba calcs

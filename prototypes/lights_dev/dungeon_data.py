# dungeon_data.py
"""
Defines the Dungeon jitclass data structure.
"""
import math  # Required for get_distance method

import numba
import numpy as np

# Import necessary constants (only WALL_ID needed directly by __init__)
from constants import PILLAR_ID, WALL_ID

# Jitclass specification remains the same
dungeon_spec = [
    ("width", numba.int64),
    ("height", numba.int64),
    ("tiles", numba.int8[:, :]),
    ("visible", numba.boolean[:, :]),
    ("memory_intensity", numba.float32[:, :]),
    ("last_seen_time", numba.float32[:, :]),
    ("current_time", numba.float32),
]


@numba.experimental.jitclass(dungeon_spec)
class Dungeon:
    """Holds Numba-compatible grid data including memory fade state."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.tiles = np.full((height, width), WALL_ID, dtype=np.int8)
        self.visible = np.zeros((height, width), dtype=np.bool_)
        self.memory_intensity = np.zeros((height, width), dtype=np.float32)
        self.last_seen_time = np.zeros((height, width), dtype=np.float32)
        self.current_time = 0.0

    def blocks_light(self, x: int, y: int) -> bool:
        """Checks if the tile at (x, y) blocks light/LOS."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        tile_id = self.tiles[y, x]
        # Use constants directly here, assuming they won't change dynamically
        # If they needed to be dynamic, they'd have to be passed or part of spec
        return tile_id == WALL_ID or tile_id == PILLAR_ID

    def get_distance(self, x: int, y: int) -> float:
        """Calculates Euclidean distance from origin (0,0) - used within octants."""
        # math.sqrt is compatible with nopython mode
        return math.sqrt(float(x * x + y * y))

    # update_memory_fade method removed, logic moved to standalone function

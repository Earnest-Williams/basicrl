# game/world/game_map.py
from typing import Final, NamedTuple, Set, Tuple  # Ensure Set, Tuple are imported

import math
import numpy as np
import structlog
from dataclasses import dataclass

# Ensure correct import for the memory helper and visibility class
from game.world.fov import update_memory_fade
from game.world.visibility import MyVisibility

log = structlog.get_logger()

TILE_ID_FLOOR: Final[int] = 0
TILE_ID_WALL: Final[int] = 1
# Maximum number of times a tile can strengthen its memory
MAX_MEMORY_STRENGTH: Final[float] = 5.0


class TileType(NamedTuple):
    walkable: bool
    transparent: bool
    tile_index: int
    color_fg: tuple[int, int, int]
    color_bg: tuple[int, int, int]


TILE_TYPES: Final[dict[int, TileType]] = {
    TILE_ID_FLOOR: TileType(
        walkable=True,
        transparent=True,
        tile_index=2, # Example index for floor tile in tileset
        color_fg=(200, 200, 200),
        color_bg=(10, 10, 30),
    ),
    TILE_ID_WALL: TileType(
        walkable=False,
        transparent=False,
        tile_index=38, # Example index for wall tile in tileset
        color_fg=(180, 180, 180),
        color_bg=(30, 30, 50),
    ),
    # Add other tile types here...
}


def get_transparency_map(tiles: np.ndarray) -> np.ndarray:
    """Creates a boolean array indicating transparency based on TILE_TYPES."""
    # Initialize with False (opaque)
    transparency = np.zeros_like(tiles, dtype=bool)
    # Iterate through defined types and set transparency
    for tile_id, tile_type in TILE_TYPES.items():
        transparency[tiles == tile_id] = tile_type.transparent
    return transparency


@dataclass(frozen=True)
class LightSource:
    """Simple representation of a colored light source."""
    x: int
    y: int
    radius: int
    color: tuple[int, int, int]


class GameMap:
    def __init__(self, width: int, height: int):
        """
        Initializes the game map with dimensions and default tile arrays.
        """
        if width <= 0 or height <= 0:
            log.error("Invalid map dimensions", width=width, height=height)
            raise ValueError("Map width and height must be positive integers.")
        self._width = width
        self._height = height
        log.info("Initializing GameMap", width=self._width, height=self._height)

        # Core map data arrays - Use C order for compatibility with many libraries
        self.tiles: np.ndarray = np.full(
            (height, width), fill_value=TILE_ID_WALL, dtype=np.uint8, order="C"
        )
        # Visibility/Exploration state
        self.explored: np.ndarray = np.zeros((height, width), dtype=bool, order="C")
        self.visible: np.ndarray = np.zeros((height, width), dtype=bool, order="C")
        # Cached transparency map
        self.transparent: np.ndarray = get_transparency_map(self.tiles)
        # Height and Ceiling maps
        self.height_map: np.ndarray = np.zeros(
            (height, width), dtype=np.int16, order="C"
        )
        self.ceiling_map: np.ndarray = np.zeros(
            (height, width), dtype=np.int16, order="C"
        )
        self.memory_intensity: np.ndarray = np.zeros(
            (height, width), dtype=np.float32, order="C"
        )
        self.memory_strength: np.ndarray = np.zeros(
            (height, width), dtype=np.float32, order="C"
        )
        self.last_seen_time: np.ndarray = np.zeros(
            (height, width), dtype=np.int32, order="C"
        )
        self.memory_fade_mask: np.ndarray = np.zeros(
            (height, width), dtype=bool, order="C"
        )
        self.prev_visible: np.ndarray = np.zeros(
            (height, width), dtype=bool, order="C"
        )
        # Perception layers reused across turns
        self.noise_map: np.ndarray = np.zeros(
            (height, width), dtype=np.float32, order="C"
        )
        self.scent_map: np.ndarray = np.zeros(
            (height, width), dtype=np.float32, order="C"
        )
        self.light_sources: list[LightSource] = []
        # Vertical transitions like stairs or shafts
        self.vertical_transitions: list[dict[str, int | str]] = []
        # Environmental storytelling hooks (annotations on the map)
        self.story_hooks: list[dict[str, str | int]] = []
        log.debug("GameMap arrays initialized", shape=(height, width))

    def update_tile_transparency(self) -> None:
        """Recalculates the transparency map based on current self.tiles."""
        # This should be called whenever self.tiles is modified (e.g., after digging)
        self.transparent = np.zeros((self._height, self._width), dtype=bool)
        for tile_id, tile_type in TILE_TYPES.items():
            self.transparent[self.tiles == tile_id] = tile_type.transparent
        transparent_count = np.sum(self.transparent)
        log.info("Transparency map updated", transparent_count=transparent_count)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def in_bounds(self, x: int, y: int) -> bool:
        """Checks if the given coordinates are within the map boundaries."""
        return 0 <= x < self._width and 0 <= y < self._height

    def is_walkable(self, x: int, y: int) -> bool:
        """Checks if the tile at (x, y) is walkable."""
        if not self.in_bounds(x, y):
            return False
        tile_id = self.tiles[y, x]
        tile_type = TILE_TYPES.get(tile_id)
        return tile_type.walkable if tile_type else False

    def is_transparent(self, x: int, y: int) -> bool:
        """Checks if the tile at (x, y) is transparent (for FOV)."""
        if not self.in_bounds(x, y):
            # Treat out of bounds as non-transparent for FOV calculations
            return False
        # Directly use the cached transparency map
        return self.transparent[y, x]

    # --- Memory fade helper ---
    def update_memory_fade(
        self, current_time: int, steepness: float, midpoint: float, duration: float
    ) -> None:
        """Fade remembered tiles based on elapsed time."""
        update_memory_fade(
            current_time,
            self.last_seen_time,
            self.memory_intensity,
            self.visible,
            self.memory_fade_mask,
            self.prev_visible,

            self.memory_strength,

            steepness,
            midpoint,
            duration,
        )

    # --- MODIFIED compute_fov method ---
    def compute_fov(self, x: int, y: int, radius: int) -> None:
        """Calculate field of view from ``(x, y)`` with the given ``radius``."""
        log_context = {"origin": (x, y), "radius": radius}
        if not self.in_bounds(x, y):
            log.warning(
                "FOV origin out of bounds in GameMap.compute_fov", **log_context
            )
            self.visible.fill(False)
            return

        def blocks_light(tx: int, ty: int) -> bool:
            return not self.in_bounds(tx, ty) or not self.transparent[ty, tx]

        def set_visible(tx: int, ty: int) -> None:
            if self.in_bounds(tx, ty):
                self.visible[ty, tx] = True
                self.explored[ty, tx] = True
                self.memory_strength[ty, tx] = np.minimum(
                    self.memory_strength[ty, tx] + 1.0, MAX_MEMORY_STRENGTH
                )

        def get_distance(dx: int, dy: int) -> float:
            return math.hypot(dx, dy)

        visibility = MyVisibility(
            blocks_light=blocks_light,
            set_visible=set_visible,
            get_distance=get_distance,
        )

        self.visible.fill(False)
        visibility.compute(x, y, radius)

        # Decrement memory strength for tiles not currently visible and
        # ensure values stay within the valid range.
        not_visible = ~self.visible
        if np.any(not_visible):
            self.memory_strength[not_visible] = np.maximum(
                self.memory_strength[not_visible] - 1.0, 0.0
            )

        np.clip(self.memory_strength, 0.0, MAX_MEMORY_STRENGTH, out=self.memory_strength)

    # --- END MODIFIED compute_fov method ---

    def create_test_room(self) -> None:
        """Creates a simple rectangular room for testing."""
        room_x, room_y = self.width // 4, self.height // 4
        room_w, room_h = self.width // 2, self.height // 2

        # Ensure indices are within bounds for slicing
        x_start = max(0, room_x)
        y_start = max(0, room_y)
        x_end = min(self.width, room_x + room_w)
        y_end = min(self.height, room_y + room_h)

        # Assign FLOOR tile ID to the room area
        self.tiles[y_start:y_end, x_start:x_end] = TILE_ID_FLOOR

        # --- Assign default height/ceiling to test room ---
        test_floor_height = 0
        test_ceiling_height = 6 # e.g., 3 meters high
        self.height_map[y_start:y_end, x_start:x_end] = test_floor_height
        self.ceiling_map[y_start:y_end, x_start:x_end] = test_ceiling_height
        # --- End Assignment ---

        # Update transparency after changing tiles
        self.update_tile_transparency()
        log.info(
            "Created test room",
            x_start=x_start,
            y_start=y_start,
            x_end=x_end,
            y_end=y_end,
            floor_h=test_floor_height,
            ceil_h=test_ceiling_height,
        )
        transparent_count = np.sum(self.transparent)
        log.info("Map contains transparent tiles", count=transparent_count)

    def update_fov_with_tracking(
        self, x: int, y: int, radius: int
    ) -> Set[Tuple[int, int]]: # Use Tuple, Set imported
        """
        Updates FOV and returns a set of (x, y) coordinates where
        visibility changed (either became visible or hidden).
        """
        previous_visible = self.visible.copy()
        self.compute_fov(x, y, radius) # Calls the updated method
        changed_positions = set()

        # Optimization: Use np.where for potentially faster comparison on large maps
        diff_indices = np.argwhere(previous_visible != self.visible)
        for y_idx, x_idx in diff_indices:
            changed_positions.add((int(x_idx), int(y_idx))) # Store as (x, y)

        if changed_positions:
            log.debug("Visibility changed", changed_count=len(changed_positions))
        return changed_positions

# game/world/game_map.py
import numpy as np
from typing import NamedTuple, Final

# Import the FOV function
from game.world.fov import compute_fov  # Assuming fov.py is in the same directory

TILE_ID_FLOOR: Final[int] = 0
TILE_ID_WALL: Final[int] = 1


class TileType(NamedTuple):
    walkable: bool
    transparent: bool  # ADDED for FOV
    tile_index: int
    color_fg: tuple[int, int, int]
    color_bg: tuple[int, int, int]


TILE_TYPES: Final[dict[int, TileType]] = {
    TILE_ID_FLOOR: TileType(
        walkable=True,
        transparent=True,  # Floors are transparent
        tile_index=2,
        color_fg=(200, 200, 200),
        color_bg=(10, 10, 30),
    ),
    TILE_ID_WALL: TileType(
        walkable=False,
        transparent=False,  # Walls block FOV
        tile_index=38,
        color_fg=(180, 180, 180),
        color_bg=(30, 30, 50),
    ),
    # Add other tile types here with appropriate transparency
}


# Precompute transparency map for faster FOV lookups
def get_transparency_map(tiles: np.ndarray) -> np.ndarray:
    # Array shape is (height, width)
    transparency = np.zeros_like(tiles, dtype=bool)
    for tile_id, tile_type in TILE_TYPES.items():
        # Use boolean indexing which works correctly regardless of order
        transparency[tiles == tile_id] = tile_type.transparent
    return transparency


class GameMap:
    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Map width and height must be positive integers.")

        self._width = width
        self._height = height
        # CORRECTED SHAPE: (height, width), CORRECTED ORDER: 'C' (default)
        self.tiles: np.ndarray = np.full(
            (height, width), fill_value=TILE_ID_WALL, dtype=np.uint8, order="C"
        )
        self.explored: np.ndarray = np.zeros((height, width), dtype=bool, order="C")
        self.visible: np.ndarray = np.zeros((height, width), dtype=bool, order="C")
        # Precompute transparency for FOV
        self.transparent: np.ndarray = get_transparency_map(self.tiles)

    def update_tile_transparency(self) -> None:
        """Recalculates the transparency map based on current tiles."""
        # Array shape is (height, width)
        # CORRECTED SHAPE: (height, width)
        self.transparent = np.zeros((self._height, self._width), dtype=bool)

        # Then set transparency based on tile types
        for tile_id, tile_type in TILE_TYPES.items():
            # Use boolean indexing
            self.transparent[self.tiles == tile_id] = tile_type.transparent

        print(f"Transparency map updated: {np.sum(self.transparent)} transparent tiles")

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def in_bounds(self, x: int, y: int) -> bool:
        """Checks if coordinates are within the map boundaries."""
        # Keep x for width check, y for height check
        return 0 <= x < self._width and 0 <= y < self._height

    def is_walkable(self, x: int, y: int) -> bool:
        """Checks if the tile at (x, y) allows movement."""
        if not self.in_bounds(x, y):
            return False
        # CORRECTED INDEXING: [y, x]
        tile_id = self.tiles[y, x]
        tile_type = TILE_TYPES.get(tile_id)
        # Ensure tile_type is not None before accessing walkable
        return tile_type.walkable if tile_type else False

    # ADDED: Helper for FOV
    def is_transparent(self, x: int, y: int) -> bool:
        """Checks if the tile at (x, y) allows light to pass through."""
        if not self.in_bounds(x, y):
            return False  # Treat out of bounds as blocking light

        # Use the precomputed transparency map
        # CORRECTED INDEXING: [y, x]
        return self.transparent[y, x]

    def compute_fov(self, x: int, y: int, radius: int) -> None:
        """Calculates FOV from (x, y) using the imported function."""
        # Ensure transparency map is up-to-date if tiles could have changed
        # self.update_tile_transparency() # Uncomment if map tiles change dynamically
        compute_fov(self, x, y, radius)

    def create_test_room(self) -> None:
        """Generates a simple test room by setting floor tiles."""
        room_x, room_y = self.width // 4, self.height // 4
        room_w, room_h = self.width // 2, self.height // 2

        # Keep these coordinates as they define the rectangular region
        x_start = max(0, room_x)
        y_start = max(0, room_y)
        x_end = min(self.width, room_x + room_w)
        y_end = min(self.height, room_y + room_h)

        # Set floor tiles using correct slicing [row_slice, col_slice] -> [y_slice, x_slice]
        # CORRECTED SLICING: [y_start:y_end, x_start:x_end]
        self.tiles[y_start:y_end, x_start:x_end] = TILE_ID_FLOOR

        # Update transparency map after changing tiles
        self.update_tile_transparency()
        print(f"Created test room from ({x_start},{y_start}) to ({x_end},{y_end})")
        print(f"Map contains {np.sum(self.transparent)} transparent tiles")

    def update_fov_with_tracking(
        self, x: int, y: int, radius: int
    ) -> set[tuple[int, int]]:
        """
        Calculate FOV and return the set of positions where visibility changed.
        """
        # Store the previous visibility map
        previous_visible = self.visible.copy()  # Shape (height, width)

        # Calculate new FOV (modifies self.visible in place)
        compute_fov(self, x, y, radius)

        # Find tiles where visibility changed using correct iteration/indexing
        changed_positions = set()
        # Iterate height (rows) then width (cols)
        for map_y in range(self.height):
            for map_x in range(self.width):
                # CORRECTED INDEXING: [map_y, map_x]
                if previous_visible[map_y, map_x] != self.visible[map_y, map_x]:
                    changed_positions.add((map_x, map_y))  # Store as (x, y)

        return changed_positions

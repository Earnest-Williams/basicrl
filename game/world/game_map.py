# game/world/game_map.py
import numpy as np
from typing import NamedTuple, Final, Set, Tuple  # Ensure Set, Tuple are imported
import structlog

# Ensure correct import for the *new* FOV function
from game.world.fov import compute_fov

log = structlog.get_logger()

TILE_ID_FLOOR: Final[int] = 0
TILE_ID_WALL: Final[int] = 1


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
        tile_index=2,
        color_fg=(200, 200, 200),
        color_bg=(10, 10, 30),
    ),
    TILE_ID_WALL: TileType(
        walkable=False,
        transparent=False,
        tile_index=38,
        color_fg=(180, 180, 180),
        color_bg=(30, 30, 50),
    ),
}


def get_transparency_map(tiles: np.ndarray) -> np.ndarray:
    # (Implementation unchanged)
    transparency = np.zeros_like(tiles, dtype=bool)
    for tile_id, tile_type in TILE_TYPES.items():
        transparency[tiles == tile_id] = tile_type.transparent
    return transparency


class GameMap:
    def __init__(self, width: int, height: int):
        # (Implementation unchanged)
        if width <= 0 or height <= 0:
            log.error("Invalid map dimensions", width=width, height=height)
            raise ValueError("Map width and height must be positive integers.")
        self._width = width
        self._height = height
        log.info("Initializing GameMap", width=self._width, height=self._height)
        self.tiles: np.ndarray = np.full(
            (height, width), fill_value=TILE_ID_WALL, dtype=np.uint8, order="C"
        )
        self.explored: np.ndarray = np.zeros((height, width), dtype=bool, order="C")
        self.visible: np.ndarray = np.zeros((height, width), dtype=bool, order="C")
        self.transparent: np.ndarray = get_transparency_map(self.tiles)
        self.height_map: np.ndarray = np.zeros(
            (height, width), dtype=np.int16, order="C"
        )
        self.ceiling_map: np.ndarray = np.zeros(
            (height, width), dtype=np.int16, order="C"
        )
        log.debug("GameMap arrays initialized", shape=(height, width))

    def update_tile_transparency(self) -> None:
        # (Implementation unchanged)
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
        return 0 <= x < self._width and 0 <= y < self._height

    def is_walkable(self, x: int, y: int) -> bool:
        # (Implementation unchanged)
        if not self.in_bounds(x, y):
            return False
        tile_id = self.tiles[y, x]
        tile_type = TILE_TYPES.get(tile_id)
        return tile_type.walkable if tile_type else False

    def is_transparent(self, x: int, y: int) -> bool:
        # (Implementation unchanged)
        if not self.in_bounds(x, y):
            return False
        return self.transparent[y, x]

    # --- MODIFIED compute_fov method ---
    def compute_fov(self, x: int, y: int, radius: int) -> None:
        """
        Calculates FOV from (x, y) by calling the updated FOV function,
        passing all necessary map data correctly (including explored map).
        """
        log_context = {"origin": (x, y), "radius": radius}
        if not self.in_bounds(x, y):
            log.warning(
                "FOV origin out of bounds in GameMap.compute_fov", **log_context
            )
            self.visible.fill(False)  # Clear visibility if origin invalid
            return

        # Prepare arguments for the imported compute_fov function
        origin_xy = (x, y)
        range_limit = radius
        opaque_grid = ~self.transparent  # Invert transparency map
        height_map = self.height_map
        ceiling_map = self.ceiling_map
        visible_grid = self.visible  # Pass visible grid to be modified
        explored_grid = self.explored  # Pass explored grid to be modified

        # No need to clear visible_grid here, compute_fov handles it.
        # explored_grid should persist between calls.

        try:
            origin_height = int(height_map[y, x])

            # Call the imported FOV function with all arguments
            compute_fov(
                origin_xy=origin_xy,
                range_limit=range_limit,
                opaque_grid=opaque_grid,
                height_map=height_map,
                ceiling_map=ceiling_map,
                origin_height=origin_height,
                visible_grid=visible_grid,  # Pass visible grid
                explored_grid=explored_grid,  # Pass explored grid
            )

        except IndexError:
            log.error(
                "IndexError getting origin height in GameMap.compute_fov", **log_context
            )
            self.visible.fill(False)
            if self.in_bounds(x, y):
                self.visible[y, x] = True
        except Exception as e:
            log.error(
                "Unexpected error during compute_fov call",
                error=str(e),
                exc_info=True,
                **log_context
            )
            self.visible.fill(False)
            if self.in_bounds(x, y):
                self.visible[y, x] = True

    # --- END MODIFIED compute_fov method ---

    def create_test_room(self) -> None:
        # (Implementation unchanged)
        room_x, room_y = self.width // 4, self.height // 4
        room_w, room_h = self.width // 2, self.height // 2
        x_start = max(0, room_x)
        y_start = max(0, room_y)
        x_end = min(self.width, room_x + room_w)
        y_end = min(self.height, room_y + room_h)
        self.tiles[y_start:y_end, x_start:x_end] = TILE_ID_FLOOR
        test_floor_height = 0
        test_ceiling_height = 6
        self.height_map[y_start:y_end, x_start:x_end] = test_floor_height
        self.ceiling_map[y_start:y_end, x_start:x_end] = test_ceiling_height
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
    ) -> Set[Tuple[int, int]]:  # Use Tuple, Set imported
        # (Implementation unchanged, calls self.compute_fov which is now updated)
        previous_visible = self.visible.copy()
        self.compute_fov(x, y, radius)  # Calls the updated method
        changed_positions = set()
        # Optimization: Use np.where for potentially faster comparison on large maps
        diff_indices = np.argwhere(previous_visible != self.visible)
        for y_idx, x_idx in diff_indices:
            changed_positions.add((int(x_idx), int(y_idx)))  # Store as (x, y)

        if changed_positions:
            log.debug("Visibility changed", changed_count=len(changed_positions))
        return changed_positions

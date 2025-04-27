# game/world/fov.py
"""
Field of View (FOV) calculation using a Numba-accelerated
recursive shadowcasting algorithm adapted from C# source.
"""
import numpy as np
import math
from typing import TYPE_CHECKING, NamedTuple

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: Numba not installed. FOV calculation will be significantly slower.")

    # Define a dummy decorator if Numba is not available
    def njit(func=None, **options):
        if func:
            return func
        else:

            def decorator(f):
                return f

            return decorator

    _NUMBA_AVAILABLE = False


if TYPE_CHECKING:
    from game.world.game_map import GameMap


# Using NamedTuple for slopes - Numba generally handles these well.
# Keep the class definition outside Numba contexts.
class Slope(NamedTuple):
    """Represents the slope Y/X as a rational number."""

    y: int  # Using int, assuming coordinates fit
    x: int


# --- Slope Comparison Logic (moved outside class for easier Numba use) ---
@njit(cache=True, fastmath=True, nogil=True)
def slope_greater(s_y: int, s_x: int, y_comp: int, x_comp: int) -> bool:
    """Returns True if slope s_y/s_x > y_comp / x_comp"""
    return s_y * x_comp > s_x * y_comp


@njit(cache=True, fastmath=True, nogil=True)
def slope_greater_or_equal(s_y: int, s_x: int, y_comp: int, x_comp: int) -> bool:
    """Returns True if slope s_y/s_x >= y_comp / x_comp"""
    return s_y * x_comp >= s_x * y_comp


@njit(cache=True, fastmath=True, nogil=True)
def slope_less(s_y: int, s_x: int, y_comp: int, x_comp: int) -> bool:
    """Returns True if slope s_y/s_x < y_comp / x_comp"""
    return s_y * x_comp < s_x * y_comp


# Uncomment if needed for symmetrical visibility option
# @njit(cache=True, fastmath=True, nogil=True)
# def slope_less_or_equal(s_y: int, s_x: int, y_comp: int, x_comp: int) -> bool:
#     """Returns True if slope s_y/s_x <= y_comp / x_comp"""
#     return s_y * x_comp <= s_x * y_comp
# ---------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _transform_octant(x_in: int, y_in: int, octant: int) -> tuple[int, int]:
    """Transforms relative octant coordinates (x, y) to absolute map offsets (dx, dy)."""
    if octant == 0:
        return x_in, -y_in
    if octant == 1:
        return y_in, -x_in
    if octant == 2:
        return -y_in, -x_in
    if octant == 3:
        return -x_in, -y_in
    if octant == 4:
        return -x_in, y_in
    if octant == 5:
        return -y_in, x_in
    if octant == 6:
        return y_in, x_in
    if octant == 7:
        return x_in, y_in
    return 0, 0  # Should be unreachable


@njit(cache=True, fastmath=True, nogil=True)
def _distance_sq(x: int, y: int) -> int:
    """Calculate squared Euclidean distance."""
    return x * x + y * y


@njit(cache=True, fastmath=True, nogil=True)
def _blocks_light_octant(
    x_oct: int,
    y_oct: int,
    octant: int,
    origin_x: int,
    origin_y: int,
    transparent_map: np.ndarray,  # Shape (height, width)
) -> bool:
    """Checks if the transformed tile blocks light."""
    # Array shape is (height, width)
    map_height, map_width = transparent_map.shape
    dx, dy = _transform_octant(x_oct, y_oct, octant)
    map_x, map_y = origin_x + dx, origin_y + dy

    # Check bounds first (using width/height derived from array shape)
    if 0 <= map_x < map_width and 0 <= map_y < map_height:
        # Return True if it blocks light (i.e., is NOT transparent)
        # CORRECTED INDEXING: [map_y, map_x]
        return not transparent_map[map_y, map_x]
    else:
        # Out of bounds always blocks light
        return True


@njit(cache=True, fastmath=True, nogil=True)
def _set_visible_octant(
    x_oct: int,
    y_oct: int,
    octant: int,
    origin_x: int,
    origin_y: int,
    visible_map: np.ndarray,  # Shape (height, width)
    explored_map: np.ndarray,  # Shape (height, width)
) -> None:
    """Sets the transformed tile to visible and explored."""
    # Array shape is (height, width)
    map_height, map_width = visible_map.shape
    dx, dy = _transform_octant(x_oct, y_oct, octant)
    map_x, map_y = origin_x + dx, origin_y + dy

    # Set visible only if within bounds
    if 0 <= map_x < map_width and 0 <= map_y < map_height:
        # CORRECTED INDEXING: [map_y, map_x]
        visible_map[map_y, map_x] = True
        explored_map[map_y, map_x] = True


@njit(cache=True, fastmath=True, nogil=True)
def _compute_octant(
    octant: int,
    origin_x: int,
    origin_y: int,
    range_limit_sq: int,
    x_start: int,
    # Pass slope components directly for Numba compatibility
    top_y_slope: int,
    top_x_slope: int,
    bottom_y_slope: int,
    bottom_x_slope: int,
    transparent_map: np.ndarray,  # Shape (height, width)
    visible_map: np.ndarray,  # Shape (height, width)
    explored_map: np.ndarray,  # Shape (height, width)
) -> None:
    """
    Numba-accelerated recursive shadowcasting computation for a single octant.
    Directly modifies visible_map and explored_map. Assumes arrays are (height, width).
    """
    # Need mutable slope values within the function scope for updates
    current_top_y, current_top_x = top_y_slope, top_x_slope
    current_bottom_y, current_bottom_x = bottom_y_slope, bottom_x_slope

    # Calculate maximum relevant x based on range limit
    max_x = (
        int(math.sqrt(max(0, range_limit_sq))) + 2 if range_limit_sq >= 0 else 2**16
    )

    for x in range(x_start, max_x):
        # --- Calculate Top Y ---
        top_y: int
        if current_top_x == 1:
            top_y = x
        else:
            top_y = ((x * 2 - 1) * current_top_y + current_top_x) // (current_top_x * 2)
            # Pass arrays with correct indexing expectations to helper
            if _blocks_light_octant(
                x, top_y, octant, origin_x, origin_y, transparent_map
            ):
                if slope_greater_or_equal(
                    current_top_y, current_top_x, top_y * 2 + 1, x * 2
                ) and not _blocks_light_octant(
                    x, top_y + 1, octant, origin_x, origin_y, transparent_map
                ):
                    top_y += 1
            else:
                ax = x * 2
                if _blocks_light_octant(
                    x + 1, top_y + 1, octant, origin_x, origin_y, transparent_map
                ):
                    ax += 1
                if slope_greater(current_top_y, current_top_x, top_y * 2 + 1, ax):
                    top_y += 1

        # --- Calculate Bottom Y ---
        bottom_y: int
        if current_bottom_y == 0:
            bottom_y = 0
        else:
            bottom_y = ((x * 2 - 1) * current_bottom_y + current_bottom_x) // (
                current_bottom_x * 2
            )
            # Pass arrays with correct indexing expectations to helper
            if (
                slope_greater_or_equal(
                    current_bottom_y, current_bottom_x, bottom_y * 2 + 1, x * 2
                )
                and _blocks_light_octant(
                    x, bottom_y, octant, origin_x, origin_y, transparent_map
                )
                and not _blocks_light_octant(
                    x, bottom_y + 1, octant, origin_x, origin_y, transparent_map
                )
            ):
                bottom_y += 1

        # --- Process Tiles in Column ---
        was_opaque: int = -1
        for y in range(top_y, bottom_y - 1, -1):
            current_dist_sq = _distance_sq(x, y)
            if range_limit_sq >= 0 and current_dist_sq > range_limit_sq:
                continue

            # Pass arrays with correct indexing expectations to helper
            is_opaque = _blocks_light_octant(
                x, y, octant, origin_x, origin_y, transparent_map
            )

            is_visible = is_opaque or (
                (
                    y != top_y
                    or slope_greater(current_top_y, current_top_x, y * 4 - 1, x * 4 + 1)
                )
                and (
                    y != bottom_y
                    or slope_less(
                        current_bottom_y, current_bottom_x, y * 4 + 1, x * 4 - 1
                    )
                )
            )

            if is_visible:
                # Pass arrays with correct indexing expectations to helper
                _set_visible_octant(
                    x, y, octant, origin_x, origin_y, visible_map, explored_map
                )

            if range_limit_sq < 0 or x * x < range_limit_sq:
                if is_opaque:
                    if was_opaque == 0:
                        nx, ny = x * 2, y * 2 + 1
                        # Pass arrays with correct indexing expectations to helper
                        if _blocks_light_octant(
                            x, y + 1, octant, origin_x, origin_y, transparent_map
                        ):
                            nx -= 1
                        if slope_greater(current_top_y, current_top_x, ny, nx):
                            if y == bottom_y:
                                current_bottom_y, current_bottom_x = ny, nx
                                break
                            else:
                                # RECURSIVE CALL: Pass arrays through
                                _compute_octant(
                                    octant,
                                    origin_x,
                                    origin_y,
                                    range_limit_sq,
                                    x + 1,
                                    current_top_y,
                                    current_top_x,
                                    ny,
                                    nx,
                                    transparent_map,
                                    visible_map,
                                    explored_map,
                                )
                        elif y == bottom_y:
                            return
                    was_opaque = 1
                else:  # Tile is clear
                    if was_opaque > 0:
                        nx, ny = x * 2, y * 2 + 1
                        # Pass arrays with correct indexing expectations to helper
                        if _blocks_light_octant(
                            x + 1, y + 1, octant, origin_x, origin_y, transparent_map
                        ):
                            nx += 1
                        if slope_greater_or_equal(
                            current_bottom_y, current_bottom_x, ny, nx
                        ):
                            return
                        current_top_y, current_top_x = ny, nx
                    was_opaque = 0
        # --- End of Column Processing ---
        if was_opaque != 0:
            break


# --- Fallback Simple FOV (Corrected Indexing) ---
def simple_circle_fov(game_map, center_x, center_y, radius):
    """
    A very simple circular FOV algorithm that just shows all tiles within radius.
    Used as an emergency fallback to ensure visibility. Assumes map arrays are (height, width).
    """
    print(
        f"Using simple_circle_fov with center ({center_x}, {center_y}) and radius {radius}"
    )

    # Reset visibility (shape height, width)
    game_map.visible[:] = False

    radius_sq = radius * radius
    # Iterate rows (y) then columns (x)
    for y in range(
        max(0, center_y - radius), min(game_map.height, center_y + radius + 1)
    ):
        for x in range(
            max(0, center_x - radius), min(game_map.width, center_x + radius + 1)
        ):
            dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
            if dist_sq <= radius_sq:
                # Always mark origin visible
                if x == center_x and y == center_y:
                    # CORRECTED INDEXING: [y, x]
                    game_map.visible[y, x] = True
                    game_map.explored[y, x] = True
                # Use the map's is_transparent method which has correct indexing
                elif game_map.is_transparent(x, y):
                    # Simple raycast check (also needs correct indexing)
                    line_is_clear = True
                    dx = x - center_x
                    dy = y - center_y
                    distance = max(abs(dx), abs(dy))
                    if distance > 1:
                        for step in range(1, distance):
                            step_ratio = step / distance
                            check_x = int(center_x + dx * step_ratio)
                            check_y = int(center_y + dy * step_ratio)
                            # Use map's method for check
                            if not game_map.is_transparent(check_x, check_y):
                                line_is_clear = False
                                break
                    if line_is_clear:
                        # CORRECTED INDEXING: [y, x]
                        game_map.visible[y, x] = True
                        game_map.explored[y, x] = True


# --- Main FOV Function (Corrected Array Passing) ---
def compute_fov(game_map: "GameMap", x: int, y: int, radius: int) -> None:
    """
    Calculate Field of View from point (x, y) with a given radius.
    Updates the game_map.visible and game_map.explored arrays in place
    using Numba-accelerated recursive shadowcasting. Assumes map arrays are (height, width).

    Args:
        game_map: The GameMap object containing map data arrays (shape height, width).
        x: The x-coordinate of the FOV origin (column index).
        y: The y-coordinate of the FOV origin (row index).
        radius: The maximum visibility radius. Use negative value (e.g., -1) for infinite range.
    """
    if not _NUMBA_AVAILABLE:
        print("WARNING: Numba not found, FOV will be very slow. Falling back.")
        simple_circle_fov(game_map, x, y, radius)  # Use fallback immediately
        return

    if not game_map.in_bounds(x, y):
        print(
            f"Warning: FOV origin ({x},{y}) is out of map bounds ({game_map.width}x{game_map.height})."
        )
        game_map.visible[:] = False
        return

    try:
        # Reset visibility grid (shape height, width)
        game_map.visible[:] = False

        # Mark origin as visible and explored
        # CORRECTED INDEXING: [y, x]
        game_map.visible[y, x] = True
        game_map.explored[y, x] = True

        radius_squared = radius * radius if radius >= 0 else -1

        # Pre-fetch arrays (already correct shape: height, width)
        transparent_map = game_map.transparent
        visible_map = game_map.visible
        explored_map = game_map.explored

        # Debugging transparency
        transparent_count = np.sum(transparent_map)
        # Corrected index
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy

        # Iterate through octants, calling the Numba-accelerated function
        for octant in range(8):
            _compute_octant(
                octant=octant,
                origin_x=x,
                origin_y=y,
                range_limit_sq=radius_squared,
                x_start=1,
                top_y_slope=1,
                top_x_slope=1,
                bottom_y_slope=0,
                bottom_x_slope=1,
                transparent_map=transparent_map,  # Pass (height, width) array
                visible_map=visible_map,  # Pass (height, width) array
                explored_map=explored_map,  # Pass (height, width) array
            )

        visible_count = np.sum(visible_map)

        if visible_count <= 1:
            print(
                "WARNING: Shadowcasting FOV failed (<=1 visible), using simple FOV fallback"
            )
            simple_circle_fov(game_map, x, y, radius)
            visible_count = np.sum(visible_map)
            print(f"Simple FOV fallback generated {visible_count} visible tiles")

    except Exception as e:
        print(f"ERROR in FOV calculation: {e}")
        import traceback

        traceback.print_exc()
        print("Using simple FOV fallback due to error")
        try:
            simple_circle_fov(game_map, x, y, radius)
        except Exception as e2:
            print(f"ERROR in simple FOV fallback: {e2}")
            if game_map.in_bounds(x, y):  # Check bounds before fallback assignment
                game_map.visible[y, x] = True  # Corrected index
                game_map.explored[y, x] = True  # Corrected index

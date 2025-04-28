# fov.py
# PEP 8 Compliant, PEP 604 Type Hints, structlog Logging
# Numba-accelerated Iterative Shadowcasting FOV with Height Checks
# Now updates explored map as well.

import time
import math
from collections import deque
from typing import TypeAlias

import numba
import numpy as np
import structlog
from numba.typed import List as NumbaList  # Explicitly typed list for Numba

# --- Type Aliases (PEP 604) ---
Point: TypeAlias = tuple[int, int]
Slope: TypeAlias = tuple[int, int]  # (y, x) representation
sector_type = numba.types.Tuple(
    (
        numba.int64,  # octant
        numba.int64,  # x
        numba.types.Tuple((numba.int64, numba.int64)),  # top slope (y, x)
        numba.types.Tuple((numba.int64, numba.int64)),  # bottom slope (y, x)
    )
)

# --- Structlog Configuration ---
log = structlog.get_logger()

# --- Height Check Heuristic Parameters ---
BASE_THRESHOLD: int = 1
CLOSE_RANGE_SQ_THRESHOLD: int = 16
CLOSE_RANGE_DIVISOR: int = 8
FAR_RANGE_DIVISOR: int = 16
_THRESHOLD_AT_CUTOFF: int = CLOSE_RANGE_SQ_THRESHOLD // CLOSE_RANGE_DIVISOR
# --- End Heuristic Parameters ---


# --- Numba Helper Functions ---
# (Slope functions unchanged)
@numba.njit(cache=True, inline="always")
def slope_greater(slope1_yx: Slope, y: int, x: int) -> bool:
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0:
        return slope_y > y
    if slope_x == 0:
        return True if x > 0 else False
    if x == 0:
        return False if slope_x > 0 else True
    return slope_y * x > slope_x * y


@numba.njit(cache=True, inline="always")
def slope_greater_or_equal(slope1_yx: Slope, y: int, x: int) -> bool:
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0:
        return slope_y >= y
    if slope_x == 0:
        return True if x >= 0 else False
    if x == 0:
        return False if slope_x >= 0 else True
    return slope_y * x >= slope_x * y


@numba.njit(cache=True, inline="always")
def slope_less(slope1_yx: Slope, y: int, x: int) -> bool:
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0:
        return slope_y < y
    if slope_x == 0:
        return False if x > 0 else True
    if x == 0:
        return True if slope_x > 0 else False
    return slope_y * x < slope_x * y


@numba.njit(cache=True, inline="always")
def slope_less_or_equal(slope1_yx: Slope, y: int, x: int) -> bool:
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0:
        return slope_y <= y
    if slope_x == 0:
        return False if x > 0 else True
    if x == 0:
        return True if slope_x > 0 else False
    return slope_y * x <= slope_x * y


@numba.njit(cache=True)
def _transform_coords(
    octant_x: int, octant_y: int, octant: int, origin_xy: Point
) -> Point:
    # (Unchanged)
    ox, oy = origin_xy
    nx, ny = ox, oy
    if octant == 0:
        nx += octant_x
        ny -= octant_y
    elif octant == 1:
        nx += octant_y
        ny -= octant_x
    elif octant == 2:
        nx -= octant_y
        ny -= octant_x
    elif octant == 3:
        nx -= octant_x
        ny -= octant_y
    elif octant == 4:
        nx -= octant_x
        ny += octant_y
    elif octant == 5:
        nx -= octant_y
        ny += octant_x
    elif octant == 6:
        nx += octant_y
        ny += octant_x
    elif octant == 7:
        nx += octant_x
        ny += octant_y
    return nx, ny


@numba.njit(cache=True)
def blocks_light_at(
    octant_x: int,
    octant_y: int,
    octant: int,
    origin_xy: Point,
    grid_shape: tuple[int, int],
    opaque_grid: np.ndarray,
    height_map: np.ndarray,
    ceiling_map: np.ndarray,
    origin_height: int,
) -> bool:
    # (Unchanged - includes height check)
    nx, ny = _transform_coords(octant_x, octant_y, octant, origin_xy)
    width, height = grid_shape
    if not (0 <= nx < width and 0 <= ny < height):
        return True
    if opaque_grid[ny, nx]:
        return True
    distance_squared = octant_x * octant_x + octant_y * octant_y
    vertical_threshold: int
    if distance_squared <= CLOSE_RANGE_SQ_THRESHOLD:
        vertical_threshold = BASE_THRESHOLD + (distance_squared // CLOSE_RANGE_DIVISOR)
    else:
        additional_distance_sq = distance_squared - CLOSE_RANGE_SQ_THRESHOLD
        vertical_threshold = (
            BASE_THRESHOLD
            + _THRESHOLD_AT_CUTOFF
            + (additional_distance_sq // FAR_RANGE_DIVISOR)
        )
    target_floor_h = height_map[ny, nx]
    target_ceil_h = ceiling_map[ny, nx]
    if target_floor_h > origin_height + vertical_threshold:
        return True
    if target_ceil_h < origin_height - vertical_threshold:
        return True
    return False


# --- MODIFIED set_visible_at signature ---
@numba.njit(cache=True)
def set_visible_at(
    octant_x: int,
    octant_y: int,
    octant: int,
    origin_xy: Point,
    grid_shape: tuple[int, int],
    visible_grid: np.ndarray,  # Output grid 1
    explored_grid: np.ndarray,  # Output grid 2 (ADDED)
):
    # --- END MODIFIED signature ---
    """Numba-optimized version modifying visibility and explored grids."""
    nx, ny = _transform_coords(octant_x, octant_y, octant, origin_xy)
    width, height = grid_shape
    if 0 <= nx < width and 0 <= ny < height:
        # --- ADDED explored_grid update ---
        visible_grid[ny, nx] = True
        explored_grid[ny, nx] = True  # Mark as explored when visible
        # --- END ADDED update ---


# --- END MODIFIED set_visible_at ---


@numba.njit(cache=True)
def is_in_range(octant_x: int, octant_y: int, range_limit_sq: int | float) -> bool:
    # (Unchanged)
    return (octant_x * octant_x + octant_y * octant_y) <= range_limit_sq


# --- MODIFIED Numba Core Logic signature ---
@numba.njit(cache=True)
def _compute_fov_iterative_numba(
    origin_xy: Point,
    range_limit: int,
    opaque_grid: np.ndarray,
    height_map: np.ndarray,
    ceiling_map: np.ndarray,
    origin_height: int,
    visible_grid: np.ndarray,
    explored_grid: np.ndarray,  # ADDED explored_grid parameter
):
    # --- END MODIFIED signature ---
    """
    Core iterative FOV calculation. Now passes explored_grid.
    """
    height, width = opaque_grid.shape
    grid_shape = (width, height)
    range_limit_sq = range_limit * range_limit
    active_sectors = NumbaList.empty_list(sector_type)
    initial_top: Slope = (1, 1)
    initial_bottom: Slope = (0, 1)
    for octant in range(8):
        active_sectors.append((octant, 1, initial_top, initial_bottom))

    while len(active_sectors) > 0:
        current_octant, current_x, current_top, current_bottom = active_sectors.pop()
        for x in range(current_x, range_limit + 1):
            # Calculate top_y and bottom_y (unchanged logic)
            top_y: int
            top_slope_y, top_slope_x = current_top
            # ... (top_y calculation unchanged) ...
            if top_slope_x == 1:
                top_y = x
            elif top_slope_x == 0:
                top_y = 0
            else:
                top_y = ((x * 2 - 1) * top_slope_y + top_slope_x) // (top_slope_x * 2)
                if blocks_light_at(
                    x,
                    top_y,
                    current_octant,
                    origin_xy,
                    grid_shape,
                    opaque_grid,
                    height_map,
                    ceiling_map,
                    origin_height,
                ):
                    if slope_greater_or_equal(
                        current_top, top_y * 2 + 1, x * 2
                    ) and not blocks_light_at(
                        x,
                        top_y + 1,
                        current_octant,
                        origin_xy,
                        grid_shape,
                        opaque_grid,
                        height_map,
                        ceiling_map,
                        origin_height,
                    ):
                        top_y += 1
                else:
                    ax = x * 2
                    if blocks_light_at(
                        x + 1,
                        top_y + 1,
                        current_octant,
                        origin_xy,
                        grid_shape,
                        opaque_grid,
                        height_map,
                        ceiling_map,
                        origin_height,
                    ):
                        ax += 1
                    if slope_greater(current_top, top_y * 2 + 1, ax):
                        top_y += 1
            bottom_y: int
            bottom_slope_y, bottom_slope_x = current_bottom
            if bottom_slope_y == 0:
                bottom_y = 0
            elif bottom_slope_x == 0:
                bottom_y = 0
            else:
                bottom_y = ((x * 2 - 1) * bottom_slope_y + bottom_slope_x) // (
                    bottom_slope_x * 2
                )  # Corrected formula
                if (
                    slope_greater_or_equal(current_bottom, bottom_y * 2 + 1, x * 2)
                    and blocks_light_at(
                        x,
                        bottom_y,
                        current_octant,
                        origin_xy,
                        grid_shape,
                        opaque_grid,
                        height_map,
                        ceiling_map,
                        origin_height,
                    )
                    and not blocks_light_at(
                        x,
                        bottom_y + 1,
                        current_octant,
                        origin_xy,
                        grid_shape,
                        opaque_grid,
                        height_map,
                        ceiling_map,
                        origin_height,
                    )
                ):
                    bottom_y += 1

            # Process column y range
            was_opaque = -1
            for y in range(top_y, bottom_y - 1, -1):
                if is_in_range(x, y, range_limit_sq):
                    is_tile_opaque = blocks_light_at(
                        x,
                        y,
                        current_octant,
                        origin_xy,
                        grid_shape,
                        opaque_grid,
                        height_map,
                        ceiling_map,
                        origin_height,
                    )
                    top_vis_check = y != top_y or slope_greater(
                        current_top, y * 4 - 1, x * 4 + 1
                    )
                    bottom_vis_check = y != bottom_y or slope_less(
                        current_bottom, y * 4 + 1, x * 4 - 1
                    )
                    is_visible = is_tile_opaque or (top_vis_check and bottom_vis_check)

                    if is_visible:
                        # --- Pass explored_grid to set_visible_at ---
                        set_visible_at(
                            x,
                            y,
                            current_octant,
                            origin_xy,
                            grid_shape,
                            visible_grid,
                            explored_grid,
                        )
                        # --- End Pass ---

                    # Transition Logic (with correct beveling, unchanged logic)
                    if x < range_limit:
                        if is_tile_opaque:
                            if was_opaque == 0:  # Clear -> Opaque
                                nx = x * 2
                                ny = y * 2 + 1
                                if blocks_light_at(
                                    x,
                                    y + 1,
                                    current_octant,
                                    origin_xy,
                                    grid_shape,
                                    opaque_grid,
                                    height_map,
                                    ceiling_map,
                                    origin_height,
                                ):
                                    nx -= 1
                                new_bottom_slope: Slope = (ny, nx)
                                if slope_greater(current_top, ny, nx):
                                    active_sectors.append(
                                        (
                                            current_octant,
                                            x + 1,
                                            current_top,
                                            new_bottom_slope,
                                        )
                                    )
                                current_bottom = new_bottom_slope
                                break
                            was_opaque = 1
                        else:  # Clear tile
                            if was_opaque > 0:  # Opaque -> Clear
                                nx = x * 2
                                ny = y * 2 + 1
                                if blocks_light_at(
                                    x + 1,
                                    y + 1,
                                    current_octant,
                                    origin_xy,
                                    grid_shape,
                                    opaque_grid,
                                    height_map,
                                    ceiling_map,
                                    origin_height,
                                ):
                                    nx += 1
                                new_top_slope: Slope = (ny, nx)
                                if slope_greater_or_equal(current_bottom, ny, nx):
                                    was_opaque = -2
                                    break
                                current_top = new_top_slope
                            was_opaque = 0
            # End of y loop
            if was_opaque == 1 or was_opaque == -2:
                break  # Exit x loop
        # End of x loop
    # End of while loop


# --- END MODIFIED _compute_fov_iterative_numba ---


# --- MODIFIED Public Interface Function Signature ---
def compute_fov(
    origin_xy: Point,
    range_limit: int,
    opaque_grid: np.ndarray,
    height_map: np.ndarray,
    ceiling_map: np.ndarray,
    origin_height: int,
    # Accept BOTH output arrays
    visible_grid: np.ndarray,
    explored_grid: np.ndarray,  # ADDED explored_grid parameter
) -> None:  # Returns None, modifies arrays in-place
    # --- END MODIFIED signature ---
    """
    Computes Field of View using the iterative shadowcasting algorithm.
    Modifies visible_grid and explored_grid in-place.
    """
    func_log = log.bind(
        origin=origin_xy, range_limit=range_limit, grid_shape=opaque_grid.shape
    )
    func_log.info("Starting FOV computation (Iterative w/ Explored)")
    start_time = time.perf_counter()

    # Input validation
    if not isinstance(opaque_grid, np.ndarray) or opaque_grid.ndim != 2:
        raise TypeError("opaque_grid must be a 2D NumPy array.")
    if height_map.shape != opaque_grid.shape:
        raise ValueError("height_map must have same shape.")
    if ceiling_map.shape != opaque_grid.shape:
        raise ValueError("ceiling_map must have same shape.")
    if visible_grid.shape != opaque_grid.shape:
        raise ValueError("visible_grid must have same shape.")
    if explored_grid.shape != opaque_grid.shape:
        raise ValueError("explored_grid must have same shape.")
    if visible_grid.dtype != np.bool_:
        raise TypeError("visible_grid must be boolean.")
    if explored_grid.dtype != np.bool_:
        raise TypeError("explored_grid must be boolean.")
    if range_limit < 0:
        raise ValueError("range_limit must be non-negative.")

    height, width = opaque_grid.shape
    ox, oy = origin_xy
    if not (0 <= ox < width and 0 <= oy < height):
        raise ValueError("Origin coordinates are outside grid bounds.")

    # Clear *only* the visible grid, leave explored grid intact
    visible_grid.fill(False)
    visible_grid[oy, ox] = True
    explored_grid[oy, ox] = True  # Also ensure origin is explored

    # Origin height fetch (already validated origin bounds)
    # Removed internal _origin_height variable, use parameter directly
    # origin_height = int(height_map[oy, ox])

    # Call the Numba core logic
    try:
        _compute_fov_iterative_numba(
            origin_xy,
            range_limit,
            opaque_grid.astype(np.bool_),
            height_map,
            ceiling_map,
            origin_height,
            visible_grid,
            explored_grid,  # Pass both output arrays
        )
    except Exception as e:
        log.error("Error during Numba FOV calculation", error=str(e), exc_info=True)
        # Consider fallback or re-raising depending on desired robustness

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    # Log visible count from the grid that was modified
    func_log.info(
        "FOV computation finished",
        duration_ms=f"{duration_ms:.2f}",
        visible_count=np.sum(visible_grid),
    )
    # No return value needed as arrays are modified in-place


# --- END MODIFIED Public Interface ---


# --- Example Usage (Updated call) ---
if __name__ == "__main__":
    MAP_WIDTH = 30
    MAP_HEIGHT = 20
    map_opaque = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.bool_)
    map_height = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)
    map_ceiling = np.full((MAP_HEIGHT, MAP_WIDTH), 10, dtype=np.int16)
    map_opaque[0, :] = True
    map_opaque[MAP_HEIGHT - 1, :] = True
    map_opaque[:, 0] = True
    map_opaque[:, MAP_WIDTH - 1] = True
    map_height[0, :] = 1
    map_height[MAP_HEIGHT - 1, :] = 1
    map_height[:, 0] = 1
    map_height[:, MAP_WIDTH - 1] = 1
    map_opaque[MAP_HEIGHT // 2, 5 : MAP_WIDTH - 5] = True
    map_height[MAP_HEIGHT // 2, 5 : MAP_WIDTH - 5] = 4
    map_ceiling[MAP_HEIGHT // 2, 5 : MAP_WIDTH - 5] = 8
    map_opaque[5 : MAP_HEIGHT - 5, MAP_WIDTH // 2] = True
    map_height[5 : MAP_HEIGHT - 5, MAP_WIDTH // 2] = 0
    map_ceiling[5 : MAP_HEIGHT - 5, MAP_WIDTH // 2] = 10
    map_height[MAP_HEIGHT // 2 + 3 : MAP_HEIGHT // 2 + 6, 5:10] = 4
    map_ceiling[MAP_HEIGHT // 2 + 3 : MAP_HEIGHT // 2 + 6, 5:10] = 14
    map_ceiling[2:5, MAP_WIDTH - 10 : MAP_WIDTH - 5] = 2
    map_height[2:5, MAP_WIDTH - 10 : MAP_WIDTH - 5] = 0
    player_origin: Point = (MAP_WIDTH // 4, MAP_HEIGHT // 4)
    map_height[player_origin[1], player_origin[0]] = 0
    map_ceiling[player_origin[1], player_origin[0]] = 10
    view_radius = 12

    # --- Create both output arrays ---
    visibility_map = np.zeros_like(map_opaque, dtype=np.bool_)
    explored_map = np.zeros_like(map_opaque, dtype=np.bool_)  # Create explored map

    print("Running Optimized Iterative FOV with Height & Explored...")
    example_origin_height = int(map_height[player_origin[1], player_origin[0]])
    # --- Pass explored_map to compute_fov ---
    compute_fov(
        player_origin,
        view_radius,
        map_opaque,
        map_height,
        map_ceiling,
        example_origin_height,
        visible_grid=visibility_map,  # Pass visible grid
        explored_grid=explored_map,  # Pass explored grid
    )
    print("Done.")

    # --- Print Output (show explored) ---
    print(f"\nMap Legend: @=Player (H=0), #=Wall, .=Visible Floor, '=Explored Floor")
    print(f"            ^=Vis Raised Floor, ~=Vis Low Ceiling, X=Explored Wall")
    output_str = ""
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            is_visible = visibility_map[y, x]
            is_explored = explored_map[y, x]
            is_opaque_tile = map_opaque[y, x]
            h = map_height[y, x]
            c = map_ceiling[y, x]

            if (x, y) == player_origin:
                char = "@"
            elif is_visible:
                if is_opaque_tile:
                    char = "#"  # Visible wall (should be rare unless inside wall?)
                elif h >= 4:
                    char = "^"
                elif c <= 2:
                    char = "~"
                else:
                    char = "."
            elif is_explored:
                if is_opaque_tile:
                    char = "X"  # Explored wall
                else:
                    char = "'"  # Explored floor
            else:
                char = " "  # Unseen
            output_str += char
        output_str += "\n"
    print(output_str)
    # --- End Print ---

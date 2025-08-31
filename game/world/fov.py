# game/world/fov.py
"""
Field of View (FOV) and Line of Sight (LOS) calculations.
Uses Numba-accelerated Iterative Shadowcasting for FOV and Bresenham for LOS.
Includes height/ceiling checks and explored tile tracking.
"""

import time
import math
from collections import deque
from typing import TypeAlias, List, Optional

import numba
import numpy as np
import structlog
from numba.typed import List as NumbaList

# --- Type Aliases ---
Point: TypeAlias = tuple[int, int]
Slope: TypeAlias = tuple[int, int]  # (y, x) representation

# Numba type definitions
sector_type = numba.types.Tuple((
    numba.int64,  # octant
    numba.int64,  # x
    numba.types.Tuple((numba.int64, numba.int64)),  # top slope (y, x)
    numba.types.Tuple((numba.int64, numba.int64)),  # bottom slope (y, x)
))

# --- Logging Setup ---
log = structlog.get_logger(__name__)

# --- Configuration Constants ---
BASE_THRESHOLD: int = 1
CLOSE_RANGE_SQ_THRESHOLD: int = 16
CLOSE_RANGE_DIVISOR: int = 8
FAR_RANGE_DIVISOR: int = 16
_THRESHOLD_AT_CUTOFF: int = CLOSE_RANGE_SQ_THRESHOLD // CLOSE_RANGE_DIVISOR
MAX_SECTORS: int = 10000  # Safety limit for sector processing

# --- Numba Helper Functions ---
@numba.njit(cache=True, inline="always")
def slope_greater(slope1_yx: Slope, y: int, x: int) -> bool:
    """Check if slope1 is greater than the slope to point (y,x)."""
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0: return slope_y > y
    if slope_x == 0: return True if x > 0 else False
    if x == 0: return False if slope_x > 0 else True
    return slope_y * x > slope_x * y

@numba.njit(cache=True, inline="always")
def slope_greater_or_equal(slope1_yx: Slope, y: int, x: int) -> bool:
    """Check if slope1 is greater than or equal to the slope to point (y,x)."""
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0: return slope_y >= y
    if slope_x == 0: return True if x >= 0 else False
    if x == 0: return False if slope_x >= 0 else True
    return slope_y * x >= slope_x * y

@numba.njit(cache=True, inline="always")
def slope_less(slope1_yx: Slope, y: int, x: int) -> bool:
    """Check if slope1 is less than the slope to point (y,x)."""
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0: return slope_y < y
    if slope_x == 0: return False if x > 0 else True
    if x == 0: return True if slope_x > 0 else False
    return slope_y * x < slope_x * y

@numba.njit(cache=True, inline="always")
def slope_less_or_equal(slope1_yx: Slope, y: int, x: int) -> bool:
    """Check if slope1 is less than or equal to the slope to point (y,x)."""
    slope_y, slope_x = slope1_yx
    if slope_x == 0 and x == 0: return slope_y <= y
    if slope_x == 0: return False if x > 0 else True
    if x == 0: return True if slope_x > 0 else False
    return slope_y * x <= slope_x * y

@numba.njit(cache=True)
def _transform_coords(octant_x: int, octant_y: int, octant: int, origin_xy: Point) -> Point:
    """Transform coordinates based on octant."""
    ox, oy = origin_xy
    nx, ny = ox, oy
    if octant == 0: nx += octant_x; ny -= octant_y
    elif octant == 1: nx += octant_y; ny -= octant_x
    elif octant == 2: nx -= octant_y; ny -= octant_x
    elif octant == 3: nx -= octant_x; ny -= octant_y
    elif octant == 4: nx -= octant_x; ny += octant_y
    elif octant == 5: nx -= octant_y; ny += octant_x
    elif octant == 6: nx += octant_y; ny += octant_x
    elif octant == 7: nx += octant_x; ny += octant_y
    return nx, ny

@numba.njit(cache=True)
def blocks_light_at(
    octant_x: int, octant_y: int, octant: int, origin_xy: Point,
    grid_shape: tuple[int, int], opaque_grid: np.ndarray,
    height_map: np.ndarray, ceiling_map: np.ndarray, origin_height: int
) -> bool:
    """Check if a tile blocks light, considering bounds, opacity, and height."""
    nx, ny = _transform_coords(octant_x, octant_y, octant, origin_xy)
    width, height = grid_shape
    
    if not (0 <= nx < width and 0 <= ny < height):
        return True
    
    if opaque_grid[ny, nx]:
        return True
    
    # Height and ceiling checks
    target_h = height_map[ny, nx]
    target_ceiling = ceiling_map[ny, nx]
    
    if target_ceiling <= origin_height:
        return True  # Ceiling too low
    
    # Calculate height difference threshold based on distance
    dist_sq = octant_x * octant_x + octant_y * octant_y
    threshold = BASE_THRESHOLD
    
    if dist_sq <= CLOSE_RANGE_SQ_THRESHOLD:
        threshold = dist_sq // CLOSE_RANGE_DIVISOR
    else:
        threshold = _THRESHOLD_AT_CUTOFF + (dist_sq - CLOSE_RANGE_SQ_THRESHOLD) // FAR_RANGE_DIVISOR
    
    return abs(target_h - origin_height) > threshold

@numba.njit(cache=True)
def set_visible_at(
    octant_x: int, octant_y: int, octant: int, origin_xy: Point,
    grid_shape: tuple[int, int], visible_grid: np.ndarray, explored_grid: np.ndarray
) -> None:
    """Mark a tile as visible and explored."""
    nx, ny = _transform_coords(octant_x, octant_y, octant, origin_xy)
    width, height = grid_shape
    if 0 <= nx < width and 0 <= ny < height:
        visible_grid[ny, nx] = True
        explored_grid[ny, nx] = True

@numba.njit(cache=True)
def is_in_range(octant_x: int, octant_y: int, range_limit_sq: int | float) -> bool:
    """Check if coordinates are within the range limit."""
    return (octant_x * octant_x + octant_y * octant_y) <= range_limit_sq


@numba.njit(cache=True)
def _compute_fov_numba_core(
    origin_xy: Point, range_limit: int, opaque_grid: np.ndarray,
    height_map: np.ndarray, ceiling_map: np.ndarray, origin_height: int,
    visible_grid: np.ndarray, explored_grid: np.ndarray
) -> None:
    """Numba-optimized FOV core computation."""
    range_limit_sq = range_limit * range_limit
    grid_shape = opaque_grid.shape
    
    # Initialize with Numba-compatible list
    sectors = NumbaList()
    for octant in range(8):
        sectors.append((octant, 1, (1, 1), (0, 1)))
    
    sector_count = 0
    while len(sectors) > 0 and sector_count < MAX_SECTORS:
        sector_count += 1
        
        # Pop first element (FIFO behavior)
        current = sectors.pop(0)
        octant, current_x = current[0], current[1]
        top_slope, bottom_slope = current[2], current[3]
        
        blocked = False
        for current_y in range(current_x, range_limit + 1):
            if not is_in_range(current_x, current_y, range_limit_sq):
                break

            cell_top_y = 2 * current_y + 1
            cell_bottom_y = 2 * current_y - 1
            cell_x = 2 * current_x
            center_y, center_x = current_y, current_x

            if slope_less(top_slope, center_y, center_x) or \
               slope_less_or_equal(bottom_slope, center_y, center_x):
                continue

            set_visible_at(current_x, current_y, octant, origin_xy, 
                         grid_shape, visible_grid, explored_grid)

            cell_blocks = blocks_light_at(
                current_x, current_y, octant, origin_xy, grid_shape,
                opaque_grid, height_map, ceiling_map, origin_height
            )

            if blocked:
                if cell_blocks:
                    continue
                else:
                    blocked = False
                    bottom_slope = (cell_top_y, cell_x)
            else:
                if cell_blocks:
                    blocked = True
                    if slope_greater(top_slope, cell_bottom_y, cell_x):
                        sectors.append((octant, current_x + 1, 
                                     top_slope, (cell_bottom_y, cell_x)))
                    top_slope = (cell_top_y, cell_x)

        if not blocked:
            sectors.append((octant, current_x + 1, top_slope, bottom_slope))

def _compute_fov_python_fallback(
    origin_xy: Point, range_limit: int, opaque_grid: np.ndarray,
    height_map: np.ndarray, ceiling_map: np.ndarray, origin_height: int,
    visible_grid: np.ndarray, explored_grid: np.ndarray
) -> None:
    """Python fallback implementation for debugging and fallback."""
    range_limit_sq = range_limit * range_limit
    grid_shape = opaque_grid.shape
    
    sectors: deque = deque()
    for octant in range(8):
        sectors.append((octant, 1, (1, 1), (0, 1)))
    
    sector_count = 0
    while sectors and sector_count < MAX_SECTORS:
        sector_count += 1
        
        octant, current_x, top_slope, bottom_slope = sectors.popleft()
        
        blocked = False
        for current_y in range(current_x, range_limit + 1):
            if not is_in_range(current_x, current_y, range_limit_sq):
                break

            cell_top_y = 2 * current_y + 1
            cell_bottom_y = 2 * current_y - 1
            cell_x = 2 * current_x
            center_y, center_x = current_y, current_x

            if slope_less(top_slope, center_y, center_x) or \
               slope_less_or_equal(bottom_slope, center_y, center_x):
                continue

            set_visible_at(current_x, current_y, octant, origin_xy,
                         grid_shape, visible_grid, explored_grid)

            cell_blocks = blocks_light_at(
                current_x, current_y, octant, origin_xy, grid_shape,
                opaque_grid, height_map, ceiling_map, origin_height
            )

            if blocked:
                if cell_blocks:
                    continue
                else:
                    blocked = False
                    bottom_slope = (cell_top_y, cell_x)
            else:
                if cell_blocks:
                    blocked = True
                    if slope_greater(top_slope, cell_bottom_y, cell_x):
                        sectors.append((octant, current_x + 1,
                                     top_slope, (cell_bottom_y, cell_x)))
                    top_slope = (cell_top_y, cell_x)

        if not blocked:
            sectors.append((octant, current_x + 1, top_slope, bottom_slope))


def compute_fov(
    origin_xy: Point, range_limit: int, opaque_grid: np.ndarray,
    height_map: np.ndarray, ceiling_map: np.ndarray, origin_height: int,
    visible_grid: np.ndarray, explored_grid: np.ndarray
) -> None:
    """
    Public interface for FOV computation.
    Attempts Numba-optimized version first, falls back to Python implementation.
    """
    func_log = log.bind(
        origin=origin_xy,
        range_limit=range_limit,
        grid_shape=opaque_grid.shape
    )
    func_log.info("Starting FOV computation")
    start_time = time.perf_counter()

    # Input validation
    if not isinstance(opaque_grid, np.ndarray) or opaque_grid.ndim != 2:
        raise TypeError("opaque_grid must be a 2D NumPy array")
    if opaque_grid.shape != height_map.shape or opaque_grid.shape != ceiling_map.shape:
        raise ValueError("Grid shapes must match")
    if not np.issubdtype(opaque_grid.dtype, np.bool_):
        opaque_grid = opaque_grid.astype(np.bool_)
    if not np.issubdtype(visible_grid.dtype, np.bool_) or \
       not np.issubdtype(explored_grid.dtype, np.bool_):
        raise TypeError("visible_grid and explored_grid must be boolean arrays")
    
    height, width = opaque_grid.shape
    ox, oy = origin_xy
    if not (0 <= ox < width and 0 <= oy < height):
        raise ValueError("Origin coordinates out of bounds")

    # Initialize visibility
    visible_grid.fill(False)
    visible_grid[oy, ox] = True
    explored_grid[oy, ox] = True

    try:
        # Try Numba-optimized version first
        _compute_fov_numba_core(
            origin_xy, range_limit, opaque_grid,
            height_map, ceiling_map, origin_height,
            visible_grid, explored_grid
        )
    except Exception as e:
        func_log.warning("Numba FOV failed, falling back to Python", error=str(e))
        try:
            # Fall back to Python version
            _compute_fov_python_fallback(
                origin_xy, range_limit, opaque_grid,
                height_map, ceiling_map, origin_height,
                visible_grid, explored_grid
            )
        except Exception as e:
            func_log.error("FOV calculation failed", error=str(e), exc_info=True)
            visible_grid.fill(False)
            visible_grid[oy, ox] = True

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    func_log.info(
        "FOV computation finished",
        duration_ms=f"{duration_ms:.2f}",
        visible_count=np.sum(visible_grid)
    )

@numba.njit(cache=True)
def line_of_sight(
    x0: int, y0: int, x1: int, y1: int, transparency_map: np.ndarray
) -> bool:
    """
    Bresenham line-of-sight implementation.
    Returns True if there's a clear line of sight between points.
    """
    height, width = transparency_map.shape
    if not (0 <= y0 < height and 0 <= x0 < width and \
           0 <= y1 < height and 0 <= x1 < width):
        return False

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    
    xi, yi = x0, y0
    n_steps = max(dx, -dy)
    
    for _ in range(n_steps):
        e2 = 2 * err
        next_xi, next_yi = xi, yi
        step_x, step_y = False, False
        
        if e2 >= dy:
            if xi == x1:
                break
            err += dy
            next_xi += sx
            step_x = True
            
        if e2 <= dx:
            if yi == y1:
                break
            err += dx
            next_yi += sy
            step_y = True
            
        check_x, check_y = xi, yi
        if step_x:
            check_x += sx
        if step_y:
            check_y += sy
            
        if not transparency_map[check_y, check_x]:
            return False
            
        xi, yi = check_x, check_y
        if xi == x1 and yi == y1:
            break
            
    return True


# game/world/fov.py
"""
Field of View (FOV) and Line of Sight (LOS) calculations.
Uses Numba-accelerated Iterative Shadowcasting for FOV and Bresenham for LOS.
Includes height/ceiling checks and explored tile tracking.
"""

import time
from collections import deque
from typing import TypeAlias, List, Optional

import numba
import numpy as np
import structlog
from numba.typed import List as NumbaList
from .los import line_of_sight as _los_line_of_sight


def line_of_sight(x0: int, y0: int, x1: int, y1: int, transparency_map):
    """Return True if tiles (x0, y0) and (x1, y1) have clear line of sight."""
    return _los_line_of_sight(y0, x0, y1, x1, transparency_map)

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

def update_memory_fade(
    current_time: int,
    last_seen_time: np.ndarray,
    memory_intensity: np.ndarray,
    visible: np.ndarray,
    needs_update_mask: np.ndarray,
    prev_visible: np.ndarray,
    memory_strength: np.ndarray,
    steepness: float,
    midpoint: float,
    duration: float,
) -> None:
    """Sigmoid-based fading of remembered tiles.

    Only tiles tracked in ``needs_update_mask`` are processed.  The mask is
    dynamically updated to include tiles that transition from visible to
    invisible and pruned when tiles become visible again or their intensity
    reaches zero.
    """

    # Remove tiles that no longer need fading
    needs_update_mask[visible] = False
    needs_update_mask[memory_intensity <= 0.0] = False

    # Add tiles that have just become invisible with non-zero intensity
    became_invisible = prev_visible & (~visible) & (memory_intensity > 0.0)
    needs_update_mask |= became_invisible

    # Prepare for next call
    prev_visible[:] = visible

    if not np.any(needs_update_mask):
        return

    ys, xs = np.where(needs_update_mask)
    elapsed_time = current_time - last_seen_time[ys, xs]
    elapsed_time = np.maximum(elapsed_time, 0.0)

    strength = memory_strength[ys, xs]
    scale = 1.0 + strength
    decay_rate = steepness / scale
    midpoint_scaled = midpoint * scale
    exponent = decay_rate * (elapsed_time - midpoint_scaled)


    new_intensity = np.zeros_like(elapsed_time, dtype=np.float32)
    mask = exponent < 70.0
    safe_exp = np.exp(np.minimum(exponent[mask], 70.0))
    denom = 1.0 + safe_exp
    new_intensity[mask] = np.where(denom > 1e-9, 1.0 / denom, 0.0)

    memory_intensity[ys, xs] = np.maximum(0.0, new_intensity)


    # Prune tiles that have faded completely
    needs_update_mask[ys, xs] = memory_intensity[ys, xs] > 0.0

    return



@numba.njit(cache=True)
def compute_light_color_array(
    origin_xy: Point,
    range_limit: int,
    opaque_grid: np.ndarray,
    height_map: np.ndarray,
    ceiling_map: np.ndarray,
    origin_height: int,
    target_rgb_array: np.ndarray,
    base_color_rgb: tuple[int, int, int],
) -> None:
    """Accumulate colored light from a source into target_rgb_array."""
    h, w = opaque_grid.shape
    temp_visible = np.zeros((h, w), dtype=np.bool_)
    temp_explored = np.zeros((h, w), dtype=np.bool_)
    _compute_fov_numba_core(
        origin_xy,
        range_limit,
        opaque_grid,
        height_map,
        ceiling_map,
        origin_height,
        temp_visible,
        temp_explored,
    )

    ox, oy = origin_xy
    radius_sq = range_limit * range_limit
    r = float(base_color_rgb[0])
    g = float(base_color_rgb[1])
    b = float(base_color_rgb[2])

    for y in range(h):
        for x in range(w):
            if temp_visible[y, x]:
                dx = x - ox
                dy = y - oy
                dist_sq = dx * dx + dy * dy
                if dist_sq <= radius_sq:
                    intensity = 1.0 - (dist_sq / radius_sq)
                    if intensity > 0.0:
                        target_rgb_array[y, x, 0] += r * intensity
                        target_rgb_array[y, x, 1] += g * intensity
                        target_rgb_array[y, x, 2] += b * intensity


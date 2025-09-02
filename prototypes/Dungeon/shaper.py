# Dungeon/shaper.py - Revised for main.py orchestration & GameRNG

from common.constants import (
    MAT_SOLID_ROCK,
    MAT_CAVE_FLOOR,
    MAT_SHAFT_OPENING,
    MAT_CLIFF_EDGE,
)
import math

# Removed 'random' import
import time
import traceback

# Use modern type hints
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import polars as pl
from numba import njit
import structlog

log = structlog.get_logger(__name__)

# Import GameRNG using relative path
try:
    # Adjust path relative to main.py's location
    from game_rng import GameRNG
except ImportError:
    # Fallback for running shaper.py directly for tests (requires PYTHONPATH)
    log.warning(
        "Relative import failed. Ensure PYTHONPATH includes project root or run via main.py."
    )
    try:
        from game_rng import GameRNG  # type: ignore # noqa
    except ImportError:
        log.error("GameRNG not found via absolute path either", exc_info=True)
        raise

# === Dependency Imports ===
try:
    from scipy.ndimage import label as ndi_label  # Use specific name
    from scipy.signal import convolve2d

    HAS_SCIPY = True
except ImportError:
    log.warning(
        "SciPy not found. CA and Chamber ID calculation will be limited.")
    HAS_SCIPY = False

    # Define dummy label if SciPy not found to avoid NameError
    def ndi_label(*args, **kwargs):  # type: ignore # noqa
        log.warning("SciPy not found, returning dummy labels.")
        # Return dummy grid of zeros and 0 labels
        if args and isinstance(args[0], np.ndarray):
            return np.zeros_like(args[0], dtype=int), 0
        return np.array([[]], dtype=int), 0


try:
    from skimage.draw import disk as sk_draw_disk
    from skimage.draw import ellipse as sk_ellipse
    from skimage.draw import line as sk_line
    from skimage.draw import polygon as sk_polygon

    try:
        from skimage.draw import ellipse_perimeter as sk_ellipse_perimeter

        HAS_ELLIPSE_PERIMETER = True
    except ImportError:
        HAS_ELLIPSE_PERIMETER = False
        log.info("skimage.draw.ellipse_perimeter not found (optional).")
    from skimage.morphology import binary_dilation
    from skimage.morphology import disk as sk_morphology_disk

    HAS_SKIMAGE = True
except ImportError:
    log.error("scikit-image not found. Rasterization/Morphology cannot proceed.")
    HAS_SKIMAGE = False  # Essential for drawing

# Removed perlin-noise import
# try:
#     from perlin_noise import PerlinNoise
#     HAS_PERLIN = True
# except ImportError:
#     HAS_PERLIN = False # Noise blobs will fallback

# === Constants & Configuration ===

GRID_RESOLUTION: float = 1.0

CA_ITERATIONS: int = 8
CA_BIRTH_THRESHOLD: int = 5
CA_SURVIVAL_THRESHOLD: int = 4
DEFAULT_HEIGHT_M: Tuple[float, float] = (2.0, 4.0)  # Default height range

CHUNK_PROPERTIES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "smooth": {"width_m": (3, 5), "height_m": (2.5, 3.5)},
    "flat": {"width_m": (4, 7), "height_m": (2.5, 3.5)},
    "steep": {"width_m": (2, 4), "height_m": (3, 5)},
    "rise": {"width_m": (2, 4), "height_m": (3, 4)},
    "cliff": {"width_m": (1, 3), "height_m": (2, 5)},
    "shaft": {"width_m": (1, 2), "height_m": (2, 4)},
    "big_room": {"width_m": (15, 30), "height_m": (4, 8)},
}
# --- NPY DEBUG DUMP DISABLED ---
ENABLE_NPY_DEBUG_DUMP: bool = False
DEBUG_DUMP_WINDOW_RADIUS: int = 30


# ============================================================
# === HELPER FUNCTIONS (Defined FIRST) ===
# ============================================================


def _get_chunk_properties(type_name: str) -> Dict[str, Any]:  # Unchanged
    """Safely gets properties, defaulting to 'smooth' if unknown."""
    base_type = type_name.split(":")[0]
    props = CHUNK_PROPERTIES.get(base_type)
    # Ensure a default height exists even if type is unknown
    default_props = CHUNK_PROPERTIES["smooth"].copy()
    if not props:
        props = default_props
    elif "height_m" not in props:
        props["height_m"] = default_props["height_m"]
    return props


def _assign_chunk_type_and_subtype(  # Unchanged logic
    parent_data: Dict, child_data: Dict
) -> Tuple[str, Optional[str]]:
    """Assigns base chunk type & potential subtype based on features or geometry."""
    feature = parent_data.get("feature")
    if feature and feature.startswith("big_room:"):
        parts = feature.split(":", 1)
        subtype = parts[1] if len(parts) > 1 else None
        return "big_room", subtype
    if feature == "cliff_edge":
        return "cliff", None
    if feature == "shaft_opening":
        return "shaft", None

    # Determine type based on incline
    incline = child_data.get("segment_incline_rate", 0.0)
    if incline < -0.25:
        return "rise", None
    if incline > 0.30:
        return "steep", None
    if abs(incline) < 0.05:
        return "flat", None
    return "smooth", None


def _rasterize_thick_line(  # Unchanged logic
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    thickness_cells: int,
    value: int,
    chunk_type_mask: Optional[np.ndarray] = None,
    type_id: int = 0,
):
    """Draws a thick line using skimage line and dilation."""
    if not HAS_SKIMAGE:
        log.warning("Thick line requires scikit-image.")
        return
    # Ensure integer inputs
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    line_mask = np.zeros_like(grid, dtype=bool)
    try:
        rr, cc = sk_line(r0, c0, r1, c1)
    except Exception as e:
        log.error("Error in sk_line", error=str(e), exc_info=True)
        return

    # Filter coordinates to be within bounds
    valid = (rr >= 0) & (rr < grid.shape[0]) & (cc >= 0) & (cc < grid.shape[1])
    rr, cc = rr[valid], cc[valid]
    if len(rr) == 0:
        return  # No valid points on the line

    # Create line mask and dilate if thickness > 1
    line_mask[rr, cc] = True
    radius = max(0, thickness_cells // 2)
    thick_mask = line_mask
    if radius > 0:
        try:
            # Ensure dilation element fits within grid dimensions if small
            # (though disk() usually handles this)
            selem = sk_morphology_disk(radius)
            thick_mask = binary_dilation(line_mask, footprint=selem)
        except Exception as e:
            log.error("Error during dilation", error=str(e), exc_info=True)
            # Fallback to just the thin line mask? Or return?
            thick_mask = line_mask  # Fallback to thin line

    # Apply value to the grid and type mask
    try:
        grid[thick_mask] = value
        if chunk_type_mask is not None and chunk_type_mask.shape == grid.shape:
            chunk_type_mask[thick_mask] = type_id
    except IndexError:
        # This might happen if dilation near edge creates out-of-bounds indices somehow
        log.warning(
            "Index error assigning thick line value near endpoints",
            start=(r0, c0),
            end=(r1, c1),
        )
    except Exception as e:
        log.error(
            "Error assigning value in rasterize_thick_line", error=str(e), exc_info=True
        )


def _rasterize_polygon(  # Unchanged logic
    grid: np.ndarray,
    r_coords: np.ndarray,
    c_coords: np.ndarray,
    value: int,
    chunk_type_mask: Optional[np.ndarray] = None,
    type_id: int = 0,
):
    """Rasterizes a filled polygon using skimage."""
    if not HAS_SKIMAGE:
        log.warning("Polygon rasterization requires scikit-image.")
        return
    try:
        # Round coordinates and ensure they are integers for skimage
        rr, cc = sk_polygon(
            np.round(r_coords).astype(int),
            np.round(c_coords).astype(int),
            shape=grid.shape,
        )
        grid[rr, cc] = value
        if chunk_type_mask is not None and chunk_type_mask.shape == grid.shape:
            chunk_type_mask[rr, cc] = type_id
    except Exception as e:
        log.error("Error rasterizing polygon", error=str(e), exc_info=True)


# --- Cavern Rasterization Helpers (Modified to accept and use 'rng') ---
def _rasterize_ellipse_cavern(  # Added rng parameter
    grid: np.ndarray,
    r_center: int,
    c_center: int,
    properties: Dict,
    orientation_rad: float,
    value: int,
    chunk_type_mask: np.ndarray,
    type_id: int,
    rng: GameRNG,
):
    """Rasterizes an ellipse, using rng."""
    if not HAS_SKIMAGE:
        return
    # USE rng for random values
    diam1 = rng.get_float(*properties["width_m"])
    diam2 = rng.get_float(*properties["width_m"])
    ry = max(2, int(round(max(diam1, diam2) / 2.0 / GRID_RESOLUTION)))
    rx = max(1, int(round(min(diam1, diam2) / 2.0 / GRID_RESOLUTION)))
    try:
        # Call WITHOUT orientation for compatibility/simplicity
        rr, cc = sk_ellipse(r_center, c_center, ry, rx, shape=grid.shape)
        grid[rr, cc] = value
        chunk_type_mask[rr, cc] = type_id
    except Exception as e:
        log.error("Error drawing ellipse cavern", error=str(e), exc_info=True)


def _rasterize_rect_cavern(  # Added rng parameter
    grid: np.ndarray,
    r_center: int,
    c_center: int,
    properties: Dict,
    orientation_rad: float,
    value: int,
    chunk_type_mask: np.ndarray,
    type_id: int,
    rng: GameRNG,
):
    """Rasterizes a rectangle, using rng."""
    if not HAS_SKIMAGE:
        return
    # USE rng for random values
    rect_len = rng.get_float(*properties["width_m"])
    rect_wid = rng.get_float(*properties["width_m"]) * rng.get_float(0.5, 0.9)
    h_cells = max(3, int(round(rect_len / GRID_RESOLUTION)))
    w_cells = max(3, int(round(rect_wid / GRID_RESOLUTION)))
    cos_o = math.cos(orientation_rad)
    sin_o = math.sin(orientation_rad)
    hh = h_cells / 2.0
    hw = w_cells / 2.0
    # Calculate corner coordinates
    corners_r = np.array(
        [
            r_center + hh * sin_o - hw * cos_o,
            r_center + hh * sin_o + hw * cos_o,
            r_center - hh * sin_o + hw * cos_o,
            r_center - hh * sin_o - hw * cos_o,
        ]
    )
    corners_c = np.array(
        [
            c_center + hh * cos_o + hw * sin_o,
            c_center + hh * cos_o - hw * sin_o,
            c_center - hh * cos_o - hw * sin_o,
            c_center - hh * cos_o + hw * sin_o,
        ]
    )
    # Call polygon rasterizer
    _rasterize_polygon(grid, corners_r, corners_c,
                       value, chunk_type_mask, type_id)


def _rasterize_multicircle_cavern(  # Added rng parameter
    grid: np.ndarray,
    r_center: int,
    c_center: int,
    properties: Dict,
    value: int,
    chunk_type_mask: np.ndarray,
    type_id: int,
    rng: GameRNG,
):
    """Rasterizes multiple overlapping circles, using rng."""
    if not HAS_SKIMAGE:
        return
    # USE rng for random values
    num_lobes = rng.get_int(2, 5)
    base_radius_m = rng.get_float(
        *properties["width_m"]) / (1.0 + 0.5 * num_lobes)
    base_radius_cells = max(3, int(round(base_radius_m / GRID_RESOLUTION)))
    max_offset = base_radius_cells * 0.8
    temp_mask = np.zeros_like(grid, dtype=bool)
    for _ in range(num_lobes):
        offset_angle = rng.get_float(0, 2 * math.pi)
        offset_dist = rng.get_float(0, max_offset)
        lobe_r = r_center + offset_dist * math.sin(offset_angle)
        lobe_c = c_center + offset_dist * math.cos(offset_angle)
        lobe_rad = int(round(base_radius_cells * rng.get_float(0.6, 1.4)))
        try:
            rr, cc = sk_draw_disk(
                (int(lobe_r), int(lobe_c)), radius=max(1, lobe_rad), shape=grid.shape
            )
            temp_mask[rr, cc] = True
        except Exception as e:
            log.error("Error drawing disk lobe", error=str(e), exc_info=True)
    grid[temp_mask] = value
    chunk_type_mask[temp_mask] = type_id


# Removed _get_noise_func (now uses rng.noise_2d)


def _rasterize_noisy_ellipse_cavern(  # Added rng parameter
    grid: np.ndarray,
    r_center: int,
    c_center: int,
    properties: Dict,
    orientation_rad: float,  # Keep orientation for noise calculation if needed
    value: int,
    chunk_type_mask: np.ndarray,
    type_id: int,
    rng: GameRNG,
):
    """Rasterizes a noise-perturbed ellipse, using rng for noise."""
    if not HAS_SKIMAGE:
        return
    # USE rng for random values
    ry = max(
        3, int(
            round(rng.get_float(*properties["height_m"]) / 2.0 / GRID_RESOLUTION))
    )
    rx = max(
        3, int(
            round(rng.get_float(*properties["width_m"]) / 2.0 / GRID_RESOLUTION))
    )

    base_rr, base_cc = np.array([]), np.array([])
    try:  # Generate perimeter points (axis-aligned)
        if HAS_ELLIPSE_PERIMETER:
            # Generate perimeter without orientation for base shape
            base_rr, base_cc = sk_ellipse_perimeter(
                r_center, c_center, ry, rx, shape=grid.shape
            )
        else:  # Fallback using linspace
            t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
            st, ct = np.sin(t), np.cos(t)
            base_c_raw = c_center + rx * ct
            base_r_raw = r_center + ry * st
            base_r_int, base_c_int = np.round(base_r_raw).astype(int), np.round(
                base_c_raw
            ).astype(int)
            # Filter points within bounds
            valid = (
                (base_r_int >= 0)
                & (base_r_int < grid.shape[0])
                & (base_c_int >= 0)
                & (base_c_int < grid.shape[1])
            )
            if np.any(valid):
                # Get unique valid points
                points = np.unique(
                    np.column_stack((base_r_int[valid], base_c_int[valid])), axis=0
                )
                base_rr, base_cc = points[:, 0], points[:, 1]
    except Exception as e:
        log.error("Error generating ellipse perimeter",
                  error=str(e), exc_info=True)

    if len(base_rr) == 0:
        log.warning("No points for noisy ellipse boundary.")
        return

    perturbed_r, perturbed_c = base_rr.astype(float), base_cc.astype(float)
    try:
        # USE GameRNG noise
        scale = max(ry, rx) * rng.get_float(0.05, 0.20)
        freq = rng.get_float(0.05, 0.2)
        # Use distinct offsets or dimensions for r and c noise
        # Simple offset based on position
        noise_seed_offset = int(r_center + c_center)

        displace_r = scale * np.array(
            [
                rng.noise_2d(r, c, scale=1.0 / freq,
                             seed_offset=noise_seed_offset)
                for r, c in zip(base_rr, base_cc)
            ]
        )
        # Add a different offset for c displacement for variety
        displace_c = scale * np.array(
            [
                rng.noise_2d(
                    c, r, scale=1.0 / freq, seed_offset=noise_seed_offset + 100
                )
                for r, c in zip(base_rr, base_cc)
            ]
        )

        perturbed_r += displace_r
        perturbed_c += displace_c
    except Exception as e:
        log.error("Error applying GameRNG noise", error=str(e), exc_info=True)

    _rasterize_polygon(grid, perturbed_r, perturbed_c,
                       value, chunk_type_mask, type_id)


def _rasterize_noise_blob_cavern(  # Added rng parameter
    grid: np.ndarray,
    r_center: int,
    c_center: int,
    properties: Dict,
    value: int,
    chunk_type_mask: np.ndarray,
    type_id: int,
    rng: GameRNG,
):
    """Rasterizes a noise blob using GameRNG noise."""
    # USE GameRNG for size
    size_m = rng.get_float(*properties["width_m"])
    size_cells = max(10, int(round(size_m / GRID_RESOLUTION)))
    radius = size_cells // 2

    # Calculate bounds safely
    r_min, r_max = max(
        0, r_center - radius), min(grid.shape[0], r_center + radius + 1)
    c_min, c_max = max(
        0, c_center - radius), min(grid.shape[1], c_center + radius + 1)
    if r_min >= r_max or c_min >= c_max:
        return  # Area is zero or invalid

    # Generate noise values using GameRNG noise
    noise_vals = np.zeros((r_max - r_min, c_max - c_min), dtype=np.float64)
    freq = rng.get_float(3.0, 8.0) / size_cells  # Use rng
    noise_seed_offset = int(r_center + c_center)  # Simple offset
    try:
        for r_idx, r_world in enumerate(range(r_min, r_max)):
            for c_idx, c_world in enumerate(range(c_min, c_max)):
                noise_vals[r_idx, c_idx] = rng.noise_2d(
                    r_world, c_world, scale=1.0 / freq, seed_offset=noise_seed_offset
                )
    except Exception as e:
        log.error("Error generating GameRNG noise field",
                  error=str(e), exc_info=True)
        # Fallback? Or just continue with potentially zeros?
        # Let's continue, might result in smaller blob

    # Create blob mask
    threshold = rng.get_float(0.0, 0.25)  # Use rng
    blob_mask = noise_vals > threshold

    # Ensure center point is included
    center_r_idx, center_c_idx = r_center - r_min, c_center - c_min
    if (
        0 <= center_r_idx < blob_mask.shape[0]
        and 0 <= center_c_idx < blob_mask.shape[1]
    ):
        blob_mask[center_r_idx, center_c_idx] = True

    # Apply mask to grid and type_mask
    grid_slice = (slice(r_min, r_max), slice(c_min, c_max))
    try:
        grid[grid_slice][blob_mask] = value
        chunk_type_mask[grid_slice][blob_mask] = type_id
    except IndexError:
        log.warning(
            "Index error assigning noise blob mask",
            grid_slice=grid_slice,
            mask_shape=blob_mask.shape,
        )
    except Exception as e:
        log.error("Error assigning noise blob mask",
                  error=str(e), exc_info=True)


# --- Map cavern subtype names to functions ---
CAVERN_GENERATORS: Dict[str, Callable] = {
    "ellipse": _rasterize_ellipse_cavern,
    "rectangle": _rasterize_rect_cavern,
    "multi_circle": _rasterize_multicircle_cavern,
    "noisy_ellipse": _rasterize_noisy_ellipse_cavern,
    "noise_blob": _rasterize_noise_blob_cavern,
}

# --- NPY Dump Helper (Removed) ---


# --- Grid Initialization Helpers (Unchanged Logic) ---
def _calculate_grid_bounds(
    nodes: list[Dict],
) -> Optional[Tuple[int, int, int, int]]:  # Unchanged logic
    """Calculates the required grid bounds based on node coordinates."""
    try:
        all_x = [n["x"]
                 for n in nodes if not math.isnan(n.get("x", float("nan")))]
        all_y = [n["y"]
                 for n in nodes if not math.isnan(n.get("y", float("nan")))]
        if not all_x or not all_y:
            log.error("No valid coordinates found for bounds calculation.")
            return None
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
    except Exception as e:
        log.error("Error calculating bounds", error=str(e), exc_info=True)
        return None

    # Add padding around the min/max coordinates
    padding = 15
    origin_offset_x = int(np.floor(min_x / GRID_RESOLUTION)) - padding
    origin_offset_y = int(np.floor(min_y / GRID_RESOLUTION)) - padding
    max_gx = int(np.ceil(max_x / GRID_RESOLUTION)) + padding
    max_gy = int(np.ceil(max_y / GRID_RESOLUTION)) + padding

    grid_width = max_gx - origin_offset_x
    grid_height = max_gy - origin_offset_y

    print(
        f"Grid Params: Offset=({origin_offset_x}, {origin_offset_y}), "
        f"Width={grid_width}, Height={grid_height}"
    )

    if grid_width <= 0 or grid_height <= 0:
        log.error("Invalid grid dimensions calculated",
                  grid_width=grid_width, grid_height=grid_height)
        return None

    return grid_width, grid_height, origin_offset_x, origin_offset_y


def _initialize_grids(
    height: int, width: int
) -> Optional[
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]
]:  # Unchanged logic
    """Initializes the main grid, depth grid, and type grid."""
    try:
        grid = np.full((height, width), MAT_SOLID_ROCK, dtype=np.int8)
        depth_grid = np.full((height, width), np.nan, dtype=np.float32)
        # Map chunk type names to integer IDs for the type grid
        chunk_type_names = list(CHUNK_PROPERTIES.keys())
        chunk_type_map = {name: i + 1 for i,
                          name in enumerate(chunk_type_names)}
        type_grid = np.zeros((height, width), dtype=np.uint8)
        return grid, depth_grid, type_grid, chunk_type_map
    except (MemoryError, ValueError) as e:
        log.error(
            "Error allocating grids", height=height, width=width, error=str(e), exc_info=True
        )
        return None


# --- Rasterize Segment (Modified to accept/pass rng, removed NPY dumps) ---
def _rasterize_segment(  # Added rng parameter
    segment_index: int,
    parent_data: Dict,
    child_node_data: Dict,
    grid: np.ndarray,
    depth_grid: np.ndarray,
    type_grid: np.ndarray,
    chunk_type_map: Dict[str, int],
    origin_offset_x: int,
    origin_offset_y: int,
    rng: GameRNG,
):
    """Determines type, calculates coords, dispatches rasterization, interpolates depth."""
    # Removed global debug dump variables
    grid_height, grid_width = grid.shape
    try:
        # Determine segment type and properties
        chunk_base_type, chunk_subtype = _assign_chunk_type_and_subtype(
            parent_data, child_node_data
        )
        properties = _get_chunk_properties(chunk_base_type)
        chunk_type_id = chunk_type_map.get(
            chunk_base_type, 0)  # 0 if type unknown

        # Get parent and child coordinates, check for NaN
        p_x, p_y = parent_data.get("x", 0.0), parent_data.get("y", 0.0)
        c_x, c_y = child_node_data.get("x", p_x), child_node_data.get("y", p_y)
        if math.isnan(p_x) or math.isnan(p_y) or math.isnan(c_x) or math.isnan(c_y):
            log.warning(
                "NaN coords in segment; skipping",
                segment=f"{parent_data['id']}->{child_node_data['id']}",
            )
            return

        # Convert world coords to grid coords, clamping to bounds
        p_gx_clamped = max(
            0,
            min(grid_width - 1, int(round((p_x / GRID_RESOLUTION) - origin_offset_x))),
        )
        p_gy_clamped = max(
            0,
            min(grid_height - 1, int(round((p_y / GRID_RESOLUTION) - origin_offset_y))),
        )
        c_gx_clamped = max(
            0,
            min(grid_width - 1, int(round((c_x / GRID_RESOLUTION) - origin_offset_x))),
        )
        c_gy_clamped = max(
            0,
            min(grid_height - 1, int(round((c_y / GRID_RESOLUTION) - origin_offset_y))),
        )

        # --- Removed NPY Debug Dump Logic ---

        width_cells = 0  # Default width
        if chunk_base_type != "big_room":
            # USE rng for random values
            width_m = rng.get_float(*properties["width_m"])
            width_cells = max(1, int(round(width_m / GRID_RESOLUTION)))

        # --- Rasterization based on type ---
        if chunk_base_type == "cliff":
            # Simple rectangle for cliff edge point
            r_min = max(0, p_gy_clamped - width_cells // 2)
            r_max = min(grid_height, p_gy_clamped + width_cells // 2 + 1)
            c_min = max(0, p_gx_clamped - width_cells // 2)
            c_max = min(grid_width, p_gx_clamped + width_cells // 2 + 1)
            if r_max > r_min and c_max > c_min:
                grid[r_min:r_max, c_min:c_max] = MAT_CLIFF_EDGE
                depth_grid[r_min:r_max, c_min:c_max] = parent_data["depth_m"]
                type_grid[r_min:r_max, c_min:c_max] = chunk_type_id
        elif chunk_base_type == "shaft":
            # Simple rectangle for shaft opening point
            r_min = max(0, p_gy_clamped - width_cells // 2)
            r_max = min(grid_height, p_gy_clamped + width_cells // 2 + 1)
            c_min = max(0, p_gx_clamped - width_cells // 2)
            c_max = min(grid_width, p_gx_clamped + width_cells // 2 + 1)
            if r_max > r_min and c_max > c_min:
                grid[r_min:r_max, c_min:c_max] = MAT_SHAFT_OPENING
                depth_grid[r_min:r_max, c_min:c_max] = parent_data["depth_m"]
                type_grid[r_min:r_max, c_min:c_max] = chunk_type_id
        elif chunk_base_type == "big_room":
            cavern_func = CAVERN_GENERATORS.get(chunk_subtype)
            if cavern_func:
                orientation = math.atan2(
                    c_gy_clamped - p_gy_clamped, c_gx_clamped - p_gx_clamped
                )
                # --- Pass rng down to cavern function ---
                # Adjusted dispatch logic slightly for clarity
                if chunk_subtype in [
                    "ellipse",
                    "rectangle",
                    "noisy_ellipse",
                ]:
                    cavern_func(
                        grid,
                        p_gy_clamped,
                        p_gx_clamped,
                        properties,
                        orientation,
                        MAT_CAVE_FLOOR,
                        type_grid,
                        chunk_type_id,
                        rng,  # Pass rng
                    )
                elif chunk_subtype in ["multi_circle", "noise_blob"]:
                    cavern_func(
                        grid,
                        p_gy_clamped,
                        p_gx_clamped,
                        properties,
                        MAT_CAVE_FLOOR,
                        type_grid,
                        chunk_type_id,
                        rng,  # Pass rng
                    )
                else:
                    log.error(
                        "Unhandled cavern function for subtype",
                        subtype=chunk_subtype,
                    )
                # Assign depth to newly painted floor cells within the cavern
                newly_painted_mask = (
                    (type_grid == chunk_type_id)
                    & np.isnan(depth_grid)
                    & (grid == MAT_CAVE_FLOOR)
                )
                depth_grid[newly_painted_mask] = parent_data.get(
                    "depth_m", 0.0)
            else:  # Fallback if subtype unknown
                log.warning(
                    "Unknown big_room subtype; drawing wide passage",
                    subtype=chunk_subtype,
                )
                width_m = rng.get_float(*properties["width_m"])  # Use rng
                width_cells = max(1, int(round(width_m / GRID_RESOLUTION)))
                _rasterize_thick_line(
                    grid,
                    p_gy_clamped,
                    p_gx_clamped,
                    c_gy_clamped,
                    c_gx_clamped,
                    width_cells,
                    MAT_CAVE_FLOOR,
                    type_grid,
                    chunk_type_id,
                )
                # Depth handled below for passages
        else:  # Natural cave passage types (smooth, flat, steep, rise)
            _rasterize_thick_line(
                grid,
                p_gy_clamped,
                p_gx_clamped,
                c_gy_clamped,
                c_gx_clamped,
                width_cells,
                MAT_CAVE_FLOOR,
                type_grid,
                chunk_type_id,
            )

        # --- Depth Interpolation (for passages and big_room fallback) ---
        # Skip for cliffs/shafts (depth assigned at point) and successfully generated big rooms
        if chunk_base_type not in ["cliff", "shaft"] and not (
            chunk_base_type == "big_room" and cavern_func and chunk_subtype
        ):
            # Ensure width_cells has a value
            if width_cells == 0:  # Fallback if big_room had unknown subtype
                default_props = _get_chunk_properties("smooth")
                width_m = rng.get_float(*default_props["width_m"])  # Use rng
                width_cells = max(1, int(round(width_m / GRID_RESOLUTION)))

            # Calculate bounding box for interpolation efficiently
            r_min_bb = max(0, min(p_gy_clamped, c_gy_clamped) - width_cells)
            r_max_bb = min(
                grid_height, max(p_gy_clamped, c_gy_clamped) + width_cells + 1
            )
            c_min_bb = max(0, min(p_gx_clamped, c_gx_clamped) - width_cells)
            c_max_bb = min(
                grid_width, max(p_gx_clamped, c_gx_clamped) + width_cells + 1
            )

            if r_min_bb >= r_max_bb or c_min_bb >= c_max_bb:
                return  # Skip if bounding box is invalid

            sub_grid_slice = (slice(r_min_bb, r_max_bb),
                              slice(c_min_bb, c_max_bb))

            # Find relevant cells within the bounding box
            mask = (
                (grid[sub_grid_slice] == MAT_CAVE_FLOOR)
                & (type_grid[sub_grid_slice] == chunk_type_id)
                & (np.isnan(depth_grid[sub_grid_slice]))
            )
            potential_indices_r, potential_indices_c = np.where(mask)

            if len(potential_indices_r) > 0:
                # Convert sub-grid indices back to full grid indices
                painted_r = potential_indices_r + r_min_bb
                painted_c = potential_indices_c + c_min_bb

                # Perform interpolation (vectorized)
                seg_vec_r = float(c_gy_clamped - p_gy_clamped)
                seg_vec_c = float(c_gx_clamped - p_gx_clamped)
                seg_len_sq = seg_vec_r**2 + seg_vec_c**2
                interp_factor = 0.0  # Default if segment length is zero

                if seg_len_sq > 1e-6:  # Avoid division by zero
                    dot_product = (painted_r - p_gy_clamped) * seg_vec_r + (
                        painted_c - p_gx_clamped
                    ) * seg_vec_c
                    interp_factor = np.clip(dot_product / seg_len_sq, 0.0, 1.0)

                parent_depth = parent_data.get("depth_m", 0.0)
                child_depth = child_node_data.get("depth_m", parent_depth)
                interpolated_depth = (
                    parent_depth + (child_depth - parent_depth) * interp_factor
                )

                depth_grid[painted_r, painted_c] = interpolated_depth

        # --- Removed NPY Dump Call ---

    except Exception as e:
        print(
            f"Error rasterizing segment {segment_index} "
            f"({parent_data.get('id', '?')}- >{child_node_data.get('id', '?')})"
            f": {e}"
        )
        # traceback.print_exc() # Uncomment for full traceback during debugging


# ============================================================
# === MAIN ORCHESTRATING FUNCTIONS (Modified to accept/pass rng) ===
# ============================================================


def initialize_cave_grid(  # Added rng parameter
    augmented_nodes: list[Dict], augmented_node_map: Dict[int, Dict], rng: GameRNG
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int]
]:
    """Creates initial 2D grid footprint by orchestrating helper functions."""
    # Removed global debug dump variables
    if not augmented_nodes:
        log.error("No nodes provided for grid initialization.")
        return None, None, None, (0, 0)

    print("Initializing cave grid (refactored)...")
    bounds_info = _calculate_grid_bounds(augmented_nodes)
    if bounds_info is None:
        return None, None, None, (0, 0)

    grid_width, grid_height, origin_offset_x, origin_offset_y = bounds_info
    grid_data = _initialize_grids(grid_height, grid_width)
    if grid_data is None:
        return None, None, None, (0, 0)

    grid, depth_grid, type_grid, chunk_type_map = grid_data
    print("Rasterizing segments...")
    segments_processed = 0
    # Iterate through child nodes to find parent->child links
    for child_node_data in augmented_nodes:
        parent_id = child_node_data.get("parent_id")
        # Skip root or nodes without valid parent links
        if parent_id is None or parent_id not in augmented_node_map:
            continue

        segments_processed += 1
        parent_data = augmented_node_map[parent_id]
        # Call the main segment rasterization logic, passing rng
        _rasterize_segment(
            segments_processed,
            parent_data,
            child_node_data,
            grid,
            depth_grid,
            type_grid,
            chunk_type_map,
            origin_offset_x,
            origin_offset_y,
            rng,  # Pass rng
        )

    print(f"Finished rasterizing {segments_processed} segments.")
    return grid, depth_grid, type_grid, (origin_offset_x, origin_offset_y)


@njit(cache=True)  # CA Step (unchanged)
def _run_ca_step_numpy(grid: np.ndarray) -> np.ndarray:
    """Numba CA step using explicit loops."""
    new_grid = grid.copy()
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            non_solid_neighbors = 0
            # Check 8 neighbors
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    # Check bounds before accessing grid
                    if 0 <= nr < height and 0 <= nc < width:
                        # Use central constant
                        if grid[nr, nc] != MAT_SOLID_ROCK:
                            non_solid_neighbors += 1

            # Apply CA rules
            # Use central constant
            if grid[r, c] == MAT_SOLID_ROCK:
                if non_solid_neighbors >= CA_BIRTH_THRESHOLD:
                    # Use central constant
                    new_grid[r, c] = MAT_CAVE_FLOOR
            # Use central constant
            elif grid[r, c] != MAT_SOLID_ROCK:
                if non_solid_neighbors < CA_SURVIVAL_THRESHOLD:
                    # Use central constant
                    new_grid[r, c] = MAT_SOLID_ROCK
            # Else (other non-solid types remain unchanged by these basic rules)

    return new_grid


def _run_ca_step_scipy(grid: np.ndarray) -> np.ndarray:  # CA Step (unchanged)
    """SciPy CA step using convolution."""
    # Kernel counts 8 neighbors
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # Use central constant
    non_solid_mask = grid != MAT_SOLID_ROCK
    # Count non-solid neighbors using convolution
    non_solid_neighbor_count = convolve2d(
        non_solid_mask, kernel, mode="same", boundary="fill", fillvalue=0
    )

    # Apply rules based on neighbor count
    # Use central constant
    born = (grid == MAT_SOLID_ROCK) & (
        non_solid_neighbor_count >= CA_BIRTH_THRESHOLD)
    # Use central constant
    survived = (grid != MAT_SOLID_ROCK) & (
        non_solid_neighbor_count >= CA_SURVIVAL_THRESHOLD
    )

    # Create new grid based on rules
    # Use central constant
    new_grid = np.full(grid.shape, MAT_SOLID_ROCK, dtype=grid.dtype)
    new_grid[born] = MAT_CAVE_FLOOR  # Use central constant
    # Preserve original tile type for survivors
    new_grid[survived] = grid[survived]
    return new_grid


def run_cellular_automata(  # Unchanged logic
    initial_grid: np.ndarray, iterations: int = CA_ITERATIONS
) -> np.ndarray | None:
    """Applies CA rules using best available method."""
    if initial_grid is None:
        return None
    print(f"Running CA for {iterations} iterations...")
    grid = initial_grid.copy()
    ca_step_func = None
    if HAS_SCIPY:
        print("Using SciPy convolution for CA.")
        ca_step_func = _run_ca_step_scipy
    # elif HAS_NUMBA: # Numba JIT for CA step might need explicit type signatures
    #     print("Using Numba loop for CA.")
    #     ca_step_func = _run_ca_step_numpy
    else:
        log.warning("SciPy not found. Using slower NumPy loop for CA.")
        # Fallback to a pure python/numpy loop if Numba isn't easily applied or available
        # For simplicity, let's assume SciPy exists or fail if not.
        # If a pure Python fallback is needed, implement _run_ca_step_numpy without @njit
        log.error("SciPy required for CA step.")
        return initial_grid  # Or return None?

    if ca_step_func:
        for i in range(iterations):
            try:
                grid = ca_step_func(grid)
            except Exception as e:
                log.error("Error during CA step", step=i +
                          1, error=str(e), exc_info=True)
                traceback.print_exc()  # Show full traceback
                return grid  # Return current state on error
    print("CA finished.")
    return grid


def create_map_dataframe(  # Added rng parameter
    final_grid: np.ndarray,
    depth_grid: np.ndarray,
    type_grid: np.ndarray,
    origin_offset: Tuple[int, int],
    rng: GameRNG,
    higher_final_grid: Optional[np.ndarray] = None,
) -> Optional[pl.DataFrame]:
    """Creates final Polars DataFrame including chamber IDs and open_above calculation."""
    if final_grid is None or depth_grid is None or type_grid is None:
        return None
    print("Creating DataFrame from final grid...")
    height, width = final_grid.shape
    ox, oy = origin_offset

    # Build reverse map for getting chunk type names (used for height lookup)
    chunk_type_names = list(CHUNK_PROPERTIES.keys())
    reverse_type_map = {i + 1: name for i, name in enumerate(chunk_type_names)}

    # Find indices of non-rock cells efficiently
    non_rock_y, non_rock_x = np.where(final_grid != MAT_SOLID_ROCK)
    num_non_rock = len(non_rock_y)
    if num_non_rock == 0:
        log.warning("No non-rock cells found to create DataFrame.")
        return pl.DataFrame()  # Return empty DataFrame

    print(f"Found {num_non_rock} non-rock cells.")

    # Extract data for non-rock cells
    mat_ids = final_grid[non_rock_y, non_rock_x]
    floor_depths_extracted = depth_grid[non_rock_y, non_rock_x]
    type_ids = type_grid[non_rock_y, non_rock_x]

    # Calculate world coordinates
    col_x = (non_rock_x + ox) * GRID_RESOLUTION
    col_y = (non_rock_y + oy) * GRID_RESOLUTION

    # Prepare basic columns
    col_floor_depth = np.nan_to_num(floor_depths_extracted, nan=0.0).round(2)
    # Ensure material ID uses an appropriate unsigned type
    col_mat_id = mat_ids.astype(np.uint16)
    # Define walkable based on material ID (e.g., excluding rock, cliffs)
    # Use central constants
    walkable_mask = (col_mat_id != MAT_SOLID_ROCK) & (
        col_mat_id != MAT_CLIFF_EDGE)
    col_walkable = walkable_mask.astype(bool)

    # Calculate heights (vectorized where possible)
    col_height = np.zeros(num_non_rock, dtype=np.float32)
    unknown_type_count = 0

    # Batch generate random heights within the default range first
    min_h_def, max_h_def = DEFAULT_HEIGHT_M
    generated_heights = rng.get_floats(
        min_h_def, max_h_def, count=num_non_rock)

    # Apply specific heights based on type_id, using pre-generated random values
    for i in range(num_non_rock):
        type_name = reverse_type_map.get(type_ids[i])
        if type_name:
            properties = _get_chunk_properties(type_name)
            min_h, max_h = properties.get("height_m", DEFAULT_HEIGHT_M)
            # Scale the pre-generated [0,1) value (approx) from default range
            # This is an approximation; true batching per type is complex
            # Re-generate if needed for accuracy per type
            # Let's regenerate for now, batching was complex:
            col_height[i] = round(rng.get_float(min_h, max_h), 2)
        else:
            # Use default pre-generated height if type unknown
            col_height[i] = round(generated_heights[i], 2)
            unknown_type_count += 1

    if unknown_type_count > 0:
        log.warning(
            "Assigned default height range to cells", count=unknown_type_count
        )

    # Calculate ceiling depth
    col_ceil_depth = (col_floor_depth - col_height).round(2)

    # --- Chamber ID Calculation ---
    col_chamber_id = np.zeros(
        num_non_rock, dtype=np.int32)  # Default to 0 or -1?
    if HAS_SCIPY:
        print("Calculating Chamber IDs using SciPy (8-connectivity)...")
        # Create boolean grid based on walkable status from DataFrame column
        walkable_grid_for_label = np.zeros(final_grid.shape, dtype=bool)
        walkable_grid_for_label[non_rock_y, non_rock_x] = (
            col_walkable  # Use derived walkable status
        )

        # Define 8-connectivity structure
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        try:
            labeled_grid, num_chambers = ndi_label(
                walkable_grid_for_label, structure=structure
            )
            print(f"Found {num_chambers} distinct chambers.")
            # Extract chamber IDs for the non-rock cells
            col_chamber_id = labeled_grid[non_rock_y,
                                          non_rock_x].astype(np.int32)
        except Exception as e:
            log.error(
                "Error during Chamber ID calculation", error=str(e), exc_info=True
            )
            col_chamber_id.fill(-1)  # Indicate error
    else:
        log.warning("SciPy not found, cannot calculate Chamber IDs.")
        col_chamber_id.fill(-1)  # Indicate missing

    # --- Create Polars DataFrame ---
    try:
        # Determine whether each cell is open above
        if higher_final_grid is not None and higher_final_grid.shape == final_grid.shape:
            col_open_above = higher_final_grid[non_rock_y,
                                               non_rock_x] != MAT_SOLID_ROCK
        else:
            if higher_final_grid is not None:
                log.warning(
                    "Higher stratum grid shape mismatch; defaulting open_above to True")
            col_open_above = np.ones(num_non_rock, dtype=bool)

        map_data = pl.DataFrame(
            {
                "x": col_x,  # Should be float for world coords
                "y": col_y,  # Should be float for world coords
                "floor_depth": col_floor_depth,
                "height": col_height,
                "ceiling_depth": col_ceil_depth,
                "material_id": col_mat_id,
                "walkable": col_walkable,
                "chamber_id": col_chamber_id,
                "open_above": col_open_above,
                "features": np.uint32(0),  # Placeholder bitmask
                "tags": np.uint32(0),  # Placeholder bitmask
                "stratum_index": np.uint8(0),  # Placeholder - Assume stratum 0
            }
        )
        print("Initial DataFrame created.")

    except Exception as e:
        log.error(
            "Error creating DataFrame or calculating open_above",
            error=str(e),
            exc_info=True,
        )
        return None

    return map_data


# === Main Orchestration Function (Modified to accept/pass rng) ===
def generate_shaped_cave(  # Added rng parameter
    augmented_nodes: list[Dict],
    augmented_node_map: Dict[int, Any],
    rng: GameRNG,
    ca_iterations: int = CA_ITERATIONS,
) -> Optional[pl.DataFrame]:
    """Orchestrates the cave shaping process using augmented node data."""
    # Removed global debug dump variables
    print("--- Stage: Initializing Cave Grid ---")
    # Pass rng down
    init_result = initialize_cave_grid(
        augmented_nodes, augmented_node_map, rng)
    # Check if grid creation failed
    if init_result is None or init_result[0] is None:
        log.error("Failed to initialize cave grid.")
        return None
    initial_grid, depth_grid, type_grid, origin = init_result
    print(f"Grid Initialized. Size: {
          initial_grid.shape}. Origin Offset: {origin}")

    print("\n--- Stage: Running Cellular Automata ---")
    final_grid = run_cellular_automata(initial_grid, iterations=ca_iterations)
    if final_grid is None:
        log.error("Cellular Automata failed.")
        return None
    print("Cellular Automata completed.")

    # Removed NPY dump call for post-CA state

    # Optionally save debug images (keep this for development)
    SAVE_DEBUG_IMAGES = True
    if SAVE_DEBUG_IMAGES:
        print("\n--- Stage: Saving Debug Images ---")
        try:
            import matplotlib.pyplot as plt  # Local import

            def save_img(filename, data, cmap):
                if np.any(data != 0) and np.any(
                    np.isfinite(data)
                ):  # Avoid saving empty/all-NaN
                    plt.imsave(filename, np.nan_to_num(data), cmap=cmap)
                    print(f"Saved {filename}")
                else:
                    print(f"Skipping save of empty/invalid {filename}.")

            save_img("debug_initial_grid.png", initial_grid, cmap="viridis")
            save_img("debug_final_grid.png", final_grid, cmap="viridis")
            save_img("debug_depth_grid.png", depth_grid, cmap="magma")
            save_img("debug_type_grid.png", type_grid, cmap="tab20")
        except ImportError:
            log.warning("Install matplotlib to save debug images.")
        except Exception as e:
            log.error("Error saving debug images", error=str(e), exc_info=True)

    print("\n--- Stage: Creating DataFrame ---")
    # Pass rng down
    map_dataframe = create_map_dataframe(
        final_grid, depth_grid, type_grid, origin, rng)
    if map_dataframe is not None:
        print(
            f"\nShaping complete. Generated DataFrame with {
                len(map_dataframe)} rows."
        )
    else:
        log.error("Failed to create DataFrame.")

    return map_dataframe


# === Example Usage (Modified for isolated testing) ===
if __name__ == "__main__":
    # This block should now only contain code for testing *shaper.py in isolation*
    print("Running shaper.py in isolation for testing...")
    # It needs to load test data (e.g., from a processed_cave_data.json)
    # and instantiate its own GameRNG.
    try:
        import json
        import time

        test_start_time = time.time()
        INPUT_JSON_FILE = "processed_cave_data.json"  # Assumes this exists
        OUTPUT_ARROW_FILE = "shaper_test_output.arrow"
        print(f"--- Loading Test Data from {INPUT_JSON_FILE} ---")

        with open(INPUT_JSON_FILE, "r") as f:
            processed_data = json.load(f)

        processed_nodes_list = processed_data.get("nodes", [])
        if not processed_nodes_list:
            log.error("No 'nodes' in test JSON.")
            sys.exit(1)

        processed_node_map = {n["id"]: n for n in processed_nodes_list}
        print(f"Loaded {len(processed_nodes_list)} test nodes.")

        # Instantiate a local RNG for testing
        test_rng_seed = 2  # Use a fixed seed for test consistency
        test_rng = GameRNG(seed=test_rng_seed)
        print(f"Using Test RNG Seed: {test_rng_seed}")

        print("\n--- Running Test Shaping Pipeline ---")
        # Call generate_shaped_cave with test data and RNG
        test_map = generate_shaped_cave(
            processed_nodes_list,
            processed_node_map,
            rng=test_rng,
            ca_iterations=4,  # Fewer iterations for faster test
        )

        if test_map is not None and not test_map.is_empty():
            print("\n--- Test Map DataFrame (Sample) ---")
            print(test_map.head())
            try:
                print(
                    f"\n--- Saving Test DataFrame to {OUTPUT_ARROW_FILE} ---")
                test_map.write_ipc(OUTPUT_ARROW_FILE)
                print("Test map saved successfully.")
            except Exception as e:
                log.error("Error saving test map DataFrame",
                          error=str(e), exc_info=True)
        elif test_map is not None:
            log.warning("Test map DataFrame is empty.")
        else:
            log.error("Test map generation pipeline failed.")

        test_end_time = time.time()
        print(
            "\nTotal shaper.py isolation test time: "
            f"{test_end_time - test_start_time:.2f} seconds"
        )

    except FileNotFoundError:
        log.error("Test input file not found", file=INPUT_JSON_FILE)
        print("Please generate it by running core.py and processor.py first.")
        print("Skipping isolated shaper test.")
    except Exception as e:
        log.error(
            "Error during isolated shaper test", error=str(e), exc_info=True
        )

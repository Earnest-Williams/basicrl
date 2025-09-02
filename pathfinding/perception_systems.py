#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perception_systems.py

Foundational implementation of noise, smell, and perception systems
inspired by Sil, optimized for Python using NumPy, Numba, Polars, and Joblib.

Designed as a starting point for iteration with custom map data.
"""

import os  # For determining CPU count
import time
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from game.world.los import line_of_sight
import structlog

from game.constants import FeatureType, FlowType, MAX_FLOWS
from game_rng import GameRNG

log = structlog.get_logger(__name__)

# --- Constants ---

# --- Configuration ---
# Noise/Smell parameters (from Sil analysis)
BASE_FLOW_CENTER: int = 100
NOISE_STRENGTH: int = 80  # Max propagation distance for noise cost calculation
NOISE_MAX_DIST: int = 200  # Clamping value for get_noise_dist
SMELL_STRENGTH: int = 80  # Threshold for scent aging/reset
SCENT_RESET_AGE: int = 250  # Base age for scent reset cycle
# Parallelism
# Determine job count with safe fallback when CPU count is unknown


def _job_count() -> int:
    """Half of available cores, or 1 if ``os.cpu_count()`` is unavailable."""
    try:
        cpu_count = os.cpu_count()
    except Exception:  # pragma: no cover - extremely unlikely
        cpu_count = 1
    return max(1, cpu_count // 2) if cpu_count else 1


N_JOBS: int = _job_count()


# --- Data Types ---
# Using NumPy structured dtype for queue items in Numba can sometimes be faster
# than tuples for simple types, but tuples are generally fine.
QueueItem = Tuple[int, int, int]  # (y, x, cost)

# --- Utility Functions (Numba Accelerated) ---

# Precompute neighbor offsets for speed within Numba loops
# Using a tuple of tuples is Numba-friendly
NEIGHBORS_8: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@njit(cache=True, fastmath=True)
def in_bounds(y: int, x: int, height: int, width: int) -> bool:
    """Checks if coordinates are within map bounds."""
    return 0 <= y < height and 0 <= x < width


@njit(cache=True, fastmath=True)
def cave_closed_door(feature_type: int) -> bool:
    """
    Checks if the feature type represents a closed or secret door.
    *** PLACEHOLDER - ADAPT TO YOUR FEATURE DEFINITIONS ***
    """
    # Example implementation - replace with your logic
    # Assumes FeatureType enum values are used in terrain_map
    return (
        feature_type == FeatureType.CLOSED_DOOR
        or feature_type == FeatureType.SECRET_DOOR
    )


# --- Noise System ---


@njit(cache=True, fastmath=True)
def _propagate_noise_kernel(
    # Single slice for the current flow; shape (height, width)
    cost_grid: np.ndarray,
    terrain_map: np.ndarray,  # Map features; shape (height, width)
    start_y: int,
    start_x: int,  # Origin of the noise
    start_cost: int,  # e.g., BASE_FLOW_CENTER
    max_dist_prop: int,  # e.g., NOISE_STRENGTH
    door_pass_penalty: int,  # Penalty for PASS_DOORS flow
    door_real_penalty: int,  # Penalty for REAL_NOISE/MONSTER_NOISE flows
    flow_type: int,  # Integer value from FlowType enum
) -> None:
    """
    Core Numba kernel for propagating noise cost via BFS/Dijkstra-like approach.
    Modifies cost_grid in-place.

    Args:
        cost_grid: 2D NumPy array to store costs for this flow type.
        terrain_map: 2D NumPy array with feature types for the map.
        start_y, start_x: Coordinates of the noise source.
        start_cost: Initial cost value at the source.
        max_dist_prop: Maximum cost difference from start_cost to propagate.
        door_pass_penalty: Added cost for FLOW_PASS_DOORS through closed doors.
        door_real_penalty: Added cost for FLOW_REAL_NOISE/MONSTER_NOISE through closed doors.
        flow_type: The type of flow being calculated (determines door handling).
    """
    height, width = cost_grid.shape
    # Use a large int value represent unreachable/infinity, avoid overflow
    infinity: int = np.iinfo(cost_grid.dtype).max // 2
    cost_grid.fill(infinity)

    if not in_bounds(start_y, start_x, height, width):
        return  # Start is outside map

    # Check if start location itself is blocked (e.g., inside a wall)
    if terrain_map[start_y, start_x] == FeatureType.WALL:
        # Cannot start noise inside a wall - might need refinement based on game rules
        # Or maybe allow it for specific effects? For now, just exit.
        return

    cost_grid[start_y, start_x] = start_cost
    # Numba doesn't directly support deque with heterogeneous types easily,
    # but works well with deque of simple tuples like our QueueItem.
    queue: Deque[QueueItem] = deque([(start_y, start_x, start_cost)])

    while queue:
        y, x, current_cost = queue.popleft()

        # Check propagation distance limit relative to the starting cost
        if current_cost - start_cost >= max_dist_prop:
            continue

        for dy, dx in NEIGHBORS_8:
            ny, nx = y + dy, x + dx

            # Check bounds for the neighbor
            if in_bounds(ny, nx, height, width):
                terrain_feature: int = terrain_map[ny, nx]
                cost_increase: int = 1  # Base cost to move to adjacent tile

                # --- Cost Calculation Logic ---
                # 1. Check for impassable terrain (walls)
                if terrain_feature == FeatureType.WALL:
                    continue  # Blocked by wall

                # 2. Check for doors and apply penalties/blocking based on flow
                # type
                is_closed: bool = cave_closed_door(terrain_feature)
                if is_closed:
                    if flow_type == FlowType.NO_DOORS:
                        continue  # Blocked for this flow type
                    elif flow_type == FlowType.PASS_DOORS:
                        # Note: adds penalty ON TOP of base cost_increase
                        cost_increase += door_pass_penalty
                    elif (
                        flow_type == FlowType.REAL_NOISE
                        or flow_type == FlowType.MONSTER_NOISE
                    ):
                        cost_increase += door_real_penalty
                    # Else: Open doors or other features have default
                    # cost_increase = 1

                # 3. Calculate potential new cost
                # Diagonal movement could optionally cost more (e.g., sqrt(2) ~ 1.41)
                # For simplicity here, diagonal = 1, same as orthogonal
                # If using floats: cost_increase = 1.414 if abs(dx) == 1 and
                # abs(dy) == 1 else 1.0
                new_cost: int = current_cost + cost_increase
                # --- End Cost Calculation ---

                # If we found a cheaper path, update cost and add to queue
                if new_cost < cost_grid[ny, nx]:
                    cost_grid[ny, nx] = new_cost
                    queue.append((ny, nx, new_cost))


def update_noise(
    cave_cost: np.ndarray,  # Full 3D cost array: (MAX_FLOWS, height, width)
    # Array to store centers: (MAX_FLOWS, 2) for y, x
    flow_centers: np.ndarray,
    terrain_map: np.ndarray,  # Map features: (height, width)
    cy: int,
    cx: int,  # Center coordinates for this update
    which_flow: FlowType,  # Which flow map to update
    penalties: Dict[str, int],  # Dict with 'pass' and 'real' penalties
) -> None:
    """
    Updates a specific noise flow map by calling the Numba kernel.

    Args:
        cave_cost: The main 3D array holding all cost maps.
        flow_centers: Array where the center (cy, cx) of this flow will be stored.
        terrain_map: The 2D map terrain data.
        cy, cx: The origin coordinates of the noise source.
        which_flow: The FlowType enum member indicating which map slice to update.
        penalties: Dictionary containing 'pass' and 'real' door cost penalties.
    """
    flow_idx: int = int(which_flow)  # Get integer index from Enum

    if not (0 <= flow_idx < MAX_FLOWS):
        log.error("Invalid flow index", flow_idx=flow_idx)
        return

    # Select the specific 2D cost grid slice for this flow type
    cost_grid_slice: np.ndarray = cave_cost[flow_idx]

    # Store the center of this flow field
    flow_centers[flow_idx, 0] = cy
    flow_centers[flow_idx, 1] = cx

    # Call the Numba kernel to perform the propagation
    _propagate_noise_kernel(
        cost_grid=cost_grid_slice,
        terrain_map=terrain_map,
        start_y=cy,
        start_x=cx,
        start_cost=BASE_FLOW_CENTER,
        max_dist_prop=NOISE_STRENGTH,
        # Default from Sil if missing
        door_pass_penalty=penalties.get("pass", 3),
        # Default from Sil if missing
        door_real_penalty=penalties.get("real", 5),
        flow_type=flow_idx,  # Pass the integer value
    )


def compute_noise_map(
    terrain_map: np.ndarray,
    cy: int,
    cx: int,
    map_size: Tuple[int, int],
    which_flow: FlowType = FlowType.REAL_NOISE,
    penalties: Dict[str, int] | None = None,
) -> np.ndarray:
    """Convenience wrapper that returns a noise map slice.

    Args:
        terrain_map: The terrain feature map.
        cy, cx: Origin coordinates of the noise source.
        map_size: ``(height, width)`` describing the map dimensions.
        which_flow: Flow type to compute.
        penalties: Door penalty configuration.
    """

    height, width = map_size

    try:
        cave_cost = np.zeros((MAX_FLOWS, height, width), dtype=np.int32)
        flow_centers = np.zeros((MAX_FLOWS, 2), dtype=np.int32)
        update_noise(
            cave_cost,
            flow_centers,
            terrain_map,
            cy,
            cx,
            which_flow,
            penalties or {},
        )
        return cave_cost[int(which_flow)]
    except Exception:
        # Fallback to a simple Python implementation if the Numba kernel is
        # unavailable (e.g., during testing environments without Numba
        # compilation support).
        return _compute_noise_map_python(
            terrain_map, cy, cx, map_size, which_flow, penalties or {}
        )


def _compute_noise_map_python(
    terrain_map: np.ndarray,
    cy: int,
    cx: int,
    map_size: Tuple[int, int],
    which_flow: FlowType,
    penalties: Dict[str, int],
) -> np.ndarray:
    height, width = map_size
    infinity = np.iinfo(np.int32).max // 2
    cost_grid = np.full((height, width), infinity, dtype=np.int32)
    if not in_bounds(cy, cx, height, width):
        return cost_grid
    if terrain_map[cy, cx] == FeatureType.WALL:
        return cost_grid

    cost_grid[cy, cx] = BASE_FLOW_CENTER
    queue: Deque[QueueItem] = deque([(cy, cx, BASE_FLOW_CENTER)])

    while queue:
        y, x, current = queue.popleft()
        if current - BASE_FLOW_CENTER >= NOISE_STRENGTH:
            continue
        for dy, dx in NEIGHBORS_8:
            ny, nx = y + dy, x + dx
            if not in_bounds(ny, nx, height, width):
                continue
            terrain_feature = terrain_map[ny, nx]
            if terrain_feature == FeatureType.WALL:
                continue
            cost_increase = 1
            is_closed = cave_closed_door(terrain_feature)
            if is_closed:
                if which_flow == FlowType.NO_DOORS:
                    continue
                elif which_flow == FlowType.PASS_DOORS:
                    cost_increase += penalties.get("pass", 3)
                elif which_flow in (FlowType.REAL_NOISE, FlowType.MONSTER_NOISE):
                    cost_increase += penalties.get("real", 5)
            new_cost = current + cost_increase
            if new_cost < cost_grid[ny, nx]:
                cost_grid[ny, nx] = new_cost
                queue.append((ny, nx, new_cost))
    return cost_grid


def get_noise_dist(
    cave_cost: np.ndarray,  # Full 3D cost array
    flow_centers: np.ndarray,  # Array containing flow centers
    which_flow: FlowType,  # Which flow map to query
    y: int,
    x: int,  # Coordinates to get the distance for
) -> int:
    """
    Calculates the noise distance from the flow's center to grid (y, x).
    Handles boundary checks and clamping based on Sil logic.

    Args:
        cave_cost: The main 3D array holding all cost maps.
        flow_centers: Array storing the origin coordinates for each flow type.
        which_flow: The FlowType enum member indicating which map slice to query.
        y, x: The target coordinates.

    Returns:
        The calculated noise distance, clamped to NOISE_MAX_DIST if negative or out of bounds.
    """
    flow_idx: int = int(which_flow)
    height, width = cave_cost.shape[1], cave_cost.shape[2]

    if not (0 <= flow_idx < MAX_FLOWS):
        log.error("Invalid flow index in get_noise_dist", flow_idx=flow_idx)
        return NOISE_MAX_DIST  # Return max distance on error

    if not in_bounds(y, x, height, width):
        return NOISE_MAX_DIST  # Out of bounds

    # Retrieve cost at target and cost at center for this flow
    cost_at_target: int = cave_cost[flow_idx, y, x]
    # Center cost is fixed at BASE_FLOW_CENTER when update_noise runs
    cost_at_center: int = BASE_FLOW_CENTER

    # If cost_at_target is infinity, it's unreachable
    infinity: int = np.iinfo(cave_cost.dtype).max // 2
    if cost_at_target >= infinity:
        return NOISE_MAX_DIST

    # Calculate distance relative to the center cost
    noise_dist: int = cost_at_target - cost_at_center

    # Clamp negative results to NOISE_MAX_DIST (as per Sil C code)
    # This happens if the target is "behind" the propagation start direction implicitly
    # or if cost somehow became lower than start_cost (shouldn't happen with
    # BFS)
    if noise_dist < 0:
        noise_dist = NOISE_MAX_DIST

    return noise_dist


# --- Smell (Scent) System ---

# Precompute scent adjustments relative to player center (from Sil analysis)
# 0=center, 1=adjacent, 2=diagonal/further, 250=ignore
# Using NumPy for potential future vectorization if needed, though loop is
# small.
SCENT_ADJUST_TABLE = np.array(
    [
        [250, 2, 2, 2, 250],
        [2, 1, 1, 1, 2],
        [2, 1, 0, 1, 2],
        [2, 1, 1, 1, 2],
        [250, 2, 2, 2, 250],
    ],
    dtype=np.int32,
)


@njit(
    cache=True, parallel=False
)  # Parallel=False as loop is small and writes to shared array
def _lay_scent_kernel(
    cave_when: np.ndarray,  # The 2D scent age map
    terrain_map: np.ndarray,  # Map features
    py: int,
    px: int,  # Player (scent source) coordinates
    current_scent_when: int,  # Current global scent timer value
    scent_adjust: np.ndarray,  # The 5x5 adjustment table
) -> None:
    """
    Numba kernel to lay down scent in a 5x5 area around the player.
    Modifies cave_when in-place.

    Args:
        cave_when: 2D NumPy array storing scent age.
        terrain_map: 2D NumPy array with map feature types.
        py, px: Player coordinates.
        current_scent_when: The current global scent age value.
        scent_adjust: The 5x5 NumPy array defining relative scent freshness.
    """
    height, width = cave_when.shape
    table_size = scent_adjust.shape[0]  # Should be 5
    offset = table_size // 2  # Should be 2

    for i in range(table_size):
        for j in range(table_size):
            y: int = py + i - offset
            x: int = px + j - offset

            # Check bounds first
            if not in_bounds(y, x, height, width):
                continue

            adjustment: int = scent_adjust[i, j]

            # Ignore grids marked as too far in the table
            if adjustment == 250:
                continue

            # Walls cannot hold scent (adapt check to your FeatureType enum)
            if terrain_map[y, x] == FeatureType.WALL:
                continue

            # Grid must not be blocked by walls from the character
            if not line_of_sight(py, px, y, x, terrain_map != FeatureType.WALL):
                continue

            # Mark the grid with new scent age
            # Newer scent = lower adjustment value added to base 'scent_when'
            cave_when[y, x] = current_scent_when + adjustment


def update_smell(
    cave_when: np.ndarray,  # The 2D scent age map (modified in-place)
    terrain_map: np.ndarray,  # Map features
    py: int,
    px: int,  # Player coordinates
    global_scent_when: int,  # Current global scent timer value (mutable!)
) -> int:
    """
    Updates the scent map: ages existing scent and lays down new scent.

    Args:
        cave_when: 2D NumPy array storing scent age. Will be modified.
        terrain_map: 2D NumPy array with map feature types.
        py, px: Player coordinates for laying new scent.
        global_scent_when: The current global scent timer.

    Returns:
        The updated global_scent_when value.
    """
    # 1. Age the global timer
    global_scent_when -= 1

    # 2. Handle scent cycle reset using NumPy vectorization
    if global_scent_when <= 0:
        # Determine the reset offset based on constants
        reset_offset: int = SCENT_RESET_AGE - SMELL_STRENGTH

        # Identify scent older than the strength threshold
        # Note: In Sil C code, '>' was used. If age decreases, this should be '<='.
        # Assuming lower value means older scent. Adjust if higher value means older.
        # Let's assume lower value = older (needs confirmation from Sil execution trace)
        # If scent_when counts down, a smaller value is older.
        # cave_when[y][x] <= SMELL_STRENGTH would mean it's old.

        # *** Reinterpreting Sil C code based on variable names ***
        # 'scent_when' decreases. cave_when[y][x] stores 'scent_when + adjustment'.
        # Higher cave_when values are newer.
        # Reset logic from Sil C code:
        # if (cave_when[y][x] > SMELL_STRENGTH) cave_when[y][x] = 0; // Erase old
        # else cave_when[y][x] = 250 - SMELL_STRENGTH + cave_when[y][x]; //
        # Reset recent

        # Vectorized equivalent:
        is_old_scent: np.ndarray = cave_when > SMELL_STRENGTH
        is_recent_scent: np.ndarray = (cave_when > 0) & (~is_old_scent)

        # Erase old scent
        cave_when[is_old_scent] = 0

        # Reset age of recent scent relative to the new cycle start
        # cave_when[is_recent_scent] = reset_offset + cave_when[is_recent_scent] # This was slightly off
        # The C code implies resetting based on the value *before* the new cycle adjustment.
        # It seems like it's shifting the valid range. Let's try to match that:
        # New value = (New Cycle Base) + (How far it was into the *old* valid range)
        # Old valid range was roughly [1, SMELL_STRENGTH].
        # New cycle base is 'reset_offset'.
        # Let's stick closer to the C formula:
        cave_when[is_recent_scent] = reset_offset + cave_when[is_recent_scent]
        # Caveat: This needs careful testing against C implementation behavior.

        # Reset the global timer
        global_scent_when = reset_offset
        print(f"Scent cycle reset. New base age: {global_scent_when}")

    # 3. Lay down new scent using the Numba kernel
    _lay_scent_kernel(
        cave_when=cave_when,
        terrain_map=terrain_map,
        py=py,
        px=px,
        current_scent_when=global_scent_when,
        scent_adjust=SCENT_ADJUST_TABLE,
    )

    return global_scent_when


def get_scent(cave_when: np.ndarray, y: int, x: int) -> int:
    """
    Gets the scent age at a specific location.

    Args:
        cave_when: 2D NumPy array storing scent age.
        y, x: Target coordinates.

    Returns:
        Scent age value, or -1 if out of bounds or no scent (age 0).
    """
    height, width = cave_when.shape
    if not in_bounds(y, x, height, width):
        return -1

    age: int = cave_when[y, x]

    # Return -1 if no scent (age 0) or if technically invalid
    return age if age > 0 else -1


# --- Monster Data Representation (Polars DataFrame) ---


def initialize_monsters(num_monsters: int, height: int, width: int) -> pl.DataFrame:
    """Creates a Polars DataFrame with sample monster data."""
    monster_data = {
        "id": range(num_monsters),
        # Example race index
        "r_idx": np.random.randint(1, 5, size=num_monsters),
        "fy": np.random.randint(0, height, size=num_monsters),
        "fx": np.random.randint(0, width, size=num_monsters),
        "is_dead": np.zeros(num_monsters, dtype=bool),
        "perception_stat": np.random.randint(
            5, 15, size=num_monsters
        ),  # Base perception
        "alertness": np.random.randint(
            -10, 1, size=num_monsters
        ),  # Example: mostly unwary/asleep
        # Add flags based on Sil RF flags if needed (e.g., RF2_SHORT_SIGHTED)
        # Placeholder for bitflags
        "flags": np.zeros(num_monsters, dtype=np.uint32),
        "heard_player": np.zeros(
            num_monsters, dtype=bool
        ),  # Track if detected this turn
    }
    # Create Polars DataFrame
    # Using lazy API allows for potential optimizations on larger datasets
    # but for this example, eager is fine.
    monster_df = pl.DataFrame(monster_data)
    return monster_df


# --- Perception System (Parallelized) ---


def skill_check(
    rng: GameRNG, actor_skill: int, difficulty: int, target_skill: int
) -> bool:
    """Deterministic d100 skill check using :class:`GameRNG`.

    The check succeeds if a d100 roll plus the actor's skill exceeds the
    combined difficulty and target skill.

    Args:
        rng: The deterministic random number generator.
        actor_skill: Skill value of the acting entity (e.g., monster perception).
        difficulty: Difficulty modifier (e.g., based on noise distance).
        target_skill: Skill value of the resisting entity (e.g., player stealth).

    Returns:
        True if the check succeeds, False otherwise.
    """
    roll = rng.get_int(1, 100)
    threshold = 50 + actor_skill - (difficulty + target_skill)
    return roll <= threshold


def _process_monster_perception_chunk(
    monster_df_chunk: pl.DataFrame,  # Chunk of the monster DataFrame
    cave_cost: np.ndarray,  # Full 3D noise cost map
    flow_centers: np.ndarray,  # Noise flow centers
    player_stealth_skill: int,  # Player's current stealth value
    noise_flow_type: FlowType,  # Which noise map to use (e.g., REAL_NOISE)
    rng: GameRNG,
) -> List[int]:  # Return list of IDs of monsters that detected player
    """
    Processes perception checks for a chunk of monsters.
    Called by Joblib workers.

    Args:
        monster_df_chunk: A Polars DataFrame subset containing monsters to process.
        cave_cost: The 3D NumPy array of noise costs.
        flow_centers: NumPy array of flow center coordinates.
        player_stealth_skill: The player's relevant skill (e.g., stealth).
        noise_flow_type: The FlowType to use for noise distance checks.

    Returns:
        A list of monster IDs from this chunk that successfully detected the player.
    """
    alerted_monster_ids: List[int] = []

    # Iterate through the rows of the DataFrame chunk
    for row in monster_df_chunk.iter_rows(named=True):
        monster_id: int = row["id"]
        m_y: int = row["fy"]
        m_x: int = row["fx"]
        perception: int = row["perception_stat"]
        # Add checks for flags like RF2_SHORT_SIGHTED if implemented

        # 1. Get Noise Distance
        noise_dist: int = get_noise_dist(
            cave_cost, flow_centers, noise_flow_type, m_y, m_x
        )

        # Optimization: If noise distance is max, monster can't hear
        if noise_dist >= NOISE_MAX_DIST:
            continue

        # 2. Perform Perception Check (using placeholder skill_check)
        # Difficulty increases with noise distance (harder to hear far away)
        # Player stealth skill resists the check.
        difficulty_mod: int = noise_dist  # Simple difficulty based on distance

        # Add other factors: lighting, player state (singing?), monster state?

        if skill_check(
            rng,
            actor_skill=perception,
            difficulty=difficulty_mod,
            target_skill=player_stealth_skill,
        ):
            # --- Perception Success ---
            alerted_monster_ids.append(monster_id)
            # Potential actions:
            # - Set monster's heard_player flag (if modifying DF directly - tricky with parallel)
            # - Update monster alertness (needs careful handling with parallel results)
            # - Trigger AI state change (best handled after collecting all results)
            # print(f"  Monster {monster_id} heard player! (Noise Dist:
            # {noise_dist})") # DEBUG

    return alerted_monster_ids


def monster_perception(
    monster_df: pl.DataFrame,  # Full monster DataFrame
    cave_cost: np.ndarray,  # Full 3D noise cost map
    flow_centers: np.ndarray,  # Noise flow centers
    player_y: int,
    player_x: int,  # Player position (needed?) - implicitly noise center
    player_stealth_skill: int,  # Player's current stealth value
    rng: GameRNG,
    noise_flow_type: FlowType = FlowType.REAL_NOISE,
    # Add other relevant params: player_attacked_recently?, main_roll?,
    # difficulty?
) -> List[int]:
    """
    Updates monster perception based on noise. Parallelized using Joblib.

    Args:
        monster_df: The Polars DataFrame containing all monster data.
        cave_cost: The 3D NumPy array of noise costs.
        flow_centers: NumPy array of flow center coordinates.
        player_y, player_x: Current player coordinates (primarily for context).
        player_stealth_skill: Player's relevant skill value.
        rng: Deterministic random number generator.
        noise_flow_type: Which noise flow map to use for perception.

    Returns:
        A list containing the IDs of all monsters that detected the player this turn.
    """
    start_time = time.time()

    # --- Filter Monsters ---
    # Select only monsters that are alive and potentially relevant
    # (e.g., within a max perception range if known, or just !is_dead)
    active_monsters_df = monster_df.filter(pl.col("is_dead") is False)

    num_monsters = active_monsters_df.height
    if num_monsters == 0:
        return []

    # --- Parallel Processing ---
    # Split the DataFrame into chunks for parallel processing
    # Adjust n_chunks based on N_JOBS and number of monsters
    # Heuristic: more chunks than jobs
    n_chunks = min(num_monsters, N_JOBS * 4)
    if n_chunks <= 0:
        n_chunks = 1

    chunk_size = (num_monsters + n_chunks - 1) // n_chunks  # Ceiling division

    # Create list of DataFrame chunks (this involves copying data, consider
    # indices if memory critical)
    df_chunks = [
        active_monsters_df.slice(i * chunk_size, chunk_size) for i in range(n_chunks)
    ]

    # Use Joblib to process chunks in parallel
    # backend="loky" is default and generally robust
    # backend="threading" might be suitable if skill_check involves I/O or GIL-releasing C code
    # backend="multiprocessing" is another option
    results: List[List[int]] = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_process_monster_perception_chunk)(
            chunk,
            cave_cost,
            flow_centers,
            player_stealth_skill,
            noise_flow_type,
            rng,
        )
        for chunk in df_chunks
    )

    # --- Aggregate Results ---
    all_alerted_ids: List[int] = [item for sublist in results for item in sublist]

    end_time = time.time()
    print(
        f"Monster perception for {num_monsters} monsters took "
        f"{end_time - start_time:.4f}s using {N_JOBS} jobs."
    )

    # --- Update Monster State (Post-Parallel) ---
    # Now that we have all alerted IDs, update the main DataFrame or trigger AI
    if all_alerted_ids:
        print(f"Monsters alerted: {all_alerted_ids}")
        # Example: Update a 'heard_player' flag in the main DataFrame
        # This is safer than trying to modify the DF from parallel workers.
        # monster_df = monster_df.with_columns(
        #     pl.when(pl.col('id').is_in(all_alerted_ids))
        #     .then(True)
        #     .otherwise(pl.col('heard_player'))
        #     .alias('heard_player')
        # )
        # Note: Reassignment needed as Polars DataFrames are immutable by default.
        # Or, pass these IDs to an AI system to update alertness/state.

    return all_alerted_ids  # Return the list of IDs


# --- Main Execution / Example Usage ---

if __name__ == "__main__":
    print("Initializing perception systems demo...")

    # --- Initialize Map Data (Placeholders) ---
    # Replace with your actual map loading/generation
    height, width = 64, 64
    # Terrain map: integers representing FeatureType
    terrain_map = np.full((height, width), FeatureType.FLOOR, dtype=np.int32)
    # Add some walls and a closed door for testing noise propagation
    terrain_map[height // 2, width // 4 : 3 * width // 4] = FeatureType.WALL
    terrain_map[height // 2 + 5, width // 2] = FeatureType.CLOSED_DOOR
    # Add boundary walls
    terrain_map[0, :] = FeatureType.WALL
    terrain_map[height - 1, :] = FeatureType.WALL
    terrain_map[:, 0] = FeatureType.WALL
    terrain_map[:, width - 1] = FeatureType.WALL

    # Noise cost map: Init with a large value (infinity)
    infinity_val = np.iinfo(np.int32).max // 2
    cave_cost = np.full((MAX_FLOWS, height, width), infinity_val, dtype=np.int32)
    # Store (y, x) centers
    flow_centers = np.zeros((MAX_FLOWS, 2), dtype=np.int32)

    # Scent map: Init with 0 (no scent)
    cave_when = np.zeros((height, width), dtype=np.int32)
    global_scent_when: int = SCENT_RESET_AGE  # Initial scent timer

    # Define door penalties
    door_penalties = {"pass": 3, "real": 5}

    # --- Initialize Player and Monsters ---
    player_y, player_x = height // 2 + 10, width // 2
    player_stealth_skill = 10  # Example skill value
    rng = GameRNG(seed=123)  # Deterministic RNG for demo

    num_monsters = 500  # Example number of monsters for testing parallelism
    monster_df = initialize_monsters(num_monsters, height, width)
    # Ensure monsters don't start inside walls (simple repositioning)
    monster_df = monster_df.with_columns(
        pl.when(terrain_map[pl.col("fy"), pl.col("fx")] == FeatureType.WALL)
        .then(player_y)  # Move to player Y if in wall (crude fix)
        .otherwise(pl.col("fy"))
        .alias("fy"),
        pl.when(terrain_map[pl.col("fy"), pl.col("fx")] == FeatureType.WALL)
        .then(player_x + 1)  # Move to player X+1 if in wall (crude fix)
        .otherwise(pl.col("fx"))
        .alias("fx"),
    )

    print(
        f"Map: {height}x{width}, Player: ({player_y}, {player_x}), Monsters: {num_monsters}"
    )
    print(f"Using {N_JOBS} parallel jobs for perception.")

    # --- Simulation Loop (Simplified Example) ---
    for turn in range(5):
        print(f"\n--- Turn {turn + 1} ---")

        # 1. Update Noise (originate from player for this example)
        noise_start_time = time.time()
        # Update the flow maps relevant for perception/pathing
        update_noise(
            cave_cost,
            flow_centers,
            terrain_map,
            player_y,
            player_x,
            FlowType.REAL_NOISE,
            door_penalties,
        )
        update_noise(
            cave_cost,
            flow_centers,
            terrain_map,
            player_y,
            player_x,
            FlowType.PASS_DOORS,
            door_penalties,
        )
        # Add update_noise for NO_DOORS if needed for pathfinding logic
        print(f"Noise update took: {time.time() - noise_start_time:.4f}s")

        # 2. Update Smell
        smell_start_time = time.time()
        global_scent_when = update_smell(
            cave_when, terrain_map, player_y, player_x, global_scent_when
        )
        print(f"Smell update took: {time.time() - smell_start_time:.4f}s")

        # 3. Monster Perception
        alerted_ids = monster_perception(
            monster_df,
            cave_cost,
            flow_centers,
            player_y,
            player_x,
            player_stealth_skill,
            rng,
        )
        # Here you would typically update the state of alerted monsters in monster_df
        # or pass alerted_ids to an AI processing system.

        # --- Example Lookups ---
        if num_monsters > 0:
            example_monster_id = 0
            m_info = monster_df.filter(pl.col("id") == example_monster_id)
            if m_info.height > 0:
                m_y, m_x = m_info.select(["fy", "fx"]).row(0)
                noise = get_noise_dist(
                    cave_cost, flow_centers, FlowType.REAL_NOISE, m_y, m_x
                )
                scent = get_scent(cave_when, m_y, m_x)
                print(
                    f"Example Monster {example_monster_id} at ({m_y}, {m_x}): Noise Dist={noise}, Scent Age={scent}"
                )

        # (Player moves, monsters move, combat happens, etc. - not implemented)
        # Simple player movement example
        player_x += 1
        if player_x >= width - 1:
            player_x = 1  # Basic wrap/reset

    print("\nDemo finished.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perception_systems.py

Foundational implementation of noise, smell, and perception systems
inspired by Sil, optimized for Python using NumPy, Numba, Polars, and Joblib.
Includes structured logging via structlog.

Designed as a starting point for iteration with custom map data.
"""

import logging  # Standard logging library backend
import logging.handlers  # For rotating file handler
import os  # For determining CPU count
import sys  # For console logging handler
import time
import uuid  # For simulation ID
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import polars as pl
import structlog  # Structured logging
from joblib import Parallel, delayed
from numba import njit
from game.world.los import line_of_sight

from game.constants import FeatureType, FlowType, MAX_FLOWS
from game_rng import GameRNG

# --- Constants ---
SIM_ID = str(uuid.uuid4())[:8]  # Unique ID for this simulation run

# --- Configuration ---
# Map dimensions (replace with actual dimensions)
MAP_HGT: int = 64
MAP_WID: int = 64
# Noise/Smell parameters (from Sil analysis)
BASE_FLOW_CENTER: int = 100
NOISE_STRENGTH: int = 80  # Max propagation distance for noise cost calculation
NOISE_MAX_DIST: int = 200  # Clamping value for get_noise_dist
SMELL_STRENGTH: int = 80  # Threshold for scent aging/reset
SCENT_RESET_AGE: int = 250  # Base age for scent reset cycle
# Parallelism
N_JOBS: int = max(1, os.cpu_count() // 2)  # Use half the CPU cores for Joblib
# Logging Configuration
LOG_LEVEL_CONSOLE = logging.INFO
LOG_LEVEL_FILE = logging.DEBUG
LOG_FILENAME = f"perception_sim_{SIM_ID}.log"


# --- Logging Setup ---
# Custom processor to add simulation context consistently
def add_sim_context(logger, method_name, event_dict):
    event_dict["sim_id"] = SIM_ID
    return event_dict


def setup_logging(
    log_level=LOG_LEVEL_CONSOLE, file_log_level=LOG_LEVEL_FILE, filename=LOG_FILENAME
):
    """Configures structlog for console and rotating file output."""
    shared_processors = [
        add_sim_context,  # Add our custom sim_id
        structlog.stdlib.add_log_level,  # Add 'level' key
        structlog.stdlib.add_logger_name,  # Add logger name
        structlog.stdlib.PositionalArgumentsFormatter(),  # Format positional args
        structlog.processors.TimeStamper(
            fmt="iso", utc=True),  # Add ISO timestamp
        structlog.processors.StackInfoRenderer(),  # Add stack info on exceptions
        structlog.processors.format_exc_info,  # Format exception info
    ]

    structlog.configure(
        processors=shared_processors
        + [
            # Prepare for stdlib formatter
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Define formatter for stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),  # Render final output as JSON
    )

    # Set up root logger (backend for structlog)
    root_logger = logging.getLogger()
    # Set root logger level to the *lowest* level needed by any handler
    root_logger.setLevel(min(log_level, file_log_level))
    # Clear existing handlers to avoid duplicates if run multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation for persistent logs
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=5_000_000, backupCount=3, encoding="utf-8"  # 5MB
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(
            f"Logging configured. Console: {
                logging.getLevelName(log_level)}, File: {
                logging.getLevelName(file_log_level)} ({filename})",
            file=sys.stderr,
        )
    except Exception as e:
        # Fallback if file logging fails (e.g., permissions)
        print(
            f"Error setting up file logger ({filename}): {
                e}. Logging to console only.",
            file=sys.stderr,
        )

    # Return a base structlog logger
    return structlog.get_logger("perception_systems")


# Initialize logger globally after setup function definition
log = setup_logging()

# --- Data Types ---
QueueItem = Tuple[int, int, int]  # (y, x, cost)

# --- Utility Functions (Numba Accelerated) ---

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
    return (
        feature_type == FeatureType.CLOSED_DOOR
        or feature_type == FeatureType.SECRET_DOOR
    )


# --- Noise System ---


@njit(cache=True, fastmath=True)
def _propagate_noise_kernel(
    cost_grid: np.ndarray,
    terrain_map: np.ndarray,
    start_y: int,
    start_x: int,
    start_cost: int,
    max_dist_prop: int,
    door_pass_penalty: int,
    door_real_penalty: int,
    flow_type: int,
) -> None:
    """
    Core Numba kernel for propagating noise cost via BFS/Dijkstra-like approach.
    Modifies cost_grid in-place. Numba prevents direct logging from here.
    """
    height, width = cost_grid.shape
    infinity: int = np.iinfo(cost_grid.dtype).max // 2
    cost_grid.fill(infinity)

    if not in_bounds(start_y, start_x, height, width):
        # Cannot log directly from njit function in nopython mode.
        # Warnings must be handled in the calling Python function if needed.
        return

    if terrain_map[start_y, start_x] == FeatureType.WALL:
        # Cannot log directly from njit function.
        return

    cost_grid[start_y, start_x] = start_cost
    queue: Deque[QueueItem] = deque([(start_y, start_x, start_cost)])

    while queue:
        y, x, current_cost = queue.popleft()

        if current_cost - start_cost >= max_dist_prop:
            continue

        for dy, dx in NEIGHBORS_8:
            ny, nx = y + dy, x + dx

            if in_bounds(ny, nx, height, width):
                terrain_feature: int = terrain_map[ny, nx]
                cost_increase: int = 1

                if terrain_feature == FeatureType.WALL:
                    continue

                is_closed: bool = cave_closed_door(terrain_feature)
                if is_closed:
                    if flow_type == FlowType.NO_DOORS:
                        continue
                    elif flow_type == FlowType.PASS_DOORS:
                        cost_increase += door_pass_penalty
                    elif (
                        flow_type == FlowType.REAL_NOISE
                        or flow_type == FlowType.MONSTER_NOISE
                    ):
                        cost_increase += door_real_penalty

                new_cost: int = current_cost + cost_increase

                if new_cost < cost_grid[ny, nx]:
                    cost_grid[ny, nx] = new_cost
                    queue.append((ny, nx, new_cost))


def update_noise(
    cave_cost: np.ndarray,
    flow_centers: np.ndarray,
    terrain_map: np.ndarray,
    cy: int,
    cx: int,
    which_flow: FlowType,
    penalties: Dict[str, int],
) -> None:
    """Updates a specific noise flow map by calling the Numba kernel."""
    flow_idx: int = int(which_flow)
    logger = log.bind(flow_type=which_flow.name, start_y=cy,
                      start_x=cx)  # Bind context

    if not (0 <= flow_idx < MAX_FLOWS):
        logger.error("invalid_flow_index", flow_index=flow_idx)
        return

    height, width = terrain_map.shape
    if not in_bounds(cy, cx, height, width):
        logger.warning("noise_start_out_of_bounds")
        # Kernel will also return early, but log here.
        # Avoid filling cost grid if start invalid? Depends on desired
        # behavior.

    elif terrain_map[cy, cx] == FeatureType.WALL:
        logger.warning("noise_start_inside_wall")
        # Kernel will also return early.

    cost_grid_slice: np.ndarray = cave_cost[flow_idx]
    flow_centers[flow_idx, 0] = cy
    flow_centers[flow_idx, 1] = cx

    # Call the Numba kernel
    _propagate_noise_kernel(
        cost_grid=cost_grid_slice,
        terrain_map=terrain_map,
        start_y=cy,
        start_x=cx,
        start_cost=BASE_FLOW_CENTER,
        max_dist_prop=NOISE_STRENGTH,
        door_pass_penalty=penalties.get("pass", 3),
        door_real_penalty=penalties.get("real", 5),
        flow_type=flow_idx,
    )
    # Note: Cannot log directly from within the Numba kernel about
    # progress/details


def get_noise_dist(
    cave_cost: np.ndarray,
    flow_centers: np.ndarray,
    which_flow: FlowType,
    y: int,
    x: int,
) -> int:
    """Calculates the noise distance from the flow's center to grid (y, x)."""
    flow_idx: int = int(which_flow)
    height, width = cave_cost.shape[1], cave_cost.shape[2]

    if not (0 <= flow_idx < MAX_FLOWS):
        # Use logger instead of print
        log.error(
            "invalid_flow_index_lookup", flow_index=flow_idx, target_y=y, target_x=x
        )
        return NOISE_MAX_DIST

    if not in_bounds(y, x, height, width):
        # Log this condition might be noisy if called often for OOB monsters
        # log.debug("noise_target_out_of_bounds", target_y=y, target_x=x, flow_type=which_flow.name)
        return NOISE_MAX_DIST

    cost_at_target: int = cave_cost[flow_idx, y, x]
    cost_at_center: int = BASE_FLOW_CENTER  # Assuming it's always this

    infinity: int = np.iinfo(cave_cost.dtype).max // 2
    if cost_at_target >= infinity:
        return NOISE_MAX_DIST  # Unreachable

    noise_dist: int = cost_at_target - cost_at_center

    if noise_dist < 0:
        # This case might indicate an issue or an edge case from the C code
        # logic
        log.warning(
            "negative_noise_distance_calculated",
            cost_target=int(cost_at_target),  # Convert numpy int
            cost_center=cost_at_center,
            calculated_dist=noise_dist,
            target_y=y,
            target_x=x,
            flow_type=which_flow.name,
        )
        noise_dist = NOISE_MAX_DIST

    return noise_dist


# --- Smell (Scent) System ---

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


@njit(cache=True, parallel=False)
def _lay_scent_kernel(
    cave_when: np.ndarray,
    terrain_map: np.ndarray,
    py: int,
    px: int,
    current_scent_when: int,
    scent_adjust: np.ndarray,
) -> None:
    """Numba kernel to lay down scent. Cannot log from here."""
    height, width = cave_when.shape
    table_size = scent_adjust.shape[0]
    offset = table_size // 2

    for i in range(table_size):
        for j in range(table_size):
            y: int = py + i - offset
            x: int = px + j - offset

            if not in_bounds(y, x, height, width):
                continue

            adjustment: int = scent_adjust[i, j]
            if adjustment == 250:
                continue

            if terrain_map[y, x] == FeatureType.WALL:
                continue

            # Line-of-sight check
            if not line_of_sight(py, px, y, x, terrain_map != FeatureType.WALL):
                continue

            cave_when[y, x] = current_scent_when + adjustment


def update_smell(
    cave_when: np.ndarray,
    terrain_map: np.ndarray,
    py: int,
    px: int,
    global_scent_when: int,
) -> int:
    """Updates the scent map: ages existing scent and lays down new scent."""
    # 1. Age the global timer
    global_scent_when -= 1
    log_ctx = log.bind(player_y=py, player_x=px)  # Bind context

    # 2. Handle scent cycle reset using NumPy vectorization
    if global_scent_when <= 0:
        log_ctx.info("scent_cycle_resetting",
                     old_scent_timer=global_scent_when)
        reset_offset: int = SCENT_RESET_AGE - SMELL_STRENGTH

        # *** Logic requires verification against C code ***
        is_old_scent: np.ndarray = cave_when > SMELL_STRENGTH
        is_recent_scent: np.ndarray = (cave_when > 0) & (~is_old_scent)

        num_old = np.sum(is_old_scent)
        num_recent = np.sum(is_recent_scent)

        # Erase old scent
        cave_when[is_old_scent] = 0
        # Reset age of recent scent
        cave_when[is_recent_scent] = reset_offset + cave_when[is_recent_scent]
        # *** End Verification Zone ***

        # Reset the global timer
        global_scent_when = reset_offset
        log_ctx.info(
            "scent_cycle_reset_complete",
            new_scent_timer=global_scent_when,
            tiles_erased=num_old,
            tiles_reset=num_recent,
        )

    # 3. Lay down new scent using the Numba kernel
    _lay_scent_kernel(
        cave_when=cave_when,
        terrain_map=terrain_map,
        py=py,
        px=px,
        current_scent_when=global_scent_when,
        scent_adjust=SCENT_ADJUST_TABLE,
    )
    # Could add debug log here if needed, e.g.,
    # log_ctx.debug("laid_scent", current_scent_timer=global_scent_when)

    return global_scent_when


def get_scent(cave_when: np.ndarray, y: int, x: int) -> int:
    """Gets the scent age at a specific location."""
    height, width = cave_when.shape
    if not in_bounds(y, x, height, width):
        # log.debug("scent_target_out_of_bounds", target_y=y, target_x=x) #
        # Potentially noisy
        return -1

    age: int = cave_when[y, x]
    return age if age > 0 else -1


# --- Monster Data Representation (Polars DataFrame) ---


def initialize_monsters(num_monsters: int, height: int, width: int) -> pl.DataFrame:
    """Creates a Polars DataFrame with sample monster data."""
    log.info(
        "initializing_monsters", num_monsters=num_monsters, map_h=height, map_w=width
    )
    monster_data = {
        "id": range(num_monsters),
        "r_idx": np.random.randint(1, 5, size=num_monsters),
        "fy": np.random.randint(0, height, size=num_monsters),
        "fx": np.random.randint(0, width, size=num_monsters),
        "is_dead": np.zeros(num_monsters, dtype=bool),
        "perception_stat": np.random.randint(5, 15, size=num_monsters),
        "alertness": np.random.randint(-10, 1, size=num_monsters),
        "flags": np.zeros(num_monsters, dtype=np.uint32),
        "heard_player": np.zeros(num_monsters, dtype=bool),
    }
    monster_df = pl.DataFrame(monster_data)
    log.debug("monster_dataframe_created", columns=monster_df.columns)
    return monster_df


# --- Perception System (Parallelized) ---


def skill_check(
    rng: GameRNG, actor_skill: int, difficulty: int, target_skill: int
) -> bool:
    """Deterministic skill check using :class:`GameRNG`."""
    roll = rng.get_int(1, 100)
    threshold = 50 + actor_skill - (difficulty + target_skill)
    return roll <= threshold


def _process_monster_perception_chunk(
    monster_df_chunk: pl.DataFrame,
    cave_cost: np.ndarray,
    flow_centers: np.ndarray,
    player_stealth_skill: int,
    noise_flow_type: FlowType,
    rng: GameRNG,
) -> List[int]:
    """Processes perception checks for a chunk of monsters. Cannot log directly."""
    alerted_monster_ids: List[int] = []

    for row in monster_df_chunk.iter_rows(named=True):
        monster_id: int = row["id"]
        m_y: int = row["fy"]
        m_x: int = row["fx"]
        perception: int = row["perception_stat"]

        noise_dist: int = get_noise_dist(
            cave_cost, flow_centers, noise_flow_type, m_y, m_x
        )

        if noise_dist >= NOISE_MAX_DIST:
            continue

        difficulty_mod: int = noise_dist
        if skill_check(
            rng,
            actor_skill=perception,
            difficulty=difficulty_mod,
            target_skill=player_stealth_skill,
        ):
            alerted_monster_ids.append(monster_id)
            # Cannot log here efficiently from parallel worker without setup

    return alerted_monster_ids


def monster_perception(
    monster_df: pl.DataFrame,
    cave_cost: np.ndarray,
    flow_centers: np.ndarray,
    player_y: int,
    player_x: int,
    player_stealth_skill: int,
    rng: GameRNG,
    noise_flow_type: FlowType = FlowType.REAL_NOISE,
) -> List[int]:
    """Updates monster perception based on noise. Parallelized using Joblib."""
    start_time = time.monotonic()
    log_ctx = log.bind(
        player_y=player_y, player_x=player_x, player_stealth=player_stealth_skill
    )

    # --- Filter Monsters ---
    # Using is_not() as recommended idiom
    active_monsters_df = monster_df.filter(pl.col("is_dead").is_not())

    num_monsters = active_monsters_df.height
    if num_monsters == 0:
        log_ctx.info("monster_perception_skipped_no_active_monsters")
        return []

    log_ctx.debug(
        "monster_perception_start",
        active_monster_count=num_monsters,
        total_monster_count=monster_df.height,
    )

    # --- Parallel Processing ---
    n_chunks = min(num_monsters, N_JOBS * 4)
    if n_chunks <= 0:
        n_chunks = 1
    chunk_size = (num_monsters + n_chunks - 1) // n_chunks

    df_chunks = [
        active_monsters_df.slice(i * chunk_size, chunk_size) for i in range(n_chunks)
    ]

    log_ctx.debug(
        "monster_perception_parallel_setup",
        num_chunks=n_chunks,
        chunk_size=chunk_size,
        num_jobs=N_JOBS,
    )

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
    all_alerted_ids: List[int] = [
        item for sublist in results for item in sublist]
    num_alerted = len(all_alerted_ids)

    end_time = time.monotonic()
    duration_ms = (end_time - start_time) * 1000

    log_ctx.info(
        "monster_perception_complete",
        duration_ms=round(duration_ms, 2),
        monsters_processed=num_monsters,
        monsters_alerted=num_alerted,
        num_jobs=N_JOBS,
    )

    if num_alerted > 0:
        # Log only IDs if many, maybe sample if extremely large?
        log_ctx.debug(
            "monsters_alerted_ids",
            alerted_ids=(
                all_alerted_ids if num_alerted < 20 else all_alerted_ids[:20] + [
                    "..."]
            ),
        )

    # Update Monster State (Post-Parallel) - Placeholder area
    if all_alerted_ids:
        # monster_df = monster_df.with_columns(
        #     pl.when(pl.col('id').is_in(all_alerted_ids))
        #     .then(True)
        #     .otherwise(pl.col('heard_player'))
        #     .alias('heard_player')
        # )
        # log_ctx.debug("monster_heard_player_flag_updated", count=num_alerted)
        pass  # Placeholder

    return all_alerted_ids


# --- Main Execution / Example Usage ---

if __name__ == "__main__":
    log.info("perception_systems_demo_start", sim_id=SIM_ID)

    # --- Initialize Map Data (Placeholders) ---
    terrain_map = np.full((MAP_HGT, MAP_WID),
                          FeatureType.FLOOR, dtype=np.int32)
    terrain_map[MAP_HGT // 2, MAP_WID //
                4: 3 * MAP_WID // 4] = FeatureType.WALL
    terrain_map[MAP_HGT // 2 + 5, MAP_WID // 2] = FeatureType.CLOSED_DOOR
    terrain_map[0, :] = FeatureType.WALL
    terrain_map[MAP_HGT - 1, :] = FeatureType.WALL
    terrain_map[:, 0] = FeatureType.WALL
    terrain_map[:, MAP_WID - 1] = FeatureType.WALL
    log.debug("terrain_map_initialized", shape=terrain_map.shape)

    # Noise cost map
    infinity_val = np.iinfo(np.int32).max // 2
    cave_cost = np.full((MAX_FLOWS, MAP_HGT, MAP_WID),
                        infinity_val, dtype=np.int32)
    flow_centers = np.zeros((MAX_FLOWS, 2), dtype=np.int32)
    log.debug("noise_cost_map_initialized", shape=cave_cost.shape)

    # Scent map
    cave_when = np.zeros((MAP_HGT, MAP_WID), dtype=np.int32)
    global_scent_when: int = SCENT_RESET_AGE
    log.debug(
        "scent_map_initialized", shape=cave_when.shape, initial_timer=global_scent_when
    )

    door_penalties = {"pass": 3, "real": 5}

    # --- Initialize Player and Monsters ---
    player_y, player_x = MAP_HGT // 2 + 10, MAP_WID // 2
    player_stealth_skill = 10
    rng = GameRNG(seed=123)
    log.info(
        "player_initialized", pos=(player_y, player_x), stealth=player_stealth_skill
    )

    num_monsters = 500
    monster_df = initialize_monsters(num_monsters, MAP_HGT, MAP_WID)

    # Ensure monsters don't start inside walls (simple repositioning)
    wall_mask = (
        terrain_map[monster_df["fy"].to_numpy(), monster_df["fx"].to_numpy()]
        == FeatureType.WALL
    )
    num_in_wall = wall_mask.sum()
    if num_in_wall > 0:
        log.warning(
            "monsters_in_walls_detected", count=num_in_wall, action="repositioning"
        )
        monster_df = monster_df.with_columns(
            pl.when(pl.lit(wall_mask))  # Use the precomputed mask
            .then(player_y)
            .otherwise(pl.col("fy"))
            .alias("fy"),
            pl.when(pl.lit(wall_mask))  # Use the precomputed mask
            .then(player_x + 1)
            .otherwise(pl.col("fx"))
            .alias("fx"),
        )
        # Verify repositioning (optional)
        # wall_mask_after = terrain_map[monster_df["fy"].to_numpy(), monster_df["fx"].to_numpy()] == FeatureType.WALL
        # log.debug("monsters_in_walls_after_reposition", count=wall_mask_after.sum())

    log.info(
        "simulation_setup_complete",
        map_dims=f"{MAP_HGT}x{MAP_WID}",
        player_pos=(player_y, player_x),
        num_monsters=num_monsters,
        num_jobs=N_JOBS,
    )

    # --- Simulation Loop (Simplified Example) ---
    max_turns = 5
    for turn in range(max_turns):
        turn_start_time = time.monotonic()
        turn_log = log.bind(turn=turn + 1)  # Bind turn context
        turn_log.info("turn_start")

        # 1. Update Noise
        noise_start_time = time.monotonic()
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
        noise_duration_ms = (time.monotonic() - noise_start_time) * 1000
        turn_log.debug("noise_updated",
                       duration_ms=round(noise_duration_ms, 2))

        # 2. Update Smell
        smell_start_time = time.monotonic()
        global_scent_when = update_smell(
            cave_when, terrain_map, player_y, player_x, global_scent_when
        )
        smell_duration_ms = (time.monotonic() - smell_start_time) * 1000
        turn_log.debug(
            "smell_updated",
            duration_ms=round(smell_duration_ms, 2),
            current_scent_timer=global_scent_when,
        )

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
        # monster_perception logs its own duration and results

        # --- Example Lookups (DEBUG level) ---
        if num_monsters > 0 and (turn % 2 == 0):  # Log less frequently
            example_monster_id = 0
            m_info = monster_df.filter(pl.col("id") == example_monster_id)
            if m_info.height > 0:
                m_y, m_x = m_info.select(["fy", "fx"]).row(0)
                noise = get_noise_dist(
                    cave_cost, flow_centers, FlowType.REAL_NOISE, m_y, m_x
                )
                scent = get_scent(cave_when, m_y, m_x)
                turn_log.debug(
                    "example_monster_status",
                    monster_id=example_monster_id,
                    pos=(m_y, m_x),
                    noise_dist=noise,
                    scent_age=scent,
                )

        # (Player moves, monsters move, combat happens, etc. - not implemented)
        player_x += 1
        if player_x >= MAP_WID - 1:
            player_x = 1
        turn_log.debug("player_moved", new_pos=(
            player_y, player_x))  # Log player move

        turn_duration_ms = (time.monotonic() - turn_start_time) * 1000
        turn_log.info("turn_end", duration_ms=round(turn_duration_ms, 2))

    log.info(
        "simulation_run_finished",
        total_turns=max_turns,
        final_player_pos=(player_y, player_x),
    )
    logging.shutdown()  # Flush and close file handlers properly

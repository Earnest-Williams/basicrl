#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Updated to use GameRNG
"""
Main script for the FOV/Light/Memory Simulation Game.

Features:
- Inverse square light falloff
- Blended COLORED light sources (True Color)
- Memory Fade (Sigmoid decay)
- Corrected Syntax
- Uses GameRNG for determinism
"""

import logging

# Removed 'random' import
import math
import os
import sys
import time
from collections import deque
from typing import List, Optional, Set, Tuple

import numba
import numpy as np

# --- Import Project Modules ---
# Assuming these are in the same directory or accessible via PYTHONPATH
try:
    import constants
    from dungeon_data import (
        Dungeon,
    )  # Assuming Dungeon is correctly imported from dungeon_data

    import dungeon_generator

    # Import GameRNG
    from game_rng import GameRNG
except ImportError as e:
    print(f"Failed to import project modules or GameRNG: {e}", file=sys.stderr)
    # Define dummies if needed for basic script parsing, but execution will fail
    WALL_ID, FLOOR_ID, PILLAR_ID = 0, 1, 2

    class Dungeon:  # type: ignore # noqa
        def __init__(self, w, h):
            self.width = w
            self.height = h
            self.tiles = np.zeros((h, w))

    class GameRNG:  # type: ignore # noqa
        def get_int(self, a, b):
            return (a + b) // 2

        def choice(self, seq):
            return seq[0] if seq else None

        def shuffle(self, seq):
            pass

        def get_float(self, a=0.0, b=1.0):
            return (a + b) / 2.0

        def weighted_choice(self, items, weights):
            return items[0] if items else None

    # Dummy dungeon generator
    class dungeon_generator:  # type: ignore # noqa
        def dungeon_generate_map_u_shape(dungeon, rng):
            pass

    # Dummy constants
    class constants:  # type: ignore # noqa
        DEFAULT_ENTITY_CATEGORY = "medium"
        TORCH_COLOR_RGB = (0, 0, 0)
        ORB_COLOR_RGB = (0, 0, 0)
        MAX_LOS_DISTANCE = 10
        MAX_LIGHT_LEVEL_FOR_VIS_CHECK = 5
        LIGHT_LEVEL_DATA = {}
        AMBIENT_COLOR_RGB = (0, 0, 0)
        MEMORY_COLOR = ""
        COLOR = {"RESET": ""}
        UNSEEN = "."
        PLAYER_CHAR = "@"
        LIGHT_CHAR = "*"
        VISIBLE_WALL = "#"
        VISIBLE_PILLAR = "O"
        VISIBLE_FLOOR = "."
        MEMORY_LEVEL_COUNT = 5
        MEMORY_WALL_LEVELS = [" "] * 5
        MEMORY_PILLAR_LEVELS = [" "] * 5
        MEMORY_FLOOR_LEVELS = [" "] * 5
        MEMORY_DECAY_RATE = 1.0
        TILE_ID_TO_CATEGORY = {}

# Expose decay rate as a plain global for Numba
MEMORY_DECAY_RATE = constants.MEMORY_DECAY_RATE


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try importing readchar
try:
    import readchar

    READCHAR_AVAILABLE = True
except ImportError:
    READCHAR_AVAILABLE = False
    print("Warning: 'readchar' library not found.", file=sys.stderr)

# ============================================================
# --- Slope jitclass (Unchanged) ---
# ============================================================
slope_spec = [("y", numba.int64), ("x", numba.int64)]


@numba.experimental.jitclass(slope_spec)
class Slope:
    def __init__(self, y: int, x: int):
        self.y = y
        self.x = x

    def greater(self, y: int, x: int) -> bool:
        return self.y * x > self.x * y

    def greater_or_equal(self, y: int, x: int) -> bool:
        return self.y * x >= self.x * y

    def less(self, y: int, x: int) -> bool:
        return self.y * x < self.x * y


# ============================================================
# --- Entity Classes (Modified LightSource) ---
# ============================================================
class Entity:  # Unchanged base class
    def __init__(
        self,
        x: int,
        y: int,
        light_radius: int = 0,
        light_level: int = 0,
        size_category: str = constants.DEFAULT_ENTITY_CATEGORY,
        base_color_rgb: Tuple[int, int, int] = (0, 0, 0),
    ):
        self.x = x
        self.y = y
        self.light_radius = max(0, light_radius)
        self.light_level = light_level
        self.size_category = size_category
        self.base_color_rgb: Tuple[int, int, int] = base_color_rgb

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)


class Player(Entity):  # Unchanged Player class
    def __init__(self, x: int, y: int, light_radius: int = 10, light_level: int = 3):
        super().__init__(
            x, y, light_radius, light_level, "medium", constants.TORCH_COLOR_RGB
        )
        self.path: List[Tuple[int, int]] = []
        self.path_index = 0

    def set_path(self, path: List[Tuple[int, int]]):
        self.path = path
        self.path_index = 0

    def move(self) -> bool:
        if not self.path or self.path_index >= len(self.path):
            return False
        self.x, self.y = self.path[self.path_index]
        self.path_index += 1
        return True


class LightSource(Entity):
    # Modified __init__ to accept rng
    def __init__(
        self,
        x: int,
        y: int,
        rng: GameRNG,
        light_radius: int = 16,
        light_level: int = 5,
        flicker: bool = False,
        base_color_rgb: Tuple[int, int, int] = constants.ORB_COLOR_RGB,
    ):
        super().__init__(x, y, light_radius, light_level, "small", base_color_rgb)
        self.flicker = flicker
        self.original_radius = max(1, light_radius)
        self.rng = rng  # Store RNG instance

    # Modified update to use self.rng
    def update(self):  # No longer needs rng passed explicitly
        if self.flicker and self.rng.get_float(0.0, 1.0) < 0.2:  # USE RNG
            self.light_radius = self.rng.get_int(  # USE RNG
                max(1, self.original_radius - 3), self.original_radius + 1
            )
        else:
            self.light_radius = self.original_radius


# ============================================================
# --- Numba Helper Functions (Unchanged) ---
# ============================================================
@numba.jit(nopython=True)
def _update_dungeon_time(dungeon_instance: Dungeon, dt: np.float32) -> None:
    dungeon_instance.current_time += dt


@numba.jit(nopython=True)
def _update_memory_fade_internal(
    current_time: np.float32,
    last_seen_time: np.ndarray,
    memory_intensity: np.ndarray,
    visible: np.ndarray,
    height: int,
    width: int,
) -> None:
    decay_rate = MEMORY_DECAY_RATE
    for y in range(height):
        for x in range(width):
            intensity = memory_intensity[y, x]
            if intensity > 0.0 and not visible[y, x]:
                elapsed_time = current_time - last_seen_time[y, x]
                if elapsed_time < 0.0:
                    elapsed_time = 0.0
                new_intensity = max(0.0, 1.0 - decay_rate * elapsed_time)
                memory_intensity[y, x] = new_intensity


@numba.jit(nopython=True)
def _transform_coordinate(
    x: int, y: int, octant: int, origin: Tuple[int, int]
) -> Tuple[int, int]:
    ox, oy = origin
    nx, ny = ox, oy
    if octant == 0:
        nx += x
        ny -= y
    elif octant == 1:
        nx += y
        ny -= x
    elif octant == 2:
        nx -= y
        ny -= x
    elif octant == 3:
        nx -= x
        ny -= y
    elif octant == 4:
        nx -= x
        ny += y
    elif octant == 5:
        nx -= y
        ny += x
    elif octant == 6:
        nx += y
        ny += x
    elif octant == 7:
        nx += x
        ny += y
    return nx, ny


@numba.jit(nopython=True)
def _blocks_light_in_octant(
    x: int, y: int, octant: int, origin: Tuple[int, int], dungeon_instance: Dungeon
) -> bool:
    nx, ny = _transform_coordinate(x, y, octant, origin)
    return dungeon_instance.blocks_light(nx, ny)


@numba.jit(nopython=True, fastmath=True)
def _calculate_top_y(
    x: int, top: Slope, octant: int, origin: Tuple[int, int], dungeon_instance: Dungeon
) -> int:
    if top.x == 1:
        return x
    top_y = ((x * 2 - 1) * top.y + top.x) // (top.x * 2)
    if _blocks_light_in_octant(x, top_y, octant, origin, dungeon_instance):
        if top.greater_or_equal(top_y * 2 + 1, x * 2) and not _blocks_light_in_octant(
            x, top_y + 1, octant, origin, dungeon_instance
        ):
            top_y += 1
    else:
        ax = x * 2
        if _blocks_light_in_octant(x + 1, top_y + 1, octant, origin, dungeon_instance):
            ax += 1
        if top.greater(top_y * 2 + 1, ax):
            top_y += 1
    return top_y


@numba.jit(nopython=True, fastmath=True)
def _calculate_bottom_y(
    x: int,
    bottom: Slope,
    octant: int,
    origin: Tuple[int, int],
    dungeon_instance: Dungeon,
) -> int:
    if bottom.y == 0:
        return 0
    bottom_y = ((x * 2 - 1) * bottom.y + bottom.x) // (bottom.x * 2)
    if (
        bottom.greater_or_equal(bottom_y * 2 + 1, x * 2)
        and _blocks_light_in_octant(x, bottom_y, octant, origin, dungeon_instance)
        and not _blocks_light_in_octant(
            x, bottom_y + 1, octant, origin, dungeon_instance
        )
    ):
        bottom_y += 1
    return bottom_y


@numba.jit(nopython=True)
def _mark_color_in_octant_array(
    x: int,
    y: int,
    octant: int,
    origin: Tuple[int, int],
    dungeon_instance: Dungeon,
    target_rgb_sum_array: np.ndarray,
    weighted_r: float,
    weighted_g: float,
    weighted_b: float,
) -> None:
    nx, ny = _transform_coordinate(x, y, octant, origin)
    if 0 <= nx < dungeon_instance.width and 0 <= ny < dungeon_instance.height:
        target_rgb_sum_array[ny, nx, 0] += weighted_r
        target_rgb_sum_array[ny, nx, 1] += weighted_g
        target_rgb_sum_array[ny, nx, 2] += weighted_b


@numba.jit(nopython=True, fastmath=True, cache=True)
def _compute_octant_for_color(
    octant: int,
    origin: Tuple[int, int],
    range_limit: int,
    x: int,
    top: Slope,
    bottom: Slope,
    dungeon_instance: Dungeon,
    target_rgb_sum_array: np.ndarray,
    base_r: int,
    base_g: int,
    base_b: int,
) -> None:
    current_top = Slope(top.y, top.x)
    current_bottom = Slope(bottom.y, bottom.x)
    effective_range = float(max(1.0, range_limit))
    effective_range_sq = effective_range * effective_range
    no_range = range_limit <= 0
    f_base_r = float(base_r)
    f_base_g = float(base_g)
    f_base_b = float(base_b)
    while x <= range_limit:
        if no_range:
            break
        top_y = _calculate_top_y(x, current_top, octant, origin, dungeon_instance)
        bottom_y = _calculate_bottom_y(
            x, current_bottom, octant, origin, dungeon_instance
        )
        was_opaque = -1
        world_check_x, _ = _transform_coordinate(x, 0, octant, origin)
        if not (0 <= world_check_x < dungeon_instance.width):
            break
        for y in range(top_y, bottom_y - 1, -1):
            distance = dungeon_instance.get_distance(x, y)
            if distance <= range_limit:
                is_opaque = _blocks_light_in_octant(
                    x, y, octant, origin, dungeon_instance
                )
                is_visible_in_los = is_opaque or (
                    (y != top_y or current_top.greater(y * 4 - 1, x * 4 + 1))
                    and (y != bottom_y or current_bottom.less(y * 4 + 1, x * 4 - 1))
                )
                if is_visible_in_los:
                    distance_sq = distance * distance
                    intensity_at_tile = max(
                        0.0, 1.0 - (distance_sq / effective_range_sq)
                    )
                    if intensity_at_tile > 0.0:
                        weighted_r = f_base_r * intensity_at_tile
                        weighted_g = f_base_g * intensity_at_tile
                        weighted_b = f_base_b * intensity_at_tile
                        _mark_color_in_octant_array(
                            x,
                            y,
                            octant,
                            origin,
                            dungeon_instance,
                            target_rgb_sum_array,
                            weighted_r,
                            weighted_g,
                            weighted_b,
                        )
                if is_opaque:
                    if was_opaque == 0:
                        nx, ny = x * 2, y * 2 + 1
                        if _blocks_light_in_octant(
                            x, y + 1, octant, origin, dungeon_instance
                        ):
                            nx -= 1
                        new_bottom = Slope(ny, nx)
                        if current_top.greater(ny, nx):
                            if y == bottom_y:
                                current_bottom = new_bottom
                                break
                            else:
                                _compute_octant_for_color(
                                    octant,
                                    origin,
                                    range_limit,
                                    x + 1,
                                    current_top,
                                    new_bottom,
                                    dungeon_instance,
                                    target_rgb_sum_array,
                                    base_r,
                                    base_g,
                                    base_b,
                                )
                        elif y == bottom_y:
                            break
                    was_opaque = 1
                else:
                    if was_opaque > 0:
                        nx, ny = x * 2, y * 2 + 1
                        if _blocks_light_in_octant(
                            x + 1, y + 1, octant, origin, dungeon_instance
                        ):
                            nx += 1
                        new_top = Slope(ny, nx)
                        if current_bottom.greater_or_equal(ny, nx):
                            return
                        current_top = new_top
                    was_opaque = 0
        if was_opaque != 0:
            break
        x += 1


@numba.jit(nopython=True)
def _mark_visible_boolean_in_octant_array(
    x: int,
    y: int,
    octant: int,
    origin: Tuple[int, int],
    dungeon_instance: Dungeon,
    target_los_array: np.ndarray,
) -> None:
    nx, ny = _transform_coordinate(x, y, octant, origin)
    if 0 <= nx < dungeon_instance.width and 0 <= ny < dungeon_instance.height:
        target_los_array[ny, nx] = True


@numba.jit(nopython=True, fastmath=True, cache=True)
def _compute_octant_for_boolean_array(
    octant: int,
    origin: Tuple[int, int],
    range_limit: int,
    x: int,
    top: Slope,
    bottom: Slope,
    dungeon_instance: Dungeon,
    target_los_array: np.ndarray,
) -> None:
    current_top = Slope(top.y, top.x)
    current_bottom = Slope(bottom.y, bottom.x)
    unlimited_range = range_limit < 0
    while unlimited_range or x <= range_limit:
        top_y = _calculate_top_y(x, current_top, octant, origin, dungeon_instance)
        bottom_y = _calculate_bottom_y(
            x, current_bottom, octant, origin, dungeon_instance
        )
        was_opaque = -1
        world_check_x, _ = _transform_coordinate(x, 0, octant, origin)
        if not (0 <= world_check_x < dungeon_instance.width):
            break
        for y in range(top_y, bottom_y - 1, -1):
            distance = dungeon_instance.get_distance(x, y)
            if unlimited_range or distance <= range_limit:
                is_opaque = _blocks_light_in_octant(
                    x, y, octant, origin, dungeon_instance
                )
                is_visible_in_los = is_opaque or (
                    (y != top_y or current_top.greater(y * 4 - 1, x * 4 + 1))
                    and (y != bottom_y or current_bottom.less(y * 4 + 1, x * 4 - 1))
                )
                if is_visible_in_los:
                    _mark_visible_boolean_in_octant_array(
                        x, y, octant, origin, dungeon_instance, target_los_array
                    )
                if is_opaque:
                    if was_opaque == 0:
                        nx, ny = x * 2, y * 2 + 1
                        if _blocks_light_in_octant(
                            x, y + 1, octant, origin, dungeon_instance
                        ):
                            nx -= 1
                        new_bottom = Slope(ny, nx)
                        if current_top.greater(ny, nx):
                            if y == bottom_y:
                                current_bottom = new_bottom
                                break
                            else:
                                _compute_octant_for_boolean_array(
                                    octant,
                                    origin,
                                    range_limit,
                                    x + 1,
                                    current_top,
                                    new_bottom,
                                    dungeon_instance,
                                    target_los_array,
                                )
                        elif y == bottom_y:
                            break
                    was_opaque = 1
                else:
                    if was_opaque > 0:
                        nx, ny = x * 2, y * 2 + 1
                        if _blocks_light_in_octant(
                            x + 1, y + 1, octant, origin, dungeon_instance
                        ):
                            nx += 1
                        new_top = Slope(ny, nx)
                        if current_bottom.greater_or_equal(ny, nx):
                            return
                        current_top = new_top
                    was_opaque = 0
        if was_opaque != 0:
            break
        x += 1


# ============================================================
# --- Top-Level Calculation Functions (Unchanged) ---
# ============================================================
def compute_illumination_color_array(
    origin: Tuple[int, int],
    range_limit: int,
    dungeon_instance: Dungeon,
    target_rgb_sum_array: np.ndarray,
    base_color_rgb: Tuple[int, int, int],
) -> None:
    ox, oy = origin
    if not (0 <= ox < dungeon_instance.width and 0 <= oy < dungeon_instance.height):
        return
    if range_limit <= 0:
        return
    target_rgb_sum_array[oy, ox, 0] += float(base_color_rgb[0])
    target_rgb_sum_array[oy, ox, 1] += float(base_color_rgb[1])
    target_rgb_sum_array[oy, ox, 2] += float(base_color_rgb[2])
    start_top = Slope(1, 1)
    start_bottom = Slope(0, 1)
    r, g, b = base_color_rgb
    for octant in range(8):
        _compute_octant_for_color(
            octant,
            origin,
            range_limit,
            1,
            start_top,
            start_bottom,
            dungeon_instance,
            target_rgb_sum_array,
            r,
            g,
            b,
        )


def compute_los_into_boolean_array(
    origin: Tuple[int, int],
    range_limit: int,
    dungeon_instance: Dungeon,
    target_los_array: np.ndarray,
) -> None:
    ox, oy = origin
    if not (0 <= ox < dungeon_instance.width and 0 <= oy < dungeon_instance.height):
        return
    target_los_array[oy, ox] = True
    start_top = Slope(1, 1)
    start_bottom = Slope(0, 1)
    for octant in range(8):
        _compute_octant_for_boolean_array(
            octant,
            origin,
            range_limit,
            1,
            start_top,
            start_bottom,
            dungeon_instance,
            target_los_array,
        )


# ============================================================
# --- BFS Pathfinding (Unchanged) ---
# ============================================================
def find_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    tiles: np.ndarray,
    width: int,
    height: int,
) -> Optional[List[Tuple[int, int]]]:
    if not (
        0 <= start[0] < width
        and 0 <= start[1] < height
        and 0 <= end[0] < width
        and 0 <= end[1] < height
    ):
        return None
    if (
        tiles[start[1], start[0]] != constants.FLOOR_ID
        or tiles[end[1], end[0]] != constants.FLOOR_ID
    ):
        return None
    if start == end:
        return [start]
    q = deque([(start, [start])])
    visited: Set[Tuple[int, int]] = {start}
    while q:
        (vx, vy), path = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = vx + dx, vy + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and tiles[ny, nx] == constants.FLOOR_ID
                and (nx, ny) not in visited
            ):
                new_path = list(path)
                new_path.append((nx, ny))
                if (nx, ny) == end:
                    return new_path
                visited.add((nx, ny))
                q.append(((nx, ny), new_path))
    return None


# ============================================================
# --- Helper Functions (Unchanged) ---
# ============================================================
def get_memory_character(tile_id: int, intensity: float) -> str:
    if intensity <= 0.0:
        return constants.UNSEEN
    level = int((1.0 - intensity) * constants.MEMORY_LEVEL_COUNT)
    index = min(constants.MEMORY_LEVEL_COUNT - 1, max(0, level))
    if tile_id == constants.WALL_ID:
        return constants.MEMORY_WALL_LEVELS[index]
    elif tile_id == constants.PILLAR_ID:
        return constants.MEMORY_PILLAR_LEVELS[index]
    elif tile_id == constants.FLOOR_ID:
        return constants.MEMORY_FLOOR_LEVELS[index]
    else:
        return constants.UNSEEN


def get_object_size_category(
    dungeon: Dungeon, x: int, y: int, entities: List[Entity]
) -> str:
    for entity in entities:
        if entity.x == x and entity.y == y:
            return entity.size_category
    if 0 <= x < dungeon.width and 0 <= y < dungeon.height:
        return constants.TILE_ID_TO_CATEGORY.get(
            dungeon.tiles[y, x], constants.DEFAULT_ENTITY_CATEGORY
        )
    return "wall"


def distance_sq(x1: int, y1: int, x2: int, y2: int) -> int:
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def _interpolate_color(
    factor: float, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    factor = max(0.0, min(1.0, factor))
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * factor)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * factor)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * factor)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _format_true_color(rgb: Tuple[int, int, int]) -> str:
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _get_brightness_from_rgb_sum(rgb_sum: np.ndarray) -> float:
    max_comp = 0.0
    if rgb_sum[0] > max_comp:
        max_comp = rgb_sum[0]
    if rgb_sum[1] > max_comp:
        max_comp = rgb_sum[1]
    if rgb_sum[2] > max_comp:
        max_comp = rgb_sum[2]
    return min(1.0, max_comp / 255.0)


# ============================================================
# --- GameState Class (Modified to use RNG) ---
# ============================================================
class GameState:
    # Modified __init__ to accept rng
    def __init__(self, width: int, height: int, rng: GameRNG):
        self.rng = rng  # Store RNG instance
        self.dungeon = Dungeon(width, height)
        self.player: Optional[Player] = None
        self.light_sources: List[LightSource] = []
        self.all_entities: List[Entity] = []
        self.current_illumination_rgb_sum: np.ndarray = np.zeros(
            (height, width, 3), dtype=np.float32
        )
        self.current_player_los: np.ndarray = np.zeros((height, width), dtype=np.bool_)

    # Modified to use self.rng
    def initialize_map_and_entities(self):  # No longer needs rng passed
        # --- Use self.rng ---
        dungeon_generator.dungeon_generate_map_u_shape(
            self.dungeon, self.rng
        )  # PASS RNG
        player_x = self.dungeon.width // 2
        player_y = 5
        search_attempts = 0
        # Player placement logic (no RNG needed here)
        while self.dungeon.tiles[player_y, player_x] != constants.FLOOR_ID:
            player_y += 1
            search_attempts += 1
            if (
                player_y >= self.dungeon.height - 1
                or search_attempts > self.dungeon.height
            ):
                player_x = 5
                player_y = 5
                search_attempts = 0
                while self.dungeon.tiles[player_y, player_x] != constants.FLOOR_ID:
                    player_x += 1
                    search_attempts += 1
                    if (
                        player_x >= self.dungeon.width - 1
                        or search_attempts > self.dungeon.width * 2
                    ):
                        logging.error("Player start failed!")
                        player_x = self.dungeon.width // 2
                        player_y = 5
                        break
                break
        self.player = Player(player_x, player_y, light_radius=10, light_level=3)

        # Light source placement (no RNG needed here for fixed positions)
        light_radius = 16
        light_level = 5
        light1_x, light1_y = 10, self.dungeon.height // 2
        light2_x, light2_y = self.dungeon.width - 11, self.dungeon.height // 2
        # Adjust positions if blocked
        if self.dungeon.tiles[light1_y, light1_x] != constants.FLOOR_ID:
            light1_x += 1
        if self.dungeon.tiles[light2_y, light2_x] != constants.FLOOR_ID:
            light2_x -= 1
        if self.dungeon.tiles[light1_y, light1_x] != constants.FLOOR_ID:
            light1_x, light1_y = 11, self.dungeon.height // 2 + 1
        if self.dungeon.tiles[light2_y, light2_x] != constants.FLOOR_ID:
            light2_x, light2_y = self.dungeon.width - 12, self.dungeon.height // 2 + 1

        # Create LightSource instances, PASSING self.rng
        self.light_sources = [
            LightSource(
                light1_x,
                light1_y,
                self.rng,
                light_radius,
                light_level,
                flicker=False,
                base_color_rgb=constants.ORB_COLOR_RGB,
            ),
            LightSource(
                light2_x,
                light2_y,
                self.rng,
                light_radius,
                light_level,
                flicker=True,
                base_color_rgb=(255, 100, 100),
            ),
        ]

        self.all_entities = ([self.player] if self.player else []) + self.light_sources
        self.generate_player_path()  # Path generation is deterministic here

    def generate_player_path(self):  # Unchanged
        if not self.player:
            return
            self.player.set_path([(self.player.x, self.player.y)])

    # Update method no longer needs rng passed, light sources have their own
    def update(self, dt: float):
        if not self.dungeon:
            return
        _update_dungeon_time(self.dungeon, np.float32(dt))
        _update_memory_fade_internal(
            self.dungeon.current_time,
            self.dungeon.last_seen_time,
            self.dungeon.memory_intensity,
            self.dungeon.visible,
            self.dungeon.height,
            self.dungeon.width,
        )
        # --- Update Entities ---
        # LightSource update method now uses its stored rng instance
        for light in self.light_sources:
            light.update()
        # Update all_entities list (no RNG needed here)
        self.all_entities = ([self.player] if self.player else []) + self.light_sources

    def update_visibility(self):  # Unchanged (visibility logic doesn't use random)
        if not self.player or not self.dungeon:
            if self.dungeon:
                self.dungeon.visible.fill(False)
            self.current_illumination_rgb_sum.fill(0.0)
            self.current_player_los.fill(False)
            return
        d = self.dungeon
        px, py = self.player.position
        self.current_player_los.fill(False)
        compute_los_into_boolean_array(
            self.player.position, constants.MAX_LOS_DISTANCE, d, self.current_player_los
        )
        self.current_illumination_rgb_sum.fill(0.0)
        compute_illumination_color_array(
            self.player.position,
            self.player.light_radius,
            d,
            self.current_illumination_rgb_sum,
            self.player.base_color_rgb,
        )
        for light in self.light_sources:
            if light.light_radius > 0:
                compute_illumination_color_array(
                    light.position,
                    light.light_radius,
                    d,
                    self.current_illumination_rgb_sum,
                    light.base_color_rgb,
                )
        final_visible = np.zeros_like(self.current_player_los)
        max_vis_check_level = constants.MAX_LIGHT_LEVEL_FOR_VIS_CHECK
        min_brightness_for_vis = 1.0 / (max_vis_check_level * 2.5)
        for y in range(d.height):
            for x in range(d.width):
                if self.current_player_los[y, x]:
                    rgb_sum = self.current_illumination_rgb_sum[y, x]
                    brightness = _get_brightness_from_rgb_sum(rgb_sum)
                    if brightness >= min_brightness_for_vis:
                        approx_level = min(
                            max_vis_check_level,
                            max(1, int(brightness * max_vis_check_level + 0.5)),
                        )
                        obj_category = get_object_size_category(
                            d, x, y, self.all_entities
                        )
                        level_str = str(approx_level)
                        required_range = 0
                        try:
                            level_info = constants.LIGHT_LEVEL_DATA.get(level_str)
                            if level_info:
                                category_info = level_info.get(obj_category)
                            if category_info:
                                required_range = category_info.get(
                                    "noticeable_range", 0
                                )
                        except Exception as e:
                            logging.error(f"Light lookup error: {e}")
                            required_range = 0
                        dist_sq_to_player = distance_sq(px, py, x, y)
                        if dist_sq_to_player <= required_range * required_range:
                            final_visible[y, x] = True
        d.visible = final_visible
        current_sim_time = d.current_time
        d.last_seen_time[final_visible] = current_sim_time
        d.memory_intensity[final_visible] = 1.0

    def render(self) -> str:  # Unchanged (render logic doesn't use random)
        if not self.dungeon:
            return "Error: Dungeon not initialized."
        if self.current_illumination_rgb_sum is None or self.current_player_los is None:
            logging.warning("Illum/LOS arrays None before render. Forcing update.")
            self.update_visibility()
            if self.current_illumination_rgb_sum is None:
                return "Error: Failed Illum calc."
        d = self.dungeon
        result = []
        player_pos = self.player.position if self.player else (-1, -1)
        rgb_sum_array = self.current_illumination_rgb_sum
        if constants.DEBUG_RENDER_MODE == "level":
            result.append("--- Est. Brightness (0.0-1.0+, from RGB Sum) ---")
            for y in range(d.height):
                row = [
                    (
                        f"{_get_brightness_from_rgb_sum(rgb_sum_array[y, x]):.2f}"
                        if np.any(rgb_sum_array[y, x] > 0.0)
                        else " .  "
                    )
                    for x in range(d.width)
                ]
                result.append(" ".join(row))
            return "\n".join(result) + "\n"
        if constants.DEBUG_RENDER_MODE == "intensity":
            result.append("--- Memory Intensity (DEBUG) ---")
            for y in range(d.height):
                row = [
                    (
                        f"{d.memory_intensity[y,x]:.1f}"
                        if d.memory_intensity[y, x] > 0.01
                        else " . "
                    )
                    for x in range(d.width)
                ]
                result.append(" ".join(row))
            return "\n".join(result) + "\n"
        if constants.DEBUG_RENDER_MODE == "level_color":
            result.append("--- Blended RGB True Color (DEBUG, Clamped Sum) ---")
            for y in range(d.height):
                row_chars = []
                for x in range(d.width):
                    rgb_sum = rgb_sum_array[y, x]
                    brightness = _get_brightness_from_rgb_sum(rgb_sum)
                    if brightness > 0.001:
                        r_val = int(max(0, min(255, rgb_sum[0])))
                        g_val = int(max(0, min(255, rgb_sum[1])))
                        b_val = int(max(0, min(255, rgb_sum[2])))
                        final_rgb = (r_val, g_val, b_val)
                        color_code = _format_true_color(final_rgb)
                        tile_id = d.tiles[y, x]
                        char = (
                            constants.VISIBLE_WALL
                            if tile_id == constants.WALL_ID
                            else (
                                constants.VISIBLE_PILLAR
                                if tile_id == constants.PILLAR_ID
                                else constants.VISIBLE_FLOOR
                            )
                        )
                        row_chars.append(
                            f"{color_code}{char}{constants.COLOR['RESET']}"
                        )
                    else:
                        memory_intensity = d.memory_intensity[y, x]
                        if memory_intensity > 0.0:
                            tile_id = d.tiles[y, x]
                            char = get_memory_character(tile_id, memory_intensity)
                            row_chars.append(
                                f"{constants.MEMORY_COLOR}{char}{constants.COLOR['RESET']}"
                            )
                        else:
                            row_chars.append(constants.UNSEEN)
                result.append("".join(row_chars))
            return "\n".join(result) + "\n"
        for y in range(d.height):
            row_chars = []
            for x in range(d.width):
                is_visible = d.visible[y, x]
                memory_intensity = d.memory_intensity[y, x]
                tile_id = d.tiles[y, x]
                char = constants.UNSEEN
                final_color_code = ""
                is_player_tile = x == player_pos[0] and y == player_pos[1]
                light_source_at_tile: Optional[LightSource] = None
                if not is_player_tile:
                    for light in self.light_sources:
                        if light.x == x and light.y == y:
                            light_source_at_tile = light
                            break
                if is_visible:
                    rgb_sum = rgb_sum_array[y, x]
                    brightness = _get_brightness_from_rgb_sum(rgb_sum)
                    if brightness > 0.001:
                        max_comp = max(rgb_sum[0], rgb_sum[1], rgb_sum[2], 1.0)
                        norm_r = rgb_sum[0] / max_comp
                        norm_g = rgb_sum[1] / max_comp
                        norm_b = rgb_sum[2] / max_comp
                        blend_hue_rgb = (
                            int(norm_r * 255),
                            int(norm_g * 255),
                            int(norm_b * 255),
                        )
                        final_rgb = _interpolate_color(
                            brightness, constants.AMBIENT_COLOR_RGB, blend_hue_rgb
                        )
                        final_color_code = _format_true_color(final_rgb)
                    else:
                        final_color_code = _format_true_color(
                            constants.AMBIENT_COLOR_RGB
                        )
                    if is_player_tile:
                        char = constants.PLAYER_CHAR
                    elif light_source_at_tile is not None:
                        char = constants.LIGHT_CHAR
                    else:
                        if tile_id == constants.WALL_ID:
                            char = constants.VISIBLE_WALL
                        elif tile_id == constants.PILLAR_ID:
                            char = constants.VISIBLE_PILLAR
                        else:
                            char = constants.VISIBLE_FLOOR
                elif memory_intensity > 0.0:
                    final_color_code = constants.MEMORY_COLOR
                    if is_player_tile:
                        char = get_memory_character(
                            constants.FLOOR_ID, memory_intensity
                        )
                    elif light_source_at_tile is not None:
                        char = (
                            constants.MEMORY_LIGHT
                            if memory_intensity > 0.3
                            else get_memory_character(
                                constants.FLOOR_ID, memory_intensity
                            )
                        )
                    else:
                        char = get_memory_character(tile_id, memory_intensity)
                else:
                    final_color_code = ""
                final_char = char if char != " " else constants.UNSEEN
                if final_color_code:
                    row_chars.append(
                        f"{final_color_code}{final_char}{constants.COLOR['RESET']}"
                    )
                else:
                    row_chars.append(final_char)
            result.append("".join(row_chars))
        return "\n".join(result)


# ============================================================
# --- Main Simulation Loop (Modified to use RNG) ---
# ============================================================
def run_simulation():
    is_profiling = os.environ.get("MY_PROFILER_RUNNING", "0") == "1"
    is_interactive = READCHAR_AVAILABLE and not is_profiling
    if not is_interactive and not is_profiling:
        print("ERROR: Need 'readchar' or profiler mode.")
        return

    print(f"--- Running in {'PROFILER' if is_profiling else 'INTERACTIVE'} mode ---")
    print(f"--- Debug Render Mode: {constants.DEBUG_RENDER_MODE} ---")

    # --- Instantiate RNG ONCE here ---
    # Use a fixed seed for reproducibility, or time-based
    main_seed = 12345 if is_profiling else int(time.time() * 1000)
    rng_instance = GameRNG(seed=main_seed)
    print(f"--- Using RNG Seed: {main_seed} ---")

    # Pass RNG instance to GameState constructor
    game_state = GameState(80, 30, rng_instance)
    try:
        # initialize_map uses rng stored in game_state now
        game_state.initialize_map_and_entities()
    except Exception as e:
        logging.exception("Map init failed!")
        print(f"\nERROR: {e}")
        print("\033[?25h")
        return

    # Pre-compilation (No RNG involved here)
    print("Pre-compiling Numba functions...")
    if game_state.dungeon and game_state.player:
        try:
            dummy_rgb_sum_array = np.zeros_like(game_state.current_illumination_rgb_sum)
            dummy_los_array = np.zeros_like(game_state.dungeon.tiles, dtype=np.bool_)
            dummy_mem_intensity = np.zeros_like(game_state.dungeon.memory_intensity)
            dummy_last_seen = np.zeros_like(game_state.dungeon.last_seen_time)
            compute_illumination_color_array(
                game_state.player.position,
                0,
                game_state.dungeon,
                dummy_rgb_sum_array,
                game_state.player.base_color_rgb,
            )
            compute_los_into_boolean_array(
                game_state.player.position, 0, game_state.dungeon, dummy_los_array
            )
            _update_memory_fade_internal(
                0.0,
                dummy_last_seen,
                dummy_mem_intensity,
                dummy_los_array,
                game_state.dungeon.height,
                game_state.dungeon.width,
            )
            _update_dungeon_time(game_state.dungeon, np.float32(0.0))
        except Exception as e:
            logging.exception("Numba pre-compile error!")
            print(f"\nWARNING: {e}")
    elif not game_state.player:
        print("WARNING: Player missing, skipping pre-compile.")
    print("Pre-compilation finished.")

    # --- Main Loop (No direct RNG calls here) ---
    frame_count = 0
    start_time = time.time()
    last_frame_time = start_time
    target_duration = 60 if is_profiling else 300
    total_update_vis_time = 0.0
    last_key_pressed = ""
    profiler_path: Optional[List[Tuple[int, int]]] = None
    profiler_path_index = 0
    profiler_target_x = (
        game_state.dungeon.width - 6
        if game_state.player and game_state.player.x < game_state.dungeon.width // 2
        else 5
    )
    last_profiler_move_time = start_time
    profiler_move_delay = 0.01

    try:
        while time.time() - start_time < target_duration:
            loop_start_perf = time.perf_counter()
            current_frame_time = time.time()
            dt = min(current_frame_time - last_frame_time, 0.1)
            last_frame_time = current_frame_time
            player_moved = False
            quit_flag = False

            # Input / Player Movement (No RNG here)
            if is_profiling:
                if current_frame_time - last_profiler_move_time >= profiler_move_delay:
                    if profiler_path and profiler_path_index < len(profiler_path):
                        next_pos = profiler_path[profiler_path_index]
                        if game_state.player:
                            game_state.player.x, game_state.player.y = next_pos
                        profiler_path_index += 1
                        player_moved = True
                        last_profiler_move_time = current_frame_time
                    elif game_state.player:
                        start_pos = game_state.player.position
                        target_pos = (profiler_target_x, start_pos[1])
                        if start_pos != target_pos:
                            profiler_path = find_path(
                                start_pos,
                                target_pos,
                                game_state.dungeon.tiles,
                                game_state.dungeon.width,
                                game_state.dungeon.height,
                            )
                            if profiler_path and len(profiler_path) > 1:
                                profiler_path_index = 1
                                next_pos = profiler_path[profiler_path_index]
                                game_state.player.x, game_state.player.y = next_pos
                                profiler_path_index += 1
                                player_moved = True
                                last_profiler_move_time = current_frame_time
                                profiler_target_x = (
                                    5
                                    if profiler_target_x > game_state.dungeon.width // 2
                                    else game_state.dungeon.width - 6
                                )
                            else:
                                profiler_path = None
                                profiler_target_x = (
                                    5
                                    if profiler_target_x > game_state.dungeon.width // 2
                                    else game_state.dungeon.width - 6
                                )
                        else:
                            profiler_target_x = (
                                5
                                if profiler_target_x > game_state.dungeon.width // 2
                                else game_state.dungeon.width - 6
                            )
                            profiler_path = None
            elif is_interactive:
                print("Move (WASD/Arrows/Numpad 1-9), Q to quit: ", end="", flush=True)
                key = readchar.readkey()
                last_key_pressed = key
                print(" " * 50, end="\r")
                dx, dy = 0, 0
                if key.lower() == "q":
                    quit_flag = True
                elif key == readchar.key.UP or key == "w" or key == "8":
                    dy = -1
                elif key == readchar.key.DOWN or key == "s" or key == "2":
                    dy = 1
                elif key == readchar.key.LEFT or key == "a" or key == "4":
                    dx = -1
                elif key == readchar.key.RIGHT or key == "d" or key == "6":
                    dx = 1
                elif key == "7" or key == readchar.key.HOME:
                    dx, dy = -1, -1
                elif key == "9" or key == readchar.key.PAGE_UP:
                    dx, dy = 1, -1
                elif key == "1" or key == readchar.key.END:
                    dx, dy = -1, 1
                elif key == "3" or key == readchar.key.PAGE_DOWN:
                    dx, dy = 1, 1
                elif key == "5" or key == readchar.key.CLEAR or key == ".":
                    dx, dy = 0, 0
                if quit_flag:
                    break
                if (dx != 0 or dy != 0) and game_state.player:
                    target_x = game_state.player.x + dx
                    target_y = game_state.player.y + dy
                    if (
                        0 <= target_x < game_state.dungeon.width
                        and 0 <= target_y < game_state.dungeon.height
                        and game_state.dungeon.tiles[target_y, target_x]
                        == constants.FLOOR_ID
                    ):
                        game_state.player.x = target_x
                        game_state.player.y = target_y
                        player_moved = True
                elif not quit_flag:
                    player_moved = True
            if quit_flag:
                break

            # Update State, Visibility, Render (Updates use internal RNG)
            update_start_time = time.perf_counter()
            game_state.update(dt)
            update_time = time.perf_counter() - update_start_time
            vis_updated_this_frame = False
            if is_profiling or player_moved:
                vis_start_time = time.perf_counter()
                game_state.update_visibility()
                vis_end_time = time.perf_counter()
                frame_vis_time = vis_end_time - vis_start_time
                total_update_vis_time += frame_vis_time
                vis_updated_this_frame = True
            else:
                frame_vis_time = 0
            render_start_time = time.perf_counter()
            rendered_map = game_state.render()
            render_end_time = time.perf_counter()
            render_time = render_end_time - render_start_time
            print("\033[H\033[J", end="")
            print(rendered_map)

            # Status Info (No RNG here)
            elapsed_time = current_frame_time - start_time
            update_count = frame_count + 1
            avg_vis_time_ms = (
                (total_update_vis_time / update_count) * 1000 if update_count > 0 else 0
            )
            mode_str = "PROFILER" if is_profiling else "INTERACTIVE"
            debug_str = (
                f" (Debug: {constants.DEBUG_RENDER_MODE})"
                if constants.DEBUG_RENDER_MODE != "normal"
                else ""
            )
            print(
                f"\nMode: {mode_str}{debug_str} | Sim Time: {game_state.dungeon.current_time:.1f}s / {target_duration:.0f}s | Frame: {frame_count+1}"
            )
            print(
                f"Frame Times (ms): Render={render_time*1000:.1f}, VisUpdate={frame_vis_time*1000:.2f}, StateUpdate={update_time*1000:.2f} | Avg Vis: {avg_vis_time_ms:.3f}ms | DeltaT: {dt*1000:.1f}ms"
            )
            if game_state.player:
                p_mem = 0.0
                try:
                    p_mem = game_state.dungeon.memory_intensity[
                        game_state.player.y, game_state.player.x
                    ]
                except IndexError:
                    logging.warning("Player index error mem check.")
                    p_mem = -1.0
                status_line = f"Player @ {game_state.player.position} (Lvl:{game_state.player.light_level}, R:{game_state.player.light_radius}) | Mem@P: {p_mem:.2f}"
                if is_interactive:
                    status_line += f" | Last key: '{last_key_pressed}'"
                print(status_line)
            if is_profiling:
                print("Profiler running... Press Ctrl+C to exit.")
            frame_count += 1
    except KeyboardInterrupt:
        print("\nSimulation stopped by user (Ctrl+C).")
    except Exception:
        print("\033[?25h")
        print("\n--- ERROR ---")
        logging.exception("Error during loop:")
        print("-------------")
    finally:
        print("\033[?25h")
        print("Simulation finished.")


if __name__ == "__main__":
    run_simulation()

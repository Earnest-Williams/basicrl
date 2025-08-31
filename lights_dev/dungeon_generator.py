# dungeon_generator.py


"""
Functions for generating dungeon map layouts.
"""
import random

# Import necessary constants
from constants import FLOOR_ID, PILLAR_ID, WALL_ID

# Import the data structure it operates on
from dungeon_data import Dungeon


def dungeon_create_room(dungeon: Dungeon, x: int, y: int, width: int, height: int):
    """Carves a rectangular room into the dungeon tiles."""
    y_start, y_end = max(0, y), min(dungeon.height, y + height)
    x_start, x_end = max(0, x), min(dungeon.width, x + width)
    if y_start < y_end and x_start < x_end:
        # Access the tiles array via the passed dungeon instance
        dungeon.tiles[y_start:y_end, x_start:x_end] = FLOOR_ID


def dungeon_generate_map_u_shape(dungeon: Dungeon):
    """Generates a U-shaped room with pillars in the given Dungeon instance."""
    width = dungeon.width
    height = dungeon.height
    margin = 4
    open_x_min, open_x_max = margin, width - margin - 1
    open_y_min, open_y_max = margin, height - margin - 1
    block_y_start = height // 2 - 3
    block_y_end = block_y_start + 6
    block_x_min = open_x_min + 10
    block_x_max = open_x_max - 10

    dungeon.tiles.fill(WALL_ID)  # Fill using instance attribute
    dungeon.tiles[open_y_min : open_y_max + 1, open_x_min : open_x_max + 1] = (
        FLOOR_ID  # Carve floor
    )

    if block_x_min < block_x_max and block_y_start < block_y_end:
        dungeon.tiles[
            block_y_start : block_y_end + 1, block_x_min : block_x_max + 1
        ] = WALL_ID  # Add block

    num_pillars = 25
    pillars_placed = 0
    attempts = 0
    max_attempts = num_pillars * 20
    while pillars_placed < num_pillars and attempts < max_attempts:
        px = random.randint(open_x_min, open_x_max)
        py = random.randint(open_y_min, open_y_max)
        if dungeon.tiles[py, px] == FLOOR_ID:  # Check tile on instance
            dungeon.tiles[py, px] = PILLAR_ID  # Place pillar on instance
            pillars_placed += 1
        attempts += 1

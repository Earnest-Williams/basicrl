# tile_mapper_core.py

import os
import traceback

import structlog

from PySide6.QtCore import QRect

# Assuming tile_mapper_config is in the same directory
from tile_mapper_config import (
    CURRENT_FORMAT_VERSION,
    FILE_READ_MODE,
    FILE_WRITE_MODE,
    JSON_DUMPS_KWARGS,
    JSON_HANDLER,
    JSON_LOADS_KWARGS,
    USE_ORJSON,
    JSON_DecodeError,
)


log = structlog.get_logger(__name__)


# --- TileMap Data Structure ---
class TileMap:
    """Stores and manages the grid data."""

    def __init__(self, width, height, default_tile):
        width = max(0, width)
        height = max(0, height)
        self.width = width
        self.height = height
        self.default_tile = default_tile
        # Initialize grid efficiently
        self.tiles = [[default_tile] * width for _ in range(height)]

    def set_tile(self, x, y, tile):
        """Sets the tile at the given grid coordinates. Returns True if changed."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.tiles[y][x] != tile:  # Only change if different
                self.tiles[y][x] = tile
                return True
        return False

    def get_tile(self, x, y):
        """Gets the tile at the given grid coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None  # Return None if out of bounds

    def get_unique_tiles(self):
        """Returns a set of unique tile characters used in the map."""
        unique_chars = set()
        for row in self.tiles:
            unique_chars.update(row)
        return unique_chars

    def export_map_data(self):
        """Exports just the map data part for multi-map saving."""
        return {
            "grid_width": self.width,
            "grid_height": self.height,
            "default_tile": self.default_tile,
            "rows": ["".join(row) for row in self.tiles],
        }

    def load_from_data(self, data, app_config):
        """Loads tile data from a dictionary, using app_config for default tile."""
        try:
            rows = data.get("rows", [])
            if not isinstance(rows, list):
                log.error("Invalid map format - 'rows' key is not a list")
                return False

            raw_height = data.get("grid_height")
            raw_width = data.get("grid_width")
            raw_default = data.get("default_tile")

            new_height = len(rows) if raw_height is None else int(raw_height)
            # Use passed app_config for default tile fallback
            map_default_tile = (
                app_config.get("default_tile", ".")
                if raw_default is None
                else str(raw_default)
            )
            if new_height < 0:
                new_height = 0

            new_width = 0
            if raw_width is not None:
                new_width = int(raw_width)
            elif new_height > 0 and rows and isinstance(rows[0], str):
                new_width = max(len(r) for r in rows) if rows else 0
            if new_width < 0:
                new_width = 0

            if not all(isinstance(row, str) for row in rows):
                log.error("Invalid map format - not all rows are strings")
                return False

            padded_rows = []
            row_lengths_consistent = True
            if new_width > 0:
                for row in rows:
                    row_len = len(row)
                    if row_len != new_width:
                        row_lengths_consistent = False
                        padded_rows.append(row.ljust(new_width, map_default_tile))
                    else:
                        padded_rows.append(row)
                if not row_lengths_consistent:
                    log.warning("Rows padded to width", width=new_width)
            else:
                padded_rows = rows

            if len(padded_rows) != new_height:
                log.warning(
                    "Height mismatch. Using actual row count", actual_rows=len(padded_rows)
                )
                new_height = len(padded_rows)

            self.height = new_height
            self.width = new_width
            self.default_tile = map_default_tile
            self.tiles = [[map_default_tile] * new_width for _ in range(new_height)]

            copy_h = min(len(padded_rows), new_height)
            for y in range(copy_h):
                row_list = list(padded_rows[y])
                copy_w = min(len(row_list), new_width)
                self.tiles[y][:copy_w] = row_list[:copy_w]

            log.info(
                "Loaded map data",
                width=self.width,
                height=self.height,
                default_tile=self.default_tile,
            )
            return True

        except (ValueError, TypeError) as e:
            log.error("Error processing map data values", error=str(e), exc_info=True)
            return False
        except Exception as e:
            log.error("Error loading map data", error=str(e), exc_info=True)
            return False


# --- Core Map Logic Functions ---


def get_neighbors(x, y):
    """Returns 4-directional neighbors (up, down, left, right)."""
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def draw_line(tilemap: TileMap, start_pos, end_pos, tile: str) -> bool:
    """Draws a line of tiles using Bresenham's algorithm. Returns True if changed."""
    x1, y1 = start_pos.x(), start_pos.y()
    x2, y2 = end_pos.x(), end_pos.y()
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    changed = False
    while True:
        if tilemap.set_tile(x1, y1, tile):
            changed = True
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            if x1 == x2:
                break  # Avoid infinite loop on vertical lines
            err += dy
            x1 += sx
        if e2 <= dx:
            if y1 == y2:
                break  # Avoid infinite loop on horizontal lines
            err += dx
            y1 += sy
    return changed


def fill_rectangle(tilemap: TileMap, start_pos, end_pos, tile: str) -> bool:
    """Fills a rectangle defined by start and end points. Returns True if changed."""
    x1, y1 = start_pos.x(), start_pos.y()
    x2, y2 = end_pos.x(), end_pos.y()
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    changed = False
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if tilemap.set_tile(x, y, tile):
                changed = True
    return changed


def _flood_fill_find_area(
    tilemap: TileMap, start_x: int, start_y: int, match_tile_type: str | None = None
) -> set[tuple[int, int]]:
    """Helper: Finds contiguous area based on match_tile_type. Returns set of (x, y) tuples."""
    start_tile = tilemap.get_tile(start_x, start_y)
    if start_tile is None:
        return set()

    if match_tile_type is None:  # Mode 1: Fill non-default area
        if start_tile == tilemap.default_tile:
            return set()
        target_type = start_tile
    else:  # Mode 2: Fill specific type
        if start_tile != match_tile_type:
            return set()
        target_type = match_tile_type

    seen = set()
    frontier = [(start_x, start_y)]
    connected_area = set()

    while frontier:
        cx, cy = frontier.pop()
        if (
            not (0 <= cx < tilemap.width and 0 <= cy < tilemap.height)
            or (cx, cy) in seen
        ):
            continue

        current_tile = tilemap.get_tile(cx, cy)
        if current_tile == target_type:
            seen.add((cx, cy))
            connected_area.add((cx, cy))
            for nx, ny in get_neighbors(cx, cy):
                if (nx, ny) not in seen:
                    frontier.append((nx, ny))
        else:
            seen.add((cx, cy))

    return connected_area


def flood_fill_replace(
    tilemap: TileMap, start_x: int, start_y: int, fill_tile: str
) -> int:
    """Performs a classic flood fill replace. Returns number of tiles filled."""
    original_tile = tilemap.get_tile(start_x, start_y)
    if original_tile is None or original_tile == fill_tile:
        return 0

    q = [(start_x, start_y)]
    visited = set([(start_x, start_y)])
    filled_count = 0

    while q:
        x, y = q.pop(0)
        if tilemap.set_tile(x, y, fill_tile):
            filled_count += 1
            for nx, ny in get_neighbors(x, y):
                if (
                    0 <= nx < tilemap.width
                    and 0 <= ny < tilemap.height
                    and (nx, ny) not in visited
                    and tilemap.get_tile(nx, ny) == original_tile
                ):
                    visited.add((nx, ny))
                    q.append((nx, ny))

    print(
        f"Flood Fill: Replaced {filled_count} tiles of type '{original_tile}' with '{fill_tile}'."
    )
    return filled_count


def ctrl_click_fill(tilemap: TileMap, x: int, y: int, fill_tile: str) -> bool:
    """Fills perimeter of non-default area with fill_tile. Returns True if changed."""
    connected_area = _flood_fill_find_area(tilemap, x, y, match_tile_type=None)
    if not connected_area:
        print(f"Ctrl+Click: No non-default area found at ({x},{y}).")
        return False

    perimeter_to_fill = set()
    for cx, cy in connected_area:
        for nx, ny in get_neighbors(cx, cy):
            if 0 <= nx < tilemap.width and 0 <= ny < tilemap.height:
                neighbor_tile = tilemap.get_tile(nx, ny)
                if (
                    neighbor_tile == tilemap.default_tile
                    and (nx, ny) not in perimeter_to_fill
                ):
                    perimeter_to_fill.add((nx, ny))

    if not perimeter_to_fill:
        print("Ctrl+Click: Area found, but no default tile perimeter detected.")
        return False

    painted_count = 0
    for px, py in perimeter_to_fill:
        if tilemap.set_tile(px, py, fill_tile):
            painted_count += 1

    print(f"Ctrl+Click: Painted {painted_count} perimeter tiles with '{fill_tile}'.")
    return painted_count > 0


def ctrl_shift_click_wall(
    tilemap: TileMap, x: int, y: int, wall_fill_tile: str, app_config: dict
) -> bool:
    """Walls perimeter of same-tile area, respecting walls/doors. Returns True if changed."""
    clicked_tile_type = tilemap.get_tile(x, y)
    wall_char = app_config.get("wall_tile", "#")
    door_char = app_config.get("door_tile", "+")
    protected_chars = {wall_char, door_char}

    if clicked_tile_type is None or clicked_tile_type == tilemap.default_tile:
        print(
            f"Ctrl+Shift+Click: Cannot wall from default '{tilemap.default_tile}' or outside."
        )
        return False

    connected_area = _flood_fill_find_area(
        tilemap, x, y, match_tile_type=clicked_tile_type
    )
    if not connected_area:
        print(
            f"Ctrl+Shift+Click: No contiguous area of type '{clicked_tile_type}' found."
        )
        return False

    perimeter_to_wall = set()
    for cx, cy in connected_area:
        for nx, ny in get_neighbors(cx, cy):
            if 0 <= nx < tilemap.width and 0 <= ny < tilemap.height:
                if (
                    nx,
                    ny,
                ) not in connected_area:  # Add if neighbor is outside the area
                    perimeter_to_wall.add((nx, ny))

    if not perimeter_to_wall:
        print(
            f"Ctrl+Shift+Click: Area of '{clicked_tile_type}' found, but no perimeter detected."
        )
        return False

    painted_count = 0
    skipped_count = 0
    for px, py in perimeter_to_wall:
        existing_tile = tilemap.get_tile(px, py)
        if existing_tile not in protected_chars:
            if tilemap.set_tile(px, py, wall_fill_tile):
                painted_count += 1
        else:
            skipped_count += 1

    print(
        f"Ctrl+Shift+Click: Painted {painted_count} perimeter tiles with '{wall_fill_tile}'. Skipped {skipped_count} protected."
    )
    return painted_count > 0


# --- Region Extraction / Saving Logic ---


def extract_map_region(source_tilemap: TileMap, selection: QRect) -> dict | None:
    """
    Extracts tile data from a specified rectangular region of a TileMap.
    QRect coordinates are assumed to be grid coordinates.

    Returns:
        A dictionary containing the extracted map data ('width', 'height',
        'default_tile', 'rows'), or None if selection is invalid.
    """
    start_x, start_y = selection.left(), selection.top()
    # QRect width/height represent number of cells
    new_width = selection.width()
    new_height = selection.height()
    end_x = start_x + new_width  # Exclusive end for slicing
    end_y = start_y + new_height  # Exclusive end for slicing

    # Validate coordinates fully
    if not (
        0 <= start_x < source_tilemap.width
        and 0 <= start_y < source_tilemap.height
        and 0 < end_x <= source_tilemap.width
        and 0 < end_y <= source_tilemap.height
        and new_width > 0
        and new_height > 0
    ):
        log.error(
            "Invalid selection rectangle coordinates",
            rect=selection.getRect(),
        )
        return None

    new_rows = []
    try:
        for y in range(start_y, end_y):
            # Slice the row directly
            row_segment = source_tilemap.tiles[y][start_x:end_x]
            new_rows.append("".join(row_segment))
    except IndexError:
        log.error("Index out of bounds during region extraction", exc_info=True)
        return None

    # Use the source map's default tile for the extracted region
    new_default = source_tilemap.default_tile

    return {
        "grid_width": new_width,
        "grid_height": new_height,
        "default_tile": new_default,
        "rows": new_rows,
    }


def save_extracted_map(
    target_filepath: str, new_map_key: str, extracted_map_data: dict
) -> bool:
    """Loads target JSON, adds/updates the map entry for the new key, saves back."""
    if not new_map_key:
        log.error("New map key cannot be empty")
        return False
    if not extracted_map_data:
        log.error("No extracted map data provided")
        return False

    file_data = {"format_version": CURRENT_FORMAT_VERSION, "maps": {}}
    if os.path.exists(target_filepath):
        try:
            with open(target_filepath, FILE_READ_MODE) as f:
                content = f.read()
                if content:
                    existing_data = JSON_HANDLER.loads(content, **JSON_LOADS_KWARGS)
                    # Basic validation of existing format
                    if isinstance(existing_data, dict) and isinstance(
                        existing_data.get("maps"), dict
                    ):
                        file_data = existing_data
                        # Ensure format version is updated/present
                        file_data["format_version"] = CURRENT_FORMAT_VERSION
                    else:
                        log.warning(
                            "File has invalid format. Overwriting.",
                            filepath=target_filepath,
                        )
                        # Keep file_data as the default new structure
                # else: file is empty, keep default structure
        except (JSON_DecodeError, IOError, Exception) as e:
            log.warning(
                "Could not read/parse existing file. Will overwrite.",
                filepath=target_filepath,
                error=str(e),
                exc_info=True,
            )
            # Keep file_data as the default new structure, existing data lost

    # Add/overwrite the extracted map data
    file_data["maps"][new_map_key] = extracted_map_data

    # Save back using appropriate handler
    try:
        with open(target_filepath, FILE_WRITE_MODE) as f:
            if USE_ORJSON:
                f.write(JSON_HANDLER.dumps(file_data, **JSON_DUMPS_KWARGS))
            else:
                JSON_HANDLER.dump(file_data, f, **JSON_DUMPS_KWARGS)
        log.info(
            "Extracted map saved successfully",
            map_key=new_map_key,
            filepath=target_filepath,
        )
        return True
    except Exception as e:
        log.error(
            "Error saving extracted map",
            filepath=target_filepath,
            error=str(e),
            exc_info=True,
        )
        return False

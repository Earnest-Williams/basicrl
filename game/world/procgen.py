# basicrl/game/world/procgen.py
from collections import deque
from typing import Dict, Iterator, List, NamedTuple, Tuple, Union

import numpy as np
import structlog

try:
    from utils.game_rng import GameRNG
except ImportError as e:
    structlog.get_logger().error("CRITICAL: GameRNG class not found.", error=str(e))
    raise

try:
    from game.world.game_map import TILE_ID_FLOOR, TILE_ID_WALL, GameMap
except ImportError as e:
    structlog.get_logger().error(
        "CRITICAL: GameMap class or TILE_ID_FLOOR not found.", error=str(e)
    )
    raise


log = structlog.get_logger()

# --- Configuration ---
MIN_LEAF_SIZE = 6
ROOM_MAX_SIZE_RATIO = 0.8
ROOM_MIN_SIZE = 4
MAX_BSP_DEPTH = 10
DEFAULT_ROOM_CEILING_OFFSET = 6 # 3.0 meters
DEFAULT_CORRIDOR_CEILING_OFFSET = 4 # 2.0 meters


class Rect(NamedTuple):
    """A rectangle on the map."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[int, int]:
        """Center coordinates of the rectangle."""
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return center_x, center_y

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    def intersects(self, other: "Rect") -> bool:
        """Returns True if this rectangle intersects with another one."""
        return (
            self.x1 <= other.x2
            and self.x2 >= other.x1
            and self.y1 <= other.y2
            and self.y2 >= other.y1
        )

    def carve(self, game_map: GameMap, floor_height: int, ceiling_height: int) -> None:
        """Carves this rectangle as floor tiles onto the game map with specified heights."""
        # Ensure GameMap is the correct type before proceeding
        if not isinstance(game_map, GameMap):
            log.error("Carve called with invalid GameMap object")
            return

        y_start = max(0, self.y1)
        y_end = min(game_map.height, self.y2 + 1)
        x_start = max(0, self.x1)
        x_end = min(game_map.width, self.x2 + 1)

        log_context = {
            "rect": self,
            "floor_h": floor_height,
            "ceil_h": ceiling_height,
            "y_slice": f"{y_start}:{y_end}",
            "x_slice": f"{x_start}:{x_end}",
        }

        if y_start < y_end and x_start < x_end:
            try:
                # Set tile type
                game_map.tiles[y_start:y_end, x_start:x_end] = TILE_ID_FLOOR
                # Assign Height/Ceiling
                game_map.height_map[y_start:y_end, x_start:x_end] = floor_height
                game_map.ceiling_map[y_start:y_end, x_start:x_end] = ceiling_height
                log.debug("Carved rectangle area", **log_context)
            except IndexError:
                log.error("IndexError during carving", **log_context)
            except Exception as e:
                log.error(
                    "Error during carving", error=str(e), exc_info=True, **log_context
                )
        else:
            log.warning("Attempted to carve zero-size area", **log_context)


class BSPNode:
    """Represents a node in the BSP tree."""
    def __init__(
        self, rect: Rect, base_height: int = 0
    ): # base_height represents floor height
        self.rect: Rect = rect
        self.left: Union[BSPNode, None] = None
        self.right: Union[BSPNode, None] = None
        self.room: Union[Rect, None] = None
        self.corridors: List[Rect] = []
        self.base_height: int = base_height # Floor height for this node's region

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def get_leaves(self) -> Iterator["BSPNode"]:
        if self.is_leaf:
            yield self
        else:
            if self.left:
                yield from self.left.get_leaves()
            if self.right:
                yield from self.right.get_leaves()

    def get_room(self) -> Union[Rect, None]:
        if self.room:
            return self.room
        room = None
        if self.left:
            room = self.left.get_room()
        if not room and self.right:
            room = self.right.get_room()
        return room


def _split_node_recursive(node: BSPNode, rng: GameRNG, depth: int) -> bool:
    """Recursively splits a BSP node. Returns True if split occurred."""
    # Ensure GameRNG is the correct type before proceeding
    if not isinstance(rng, GameRNG):
        log.error("Split called with invalid GameRNG object")
        return False

    log.debug(
        "Attempting BSP split",
        depth=depth,
        max_depth=MAX_BSP_DEPTH,
        is_leaf=node.is_leaf,
        rect=node.rect,
        base_h=node.base_height,
    )

    if depth >= MAX_BSP_DEPTH:
        log.debug("Split aborted: Max depth reached", depth=depth)
        return False

    # Decide split direction
    split_horizontally: bool
    if (
        node.rect.width > node.rect.height
        and node.rect.width / node.rect.height >= 1.25
    ):
        split_horizontally = False
    elif (
        node.rect.height > node.rect.width
        and node.rect.height / node.rect.width >= 1.25
    ):
        split_horizontally = True
    else:
        split_horizontally = rng.coin_flip()[0] == "heads"

    # Check if node is large enough
    max_size = node.rect.height if split_horizontally else node.rect.width
    min_req_size = MIN_LEAF_SIZE * 2
    if max_size <= min_req_size:
        log.debug(
            "Split aborted: Node too small",
            dimension_size=max_size,
            min_required=min_req_size,
            split_direction="horizontal" if split_horizontally else "vertical",
            rect=node.rect,
        )
        return False

    # Determine split position
    split_margin = MIN_LEAF_SIZE
    parent_height = node.base_height # Get parent floor height

    # Basic Height Inheritance (No variation yet)
    left_h, right_h = parent_height, parent_height

    if split_horizontally:
        split_y = rng.get_int(node.rect.y1 + split_margin, node.rect.y2 - split_margin)
        node.left = BSPNode(
            Rect(node.rect.x1, node.rect.y1, node.rect.x2, split_y - 1), left_h
        )
        node.right = BSPNode(
            Rect(node.rect.x1, split_y, node.rect.x2, node.rect.y2), right_h
        )
        log.debug(
            "Split node horizontally",
            depth=depth,
            y_split=split_y,
            left_rect=node.left.rect,
            left_h=left_h,
            right_rect=node.right.rect,
            right_h=right_h,
        )
    else: # Split vertically
        split_x = rng.get_int(node.rect.x1 + split_margin, node.rect.x2 - split_margin)
        node.left = BSPNode(
            Rect(node.rect.x1, node.rect.y1, split_x - 1, node.rect.y2), left_h
        )
        node.right = BSPNode(
            Rect(split_x, node.rect.y1, node.rect.x2, node.rect.y2), right_h
        )
        log.debug(
            "Split node vertically",
            depth=depth,
            x_split=split_x,
            left_rect=node.left.rect,
            left_h=left_h,
            right_rect=node.right.rect,
            right_h=right_h,
        )

    # Recursively split children
    _split_node_recursive(node.left, rng, depth + 1)
    _split_node_recursive(node.right, rng, depth + 1)

    return True # This node was split


def _create_rooms_in_leaves(root_node: BSPNode, rng: GameRNG):
    """Creates rooms within the leaf nodes of the BSP tree."""
    # Ensure GameRNG is the correct type before proceeding
    if not isinstance(rng, GameRNG):
        log.error("Create rooms called with invalid GameRNG object")
        return

    log.debug("Creating rooms in leaves...")
    room_count = 0
    skipped_count = 0
    for leaf in root_node.get_leaves():
        # Room size calculation
        max_w = int(leaf.rect.width * ROOM_MAX_SIZE_RATIO)
        max_h = int(leaf.rect.height * ROOM_MAX_SIZE_RATIO)
        room_w = rng.get_int(ROOM_MIN_SIZE, max(ROOM_MIN_SIZE, max_w))
        room_h = rng.get_int(ROOM_MIN_SIZE, max(ROOM_MIN_SIZE, max_h))

        # Room position calculation
        room_x1 = rng.get_int(
            leaf.rect.x1, max(leaf.rect.x1, leaf.rect.x2 - room_w + 1)
        )
        room_y1 = rng.get_int(
            leaf.rect.y1, max(leaf.rect.y1, leaf.rect.y2 - room_h + 1)
        )
        room_x2 = room_x1 + room_w - 1
        room_y2 = room_y1 + room_h - 1

        # Clamp coordinates
        room_x1 = max(leaf.rect.x1, room_x1)
        room_y1 = max(leaf.rect.y1, room_y1)
        room_x2 = min(leaf.rect.x2, room_x2)
        room_y2 = min(leaf.rect.y2, room_y2)

        # Check final size and create room
        if (
            room_x2 >= room_x1 + ROOM_MIN_SIZE - 1
            and room_y2 >= room_y1 + ROOM_MIN_SIZE - 1
        ):
            leaf.room = Rect(room_x1, room_y1, room_x2, room_y2)
            room_count += 1
            log.debug(
                "Defined room",
                room_rect=leaf.room,
                leaf_rect=leaf.rect,
                base_floor_h=leaf.base_height,
            )
        else:
            skipped_count += 1
            log.debug(
                "Skipped room creation (too small)",
                leaf_rect=leaf.rect,
                final_w=(room_x2 - room_x1 + 1),
                final_h=(room_y2 - room_y1 + 1),
            )
    log.info("Room definition finished", created=room_count, skipped=skipped_count)


def _carve_tunnel(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    start_floor_h: int,
    end_floor_h: int, # Use floor heights
    game_map: GameMap,
    rng: GameRNG,
) -> List[Rect]:
    """
    Carves an L-shaped tunnel between two points, assigning simple heights.
    Returns list of Rects carved.
    """
    # Ensure GameMap and GameRNG are valid
    if not isinstance(game_map, GameMap):
        log.error("Carve tunnel called with invalid GameMap object")
        return []
    if not isinstance(rng, GameRNG):
        log.error("Carve tunnel called with invalid GameRNG object")
        return []

    log_context = {
        "start_pos": (x1, y1),
        "end_pos": (x2, y2),
        "start_floor_h": start_floor_h,
        "end_floor_h": end_floor_h,
    }
    log.debug("Carving tunnel", **log_context)
    carved_rects = []

    # Simple height assignment: use start height for first segment, end height for second.
    # Define ceiling based on floor height.
    corridor_ceil_h1 = start_floor_h + DEFAULT_CORRIDOR_CEILING_OFFSET
    corridor_ceil_h2 = end_floor_h + DEFAULT_CORRIDOR_CEILING_OFFSET

    if rng.coin_flip()[0] == "heads": # Horizontal first, then vertical
        # Carve horizontal segment (use start_floor_h)
        cx1, cy1, cx2, cy2 = min(x1, x2), y1, max(x1, x2), y1
        h_tunnel = Rect(cx1, cy1, cx2, cy2)
        h_tunnel.carve(game_map, start_floor_h, corridor_ceil_h1)
        carved_rects.append(h_tunnel)
        log.debug(
            "Carved horizontal tunnel segment",
            rect=h_tunnel,
            floor_h=start_floor_h,
            ceil_h=corridor_ceil_h1,
        )

        # Carve vertical segment (use end_floor_h)
        vx1, vy1, vx2, vy2 = x2, min(y1, y2), x2, max(y1, y2)
        # Ensure correct range for vertical segment (don't overlap corners)
        if y1 < y2:
            vy1 = y1 + 1 # Start one step down if moving down
        if y1 > y2:
            vy2 = y1 - 1 # Start one step up if moving up
        if vy1 <= vy2: # Only carve if valid range
            v_tunnel = Rect(vx1, vy1, vx2, vy2)
            v_tunnel.carve(game_map, end_floor_h, corridor_ceil_h2)
            carved_rects.append(v_tunnel)
            log.debug(
                "Carved vertical tunnel segment",
                rect=v_tunnel,
                floor_h=end_floor_h,
                ceil_h=corridor_ceil_h2,
            )

    else: # Vertical first, then horizontal
        # Carve vertical segment (use start_floor_h)
        vx1, vy1, vx2, vy2 = x1, min(y1, y2), x1, max(y1, y2)
        v_tunnel = Rect(vx1, vy1, vx2, vy2)
        v_tunnel.carve(game_map, start_floor_h, corridor_ceil_h1)
        carved_rects.append(v_tunnel)
        log.debug(
            "Carved vertical tunnel segment",
            rect=v_tunnel,
            floor_h=start_floor_h,
            ceil_h=corridor_ceil_h1,
        )

        # Carve horizontal segment (use end_floor_h)
        cx1, cy1, cx2, cy2 = min(x1, x2), y2, max(x1, x2), y2
        # Ensure correct range for horizontal segment
        if x1 < x2:
            cx1 = x1 + 1
        if x1 > x2:
            cx2 = x1 - 1
        if cx1 <= cx2: # Only carve if valid range
            h_tunnel = Rect(cx1, cy1, cx2, cy2)
            h_tunnel.carve(game_map, end_floor_h, corridor_ceil_h2)
            carved_rects.append(h_tunnel)
            log.debug(
                "Carved horizontal tunnel segment",
                rect=h_tunnel,
                floor_h=end_floor_h,
                ceil_h=corridor_ceil_h2,
            )

    return carved_rects


def _connect_rooms(node: BSPNode, game_map: GameMap, rng: GameRNG):
    """Recursively connects rooms in sibling nodes."""
    # Ensure GameMap and GameRNG are valid
    if not isinstance(game_map, GameMap):
        log.error("Connect rooms called with invalid GameMap object")
        return
    if not isinstance(rng, GameRNG):
        log.error("Connect rooms called with invalid GameRNG object")
        return

    log.debug("Connecting rooms for node", rect=node.rect, is_leaf=node.is_leaf)
    if node.is_leaf:
        return

    # Recursively connect children first
    if node.left:
        _connect_rooms(node.left, game_map, rng)
    if node.right:
        _connect_rooms(node.right, game_map, rng)

    # Connect the rooms in the direct children (if they exist)
    left_room = node.left.get_room() if node.left else None
    right_room = node.right.get_room() if node.right else None

    if left_room and right_room:
        # Pick random points within each room to connect
        lx, ly = rng.get_int(left_room.x1, left_room.x2), rng.get_int(
            left_room.y1, left_room.y2
        )
        rx, ry = rng.get_int(right_room.x1, right_room.x2), rng.get_int(
            right_room.y1, right_room.y2
        )

        # Get Base Floor Heights
        start_h = (
            node.left.base_height if node.left else 0
        ) # Default if something went wrong
        end_h = node.right.base_height if node.right else 0

        log_context = {
            "left_center": left_room.center,
            "right_center": right_room.center,
            "connect_left": (lx, ly),
            "connect_right": (rx, ry),
            "start_floor_h": start_h,
            "end_floor_h": end_h,
        }
        log.debug("Connecting sibling rooms", **log_context)

        # Pass floor heights to carve_tunnel
        node.corridors = _carve_tunnel(lx, ly, rx, ry, start_h, end_h, game_map, rng)
    elif left_room and not right_room:
         log.debug("Skipping connection: Right child has no room", left_center=left_room.center)
    elif not left_room and right_room:
         log.debug("Skipping connection: Left child has no room", right_center=right_room.center)
    else:
         log.debug("Skipping connection: Neither child has a room.")


def _generate_bsp_dungeon(
    game_map: GameMap, map_width: int, map_height: int, rng: GameRNG
) -> Tuple[int, int]:
    """Generate a classic rooms-and-corridors layout using BSP."""
    initial_base_height = 0
    root_node = BSPNode(Rect(1, 1, map_width - 2, map_height - 2), initial_base_height)
    log.debug(
        "Created root BSP node", rect=root_node.rect, base_height=initial_base_height
    )

    log.info("Splitting BSP tree...")
    _split_node_recursive(root_node, rng, 0)

    log.info("Defining rooms...")
    _create_rooms_in_leaves(root_node, rng)

    all_rooms: List[Rect] = []
    log.info("Carving rooms onto map...")
    for leaf in root_node.get_leaves():
        if leaf.room:
            room_floor_h = leaf.base_height
            room_ceil_h = room_floor_h + DEFAULT_ROOM_CEILING_OFFSET
            leaf.room.carve(game_map, room_floor_h, room_ceil_h)
            all_rooms.append(leaf.room)
    log.info("Rooms carved", count=len(all_rooms))

    if not all_rooms:
        log.error("BSP generation failed to create any rooms!")
        raise RuntimeError("BSP generation failed to create any rooms!")

    log.info("Connecting rooms...")
    _connect_rooms(root_node, game_map, rng)

    first_room = all_rooms[0]
    player_start_x, player_start_y = first_room.center
    log.info(
        "Determined player start position",
        pos=(player_start_x, player_start_y),
        first_room_rect=first_room,
        room_center=first_room.center,
    )

    return player_start_x, player_start_y


def _generate_cavern_level(
    game_map: GameMap, map_width: int, map_height: int, rng: GameRNG
) -> Tuple[int, int]:
    """Generate a cave layout using cellular automata."""
    fill_prob = 0.45
    for y in range(1, map_height - 1):
        for x in range(1, map_width - 1):
            if rng.get_float() < fill_prob:
                game_map.tiles[y, x] = TILE_ID_FLOOR
            else:
                game_map.tiles[y, x] = TILE_ID_WALL
    for _ in range(4):
        new_tiles = game_map.tiles.copy()
        for y in range(1, map_height - 1):
            for x in range(1, map_width - 1):
                wall_count = 0
                for ny in range(y - 1, y + 2):
                    for nx in range(x - 1, x + 2):
                        if nx == x and ny == y:
                            continue
                        if game_map.tiles[ny, nx] == TILE_ID_WALL:
                            wall_count += 1
                if wall_count > 4:
                    new_tiles[y, x] = TILE_ID_WALL
                else:
                    new_tiles[y, x] = TILE_ID_FLOOR
        game_map.tiles[1 : map_height - 1, 1 : map_width - 1] = new_tiles[
            1 : map_height - 1, 1 : map_width - 1
        ]

    floor_positions = np.argwhere(game_map.tiles == TILE_ID_FLOOR)
    if floor_positions.size == 0:
        raise RuntimeError("Cavern generation produced no walkable tiles")

    # Pick a random floor tile and flood fill to find its connected region
    start_index = rng.get_int(0, len(floor_positions) - 1)
    seed_y, seed_x = floor_positions[start_index]
    queue = deque([(int(seed_y), int(seed_x))])
    visited = {(int(seed_y), int(seed_x))}
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ny, nx = cy + dy, cx + dx
            if (
                0 <= ny < map_height
                and 0 <= nx < map_width
                and game_map.tiles[ny, nx] == TILE_ID_FLOOR
                and (ny, nx) not in visited
            ):
                visited.add((ny, nx))
                queue.append((ny, nx))

    # Convert unreachable floor tiles to walls
    for (y, x) in floor_positions:
        if (int(y), int(x)) not in visited:
            game_map.tiles[y, x] = TILE_ID_WALL

    # Update height and ceiling only for reachable floors
    for (y, x) in visited:
        game_map.height_map[y, x] = 0
        game_map.ceiling_map[y, x] = DEFAULT_ROOM_CEILING_OFFSET

    visited_list = list(visited)
    spawn_y, spawn_x = visited_list[rng.get_int(0, len(visited_list) - 1)]
    return int(spawn_x), int(spawn_y)


def _apply_prefab(game_map: GameMap, x: int, y: int, prefab: List[str]) -> Tuple[int, int]:
    """Carve a prefab module onto the map and return its center."""
    for dy, row in enumerate(prefab):
        for dx, char in enumerate(row):
            tx, ty = x + dx, y + dy
            if not game_map.in_bounds(tx, ty):
                continue
            if char == ".":
                game_map.tiles[ty, tx] = TILE_ID_FLOOR
                game_map.height_map[ty, tx] = 0
                game_map.ceiling_map[ty, tx] = DEFAULT_ROOM_CEILING_OFFSET
    w, h = len(prefab[0]), len(prefab)
    return x + w // 2, y + h // 2


PREFABS = [
    ["#####", "#...#", "#...#", "#...#", "#####"],
    ["#######", "#.....#", "#.....#", "#.....#", "#######"],
]


def _generate_surface_level(
    game_map: GameMap, map_width: int, map_height: int, rng: GameRNG
) -> Tuple[int, int]:
    """Place prefab surface structures."""
    centers: List[Tuple[int, int]] = []
    for prefab in PREFABS:
        h = len(prefab)
        w = len(prefab[0])
        x = rng.get_int(1, max(1, map_width - w - 1))
        y = rng.get_int(1, max(1, map_height - h - 1))
        centers.append(_apply_prefab(game_map, x, y, prefab))
    if not centers:
        raise RuntimeError("No prefabs placed")
    return centers[0]


def _place_vertical_transitions(
    game_map: GameMap, rng: GameRNG, floor_positions: np.ndarray
) -> None:
    floor_list = [tuple(p) for p in floor_positions]
    transitions: List[Dict[str, Union[int, str]]] = []
    if floor_list:
        choices = rng.sample(floor_list, k=min(2, len(floor_list)))
        if len(choices) >= 1:
            y, x = choices[0]
            transitions.append({"type": "stairs_up", "x": int(x), "y": int(y)})
        if len(choices) >= 2:
            y, x = choices[1]
            transitions.append({"type": "stairs_down", "x": int(x), "y": int(y)})
    game_map.vertical_transitions = transitions


def _add_story_hooks(
    game_map: GameMap,
    rng: GameRNG,
    floor_positions: np.ndarray,
    count: int = 2,
) -> None:
    descriptions = [
        "bones litter the ground",
        "a campfire long extinguished",
        "scratches on the wall",
        "a mysterious rune etched into stone",
    ]
    floor_list = [tuple(p) for p in floor_positions]
    hooks: List[Dict[str, Union[int, str]]] = []
    if floor_list:
        count = min(count, len(floor_list))
        positions = rng.sample(floor_list, k=count)
        for y, x in positions:
            hooks.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "description": rng.choice(descriptions),
                }
            )
    game_map.story_hooks = hooks


def generate_dungeon(
    game_map: GameMap,
    map_width: int,
    map_height: int,
    seed: int | None = None,
    algorithm: str = "bsp",
    region: str | None = None,
) -> Tuple[int, int]:
    """Entry point for dungeon generation selecting different algorithms."""
    if not isinstance(game_map, GameMap):
        log.error("Generate dungeon called with invalid GameMap object")
        raise TypeError("Invalid GameMap object passed to generate_dungeon")

    log.info(
        "Starting dungeon generation",
        width=map_width,
        height=map_height,
        seed=seed,
        algorithm=algorithm,
        region=region,
    )

    rng = GameRNG(seed=seed)

    if algorithm == "cellular":
        player_start_x, player_start_y = _generate_cavern_level(
            game_map, map_width, map_height, rng
        )
    elif algorithm == "prefab":
        player_start_x, player_start_y = _generate_surface_level(
            game_map, map_width, map_height, rng
        )
    else:
        player_start_x, player_start_y = _generate_bsp_dungeon(
            game_map, map_width, map_height, rng
        )

    floor_positions = np.argwhere(game_map.tiles == TILE_ID_FLOOR)
    _place_vertical_transitions(game_map, rng, floor_positions)
    _add_story_hooks(game_map, rng, floor_positions)

    log.info("Updating transparency map...")
    game_map.update_tile_transparency()
    log.info(
        "Dungeon generation complete",
        player_start=(player_start_x, player_start_y),
        algorithm=algorithm,
    )
    return player_start_x, player_start_y

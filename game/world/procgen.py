# basicrl/game/world/procgen.py
import numpy as np
from typing import Iterator, Tuple, List, Union, NamedTuple

# Assuming GameRNG is importable like this based on your structure
from utils.game_rng import GameRNG
from game.world.game_map import GameMap, TILE_ID_FLOOR, TILE_ID_WALL

# --- Configuration ---
MIN_LEAF_SIZE = 6  # Minimum size (width or height) of a BSP leaf node
ROOM_MAX_SIZE_RATIO = 0.8  # Max room size relative to its leaf node size
ROOM_MIN_SIZE = 4  # Min absolute room size
MAX_BSP_DEPTH = 10  # Maximum recursion depth for splitting


class Rect(NamedTuple):
    """A rectangle on the map."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[int, int]:
        print("x1: ", self.x1, "y1: ", self.y1, "x2: ", self.x2, "y2: ", self.y2)
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

    def carve(self, game_map: GameMap) -> None:
        """Carves this rectangle as floor tiles onto the game map."""
        # Use NumPy slicing for efficiency, ensuring bounds
        y_start = max(0, self.y1)
        y_end = min(game_map.height, self.y2 + 1)  # Slice end is exclusive
        x_start = max(0, self.x1)
        x_end = min(game_map.width, self.x2 + 1)  # Slice end is exclusive

        if y_start < y_end and x_start < x_end:
            # CORRECTED INDEXING: [y_slice, x_slice]
            game_map.tiles[y_start:y_end, x_start:x_end] = TILE_ID_FLOOR


class BSPNode:
    """Represents a node in the BSP tree."""

    def __init__(self, rect: Rect):
        self.rect: Rect = rect
        self.left: Union[BSPNode, None] = None
        self.right: Union[BSPNode, None] = None
        self.room: Union[Rect, None] = None  # Room associated with this node (if leaf)
        self.corridors: List[Rect] = []  # Corridors created from this node's split

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def get_leaves(self) -> Iterator["BSPNode"]:
        """Generator to yield all leaf nodes in this subtree."""
        if self.is_leaf:
            yield self
        else:
            if self.left:
                yield from self.left.get_leaves()
            if self.right:
                yield from self.right.get_leaves()

    def get_room(self) -> Union[Rect, None]:
        """Finds the first room in this node or its descendants."""
        if self.room:
            return self.room
        room = None
        if self.left:
            room = self.left.get_room()
        if not room and self.right:
            room = self.right.get_room()
        return room


def _split_node_recursive(node: BSPNode, rng: GameRNG, depth: int) -> bool:
    print("split node recursive")
    """Recursively splits a BSP node. Returns True if split occurred."""
    print(
        "depth: ", depth, "max_bsp_depth", MAX_BSP_DEPTH, "node.is_leaf: ", node.is_leaf
    )
    if depth >= MAX_BSP_DEPTH:  # or node.is_leaf:
        return False  # Don't split further

    # Decide split direction (horizontal or vertical)
    split_horizontally: bool
    # Split wider side more often
    if (
        node.rect.width > node.rect.height
        and node.rect.width / node.rect.height >= 1.25
    ):
        split_horizontally = False  # Split vertically
    elif (
        node.rect.height > node.rect.width
        and node.rect.height / node.rect.width >= 1.25
    ):
        split_horizontally = True  # Split horizontally
    else:  # Roughly square, split randomly
        split_horizontally = rng.coin_flip()[0] == "heads"  # Access first element

    # Check if node is large enough to split in the chosen direction
    max_size = node.rect.height if split_horizontally else node.rect.width
    print(
        "max size: ",
        max_size,
        "node.rect.height: ",
        node.rect.height,
        "node.rect.width: ",
        node.rect.width,
    )
    if max_size <= MIN_LEAF_SIZE * 2:
        return False  # Too small to split meaningfully

    # Determine split position
    split_margin = MIN_LEAF_SIZE  # Ensure sub-nodes are at least MIN_LEAF_SIZE
    if split_horizontally:
        # Split between y1 + margin and y2 - margin
        split_y = rng.get_int(node.rect.y1 + split_margin, node.rect.y2 - split_margin)
        node.left = BSPNode(Rect(node.rect.x1, node.rect.y1, node.rect.x2, split_y - 1))
        node.right = BSPNode(Rect(node.rect.x1, split_y, node.rect.x2, node.rect.y2))
        print(
            f"Depth {depth}: Split H at y={split_y}. Left: {node.left.rect}, Right: {node.right.rect}"
        )
    else:  # Split vertically
        # Split between x1 + margin and x2 - margin
        split_x = rng.get_int(node.rect.x1 + split_margin, node.rect.x2 - split_margin)
        node.left = BSPNode(Rect(node.rect.x1, node.rect.y1, split_x - 1, node.rect.y2))
        node.right = BSPNode(Rect(split_x, node.rect.y1, node.rect.x2, node.rect.y2))
        print(
            f"Depth {depth}: Split V at x={split_x}. Left: {node.left.rect}, Right: {node.right.rect}"
        )

    # Recursively split children
    split_left = _split_node_recursive(node.left, rng, depth + 1)
    split_right = _split_node_recursive(node.right, rng, depth + 1)

    return True  # This node was split


def _create_rooms_in_leaves(root_node: BSPNode, rng: GameRNG):
    print("create rooms in leaves")
    """Creates rooms within the leaf nodes of the BSP tree."""
    for leaf in root_node.get_leaves():
        # Add padding / margin inside the leaf node for the room
        max_w = int(leaf.rect.width * ROOM_MAX_SIZE_RATIO)
        max_h = int(leaf.rect.height * ROOM_MAX_SIZE_RATIO)
        room_w = rng.get_int(ROOM_MIN_SIZE, max(ROOM_MIN_SIZE, max_w))
        room_h = rng.get_int(ROOM_MIN_SIZE, max(ROOM_MIN_SIZE, max_h))

        # Place room randomly within the leaf node boundaries, respecting its own size
        # Ensure room fits within the leaf node
        room_x1 = rng.get_int(
            leaf.rect.x1, max(leaf.rect.x1, leaf.rect.x2 - room_w + 1)
        )
        room_y1 = rng.get_int(
            leaf.rect.y1, max(leaf.rect.y1, leaf.rect.y2 - room_h + 1)
        )
        room_x2 = room_x1 + room_w - 1
        room_y2 = room_y1 + room_h - 1

        # Clamp room coords just in case calculation pushed them out slightly
        room_x1 = max(leaf.rect.x1, room_x1)
        room_y1 = max(leaf.rect.y1, room_y1)
        room_x2 = min(leaf.rect.x2, room_x2)
        room_y2 = min(leaf.rect.y2, room_y2)

        # Ensure room is still valid size after potential clamping
        if (
            room_x2 >= room_x1 + ROOM_MIN_SIZE - 1
            and room_y2 >= room_y1 + ROOM_MIN_SIZE - 1
        ):
            leaf.room = Rect(room_x1, room_y1, room_x2, room_y2)
            # print(f"  Created room: {leaf.room} inside leaf {leaf.rect}")
        else:
            # print(f"  Skipped room creation in leaf {leaf.rect} (too small after padding/random placement)")
            pass


def _carve_tunnel(
    x1: int, y1: int, x2: int, y2: int, game_map: GameMap, rng: GameRNG
) -> List[Rect]:
    """Carves an L-shaped tunnel between two points. Returns list of Rects carved."""
    print("carve tunnel")
    carved_rects = []
    if rng.coin_flip()[0] == "heads":  # Horizontal first, then vertical
        cx1, cy1, cx2, cy2 = min(x1, x2), y1, max(x1, x2), y1
        h_tunnel = Rect(cx1, cy1, cx2, cy2)
        h_tunnel.carve(game_map)
        carved_rects.append(h_tunnel)

        vx1, vy1, vx2, vy2 = x2, min(y1, y2), x2, max(y1, y2)
        v_tunnel = Rect(vx1, vy1, vx2, vy2)
        v_tunnel.carve(game_map)
        carved_rects.append(v_tunnel)
    else:  # Vertical first, then horizontal
        vx1, vy1, vx2, vy2 = x1, min(y1, y2), x1, max(y1, y2)
        v_tunnel = Rect(vx1, vy1, vx2, vy2)
        v_tunnel.carve(game_map)
        carved_rects.append(v_tunnel)

        cx1, cy1, cx2, cy2 = min(x1, x2), y2, max(x1, x2), y2
        h_tunnel = Rect(cx1, cy1, cx2, cy2)
        h_tunnel.carve(game_map)
        carved_rects.append(h_tunnel)
    return carved_rects


def _connect_rooms(node: BSPNode, game_map: GameMap, rng: GameRNG):
    print("connect rooms")
    """Recursively connects rooms in sibling nodes."""
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
        # print(f"  Connecting room {left_room.center} to {right_room.center} via ({lx},{ly}) and ({rx},{ry})")
        node.corridors = _carve_tunnel(lx, ly, rx, ry, game_map, rng)


def generate_dungeon(
    game_map: GameMap, map_width: int, map_height: int, seed: Union[int, None] = None
) -> Tuple[int, int]:
    """
    Generates a dungeon layout using BSP trees.

    Args:
        game_map: The GameMap object to modify.
        map_width: Width of the map.
        map_height: Height of the map.
        seed: Optional seed for the random number generator.

    Returns:
        A tuple (player_start_x, player_start_y) for the player's initial position.
    """
    print("Generating dungeon layout using BSP...")
    rng = GameRNG(seed=seed)  # Use the dedicated RNG

    # 1. Initialize map with walls
    # Assume GameMap constructor already does this, or uncomment:
    # game_map.tiles[:] = TILE_ID_WALL

    # 2. Create root BSP node covering the whole map (with a 1-tile border)
    root_node = BSPNode(Rect(1, 1, map_width - 2, map_height - 2))

    # 3. Recursively split the map
    print("  Splitting BSP tree...")
    _split_node_recursive(root_node, rng, 0)

    # 4. Create rooms in the leaf nodes
    print("  Creating rooms...")
    _create_rooms_in_leaves(root_node, rng)

    # 5. Carve rooms onto the map
    all_rooms: List[Rect] = []
    for leaf in root_node.get_leaves():
        if leaf.room:
            leaf.room.carve(game_map)
            all_rooms.append(leaf.room)

    if not all_rooms:
        raise RuntimeError("BSP generation failed to create any rooms!")
    print("all rooms", all_rooms)
    # 6. Connect rooms
    print("  Connecting rooms...")
    _connect_rooms(root_node, game_map, rng)

    # 7. Determine player start position (e.g., center of the first room)
    first_room = all_rooms[0]
    player_start_x, player_start_y = first_room.center

    # 8. Update map transparency (crucial for FOV)
    print("  Updating transparency map...")
    game_map.update_tile_transparency()

    print(
        f"Dungeon generation complete. Player start: ({player_start_x}, {player_start_y})"
    )
    return player_start_x, player_start_y

"""Adapter for community simulation AI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import structlog

from game.systems import movement_system
from game.systems.pathfinding.flowfield import FlowFieldPathfinder
from game.world.game_map import TILE_TYPES

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()


def _ensure_pathfinder(game_state: 'GameState') -> FlowFieldPathfinder:
    """Return a cached FlowFieldPathfinder tied to the current map."""

    game_map = game_state.game_map
    tiles_hash = hash(game_map.tiles.tobytes())
    pf = getattr(game_state, "_pathfinder", None)
    if pf is None or getattr(game_state, "_pf_tiles_hash", None) != tiles_hash:
        walkable_ids = [tid for tid, t in TILE_TYPES.items() if t.walkable]
        passable = np.isin(game_map.tiles, walkable_ids)
        terrain_cost = np.ones(passable.shape, dtype=np.float32)
        pf = FlowFieldPathfinder(
            passable,
            terrain_cost,
            game_map.height_map,
            max_traversable_step=1,
        )
        game_state._pathfinder = pf
        game_state._pf_tiles_hash = tiles_hash
    return pf


def take_turn(
    entity_row,
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray', 'np.ndarray', 'np.ndarray'],
    **kwargs,
) -> None:
    """Execute one turn for an entity using the community AI system.

    This placeholder implementation mirrors the GOAP adapter for now but is
    kept separate so more sophisticated social behaviours can be developed
    later.
    """
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]
    noise_map, scent_map, los_map = perception

    player_pos = game_state.player_position
    move = None
    dx = dy = 0
    if player_pos is not None:
        pathfinder = _ensure_pathfinder(game_state)
        pathfinder.compute_field([(player_pos.y, player_pos.x)])
        pdx, pdy = pathfinder.get_flow_vector(y, x)
        nx, ny = x + pdx, y + pdy
        if not (0 <= nx < game_state.map_width and 0 <= ny < game_state.map_height):
            pdx = 0 if player_pos.x == x else (-1 if player_pos.x < x else 1)
            pdy = 0 if player_pos.y == y else (-1 if player_pos.y < y else 1)
        move = (pdx, pdy)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    current_noise = noise_map[y, x]
    best_noise = current_noise
    for ndx, ndy in directions:
        nx, ny = x + ndx, y + ndy
        if 0 <= nx < noise_map.shape[1] and 0 <= ny < noise_map.shape[0]:
            if noise_map[ny, nx] > best_noise:
                best_noise = noise_map[ny, nx]
                move = (ndx, ndy)

    if move is None:
        current_scent = scent_map[y, x]
        best_scent = current_scent
        for ndx, ndy in directions:
            nx, ny = x + ndx, y + ndy
            if 0 <= nx < scent_map.shape[1] and 0 <= ny < scent_map.shape[0]:
                if scent_map[ny, nx] > best_scent:
                    best_scent = scent_map[ny, nx]
                    move = (ndx, ndy)

    if move is None:
        if hasattr(rng, "randint"):
            idx = rng.randint(0, len(directions) - 1)
            move = directions[idx]
        else:
            import random

            move = random.choice(directions)

    dx, dy = move
    moved = movement_system.try_move(entity_id, dx, dy, game_state)

    log.debug(
        "Community AI entity processed",
        entity_id=entity_id,
        pos=(x, y),
        noise=int(noise_map[y, x]) if noise_map.size else None,
        scent=int(scent_map[y, x]) if scent_map.size else None,
        visible=bool(los_map[y, x]) if los_map.size else None,
        dx=dx,
        dy=dy,
        moved=moved,
    )

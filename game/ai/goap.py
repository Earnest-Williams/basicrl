"""Adapter for Goal-Oriented Action Planning (GOAP) based AI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import structlog

from game.systems import movement_system
from game.systems.pathfinding.flowfield import FlowFieldPathfinder
from game.world.game_map import TILE_TYPES

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np<<<<<<< codex/implement-flow-field-guided-movement-for-ai
    from game.game_state import GameState
    from game_rng import GameRNG
    from polars.type_aliases import IntoExpr

log = structlog.get_logger()


def _ensure_pathfinder(game_state: 'GameState') -> FlowFieldPathfinder:
    """Return a cached FlowFieldPathfinder for the current map.

    The pathfinder is (re)created if one does not exist yet or if the map
    tiles have changed. This ensures the flow field reflects current terrain
    whenever movement or targets change.
    """

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
    entity_row,  # row from polars DataFrame
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray', 'np.ndarray', 'np.ndarray'],
) -> None:
    """Execute one turn for an entity using the GOAP AI system.

    The implementation is intentionally lightweight. It demonstrates how
    perception data (noise, scent, line-of-sight) can be supplied to the
    planner, even though this stub simply performs a random move.
    """
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]
    noise_map, scent_map, los_map = perception

    # Determine movement using flow-field pathfinding towards the player.
    player_pos = game_state.player_position
    dx = dy = 0
    if player_pos is not None:
        pathfinder = _ensure_pathfinder(game_state)
        # Compute field towards the player's location (y, x order)
        pathfinder.compute_field([(player_pos.y, player_pos.x)])
        dx, dy = pathfinder.get_flow_vector(y, x)


    # React to noise first
    current_noise = noise_map[y, x]
    best_noise = current_noise
    move = None
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < noise_map.shape[1] and 0 <= ny < noise_map.shape[0]:
            if noise_map[ny, nx] > best_noise:
                best_noise = noise_map[ny, nx]
                move = (dx, dy)


    # If no noisy direction, follow scent
    if move is None:
        current_scent = scent_map[y, x]
        best_scent = current_scent
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < scent_map.shape[1] and 0 <= ny < scent_map.shape[0]:
                if scent_map[ny, nx] > best_scent:
                    best_scent = scent_map[ny, nx]
                    move = (dx, dy)

    # Default to random movement
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
        "GOAP AI entity processed",
        entity_id=entity_id,
        pos=(x, y),
        noise=int(noise_map[y, x]) if noise_map.size else None,
        scent=int(scent_map[y, x]) if scent_map.size else None,
        visible=bool(los_map[y, x]) if los_map.size else None,
        dx=dx,
        dy=dy,
        moved=moved,
    )

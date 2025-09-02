"""Adapter for Goal-Oriented Action Planning (GOAP) based AI."""

from __future__ import annotations

from functools import partial
from typing import Callable, List, TYPE_CHECKING, Tuple

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


# Cardinal directions used for simple movement heuristics when planning
# cannot determine a better action.

# Possible movement directions (x, y offsets)

directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]


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


def _action_move_attack(
    entity_row,
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray', 'np.ndarray', 'np.ndarray'],
) -> bool:
    """Basic behaviour: move toward the player or wander."""
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]
    noise_map, scent_map, los_map = perception

    player_pos = game_state.player_position
    move = None
    if player_pos is not None:
        pathfinder = _ensure_pathfinder(game_state)
        pathfinder.compute_field([(player_pos.y, player_pos.x)])
        pdx, pdy = pathfinder.get_flow_vector(y, x)
        nx, ny = x + pdx, y + pdy
        if not (0 <= nx < game_state.map_width and 0 <= ny < game_state.map_height):
            pdx = 0 if player_pos.x == x else (-1 if player_pos.x < x else 1)
            pdy = 0 if player_pos.y == y else (-1 if player_pos.y < y else 1)
        move = (pdx, pdy)

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
        else:  # pragma: no cover - only used in interactive mode
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
    return True


def _action_seek_cover(
    entity_row,
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray', 'np.ndarray', 'np.ndarray'],
) -> bool:
    """Intermediate behaviour: attempt to move to a tile out of sight."""
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]
    _, _, los_map = perception
    if not los_map[y, x]:
        return False
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < los_map.shape[1] and 0 <= ny < los_map.shape[0]:
            if not los_map[ny, nx]:
                movement_system.try_move(entity_id, dx, dy, game_state)
                return True
    return False


def _action_coordinate(
    entity_row,
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray', 'np.ndarray', 'np.ndarray'],
) -> bool:
    """Advanced behaviour: coordinate with allies (placeholder)."""
    # For now we simply record that coordination was attempted.
    game_state.last_coordination = entity_row["entity_id"]
    return True


ACTION_TIERS: List[List[Callable]] = [
    [_action_move_attack],
    [_action_seek_cover, _action_move_attack],
    [_action_coordinate, _action_seek_cover, _action_move_attack],
]


def take_turn(
    entity_row,
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray', 'np.ndarray', 'np.ndarray'],
    plan_depth: int = 1,
    **kwargs,
) -> None:
    """Execute one turn for an entity using the GOAP AI system.

    Parameters
    ----------
    plan_depth:
        Determines the level of intelligence available to the agent. Higher
        values unlock more sophisticated actions.
    """
    tier = ACTION_TIERS[min(max(plan_depth, 1), len(ACTION_TIERS)) - 1]
    for action in tier:
        if action(entity_row, game_state, rng, perception):
            game_state._last_goap_action = action.__name__
            return


def get_goap_adapter(level: int) -> Callable:
    """Return a GOAP adapter function bound to ``plan_depth`` ``level``."""

    return partial(take_turn, plan_depth=level)

"""Adapter for Goal-Oriented Action Planning (GOAP) based AI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import structlog

from game.systems import movement_system

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG
    from polars.type_aliases import IntoExpr

log = structlog.get_logger()


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

    # Random step in one of the four cardinal directions
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

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

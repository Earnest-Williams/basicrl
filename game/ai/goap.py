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
    if hasattr(rng, "randint"):
        idx = rng.randint(0, len(directions) - 1)
        dx, dy = directions[idx]
    else:  # Fallback to Python's random module
        import random

        dx, dy = random.choice(directions)

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

"""Adapter for machine learning policy driven AI.

This is a stub implementation that mirrors the simple heuristic behaviour
until an actual trained policy is integrated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import structlog

from game.systems import movement_system

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def take_turn(
    entity_row,
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
) -> None:
    """Execute one turn for an entity using an ML policy.

    The current placeholder simply performs a random step to demonstrate the
    adapter interface.
    """
    entity_id = entity_row["entity_id"]

    if hasattr(rng, "randint"):
        dx, dy = _DIRECTIONS[rng.randint(0, len(_DIRECTIONS) - 1)]
    else:  # pragma: no cover
        import random

        dx, dy = random.choice(_DIRECTIONS)

    moved = movement_system.try_move(entity_id, dx, dy, game_state)
    log.debug(
        "ML policy AI entity processed",
        entity_id=entity_id,
        dx=dx,
        dy=dy,
        moved=moved,
    )

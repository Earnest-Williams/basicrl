"""Flying bird AI adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import structlog

from game.systems import movement_system

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()

# Allow diagonal flight directions
_DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]


def _random_direction(rng: "GameRNG") -> Tuple[int, int]:
    if hasattr(rng, "get_int"):
        idx = rng.get_int(0, len(_DIRECTIONS) - 1)
    elif hasattr(rng, "randint"):
        idx = rng.randint(0, len(_DIRECTIONS) - 1)
    else:  # pragma: no cover - fallback
        import random

        idx = random.randrange(len(_DIRECTIONS))
    return _DIRECTIONS[idx]


def take_turn(
    entity_row,
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
    **kwargs,
) -> None:
    """Execute one turn for a flying bird.

    Birds move quickly, attempting to traverse two tiles in the same
    direction each turn to simulate flight.
    """
    entity_id = entity_row["entity_id"]

    dx, dy = _random_direction(rng)
    moved = movement_system.try_move(entity_id, dx, dy, game_state)
    if moved:
        movement_system.try_move(entity_id, dx, dy, game_state)
    log.debug(
        "Bird AI entity processed",
        entity_id=entity_id,
        dx=dx,
        dy=dy,
        moved=moved,
    )

"""Pack-hunting mammal AI adapter."""

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


def _random_direction(rng: "GameRNG") -> Tuple[int, int]:
    if hasattr(rng, "get_int"):
        idx = rng.get_int(0, len(_DIRECTIONS) - 1)
    elif hasattr(rng, "randint"):
        idx = rng.randint(0, len(_DIRECTIONS) - 1)
    else:  # pragma: no cover
        import random

        idx = random.randrange(len(_DIRECTIONS))
    return _DIRECTIONS[idx]


def take_turn(
    entity_row,
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
) -> None:
    """Execute one turn for a pack-hunting mammal."""
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]

    player_pos = game_state.player_position
    if player_pos is not None:
        dxp = player_pos.x - x
        dyp = player_pos.y - y
        if abs(dxp) + abs(dyp) <= 5:
            dx = 0 if dxp == 0 else (1 if dxp > 0 else -1)
            dy = 0 if dyp == 0 else (1 if dyp > 0 else -1)
            moved = movement_system.try_move(entity_id, dx, dy, game_state)
            log.debug(
                "Mammal AI entity processed",
                entity_id=entity_id,
                dx=dx,
                dy=dy,
                moved=moved,
            )
            return

    dx, dy = _random_direction(rng)
    moved = movement_system.try_move(entity_id, dx, dy, game_state)
    log.debug(
        "Mammal AI entity processed",
        entity_id=entity_id,
        dx=dx,
        dy=dy,
        moved=moved,
    )

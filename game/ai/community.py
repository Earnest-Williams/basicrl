"""Adapter for community simulation AI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import structlog

from game.systems import movement_system

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()


def take_turn(
    entity_row,
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray', 'np.ndarray', 'np.ndarray'],
) -> None:
    """Execute one turn for an entity using the community AI system.

    This placeholder implementation mirrors the GOAP adapter for now but is
    kept separate so more sophisticated social behaviours can be developed
    later.
    """
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]
    noise_map, scent_map, los_map = perception

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if hasattr(rng, "randint"):
        idx = rng.randint(0, len(directions) - 1)
        dx, dy = directions[idx]
    else:
        import random

        dx, dy = random.choice(directions)

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

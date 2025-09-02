"""Stationary attacking plant AI adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import structlog

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()


def take_turn(
    entity_row,
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
    **kwargs,
) -> None:
    """Execute one turn for a stationary attacking plant."""
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]
    player_pos = game_state.player_position

    acted = False
    if player_pos is not None:
        if abs(player_pos.x - x) <= 1 and abs(player_pos.y - y) <= 1:
            from game.systems import combat_system

            combat_system.handle_melee_attack(
                entity_id, game_state.player_id, game_state
            )
            acted = True

    log.debug("Plant AI entity processed", entity_id=entity_id, acted=acted)

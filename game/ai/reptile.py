"""Ambushing reptile AI adapter."""

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
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
) -> None:
    """Execute one turn for an ambushing reptile."""
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]
    player_pos = game_state.player_position

    if player_pos is None:
        log.debug("Reptile AI entity idle", entity_id=entity_id)
        return

    dxp = player_pos.x - x
    dyp = player_pos.y - y

    if abs(dxp) <= 1 and abs(dyp) <= 1:
        from game.systems import combat_system

        combat_system.handle_melee_attack(entity_id, game_state.player_id, game_state)
        log.debug("Reptile AI entity attacked", entity_id=entity_id)
        return

    if abs(dxp) + abs(dyp) <= 3:
        dx = 0 if dxp == 0 else (1 if dxp > 0 else -1)
        dy = 0 if dyp == 0 else (1 if dyp > 0 else -1)
        moved = movement_system.try_move(entity_id, dx, dy, game_state)
        log.debug(
            "Reptile AI entity moved", entity_id=entity_id, dx=dx, dy=dy, moved=moved
        )
        return

    log.debug("Reptile AI entity waiting", entity_id=entity_id)

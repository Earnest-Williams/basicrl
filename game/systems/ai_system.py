"""Basic AI dispatcher used during turn advancement.

This module provides a simple dispatcher that iterates over all
AI-controlled entities and performs their actions for the current turn.
The implementation is intentionally lightweight and primarily logs
activity for debugging purposes. It is designed to be safe when no AI
entities are present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


from game_rng import GameRNG

import structlog

from game.systems import movement_system
from game.systems.pathfinding.flowfield import FlowFieldPathfinder
from game.world.game_map import TILE_TYPES

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from game.game_state import GameState

log = structlog.get_logger()


def dispatch_ai(game_state: GameState, rng: GameRNG) -> None:
    """Process all AI-controlled entities for the current turn.

    Args:
        game_state: The active :class:`GameState` instance.
        rng: Shared :class:`GameRNG` instance providing randomness.

    The dispatcher logs each entity it processes. If there are no AI
    entities, it exits quietly after logging. This function is a stub
    for future, more complex AI behavior.
    """
    entity_reg = game_state.entity_registry
    entities_df = entity_reg.entities_df

    ai_entities = [
        row
        for row in entities_df.iter_rows(named=True)
        if row.get("is_active", False) and row["entity_id"] != game_state.player_id
    ]

    if not ai_entities:
        log.debug("No AI entities to process this turn")
        return

    log.debug("Processing AI entities", count=len(ai_entities))

    # --- Build FlowFieldPathfinder toward the player ---
    player_pos = game_state.player_position
    if player_pos is None:
        log.error("Player position unavailable for AI pathfinding")
        return

    game_map = game_state.game_map
    px, py = player_pos

    try:
        passable = np.zeros_like(game_map.tiles, dtype=bool)
        for tile_id, tile_type in TILE_TYPES.items():
            passable[game_map.tiles == tile_id] = tile_type.walkable

        terrain_cost = np.ones_like(game_map.tiles, dtype=np.float32)
        pathfinder = FlowFieldPathfinder(
            passable,
            terrain_cost,
            game_map.height_map,
            max_traversable_step=1,
        )
        if not pathfinder.compute_field([(py, px)]):
            log.error("Failed to compute flow field for AI", goal=(px, py))
            return
    except Exception as e:  # pragma: no cover - defensive
        log.error("Exception during AI pathfinding setup", error=str(e), exc_info=True)
        return

    # --- Move each AI entity according to the flow field ---
    for entity in ai_entities:
        entity_id = entity["entity_id"]

        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

        moved = movement_system.try_move(entity_id, dx, dy, game_state)
        log.debug(
            "AI entity processed", entity_id=entity_id, dx=dx, dy=dy, moved=moved
        )

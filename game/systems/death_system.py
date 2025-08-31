# game/systems/death_system.py
"""Utility functions for handling entity death.

Drops carried and equipped items, removes the entity from registries and
other systems, and logs the death if visible to the player.
"""
from __future__ import annotations

import structlog

from game.game_state import GameState

log = structlog.get_logger(__name__)


def handle_entity_death(entity_id: int, gs: GameState, message: str | None = None) -> None:
    """Handles death cleanup for ``entity_id``.

    Parameters
    ----------
    entity_id:
        The entity that died.
    gs:
        The active :class:`~game.game_state.GameState` instance.
    message:
        Optional custom message describing the death.  If ``None`` a generic
        message is used.
    """
    entity_reg = gs.entity_registry
    item_reg = gs.item_registry

    pos = entity_reg.get_position(entity_id)
    name = entity_reg.get_entity_component(entity_id, "name") or f"Entity {entity_id}"

    if pos is not None:
        # Drop inventory items
        inv_df = item_reg.get_entity_inventory(entity_id)
        for row in inv_df.iter_rows(named=True):
            item_reg.move_item(row["item_id"], "ground", x=pos.x, y=pos.y)
        # Drop equipped items
        eq_df = item_reg.get_entity_equipped(entity_id)
        for row in eq_df.iter_rows(named=True):
            item_reg.move_item(row["item_id"], "ground", x=pos.x, y=pos.y)
    else:
        log.warning("Entity position unknown during death cleanup", entity_id=entity_id)

    # Remove the entity from the registry (and thus from AI/FOV systems)
    entity_reg.delete_entity(entity_id)

    # Remove from additional systems if necessary (e.g., lights)
    if hasattr(gs, "light_sources"):
        gs.light_sources = [ls for ls in gs.light_sources if getattr(ls, "owner_id", None) != entity_id]

    # Log death if visible to the player
    if pos is not None and gs.game_map.visible[pos.y, pos.x]:
        if entity_id == gs.player_id:
            death_msg = message or "You die!"
        else:
            death_msg = message or f"The {name} dies!"
        gs.add_message(death_msg, (255, 100, 100))

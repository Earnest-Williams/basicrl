"""Movement helper utilities.

This module exposes small helper functions for moving entities around the
game map.  The helpers centralise common checks such as map bounds and tile
walkability before delegating to the :class:`EntityRegistry` to update the
entity's position component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from game.entities.components import Position

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from game.game_state import GameState


def try_move(entity_id: int, dx: int, dy: int, gs: GameState) -> bool:
    """Attempt to move an entity.

    Parameters
    ----------
    entity_id:
        The identifier of the entity to move.
    dx, dy:
        Delta values to apply to the entity's current position.
    gs:
        The active :class:`~game.game_state.GameState` instance which contains
        the map and entity registry.

    Returns
    -------
    bool
        ``True`` if the movement succeeded, ``False`` otherwise.

    Notes
    -----
    This helper only checks simple bounds and walkability rules.  Systems that
    require additional behaviour (e.g. falling, bump attacks) should perform
    those checks before calling this function.
    """

    entity_reg = gs.entity_registry
    game_map = gs.game_map

    current_pos = entity_reg.get_position(entity_id)
    if current_pos is None:
        return False

    x, y = current_pos
    dest_x, dest_y = x + dx, y + dy

    if not game_map.in_bounds(dest_x, dest_y):
        return False

    if not game_map.is_walkable(dest_x, dest_y):
        return False

    moved = entity_reg.set_position(entity_id, Position(dest_x, dest_y))
    if moved:
        # Moving entities generate noise at their destination
        try:
            gs.noise_events.append((dest_x, dest_y, 10.0))
        except AttributeError:
            pass

        # Play movement sound effect if it's the player
        if entity_id == gs.player_id:
            # Import sound system dynamically to avoid circular imports
            try:
                from game.systems.sound import handle_event

                terrain_type = (
                    gs.game_map.get_tile_type_name(dest_x, dest_y)
                    if hasattr(gs.game_map, "get_tile_type_name")
                    else "floor"
                )
                sound_context = {
                    "entity": "player",
                    "terrain": terrain_type,
                    "position": (dest_x, dest_y),
                }
                handle_event("player_move", sound_context)
            except ImportError:
                pass
    return moved

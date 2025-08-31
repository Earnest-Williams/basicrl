"""Central AI dispatch system.

This module exposes :func:`dispatch_ai` which selects an appropriate AI
adapter for an entity based on its metadata.  Adapters must implement the
``take_turn(entity_row, game_state, rng, perception)`` interface so the game
can mix multiple decision making systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import structlog

from game.ai import get_adapter

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()


def dispatch_ai(
    entity_row,
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
) -> None:
    """Dispatch to the correct AI adapter for ``entity_row``.

    Parameters mirror the adapter interface so tests may monkeypatch this
    function.  The ``ai_type`` is read from the entity metadata with a
    fallback to the ``GameState`` default configuration.
    """

    ai_type = entity_row.get("ai_type") or game_state.ai_config.get("default", "goap")
    adapter = get_adapter(ai_type)
    log.debug(
        "Dispatching AI", ai_type=ai_type, entity_id=entity_row.get("entity_id")
    )
    adapter(entity_row, game_state, rng, perception)


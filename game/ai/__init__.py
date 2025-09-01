"""AI package providing adapters for different decision systems.

This package exposes helper functions that select the appropriate
AI routine for a given entity. The goal is to allow the game to mix
multiple AI implementations (e.g. GOAP for monsters and community
simulation for townsfolk) within the same simulation.
"""

from __future__ import annotations

from typing import Callable, Dict

import structlog

from . import goap, community, simple, ml_policy, strategy

if True:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:  # pragma: no cover - for type checking
        from polars import series
        from game.game_state import GameState
        from game_rng import GameRNG

log = structlog.get_logger()

# Mapping from ai_type string to the adapter function that will
# execute a turn for entities of that type.
ADAPTERS: Dict[str, Callable] = {
    "goap": goap.take_turn,
    "community": community.take_turn,
    "simple": simple.take_turn,
    "ml_policy": ml_policy.take_turn,
    "strategy": strategy.dispatch_strategy,
}


def get_adapter(ai_type: str) -> Callable:
    """Return the adapter function for the given ai_type.

    Unknown types default to the GOAP adapter so that new entity
    definitions do not cause crashes if misconfigured.
    """
    adapter = ADAPTERS.get(ai_type, goap.take_turn)
    if adapter is goap.take_turn and ai_type not in ADAPTERS:
        log.debug("Unknown ai_type, defaulting to GOAP", ai_type=ai_type)
    return adapter

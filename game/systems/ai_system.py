"""Central AI dispatch system.

This module exposes :func:`dispatch_ai` which selects an appropriate AI
adapter for an entity based on its metadata.  Adapters must implement the
``take_turn(entity_row, game_state, rng, perception)`` interface so the game
can mix multiple decision making systems.
"""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING, Tuple

import structlog
from multiprocessing.dummy import Pool as ThreadPool

from game.ai import get_adapter, goap

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()


def dispatch_ai(
    entities,
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
    batch_size: int = 4,
) -> None:
    """Execute AI adapters for a batch of entities in parallel."""

    if not isinstance(entities, Iterable) or hasattr(entities, "get"):
        entity_rows = [entities]
    else:
        entity_rows = list(entities)

    def _invoke(row):
        species = row.get("species")
        intelligence = row.get("intelligence")
        species_map = game_state.ai_config.get("species_mapping", {})
        goap_tiers = game_state.ai_config.get("intelligence_tiers", {})

        ai_type = row.get("ai_type")
        plan_depth = None
        if not ai_type:
            if species and species in species_map:
                ai_type = species_map[species]
            elif intelligence is not None:
                ai_type = "goap"
        if not ai_type:
            ai_type = game_state.ai_config.get("default", "goap")

        adapter = get_adapter(ai_type)
        if adapter is goap.take_turn:
            if intelligence is not None:
                plan_depth = goap_tiers.get(intelligence, intelligence)
            if plan_depth is None:
                plan_depth = 1
            adapter = goap.get_goap_adapter(plan_depth)
        log.debug(
            "Dispatching AI",
            ai_type=ai_type,
            species=species,
            intelligence=intelligence,
            entity_id=row.get("entity_id"),
            plan_depth=plan_depth,
        )
        adapter(
            row,
            game_state,
            rng,
            perception,
            species=species,
            intelligence=intelligence,
        )

    for i in range(0, len(entity_rows), batch_size):
        batch = entity_rows[i : i + batch_size]
        with ThreadPool(len(batch)) as pool:
            pool.map(_invoke, batch)

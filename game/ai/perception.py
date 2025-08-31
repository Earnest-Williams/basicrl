"""Lightweight perception helpers for AI modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import structlog

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from game.game_state import GameState

log = structlog.get_logger()


def gather_perception(game_state: 'GameState') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate perception maps used by AI systems.

    The current implementation is intentionally simple. It creates blank
    noise and scent maps and reuses the game's visibility map for line of
    sight. These placeholders illustrate how perception data can be
    collected and passed to AI modules without committing to a particular
    algorithm yet.
    """
    game_map = game_state.game_map
    noise_map = np.zeros_like(game_map.tiles, dtype=np.int32)
    scent_map = np.zeros_like(game_map.tiles, dtype=np.int32)
    los_map = game_map.visible.copy()
    log.debug("Perception maps generated", shape=noise_map.shape)
    return noise_map, scent_map, los_map

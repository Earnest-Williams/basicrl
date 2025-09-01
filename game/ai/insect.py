"""Swarming insect AI adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import polars as pl
import structlog

from game.systems import movement_system

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nearest_ally(entity_id: int, x: int, y: int, gs: "GameState") -> Tuple[int, int] | None:
    """Return direction towards nearest allied insect if any."""
    df = gs.entity_registry.entities_df
    others = df.filter(
        (pl.col("ai_type") == "insect") & (pl.col("entity_id") != entity_id)
    )
    if others.height == 0:
        return None
    others = others.with_columns(
        ((pl.col("x") - x).abs() + (pl.col("y") - y).abs()).alias("dist")
    ).sort("dist")
    row = others.row(0, named=True)
    dx = 0 if row["x"] == x else (1 if row["x"] > x else -1)
    dy = 0 if row["y"] == y else (1 if row["y"] > y else -1)
    return dx, dy


def _random_direction(rng: "GameRNG") -> Tuple[int, int]:
    if hasattr(rng, "get_int"):
        idx = rng.get_int(0, len(_DIRECTIONS) - 1)
    elif hasattr(rng, "randint"):
        idx = rng.randint(0, len(_DIRECTIONS) - 1)
    else:  # pragma: no cover - fallback for unexpected RNGs
        import random

        idx = random.randrange(len(_DIRECTIONS))
    return _DIRECTIONS[idx]


def take_turn(
    entity_row,
    game_state: "GameState",
    rng: "GameRNG",
    perception: Tuple["np.ndarray", "np.ndarray", "np.ndarray"],
) -> None:
    """Execute one turn for a swarming insect."""
    entity_id = entity_row["entity_id"]
    x, y = entity_row["x"], entity_row["y"]

    dir_to_ally = _nearest_ally(entity_id, x, y, game_state)
    if dir_to_ally is not None:
        dx, dy = dir_to_ally
    else:
        dx, dy = _random_direction(rng)

    moved = movement_system.try_move(entity_id, dx, dy, game_state)
    log.debug(
        "Insect AI entity processed",
        entity_id=entity_id,
        dx=dx,
        dy=dy,
        moved=moved,
    )

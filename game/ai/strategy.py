"""State-driven AI strategies.

Provides a small state machine that dispatches to behaviour functions
based on an entity's ``StrategyState``.  Strategies rely on perception
helpers to select targets and can be executed in parallel by the AI
scheduler.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Tuple

import structlog

from .perception import find_visible_enemies
from game.systems import movement_system

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np
    from polars import series
    from game.game_state import GameState
    from game_rng import GameRNG

log = structlog.get_logger()


class StrategyState(Enum):
    """Simple behaviour state machine for AI entities."""

    HOME = auto()
    CHARGE = auto()
    SMART_KOBOLD = auto()
    FLEE = auto()


def _step_towards(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    """Return a single step from ``src`` towards ``dst``."""
    sx, sy = src
    dx = 0 if dst[0] == sx else (1 if dst[0] > sx else -1)
    dy = 0 if dst[1] == sy else (1 if dst[1] > sy else -1)
    return dx, dy


def _move(entity_row: 'series', dx: int, dy: int, game_state: 'GameState') -> None:
    movement_system.try_move(entity_row['entity_id'], dx, dy, game_state)


def charge_behavior(entity_row: 'series', game_state: 'GameState', perception: Tuple['np.ndarray','np.ndarray','np.ndarray']) -> None:
    noise, scent, los = perception
    enemies = find_visible_enemies(entity_row, game_state, los)
    if not enemies:
        return
    target = enemies[0]
    dx, dy = _step_towards((entity_row.get('x'), entity_row.get('y')), (target.get('x'), target.get('y')))
    _move(entity_row, dx, dy, game_state)


def home_behavior(entity_row: 'series', game_state: 'GameState') -> None:
    home_x = entity_row.get('home_x', 0)
    home_y = entity_row.get('home_y', 0)
    dx, dy = _step_towards((entity_row.get('x'), entity_row.get('y')), (home_x, home_y))
    _move(entity_row, dx, dy, game_state)


def flee_behavior(entity_row: 'series', game_state: 'GameState', perception: Tuple['np.ndarray','np.ndarray','np.ndarray']) -> None:
    noise, scent, los = perception
    enemies = find_visible_enemies(entity_row, game_state, los)
    if not enemies:
        return
    target = enemies[0]
    sx, sy = entity_row.get('x'), entity_row.get('y')
    tx, ty = target.get('x'), target.get('y')
    dx = 0 if tx == sx else (-1 if tx > sx else 1)
    dy = 0 if ty == sy else (-1 if ty > sy else 1)
    _move(entity_row, dx, dy, game_state)


def smart_kobold_behavior(entity_row: 'series', game_state: 'GameState', perception: Tuple['np.ndarray','np.ndarray','np.ndarray']) -> None:
    hp = entity_row.get('hp', 1)
    max_hp = entity_row.get('max_hp', hp)
    if hp / max_hp < 0.3:
        flee_behavior(entity_row, game_state, perception)
    else:
        charge_behavior(entity_row, game_state, perception)


def dispatch_strategy(
    entity_row: 'series',
    game_state: 'GameState',
    rng: 'GameRNG',
    perception: Tuple['np.ndarray','np.ndarray','np.ndarray'],
) -> None:
    """Dispatch behaviour based on the entity's ``strategy_state``."""
    state = entity_row.get('strategy_state')
    if state is None:
        return
    try:
        strat = StrategyState[state.upper()]
    except KeyError:
        log.debug('Unknown strategy state', state=state)
        return
    if strat is StrategyState.CHARGE:
        charge_behavior(entity_row, game_state, perception)
    elif strat is StrategyState.HOME:
        home_behavior(entity_row, game_state)
    elif strat is StrategyState.SMART_KOBOLD:
        smart_kobold_behavior(entity_row, game_state, perception)
    elif strat is StrategyState.FLEE:
        flee_behavior(entity_row, game_state, perception)


__all__ = ['StrategyState', 'dispatch_strategy']

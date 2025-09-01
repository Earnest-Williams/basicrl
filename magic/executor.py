"""Utilities for executing magical work.

This module introduces a simple execution loop for "work" – the game's
term for a magical action.  ``execute_work`` validates the work, applies
an effect handler mapped from the ``(Art, Substance)`` pair and manages
friction accumulation per heartbeat.

The broader magic system has not been implemented yet, so most of the
checks here are placeholders.  They log what would normally happen,
allowing other parts of the engine to hook into the process later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Tuple

import structlog

from game.game_state import GameState

log = structlog.get_logger()


class Art(str, Enum):
    """Placeholder enumeration for magical arts."""

    FIRE = "fire"
    ICE = "ice"
    VOID = "void"


class Substance(str, Enum):
    """Placeholder enumeration for magical substances."""

    FLAME = "flame"
    WATER = "water"
    AETHER = "aether"


@dataclass(frozen=True)
class Work:
    """Represents a single magical working."""

    art: Art
    substance: Substance
    seals: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    vents: list[str] = field(default_factory=list)
    friction: float = 0.0
    quiver_threshold: float = 10.0
    warp_threshold: float = 20.0
    shiver_threshold: float = 30.0
    backlash_threshold: float = 40.0


EffectHandler = Callable[[Work, GameState], None]
EFFECT_HANDLERS: Dict[Tuple[Art, Substance], EffectHandler] = {}


def register_handler(art: Art, substance: Substance, handler: EffectHandler) -> None:
    """Register a handler for a given Art/Substance pair."""
    EFFECT_HANDLERS[(art, substance)] = handler
    log.debug("Registered effect handler", art=art, substance=substance)


def execute_work(work: Work, context: GameState) -> None:
    """Execute a piece of work within the game context.

    The function performs three high level steps:

    1. Validate the work (verify seals, fonts and vents).
    2. Apply the effect associated with the work's art and substance.
    3. Track and resolve friction thresholds.
    """
    log.debug("Executing work", art=work.art, substance=work.substance)

    if not _verify_seals(work, context):
        log.warning("Seal verification failed", work=work)
        return
    if not _verify_fonts(work, context):
        log.warning("Font verification failed", work=work)
        return
    if not _verify_vents(work, context):
        log.warning("Vent verification failed", work=work)
        return

    handler = EFFECT_HANDLERS.get((work.art, work.substance))
    if handler:
        log.debug("Applying effect handler", art=work.art, substance=work.substance)
        handler(work, context)
    else:
        log.info(
            "No effect handler registered for work",
            art=work.art,
            substance=work.substance,
        )

    _update_friction(work, context)


def _verify_seals(work: Work, context: GameState) -> bool:
    """Placeholder check for seals."""
    valid = bool(work.seals)
    log.debug("Verifying seals", count=len(work.seals), result=valid)
    return valid


def _verify_fonts(work: Work, context: GameState) -> bool:
    """Placeholder check for fonts."""
    valid = bool(work.fonts)
    log.debug("Verifying fonts", count=len(work.fonts), result=valid)
    return valid


def _verify_vents(work: Work, context: GameState) -> bool:
    """Placeholder check for vents."""
    valid = bool(work.vents)
    log.debug("Verifying vents", count=len(work.vents), result=valid)
    return valid


def _update_friction(work: Work, context: GameState) -> None:
    """Increase friction and resolve any thresholds reached."""
    work.friction += 1.0
    log.debug("Friction updated", value=work.friction)

    if work.friction >= work.backlash_threshold:
        _handle_backlash(work, context)
        work.friction = 0.0
    elif work.friction >= work.shiver_threshold:
        _handle_shiver(work, context)
    elif work.friction >= work.warp_threshold:
        _handle_warp(work, context)
    elif work.friction >= work.quiver_threshold:
        _handle_quiver(work, context)


def _handle_quiver(work: Work, context: GameState) -> None:
    log.info("Quiver threshold reached", friction=work.friction)


def _handle_warp(work: Work, context: GameState) -> None:
    log.info("Warp threshold reached", friction=work.friction)


def _handle_shiver(work: Work, context: GameState) -> None:
    log.info("Shiver threshold reached", friction=work.friction)


def _handle_backlash(work: Work, context: GameState) -> None:
    log.warning("Backlash threshold reached", friction=work.friction)

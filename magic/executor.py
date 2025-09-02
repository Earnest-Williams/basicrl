"""Utilities for executing magical work.

This module introduces a simple execution loop for "work" – the game's
term for a magical action. ``execute_work`` validates the work, checks any
active Wards/Counterseals, applies an effect handler mapped from the
``(Art, Substance)`` pair (or an optional callable on the Work itself),
and manages friction accumulation per heartbeat.

The broader magic system has not been implemented yet, so most of the
checks here are placeholders. They log what would normally happen,
allowing other parts of the engine to hook into the process later.

Execution helpers for Works.

The :func:`execute_work` function is a thin wrapper around calling a Work's
effect. Before execution it checks any active :class:`Ward` instances and
optional :class:`Counterseal`s to ensure the Work is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Iterable, Set, TYPE_CHECKING, List

import structlog

from .models import Art, Substance
from .wards import Counterseal, Ward, is_blocked

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from game.game_state import GameState

log = structlog.get_logger()


EffectHandler = Callable[["Work", "GameState"], None]
EFFECT_HANDLERS: Dict[Tuple[Art, Substance], EffectHandler] = {}

# ---------------------------------------------------------------------------
# Friction event callbacks
FrictionCallback = Callable[["Work", "GameState"], None]
FRICTION_CALLBACKS: Dict[str, List[FrictionCallback]] = {
    "quiver": [],
    "warp": [],
    "shiver": [],
    "backlash": [],
}


def register_friction_callback(event: str, callback: FrictionCallback) -> None:
    """Register a callback for a friction ``event``.

    ``event`` must be one of ``"quiver"``, ``"warp"``, ``"shiver"`` or
    ``"backlash"``. Callbacks receive the triggering :class:`Work` and
    :class:`GameState`.
    """

    FRICTION_CALLBACKS.setdefault(event, []).append(callback)
    log.debug("Registered friction callback",
              event_name=event, callback=callback)


def _dispatch_friction_event(event: str, work: "Work", context: "GameState") -> None:
    """Invoke callbacks registered for ``event``."""
    for cb in FRICTION_CALLBACKS.get(event, []):
        try:
            cb(work, context)
        except Exception as err:  # pragma: no cover - defensive
            log.error("Friction callback failed",
                      event_name=event, error=str(err))


def register_handler(art: Art, substance: Substance, handler: EffectHandler) -> None:
    """Register a handler for a given Art/Substance pair."""
    EFFECT_HANDLERS[(art, substance)] = handler
    log.debug("Registered effect handler", art=art, substance=substance)


@dataclass
class Work:
    """Represents a single magical working.

    Fields support both the effect-registry flow (Art/Substance pair) and the
    legacy callable flow (``func``). The ``substances`` set is retained for
    compatibility with ward checks; it is auto-populated from ``substance``.
    """
    art: Art
    substance: Substance

    # Compatibility with ward logic from master branch: wards may inspect this.
    substances: Set[Substance] = field(default_factory=set)

    # Optional direct callable (master branch behavior) when no handler exists.
    func: Callable[[], object] | None = None

    # Execution constraints (placeholder validations)
    seals: list[str] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    vents: list[str] = field(default_factory=list)

    # Friction model
    friction: float = 0.0
    quiver_threshold: float = 10.0
    warp_threshold: float = 20.0
    shiver_threshold: float = 30.0
    backlash_threshold: float = 40.0

    def __post_init__(self) -> None:
        # Ensure the compatibility set always contains the primary substance.
        if self.substance not in self.substances:
            self.substances.add(self.substance)

    def perform(self) -> object:
        """Execute the underlying callable if present."""
        if self.func is None:
            return None
        return self.func()


def execute_work(
    work: Work,
    context: GameState,
    *,
    wards: Iterable[Ward] = (),
    counterseals: Iterable[Counterseal] = (),
) -> bool:
    """Attempt to execute ``work`` within the game context.

    Order:
      1) Ward/counterseal gate.
      2) Placeholder validations (seals/fonts/vents).
      3) Effect application via registered handler or ``work.func`` fallback.
      4) Friction update and threshold handling.

    Returns:
        True if execution occurred (handler or callable ran), False otherwise.
    """
    log.debug("Executing work", art=work.art, substance=work.substance)

    # 1) Ward gate
    if is_blocked(work, wards, counterseals):
        log.info("Work blocked by ward", art=work.art,
                 substances=list(work.substances))
        return False

    # 2) Placeholder validations
    if not _verify_seals(work, context):
        log.warning("Seal verification failed", work=work)
        return False
    if not _verify_fonts(work, context):
        log.warning("Font verification failed", work=work)
        return False
    if not _verify_vents(work, context):
        log.warning("Vent verification failed", work=work)
        return False

    # 3) Apply effect
    handler = EFFECT_HANDLERS.get((work.art, work.substance))
    ran = False
    if handler:
        log.debug("Applying effect handler",
                  art=work.art, substance=work.substance)
        handler(work, context)
        ran = True
    elif work.func is not None:
        log.debug("No handler; invoking work callable")
        work.perform()
        ran = True
    else:
        log.info(
            "No effect handler or callable for work",
            art=work.art,
            substance=work.substance,
        )

    # 4) Friction processing (only if something executed)
    if ran:
        _update_friction(work, context)

    return ran


def _verify_seals(work: Work, context: GameState) -> bool:
    """Check that all required seal tags exist on the acting entity.

    The :class:`GameState` must expose a ``has_seal_tag`` method and a
    ``player_id`` attribute referencing the acting entity. Each seal listed on
    the work is verified against the game state's registry.
    """
    for seal in work.seals:
        if not context.has_seal_tag(context.player_id, seal):
            log.debug("Verifying seals", missing=seal, result=False)
            return False
    log.debug("Verifying seals", count=len(work.seals), result=True)
    return True


def _verify_fonts(work: Work, context: GameState) -> bool:
    """Ensure all fonts required by ``work`` are available to the actor.

    ``GameState`` must implement ``has_font_source`` mirroring
    :func:`GameState.has_font_source`.
    """
    for font in work.fonts:
        if not context.has_font_source(context.player_id, font):
            log.debug("Verifying fonts", missing=font, result=False)
            return False
    log.debug("Verifying fonts", count=len(work.fonts), result=True)
    return True


def _verify_vents(work: Work, context: GameState) -> bool:
    """Verify that all vent targets are present on the acting entity."""
    for vent in work.vents:
        if not context.has_vent_target(context.player_id, vent):
            log.debug("Verifying vents", missing=vent, result=False)
            return False
    log.debug("Verifying vents", count=len(work.vents), result=True)
    return True


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


def _apply_status(context: GameState, status_id: str, duration: int) -> None:
    """Apply a status effect to the acting entity if components exist."""
    try:
        statuses = context.entity_registry.get_entity_component(
            context.player_id, "status_effects"
        ) or []
        updated = list(statuses)
        updated.append({"id": status_id, "duration": duration})
        context.entity_registry.set_entity_component(
            context.player_id, "status_effects", updated
        )
    except Exception:  # pragma: no cover - defensive
        log.debug("Unable to apply status effect", status=status_id)


def _drain_player_fuel(context: GameState, amount: int) -> None:
    """Reduce the player's fuel resource if present."""
    if hasattr(context, "player_fuel"):
        context.player_fuel = max(0, context.player_fuel - amount)


def _drain_player_hp(context: GameState, amount: int) -> None:
    """Reduce the player's hit points via the entity registry."""
    try:
        hp = context.entity_registry.get_entity_component(
            context.player_id, "hp")
        if hp is not None:
            context.entity_registry.set_entity_component(
                context.player_id, "hp", max(0, hp - amount)
            )
    except Exception:  # pragma: no cover - defensive
        log.debug("Unable to drain HP", amount=amount)


def _handle_quiver(work: Work, context: GameState) -> None:
    log.info("Quiver threshold reached", friction=work.friction)
    _drain_player_fuel(context, 1)
    _apply_status(context, "quivering", 1)
    _dispatch_friction_event("quiver", work, context)


def _handle_warp(work: Work, context: GameState) -> None:
    log.info("Warp threshold reached", friction=work.friction)
    _drain_player_fuel(context, 2)
    _apply_status(context, "warped", 1)
    _dispatch_friction_event("warp", work, context)


def _handle_shiver(work: Work, context: GameState) -> None:
    log.info("Shiver threshold reached", friction=work.friction)
    _drain_player_fuel(context, 3)
    _apply_status(context, "shivering", 1)
    _dispatch_friction_event("shiver", work, context)


def _handle_backlash(work: Work, context: GameState) -> None:
    log.warning("Backlash threshold reached", friction=work.friction)
    _drain_player_fuel(context, 5)
    _drain_player_hp(context, 1)
    _apply_status(context, "backlash", 1)
    _dispatch_friction_event("backlash", work, context)


__all__ = [
    "Art",
    "Substance",
    "Work",
    "execute_work",
    "register_handler",
    "register_friction_callback",
]

# Import game effect handlers to populate the registry before any work runs
try:  # pragma: no cover - defensive import
    import game.effects  # noqa: F401
except ImportError:  # pragma: no cover - game package may be optional
    log.debug("game.effects package not available; skipping handler registration")

import pytest

from magic.executor import (
    Art,
    Substance,
    Work,
    execute_work,
    register_handler,
)
from magic import executor
from magic.wards import Ward, Counterseal


class DummyGameState:
    """Lightweight stand-in for the real GameState.

    The magic executor expects a ``player_id`` attribute and resource
    lookup methods for seals, fonts, and vents.  This stub provides simple
    set-based implementations suitable for unit tests.
    """

    def __init__(self, *, seals=("s",), fonts=("f",), vents=("v",)):
        self.player_id = 0
        self._seals = set(seals)
        self._fonts = set(fonts)
        self._vents = set(vents)

    def has_seal_tag(self, entity_id: int, tag: str) -> bool:  # pragma: no cover - trivial
        return tag in self._seals

    def has_font_source(self, entity_id: int, source: str) -> bool:  # pragma: no cover - trivial
        return source in self._fonts

    def has_vent_target(self, entity_id: int, target: str) -> bool:  # pragma: no cover - trivial
        return target in self._vents


def make_basic_work(**kwargs):
    """Helper to create a Work with mandatory validation fields populated."""
    defaults = {
        "art": Art.CREATE,
        "substance": Substance.FIRE,
        "seals": ["s"],
        "fonts": ["f"],
        "vents": ["v"],
        "func": lambda: None,
    }
    defaults.update(kwargs)
    return Work(**defaults)


def test_seal_font_vent_verifications_pass_and_fail():
    """Each validation succeeds when resources exist and fails otherwise."""

    work = make_basic_work(seals=["alpha"], fonts=["beta"], vents=["gamma"])

    # All resources present: should pass
    gs_ok = DummyGameState(seals=("alpha",), fonts=("beta",), vents=("gamma",))
    assert executor._verify_seals(work, gs_ok)
    assert executor._verify_fonts(work, gs_ok)
    assert executor._verify_vents(work, gs_ok)
    assert execute_work(work, gs_ok)

    # Missing each resource should fail the respective check and execution
    gs_missing_seal = DummyGameState(seals=(), fonts=("beta",), vents=("gamma",))
    assert not executor._verify_seals(work, gs_missing_seal)
    assert not execute_work(make_basic_work(seals=["alpha"], fonts=["beta"], vents=["gamma"]), gs_missing_seal)

    gs_missing_font = DummyGameState(seals=("alpha",), fonts=(), vents=("gamma",))
    assert not executor._verify_fonts(work, gs_missing_font)
    assert not execute_work(make_basic_work(seals=["alpha"], fonts=["beta"], vents=["gamma"]), gs_missing_font)

    gs_missing_vent = DummyGameState(seals=("alpha",), fonts=("beta",), vents=())
    assert not executor._verify_vents(work, gs_missing_vent)
    assert not execute_work(make_basic_work(seals=["alpha"], fonts=["beta"], vents=["gamma"]), gs_missing_vent)


def test_execute_work_blocked_by_ward_without_counterseal():
    work = make_basic_work()
    ward = Ward(arts={Art.CREATE})
    counterseal = Counterseal(arts={Art.DESTROY})  # does not match the ward
    result = execute_work(work, DummyGameState(), wards=[ward], counterseals=[counterseal])
    assert result is False


def test_friction_increases_and_triggers_thresholds(monkeypatch):
    calls = []

    def recorder(name):
        def _rec(work, ctx):
            calls.append(name)
        return _rec

    monkeypatch.setattr(executor, "_handle_quiver", recorder("quiver"))
    monkeypatch.setattr(executor, "_handle_warp", recorder("warp"))
    monkeypatch.setattr(executor, "_handle_shiver", recorder("shiver"))
    monkeypatch.setattr(executor, "_handle_backlash", recorder("backlash"))

    work = make_basic_work(
        quiver_threshold=1,
        warp_threshold=2,
        shiver_threshold=3,
        backlash_threshold=4,
    )
    gs = DummyGameState()

    frictions = []
    for _ in range(4):
        execute_work(work, gs)
        frictions.append(work.friction)

    assert frictions == [1.0, 2.0, 3.0, 0.0]
    assert calls == ["quiver", "warp", "shiver", "backlash"]


def test_registered_handlers_are_invoked(monkeypatch):
    called = []
    monkeypatch.setattr(executor, "EFFECT_HANDLERS", {})

    def handler(work, state):
        called.append((work, state))

    register_handler(Art.DESTROY, Substance.WATER, handler)

    work = make_basic_work(art=Art.DESTROY, substance=Substance.WATER, func=None)
    gs = DummyGameState()
    result = execute_work(work, gs)

    assert result is True
    assert called == [(work, gs)]


def test_game_effects_register_existing_handlers(monkeypatch):
    monkeypatch.setattr(executor, "EFFECT_HANDLERS", {})

    import importlib
    import game.effects

    importlib.reload(game.effects)

    assert (Art.CREATE, Substance.WATER) in executor.EFFECT_HANDLERS

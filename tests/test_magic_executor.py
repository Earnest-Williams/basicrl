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
    """Lightweight stand-in for the real GameState."""
    pass


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

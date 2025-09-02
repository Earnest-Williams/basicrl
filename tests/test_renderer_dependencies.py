import importlib
import sys
import types

import pytest


def test_renderer_errors_when_gamestate_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "engine.renderer", raising=False)
    monkeypatch.setitem(sys.modules, "game.game_state", types.ModuleType("game.game_state"))
    monkeypatch.setitem(sys.modules, "basicrl.game", types.ModuleType("basicrl.game"))
    monkeypatch.setitem(
        sys.modules, "basicrl.game.game_state", types.ModuleType("basicrl.game.game_state")
    )
    with pytest.raises(RuntimeError):
        importlib.import_module("engine.renderer")


def test_renderer_errors_when_gamerng_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "engine.renderer", raising=False)
    monkeypatch.setitem(sys.modules, "game_rng", types.ModuleType("game_rng"))
    monkeypatch.setitem(
        sys.modules, "basicrl.game_rng", types.ModuleType("basicrl.game_rng")
    )
    with pytest.raises(RuntimeError):
        importlib.import_module("engine.renderer")

import sys
import types
import pytest

# Provide lightweight stubs for heavy modules to avoid unnecessary imports
sys.modules.setdefault("game.game_state", types.ModuleType("game.game_state")).GameState = object
sys.modules.setdefault("simulation.zone_manager", types.ModuleType("simulation.zone_manager")).ZoneManager = object

# Stub out engine.action_handler and engine.renderer to avoid heavy dependencies
action_handler_stub = types.ModuleType("engine.action_handler")
action_handler_stub.process_player_action = lambda *args, **kwargs: False
sys.modules["engine.action_handler"] = action_handler_stub

renderer_stub = types.ModuleType("engine.renderer")
class RenderConfig:  # minimal placeholder
    pass
class ViewportParams:  # minimal placeholder
    pass
renderer_stub.RenderConfig = RenderConfig
renderer_stub.ViewportParams = ViewportParams
renderer_stub.render_viewport = lambda *args, **kwargs: None
sys.modules["engine.renderer"] = renderer_stub

from engine import main_loop as ml_module
MainLoop = ml_module.MainLoop


class DummyGameState:
    def __init__(self):
        self.messages = []
        self.turns = 0

    def add_message(self, msg, color):
        self.messages.append((msg, color))

    def advance_turn(self):
        self.turns += 1


class DummyWindow:
    def update_frame(self):
        pass


def create_main_loop():
    gs = DummyGameState()
    window = DummyWindow()
    ml = MainLoop(
        game_state=gs,
        window=window,
        vis_enabled_default=False,
        vis_max_diff=1,
        vis_color_high=[0, 0, 0],
        vis_color_mid=[0, 0, 0],
        vis_color_low=[0, 0, 0],
        vis_blend_factor=0.0,
        max_traversable_step=1,
        lighting_ambient=0.0,
        lighting_min_fov=0.0,
        lighting_falloff=1.0,
    )
    return ml, gs


def test_value_error_is_handled(monkeypatch):
    ml, gs = create_main_loop()

    def raise_value_error(action, gs, max_step):
        raise ValueError("bad action")

    monkeypatch.setattr(ml_module.action_handler, "process_player_action", raise_value_error)

    assert ml.handle_action({"type": "test"}) is False
    assert gs.messages


def test_unexpected_exception_propagates(monkeypatch):
    ml, _ = create_main_loop()

    def raise_runtime_error(action, gs, max_step):
        raise RuntimeError("boom")

    monkeypatch.setattr(ml_module.action_handler, "process_player_action", raise_runtime_error)

    with pytest.raises(RuntimeError):
        ml.handle_action({"type": "test"})

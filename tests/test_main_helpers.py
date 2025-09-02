import os
from PySide6.QtWidgets import QApplication

from main import Configs, load_configs, init_game_state, init_window

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_load_configs():
    configs = load_configs()
    assert isinstance(configs, Configs)
    assert "map_width" in configs.main
    assert isinstance(configs.item_templates, dict)


def test_init_game_state():
    configs = load_configs()
    game_state = init_game_state(configs)
    assert game_state.map_width == configs.main.get("map_width", 80)
    assert game_state.map_height == configs.main.get("map_height", 50)


def test_init_window():
    app = QApplication.instance() or QApplication([])
    configs = load_configs()
    game_state = init_game_state(configs)
    window = init_window(configs, game_state)
    assert hasattr(window, "main_loop")
    window.close()
    app.quit()

from game.game_state import GameState
from game.world.game_map import GameMap
from structlog.testing import capture_logs
import sys
import types

# Provide a minimal ai_system module for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module


def create_game_state():
    game_map = GameMap(width=10, height=10)
    game_map.create_test_room()
    gs = GameState(
        existing_map=game_map,
        player_start_pos=(5, 5),
        player_glyph=ord("@"),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        entity_templates={},
        effect_definitions={},
        rng_seed=42,
        memory_fade_config={"enabled": False},
    )
    return gs


def test_update_fov_missing_light_source_logs_error():
    gs = create_game_state()
    # Remove the player's light source to simulate missing entry
    gs.light_sources.pop(gs.player_light_index)

    with capture_logs() as logs:
        gs.update_fov()

    assert any(
        log["event"] == "Failed to update player light source" for log in logs
    )

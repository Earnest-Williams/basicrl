from game.game_state import GameState
from game.world.game_map import GameMap, TILE_ID_FLOOR
import sys
import types

# Minimal ai_system dispatch for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module


def make_gs():
    gm = GameMap(5, 5)
    gm.tiles[:] = TILE_ID_FLOOR
    gm.update_tile_transparency()
    return GameState(
        existing_map=gm,
        player_start_pos=(2, 2),
        player_glyph=ord("@"),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        effect_definitions={},
        rng_seed=1,
    )


def test_status_effects_expire():
    gs = make_gs()
    gs.entity_registry.set_entity_component(
        gs.player_id,
        "status_effects",
        [{"id": "poison", "duration": 1, "intensity": 1.0}],
    )
    gs.advance_turn()
    effects = gs.entity_registry.get_entity_component(gs.player_id, "status_effects")
    assert effects == []


def test_starvation_progression():
    gs = make_gs()
    gs.entity_registry.set_entity_component(gs.player_id, "fullness", 2.0)
    gs.advance_turn()
    assert gs.entity_registry.get_entity_component(gs.player_id, "fullness") == 1.0
    gs.advance_turn()
    assert gs.entity_registry.get_entity_component(gs.player_id, "fullness") == 0.0
    assert any("starving" in msg[0] for msg in gs.message_log)

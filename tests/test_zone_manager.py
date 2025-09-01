import sys
import types
from game_rng import GameRNG

# Stub ai_system to satisfy GameState imports
ai_module = types.ModuleType("game.systems.ai_system")
def dispatch_ai(*args, **kwargs):
    return None
ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

from simulation.zone_manager import ZoneManager
from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.game_state import GameState

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def test_zone_manager_far_zone_delay():
    manager = ZoneManager(map_width=100, map_height=100, zone_size=10, active_radius=1, passive_interval=3)
    called = []
    manager.schedule_event(50, 50, lambda gs: called.append(1))
    for turn in range(3):
        active = manager.get_active_zones((5, 5))
        manager.process(turn, active, None)
    assert called == []
    manager.process(3, manager.get_active_zones((5, 5)), None)
    assert called == [1]


def test_zone_manager_active_zone_immediate():
    manager = ZoneManager(map_width=100, map_height=100, zone_size=10, active_radius=1, passive_interval=3)
    called = []
    manager.schedule_event(5, 5, lambda gs: called.append(1))
    manager.process(0, manager.get_active_zones((5, 5)), None)
    assert called == [1]


def _create_game_state():
    gm = GameMap(100, 100)
    gm.tiles[:] = TILE_ID_FLOOR
    gm.update_tile_transparency()
    gs = GameState(
        existing_map=gm,
        player_start_pos=(1, 1),
        player_glyph=ord('@'),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        effect_definitions={},
        rng_seed=1,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    return gs


def test_game_state_schedules_far_event():
    gs = _create_game_state()
    triggered = []
    gs.schedule_low_detail_update(80, 80, lambda state: triggered.append(state.turn_count))
    for _ in range(4):
        gs.advance_turn()
        assert triggered == []
    gs.advance_turn()
    assert triggered == [5]


def test_zone_manager_serialization_roundtrip():
    manager = ZoneManager(
        map_width=100,
        map_height=100,
        zone_size=10,
        active_radius=1,
        passive_interval=3,
    )
    called = []
    manager.schedule_event(50, 50, lambda gs: called.append(1))
    state = manager.to_dict()
    registry_copy = dict(manager.event_registry)
    restored = ZoneManager.from_dict(state, registry_copy)
    for turn in range(3):
        restored.process(turn, restored.get_active_zones((5, 5)), None)
        assert called == []
    restored.process(3, restored.get_active_zones((5, 5)), None)
    assert called == [1]

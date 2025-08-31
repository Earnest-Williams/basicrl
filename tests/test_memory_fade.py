import numpy as np
import os
import sys
import types

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Minimal stubs for external modules
module = types.ModuleType("game_rng")


class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed

    def randint(self, a, b):
        return a


module.GameRNG = DummyRNG
sys.modules["game_rng"] = module

ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

from game.world.game_map import GameMap
from game.game_state import GameState


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
    )
    return gs


def test_memory_fade_decay_and_mask():
    gs = create_game_state()
    px, py = gs.player_position

    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()

    assert gs.game_map.memory_intensity[py, px] < 1.0
    assert gs.game_map.memory_fade_mask[py, px]


def test_memory_fade_skips_zero_intensity_tiles():
    gs = create_game_state()
    px, py = gs.player_position

    gs.entity_registry.set_entity_component(gs.player_id, "x", 0)
    gs.entity_registry.set_entity_component(gs.player_id, "y", 0)
    gs.advance_turn()

    gm = gs.game_map
    gm.memory_intensity[py, px] = 0.0
    assert gm.memory_fade_mask[py, px]

    gm.update_memory_fade(float(gs.turn_count))
    assert not gm.memory_fade_mask[py, px]
    assert gm.memory_intensity[py, px] == 0.0

    gm.update_memory_fade(float(gs.turn_count) + 1.0)
    assert gm.memory_intensity[py, px] == 0.0
    assert not gm.memory_fade_mask[py, px]


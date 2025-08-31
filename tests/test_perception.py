import os
import sys
import types

import numpy as np

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Provide a minimal game_rng module for tests
module = types.ModuleType("game_rng")


class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed

    def randint(self, a, b):
        return a


module.GameRNG = DummyRNG
sys.modules["game_rng"] = module

# Provide a minimal ai_system module for GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

from game.world.game_map import GameMap
from game.game_state import GameState
from game.ai.perception import gather_perception
from game.systems import movement_system

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


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
        effect_definitions={},
        rng_seed=42,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    return gs


def test_noise_map_updates_on_movement():
    gs = create_game_state()
    noise_before = gather_perception(gs)[0]
    assert np.all(noise_before == 0)

    moved = movement_system.try_move(gs.player_id, 1, 0, gs)
    assert moved

    noise_after = gather_perception(gs)[0]
    px, py = gs.entity_registry.get_position(gs.player_id)
    assert noise_after[py, px] > 0


def test_scent_map_updates_on_turn():
    gs = create_game_state()
    scent_before = gather_perception(gs)[1]
    assert np.all(scent_before == 0)

    gs.advance_turn()

    scent_after = gather_perception(gs)[1]
    px, py = gs.entity_registry.get_position(gs.player_id)
    assert scent_after[py, px] > 0

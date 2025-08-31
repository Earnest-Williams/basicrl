import os
import sys
import types

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Stub game_rng module for deterministic dice
module = types.ModuleType('game_rng')
class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed
    def get_int(self, a, b):
        return a
module.GameRNG = DummyRNG
sys.modules['game_rng'] = module

# Stub ai_system dispatch
ai_module = types.ModuleType('game.systems.ai_system')
def dispatch_ai(*args, **kwargs):
    return None
ai_module.dispatch_ai = dispatch_ai
sys.modules['game.systems.ai_system'] = ai_module

from game.world.game_map import GameMap
from game.game_state import GameState
from game.systems.combat_system import handle_melee_attack


def create_game_state():
    gm = GameMap(width=10, height=10)
    gm.create_test_room()
    gs = GameState(
        existing_map=gm,
        player_start_pos=(5, 5),
        player_glyph=ord('@'),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        effect_definitions={},
        rng_seed=1,
    )
    return gs


def test_strength_defense_modify_damage():
    gs = create_game_state()
    er = gs.entity_registry
    attacker = er.create_entity(
        x=1,
        y=1,
        glyph=ord('a'),
        color_fg=(255, 0, 0),
        name='Orc',
        hp=5,
        max_hp=5,
        strength=3,
    )
    defender = er.create_entity(
        x=1,
        y=2,
        glyph=ord('d'),
        color_fg=(0, 255, 0),
        name='Goblin',
        hp=10,
        max_hp=10,
        defense=1,
        armor=1,
    )
    gs.game_map.visible[:, :] = True
    handle_melee_attack(attacker, defender, gs)
    assert er.get_entity_component(defender, 'hp') == 8


def test_combat_messages_require_visibility():
    gs = create_game_state()
    er = gs.entity_registry
    attacker = er.create_entity(
        x=0,
        y=0,
        glyph=ord('a'),
        color_fg=(255, 0, 0),
        name='Orc',
        hp=5,
        max_hp=5,
        strength=1,
    )
    defender = er.create_entity(
        x=0,
        y=1,
        glyph=ord('d'),
        color_fg=(0, 255, 0),
        name='Goblin',
        hp=5,
        max_hp=5,
    )
    start_len = len(gs.message_log)
    gs.game_map.visible[:, :] = False
    handle_melee_attack(attacker, defender, gs)
    assert len(gs.message_log) == start_len
    gs.game_map.visible[0, 0] = True
    gs.game_map.visible[0, 1] = True
    handle_melee_attack(attacker, defender, gs)
    assert len(gs.message_log) > start_len

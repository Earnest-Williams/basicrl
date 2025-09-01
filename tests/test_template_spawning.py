import sys
import types
from game_rng import GameRNG

# Provide a minimal game.systems.ai_system module for tests
ai_module = types.ModuleType('game.systems.ai_system')

def dummy_dispatch_ai(*args, **kwargs):
    return None

ai_module.dispatch_ai = dummy_dispatch_ai
sys.modules['game.systems.ai_system'] = ai_module

from game.game_state import GameState
from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.effects.handlers import create_portal, attempt_spawn_entity

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def _make_game_state(entity_templates):
    game_map = GameMap(5, 5)
    game_map.tiles[:] = TILE_ID_FLOOR
    game_map.update_tile_transparency()
    return GameState(
        existing_map=game_map,
        player_start_pos=(0, 0),
        player_glyph=ord('@'),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates={},
        entity_templates=entity_templates,
        effect_definitions={},
        rng_seed=1,
        memory_fade_config=MEMORY_FADE_CFG,
    )


def test_create_portal_uses_template():
    templates = {
        'test_portal': {
            'glyph': 62,
            'color': [10, 20, 30],
            'name': 'Magic Gate',
            'blocks_movement': False,
            'target_map': 'depth2',
        }
    }
    gs = _make_game_state(templates)
    context = {'game_state': gs, 'target_pos': (1, 1)}
    params = {
        'portal_template': 'test_portal',
        'linked_positions': [{'x': 4, 'y': 4}],
    }
    create_portal(context, params)
    # New entity should be id 1 (player is id 0)
    name = gs.entity_registry.get_entity_component(1, 'name')
    glyph = gs.entity_registry.get_entity_component(1, 'glyph')
    color_r = gs.entity_registry.get_entity_component(1, 'color_fg_r')
    linked = gs.entity_registry.get_entity_component(1, 'linked_positions')
    target_map = gs.entity_registry.get_entity_component(1, 'target_map')
    assert name == 'Magic Gate'
    assert glyph == 62
    assert color_r == 10
    assert linked == [{'x': 4, 'y': 4}]
    assert target_map == 'depth2'


def test_attempt_spawn_entity_uses_template():
    templates = {
        'goblin': {
            'glyph': ord('g'),
            'color': [0, 128, 0],
            'hp': 7,
            'name': 'Goblin'
        }
    }
    gs = _make_game_state(templates)
    rng = GameRNG(seed=1)
    context = {'game_state': gs, 'target_pos': (2, 2), 'rng': rng}
    params = {'entity_template': 'goblin', 'chance': 100}
    attempt_spawn_entity(context, params)
    # Entity spawned should be id 1
    name = gs.entity_registry.get_entity_component(1, 'name')
    hp = gs.entity_registry.get_entity_component(1, 'hp')
    glyph = gs.entity_registry.get_entity_component(1, 'glyph')
    assert name == 'Goblin'
    assert hp == 7
    assert glyph == ord('g')

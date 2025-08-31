import types
import sys
import os

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Provide a minimal game_rng module for tests
module = types.ModuleType('game_rng')

class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed

    def randint(self, a, b):
        return a

module.GameRNG = DummyRNG
sys.modules['game_rng'] = module

from engine.action_handler import _handle_player_pickup
from game.game_state import GameState
from game.world.game_map import GameMap, TILE_ID_FLOOR


def test_pickup_fails_when_inventory_full():
    game_map = GameMap(5, 5)
    game_map.tiles[:] = TILE_ID_FLOOR
    game_map.update_tile_transparency()

    item_templates = {
        'test_item': {
            'name': 'Test Item',
            'glyph': 1,
            'color_fg': [255, 255, 255],
            'attributes': {}
        }
    }

    gs = GameState(
        existing_map=game_map,
        player_start_pos=(2, 2),
        player_glyph=1,
        player_start_hp=10,
        player_fov_radius=8,
        item_templates=item_templates,
        effect_definitions={},
        rng_seed=1,
    )

    player_id = gs.player_id
    gs.entity_registry.set_entity_component(player_id, 'inventory_capacity', 1)

    # Fill inventory to capacity
    gs.item_registry.create_item(
        template_id='test_item',
        location='inventory',
        owner_entity_id=player_id,
    )

    # Place another item on the ground at player's position
    px, py = gs.player_position
    gs.item_registry.create_item(
        template_id='test_item',
        location='ground',
        x=px,
        y=py,
    )

    result = _handle_player_pickup(gs)
    assert result is False
    assert gs.item_registry.get_entity_inventory(player_id).height == 1
    assert gs.item_registry.get_items_at(px, py).height == 1
    assert any('inventory is full' in msg[0].lower() for msg in gs.message_log)

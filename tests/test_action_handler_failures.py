import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from engine import action_handler
from game.world.game_map import GameMap
from game.game_state import GameState

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def create_game_state(item_templates=None):
    gm = GameMap(width=10, height=10)
    gm.create_test_room()
    gs = GameState(
        existing_map=gm,
        player_start_pos=(5, 5),
        player_glyph=ord('@'),
        player_start_hp=10,
        player_fov_radius=4,
        item_templates=item_templates or {},
        effect_definitions={},
        rng_seed=1,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    return gs


def test_combat_error_propagates(monkeypatch):
    gs = create_game_state()
    er = gs.entity_registry
    er.create_entity(x=6, y=5, glyph=ord('g'), color_fg=(255, 0, 0), name='Goblin')

    def broken_attack(attacker, defender, gs):
        raise RuntimeError('combat failed')

    monkeypatch.setattr(
        action_handler.combat_system,
        'handle_melee_attack',
        broken_attack,
    )

    with pytest.raises(RuntimeError):
        action_handler._handle_player_move(1, 0, gs, max_step=1)


def test_height_check_value_error_propagates(monkeypatch):
    gs = create_game_state()
    
    class BadHeightMap:
        def item(self, *args, **kwargs):
            raise ValueError('bad height')

    monkeypatch.setattr(gs.game_map, 'height_map', BadHeightMap())

    with pytest.raises(ValueError):
        action_handler._handle_player_move(1, 0, gs, max_step=1)


def test_height_check_index_error_returns_false(monkeypatch):
    gs = create_game_state()
    
    class BadHeightMap:
        def item(self, *args, **kwargs):
            raise IndexError('out of bounds')

    monkeypatch.setattr(gs.game_map, 'height_map', BadHeightMap())

    assert action_handler._handle_player_move(1, 0, gs, max_step=1) is False


def _setup_fall_path(gm, x: int, start_y: int) -> None:
    heights = [7, 3, 0, 0]
    for idx, h in enumerate(heights, start=1):
        gm.height_map[start_y + idx, x] = h


def test_fall_damage_message_visibility():
    gs = create_game_state()
    er = gs.entity_registry

    x, start_y = 6, 5
    _setup_fall_path(gs.game_map, x, start_y)
    entity_id = er.create_entity(x=x, y=start_y, glyph=ord('g'), color_fg=(255, 0, 0), name='Goblin', hp=10, max_hp=10)

    gs.message_log.clear()
    gs.update_fov()
    action_handler._handle_fall(entity_id, (x, start_y, 10), (x, start_y + 1, 7), gs, max_step=1)
    assert any('falling damage' in msg[0].lower() for msg in gs.message_log)

    gs = create_game_state()
    er = gs.entity_registry
    x, start_y = 9, 2
    _setup_fall_path(gs.game_map, x, start_y)
    entity_id = er.create_entity(x=x, y=start_y, glyph=ord('g'), color_fg=(255, 0, 0), name='Goblin', hp=10, max_hp=10)

    gs.message_log.clear()
    gs.update_fov()
    action_handler._handle_fall(entity_id, (x, start_y, 10), (x, start_y + 1, 7), gs, max_step=1)
    assert not any('falling damage' in msg[0].lower() for msg in gs.message_log)


def test_fall_death_cleanup():
    item_templates = {
        'test_item': {
            'name': 'Test Item',
            'glyph': 1,
            'color_fg': [255, 255, 255],
            'attributes': {},
        }
    }
    gs = create_game_state(item_templates=item_templates)
    er = gs.entity_registry
    ir = gs.item_registry

    x, start_y = 6, 5
    _setup_fall_path(gs.game_map, x, start_y)
    entity_id = er.create_entity(x=x, y=start_y, glyph=ord('g'), color_fg=(255, 0, 0), name='Goblin', hp=5, max_hp=5)
    ir.create_item('test_item', location='inventory', owner_entity_id=entity_id)

    gs.update_fov()
    action_handler._handle_fall(entity_id, (x, start_y, 10), (x, start_y + 1, 7), gs, max_step=1)

    assert er.get_position(entity_id) is None
    assert ir.get_entity_inventory(entity_id).height == 0
    assert ir.get_items_at(x, start_y).height == 1

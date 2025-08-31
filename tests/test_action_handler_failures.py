import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from engine import action_handler
from game.world.game_map import GameMap
from game.game_state import GameState

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


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

from game.systems.combat_system import handle_melee_attack
from game.game_state import GameState
from game.world.game_map import GameMap
import sys
import types
import polars as pl

# Stub ai_system dispatch
ai_module = types.ModuleType('game.systems.ai_system')


def dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dispatch_ai
sys.modules['game.systems.ai_system'] = ai_module


MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0,
                   "midpoint": 2.5, "steepness": 1.2}


def create_game_state(item_templates=None, rng_seed: int = 1):
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
        rng_seed=rng_seed,
        memory_fade_config=MEMORY_FADE_CFG,
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


def test_dual_wield_damage():
    templates = {
        'sword': {
            'name': 'Sword',
            'glyph': 47,
            'color_fg': [255, 255, 255],
            'item_type': 'Weapon',
            'equip_slot': 'main_hand',
            'flags': ['WEAPON', 'EQUIPPABLE'],
            'attributes': {'damage_dice': '2d6'},
            'effects': {},
        },
        'dagger_off': {
            'name': 'Off Dagger',
            'glyph': 47,
            'color_fg': [255, 255, 255],
            'item_type': 'Weapon',
            'equip_slot': 'off_hand',
            'flags': ['WEAPON', 'EQUIPPABLE', 'OFF_HAND'],
            'attributes': {'damage_dice': '2d4'},
            'effects': {},
        },
    }
    gs = create_game_state(item_templates=templates, rng_seed=688)
    er = gs.entity_registry
    ir = gs.item_registry

    attacker = er.create_entity(x=1, y=1, glyph=ord(
        'a'), color_fg=(255, 0, 0), name='Orc', hp=5, max_hp=5)
    defender = er.create_entity(x=1, y=2, glyph=ord(
        'd'), color_fg=(0, 255, 0), name='Goblin', hp=10, max_hp=10)

    main_id = ir.create_item('sword', location='equipped',
                             owner_entity_id=attacker, equipped_slot='main_hand')
    off_id = ir.create_item('dagger_off', location='equipped',
                            owner_entity_id=attacker, equipped_slot='off_hand')
    er.set_equipped_ids(attacker, [main_id, off_id])

    gs.game_map.visible[:, :] = True
    handle_melee_attack(attacker, defender, gs)

    # Deterministic damage with seed 688 should leave defender at 8 HP
    assert er.get_entity_component(defender, 'hp') == 8


def test_two_handed_bonus_damage():
    templates = {
        'greatsword': {
            'name': 'Greatsword',
            'glyph': 47,
            'color_fg': [255, 255, 255],
            'item_type': 'Weapon',
            'equip_slot': 'main_hand',
            'flags': ['WEAPON', 'EQUIPPABLE', 'TWO_HANDED'],
            'attributes': {'damage_dice': '2d6'},
            'effects': {},
        }
    }
    gs = create_game_state(item_templates=templates, rng_seed=11)
    er = gs.entity_registry
    ir = gs.item_registry

    attacker = er.create_entity(x=1, y=1, glyph=ord(
        'a'), color_fg=(255, 0, 0), name='Orc', hp=5, max_hp=5)
    defender = er.create_entity(x=1, y=2, glyph=ord(
        'd'), color_fg=(0, 255, 0), name='Goblin', hp=10, max_hp=10)

    main_id = ir.create_item('greatsword', location='equipped',
                             owner_entity_id=attacker, equipped_slot='main_hand')
    er.set_equipped_ids(attacker, [main_id])

    gs.game_map.visible[:, :] = True
    handle_melee_attack(attacker, defender, gs)

    # Deterministic damage with seed 11 results in 3 damage after two-handed bonus
    assert er.get_entity_component(defender, 'hp') == 7


def test_damage_types_resistance_and_vulnerability():
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
        resistances={'fire': 0.5},
        vulnerabilities={'ice': 0.5},
    )
    gs.game_map.visible[:, :] = True
    handle_melee_attack(attacker, defender, gs, damage_type='fire')
    assert er.get_entity_component(defender, 'hp') == 8

    gs2 = create_game_state()
    er2 = gs2.entity_registry
    attacker2 = er2.create_entity(
        x=1,
        y=1,
        glyph=ord('a'),
        color_fg=(255, 0, 0),
        name='Orc',
        hp=5,
        max_hp=5,
        strength=3,
    )
    defender2 = er2.create_entity(
        x=1,
        y=2,
        glyph=ord('d'),
        color_fg=(0, 255, 0),
        name='Goblin',
        hp=10,
        max_hp=10,
        resistances={'fire': 0.5},
        vulnerabilities={'ice': 0.5},
    )
    gs2.game_map.visible[:, :] = True
    handle_melee_attack(attacker2, defender2, gs2, damage_type='ice')
    assert er2.get_entity_component(defender2, 'hp') == 4


def test_xp_and_loot_on_death():
    templates = {
        'coin': {
            'name': 'Coin',
            'glyph': 36,
            'color_fg': [255, 255, 0],
            'item_type': 'misc',
            'flags': [],
            'attributes': {},
            'effects': {},
        }
    }
    gs = create_game_state(item_templates=templates)
    er = gs.entity_registry
    ir = gs.item_registry

    attacker = er.create_entity(x=1, y=1, glyph=ord('a'), color_fg=(
        255, 0, 0), name='Hero', hp=5, max_hp=5, xp=0)
    drop_table = [{'template_id': 'coin', 'chance': 1.0}]
    defender = er.create_entity(
        x=1,
        y=2,
        glyph=ord('d'),
        color_fg=(0, 255, 0),
        name='Goblin',
        hp=1,
        max_hp=1,
        xp_reward=5,
        drop_table=drop_table,
    )
    gs.game_map.visible[:, :] = True
    handle_melee_attack(attacker, defender, gs)
    assert er.get_entity_component(attacker, 'xp') == 5
    ground = ir.items_df.filter(
        (pl.col('location_type') == 'ground') & (
            pl.col('x') == 1) & (pl.col('y') == 2)
    )
    assert ground.height == 1
    assert ground.select('template_id').item() == 'coin'

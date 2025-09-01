import types
import sys

# Stub game_rng for deterministic behavior
module = types.ModuleType("game_rng")

class DummyRNG:
    def __init__(self, seed=None):
        self.initial_seed = seed
    def get_int(self, a, b):
        return a

module.GameRNG = DummyRNG
sys.modules["game_rng"] = module

from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.game_state import GameState
from game.effects.handlers import heal_target, deal_damage, apply_status

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def make_game_state():
    game_map = GameMap(5, 5)
    game_map.tiles[:] = TILE_ID_FLOOR
    game_map.update_tile_transparency()
    gs = GameState(
        existing_map=game_map,
        player_start_pos=(2, 2),
        player_glyph=ord("@"),
        player_start_hp=20,
        player_fov_radius=3,
        item_templates={},
        effect_definitions={},
        rng_seed=1,
        memory_fade_config=MEMORY_FADE_CFG,
    )
    return gs


def test_heal_target_message_only_when_visible():
    gs = make_game_state()
    rng = gs.rng_instance
    hidden_id = gs.entity_registry.create_entity(
        x=4,
        y=4,
        glyph=ord("h"),
        color_fg=(255, 0, 0),
        name="Hidden",
        hp=5,
        max_hp=10,
    )
    visible_id = gs.entity_registry.create_entity(
        x=2,
        y=3,
        glyph=ord("v"),
        color_fg=(255, 0, 0),
        name="Visible",
        hp=5,
        max_hp=10,
    )
    gs.update_fov()
    start_len = len(gs.message_log)
    context = {"game_state": gs, "target_entity_id": hidden_id, "rng": rng}
    params = {"base_heal": 5}
    heal_target(context, params)
    assert len(gs.message_log) == start_len

    context["target_entity_id"] = visible_id
    heal_target(context, params)
    assert len(gs.message_log) == start_len + 1


def test_deal_damage_message_only_when_visible():
    gs = make_game_state()
    rng = gs.rng_instance
    attacker_id = gs.entity_registry.create_entity(
        x=1,
        y=1,
        glyph=ord("a"),
        color_fg=(255, 255, 255),
        name="Attacker",
    )
    hidden_id = gs.entity_registry.create_entity(
        x=4,
        y=4,
        glyph=ord("h"),
        color_fg=(255, 0, 0),
        name="Hidden",
        hp=10,
        max_hp=10,
    )
    visible_id = gs.entity_registry.create_entity(
        x=2,
        y=3,
        glyph=ord("v"),
        color_fg=(255, 0, 0),
        name="Visible",
        hp=10,
        max_hp=10,
    )
    gs.update_fov()
    start_len = len(gs.message_log)
    context = {
        "game_state": gs,
        "target_entity_id": hidden_id,
        "source_entity_id": attacker_id,
        "rng": rng,
    }
    params = {"base_damage": 5}
    deal_damage(context, params)
    assert len(gs.message_log) == start_len
    assert len(gs.message_queue) == 1

    context["target_entity_id"] = visible_id
    deal_damage(context, params)
    assert len(gs.message_log) == start_len + 1
    assert len(gs.message_queue) == 1


def test_apply_status_message_only_when_visible():
    gs = make_game_state()
    rng = gs.rng_instance
    hidden_id = gs.entity_registry.create_entity(
        x=4,
        y=4,
        glyph=ord("h"),
        color_fg=(255, 0, 0),
        name="Hidden",
    )
    visible_id = gs.entity_registry.create_entity(
        x=2,
        y=3,
        glyph=ord("v"),
        color_fg=(255, 0, 0),
        name="Visible",
    )
    gs.update_fov()
    start_len = len(gs.message_log)
    start_queue = len(gs.message_queue)
    context = {"game_state": gs, "target_entity_id": hidden_id, "rng": rng}
    params = {"status": "burning"}
    apply_status(context, params)
    assert len(gs.message_log) == start_len
    assert len(gs.message_queue) == start_queue + 1

    context["target_entity_id"] = visible_id
    apply_status(context, params)
    assert len(gs.message_log) == start_len + 1
    assert len(gs.message_queue) == start_queue + 1


def test_deal_damage_resistance_and_vulnerability():
    gs = make_game_state()
    rng = gs.rng_instance
    res_id = gs.entity_registry.create_entity(
        x=2,
        y=3,
        glyph=ord("r"),
        color_fg=(255, 0, 0),
        name="Resistant",
        hp=20,
        max_hp=20,
        resistances={"fire": 0.5},
    )
    vuln_id = gs.entity_registry.create_entity(
        x=3,
        y=2,
        glyph=ord("v"),
        color_fg=(255, 0, 0),
        name="Vulnerable",
        hp=20,
        max_hp=20,
        vulnerabilities={"fire": 0.5},
    )
    context = {"game_state": gs, "rng": rng, "target_entity_id": res_id}
    params = {"base_damage": 10, "damage_type": "fire"}
    deal_damage(context, params)
    hp_res = gs.entity_registry.get_entity_component(res_id, "hp")
    assert hp_res == 15

    context["target_entity_id"] = vuln_id
    deal_damage(context, params)
    hp_vuln = gs.entity_registry.get_entity_component(vuln_id, "hp")
    assert hp_vuln == 5

import sys
import types

# Stub ai_system to satisfy GameState imports
ai_module = types.ModuleType("game.systems.ai_system")


def dummy_dispatch_ai(*args, **kwargs):
    return None


ai_module.dispatch_ai = dummy_dispatch_ai
sys.modules["game.systems.ai_system"] = ai_module

import polars as pl

# Patch DataFrame.row to ignore unsupported kwargs (compatibility across polars versions)
_orig_row = pl.DataFrame.row


def _row_with_default(self, index, *, named=False, **kwargs):
    return _orig_row(self, index, named=named)


pl.DataFrame.row = _row_with_default

from game.game_state import GameState
from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.systems.equipment_system import apply_passive_effects, remove_passive_effects

MEMORY_FADE_CFG = {"enabled": True, "duration": 5.0, "midpoint": 2.5, "steepness": 1.2}


def _create_basic_map():
    game_map = GameMap(5, 5)
    game_map.tiles[:] = TILE_ID_FLOOR
    game_map.update_tile_transparency()
    return game_map


def _effect_definitions():
    return {
        "increase_mana": {
            "type": "passive",
            "target_category": "user",
            "logic_handler": "modify_resource",
            "params": {"resource": "mana", "base_change": 5},
        },
        "decrease_mana": {
            "type": "passive",
            "target_category": "user",
            "logic_handler": "modify_resource",
            "params": {"resource": "mana", "base_change": -5},
        },
    }


def test_apply_and_remove_passive_effects():
    item_templates = {
        "mana_ring": {
            "name": "Mana Ring",
            "glyph": 1,
            "color_fg": [255, 255, 255],
            "attributes": {},
            "effects": {
                "passive": ["increase_mana"],
                "passive_remove": ["decrease_mana"],
            },
        }
    }

    gs = GameState(
        existing_map=_create_basic_map(),
        player_start_pos=(2, 2),
        player_glyph=1,
        player_start_hp=10,
        player_fov_radius=8,
        item_templates=item_templates,
        effect_definitions=_effect_definitions(),
        rng_seed=1,
        memory_fade_config=MEMORY_FADE_CFG,
    )

    player_id = gs.player_id
    gs.entity_registry.set_entity_component(player_id, "max_mana", 10)
    item_id = gs.item_registry.create_item(
        template_id="mana_ring", location="inventory", owner_entity_id=player_id
    )

    base_mana = gs.entity_registry.get_entity_component(player_id, "mana")

    apply_passive_effects(item_id, player_id, gs)
    assert gs.entity_registry.get_entity_component(player_id, "mana") == base_mana + 5

    remove_passive_effects(item_id, player_id, gs)
    assert gs.entity_registry.get_entity_component(player_id, "mana") == base_mana


# === basicrl/prototypes/roaming_entity_demo.py ===
import sys
import time
from pathlib import Path

# Allow imports like 'from game.*' and 'from game_rng'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from game.game_state import GameState
from game.world.game_map import GameMap


def create_roamer(entity_registry, x, y):
    return entity_registry.create_entity(
        x=x,
        y=y,
        glyph=ord("R"),
        color_fg=(255, 128, 0),
        name="Roamer",
        blocks_movement=True,
        hp=5,
        max_hp=5,
    )


def move_roamer(entity_registry, entity_id, game_map):
    dx, dy = 1, 0  # Move right each step
    x = entity_registry.get_entity_component(entity_id, "x")
    y = entity_registry.get_entity_component(entity_id, "y")
    new_x = x + dx
    new_y = y + dy

    if game_map.in_bounds(new_x, new_y) and game_map.is_walkable(new_x, new_y):
        entity_registry.set_entity_component(entity_id, "x", new_x)
        entity_registry.set_entity_component(entity_id, "y", new_y)
        return (new_x, new_y)
    return (x, y)


def main():
    width, height = 20, 10
    game_map = GameMap(width=width, height=height)
    game_map.create_test_room()

    game_state = GameState(
        existing_map=game_map,
        player_start_pos=(10, 5),
        player_glyph=ord("@"),
        player_start_hp=10,
        player_fov_radius=6,
        item_templates={},  # ✅ must be a dict
        effect_definitions={},
        rng_seed=42,
    )

    entity_registry = game_state.entity_registry
    game_map = game_state.game_map

    roamer_id = create_roamer(entity_registry, 2, 5)

    for tick in range(10):
        new_pos = move_roamer(entity_registry, roamer_id, game_map)
        game_state.turn_count += 1
        game_state.add_message(f"Roamer moved to {new_pos}", (200, 200, 0))
        print(f"[Tick {tick}] Roamer is now at {new_pos}")
        time.sleep(0.25)


if __name__ == "__main__":
    main()

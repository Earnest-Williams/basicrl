from game.world.game_map import GameMap, MAX_MEMORY_STRENGTH


def test_memory_strength_reinforce_and_decay():
    gm = GameMap(width=10, height=10)
    gm.create_test_room()
    x, y = 5, 5

    # Reinforce the tile repeatedly and ensure it is capped at MAX_MEMORY_STRENGTH.
    for _ in range(int(MAX_MEMORY_STRENGTH) + 2):
        gm.compute_fov(x, y, radius=5)
    assert gm.memory_strength[y, x] == MAX_MEMORY_STRENGTH

    # Move FOV so the tile is no longer visible and check gradual decay.
    for expected in range(int(MAX_MEMORY_STRENGTH) - 1, -1, -1):
        gm.compute_fov(0, 0, radius=1)
        assert gm.memory_strength[y, x] == expected

    # Additional turns should not push the value below zero.
    gm.compute_fov(0, 0, radius=1)
    assert gm.memory_strength[y, x] == 0.0

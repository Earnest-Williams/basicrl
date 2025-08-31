from game.ai.action_resolver import apply_actions
from game.ai.action_schema import make_move_action


def test_simple_moves():
    positions = {1: (0, 0), 2: (2, 2)}
    walkable = {(0, 0), (1, 0), (2, 2), (2, 1)}
    actions = [make_move_action(1, 1, 0), make_move_action(2, 0, -1)]
    new_pos, applied = apply_actions(actions, positions, walkable)
    assert new_pos[1] == (1, 0)
    assert new_pos[2] == (2, 1)
    assert len(applied) == 2


def test_conflict_moves():
    positions = {1: (0, 0), 2: (2, 0)}
    walkable = {(0, 0), (1, 0), (2, 0)}
    # Both attempt to move to (1,0)
    actions = [make_move_action(1, 1, 0), make_move_action(2, -1, 0)]
    new_pos, applied = apply_actions(actions, positions, walkable)
    # Actor 1 has lower actor id, so should win the spot
    assert new_pos[1] == (1, 0)
    # Actor 2's move should be skipped; remains in place
    assert new_pos[2] == (2, 0)
    assert len(applied) == 1


def test_move_to_non_walkable_is_skipped():
    positions = {1: (0, 0)}
    walkable = {(0, 0)}  # (1,0) is not walkable
    actions = [make_move_action(1, 1, 0)]
    new_pos, applied = apply_actions(actions, positions, walkable)
    assert new_pos[1] == (0, 0)
    assert len(applied) == 0
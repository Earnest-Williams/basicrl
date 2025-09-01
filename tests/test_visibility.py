import numpy as np
import pytest
from game_rng import GameRNG
import game.world.fov as MyVisibility
from game.world.fov import line_of_sight

# Alias compute function for convenience
MyVisibility.compute = MyVisibility.compute_fov


def _make_basic_maps(width: int = 5, height: int = 5):
    opaque = np.zeros((height, width), dtype=bool)
    height_map = np.zeros((height, width), dtype=np.int16)
    ceiling_map = np.full((height, width), 10, dtype=np.int16)
    return opaque, height_map, ceiling_map


def test_compute_with_wall_blocks_visibility():
    opaque, height_map, ceiling_map = _make_basic_maps()
    # Vertical wall in column 2
    opaque[:, 2] = True
    transparent = ~opaque
    visible = np.zeros_like(opaque, dtype=bool)
    explored = np.zeros_like(opaque, dtype=bool)
    origin = (1, 2)
    MyVisibility.compute(
        origin_xy=origin,
        range_limit=4,
        opaque_grid=opaque,
        height_map=height_map,
        ceiling_map=ceiling_map,
        origin_height=int(height_map[origin[1], origin[0]]),
        visible_grid=visible,
        explored_grid=explored,
    )
    assert visible[origin[1], origin[0]]
    assert not visible[origin[1], 3]
    assert not line_of_sight(origin[1], origin[0], origin[1], 3, transparent)
    assert line_of_sight(origin[1], origin[0], origin[1], 0, transparent)


def test_compute_negative_range_only_origin_visible():
    opaque, height_map, ceiling_map = _make_basic_maps()
    visible = np.zeros_like(opaque, dtype=bool)
    explored = np.zeros_like(opaque, dtype=bool)
    origin = (2, 2)
    MyVisibility.compute(
        origin_xy=origin,
        range_limit=-5,
        opaque_grid=opaque,
        height_map=height_map,
        ceiling_map=ceiling_map,
        origin_height=0,
        visible_grid=visible,
        explored_grid=explored,
    )
    assert visible.sum() == 1
    assert visible[origin[1], origin[0]]


def test_compute_out_of_bounds_raises():
    opaque, height_map, ceiling_map = _make_basic_maps()
    visible = np.zeros_like(opaque, dtype=bool)
    explored = np.zeros_like(opaque, dtype=bool)
    with pytest.raises(ValueError):
        MyVisibility.compute(
            origin_xy=(-1, -1),
            range_limit=4,
            opaque_grid=opaque,
            height_map=height_map,
            ceiling_map=ceiling_map,
            origin_height=0,
            visible_grid=visible,
            explored_grid=explored,
        )


def test_line_of_sight_out_of_bounds_returns_false():
    transparency = np.ones((5, 5), dtype=bool)
    assert line_of_sight(-1, 0, 2, 2, transparency) is False
    assert line_of_sight(0, 0, 5, 5, transparency) is False


def test_line_of_sight_blocks_diagonal_walls():
    transparency = np.ones((5, 5), dtype=bool)
    for i in range(1, 4):
        transparency[i, i] = False
    assert line_of_sight(0, 0, 4, 4, transparency) is False

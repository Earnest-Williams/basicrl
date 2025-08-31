import os
import sys
import numpy as np
import polars as pl

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from game.constants import FeatureType, FlowType, MAX_FLOWS
from game_rng import GameRNG
from pathfinding.perception_systems import (
    BASE_FLOW_CENTER,
    NOISE_MAX_DIST,
    _process_monster_perception_chunk,
    cave_closed_door,
    skill_check,
)


def test_skill_check_deterministic():
    seed = 7
    expected_roll = GameRNG(seed).get_int(1, 100)
    rng = GameRNG(seed)
    result = skill_check(rng, actor_skill=30, difficulty=10, target_skill=5)
    threshold = 50 + 30 - (10 + 5)
    assert result == (expected_roll <= threshold)


def test_cave_closed_door_feature_detection():
    assert cave_closed_door(FeatureType.CLOSED_DOOR)
    assert cave_closed_door(FeatureType.SECRET_DOOR)
    assert not cave_closed_door(FeatureType.OPEN_DOOR)
    assert not cave_closed_door(FeatureType.FLOOR)


def test_process_chunk_perception_outcome():
    rng_seed = 42
    rng = GameRNG(rng_seed)
    cave_cost = np.full((MAX_FLOWS, 5, 5), BASE_FLOW_CENTER + NOISE_MAX_DIST + 1, dtype=np.int32)
    flow_centers = np.zeros((MAX_FLOWS, 2), dtype=np.int32)
    cave_cost[FlowType.REAL_NOISE, 1, 1] = BASE_FLOW_CENTER + 5
    cave_cost[FlowType.REAL_NOISE, 1, 2] = BASE_FLOW_CENTER + 50
    monster_df = pl.DataFrame(
        {
            "id": [1, 2],
            "fy": [1, 1],
            "fx": [1, 2],
            "perception_stat": [60, 60],
        }
    )
    result = _process_monster_perception_chunk(
        monster_df, cave_cost, flow_centers, 10, FlowType.REAL_NOISE, rng
    )
    exp_rng = GameRNG(rng_seed)
    roll1 = exp_rng.get_int(1, 100)
    roll2 = exp_rng.get_int(1, 100)
    threshold1 = 50 + 60 - (5 + 10)
    threshold2 = 50 + 60 - (50 + 10)
    expected = []
    if roll1 <= threshold1:
        expected.append(1)
    if roll2 <= threshold2:
        expected.append(2)
    assert result == expected

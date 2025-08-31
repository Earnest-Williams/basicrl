import pytest
from types import SimpleNamespace
from collections import defaultdict

# Minimal stubs to match v9.py Expectations
class Behavior:
    def __init__(self, impact=None, est_energy_cost=1.0, est_time_cost=1.0):
        self.impact = impact or {}
        self.est_energy_cost = est_energy_cost
        self.est_time_cost = est_time_cost

class Habit:
    def __init__(self, sequence):
        self.sequence = sequence

    def estimate_impact(self, agent):
        total = defaultdict(float)
        for item in self.sequence:
            if isinstance(item, Behavior):
                for k, v in item.impact.items():
                    total[f"delta_{k}"] += v
                total["delta_energy"] -= item.est_energy_cost
                total["delta_time"] += item.est_time_cost
            elif isinstance(item, Habit):
                nested = item.estimate_impact(agent)
                for k, v in nested.items():
                    total[k] += v
        return dict(total)

class DummyAgent:
    def __init__(self):
        self.task_stats = {}  # placeholder for future integration


def test_estimate_simple_behavior_impact():
    b = Behavior(impact={"health": 2.0, "hunger": -1.0}, est_energy_cost=0.5, est_time_cost=2.0)
    hab = Habit([b])
    agent = DummyAgent()
    impact = hab.estimate_impact(agent)
    # Expect keys prefixed with delta_
    assert impact["delta_health"] == 2.0
    assert impact["delta_hunger"] == -1.0
    assert impact["delta_energy"] == -0.5
    assert impact["delta_time"] == pytest.approx(2.0)


def test_nested_habit_aggregation():
    b1 = Behavior(impact={"health": 1.0}, est_energy_cost=0.2, est_time_cost=1.0)
    b2 = Behavior(impact={"health": 0.5, "mood": 1.0}, est_energy_cost=0.3, est_time_cost=0.5)
    inner = Habit([b1])
    outer = Habit([inner, b2])
    impact = outer.estimate_impact(DummyAgent())
    assert impact["delta_health"] == pytest.approx(1.5)
    assert impact["delta_mood"] == pytest.approx(1.0)
    assert impact["delta_energy"] == pytest.approx(-0.5)
    assert impact["delta_time"] == pytest.approx(1.5)
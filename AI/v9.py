#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent-based simulation core featuring farming, resource management,
and a learning habit system with adaptive impact estimation for planning,
feature triggers, and contextual outcome evaluation.
Optimized with NumPy and deterministic RNG.
Version: 9.0 (Merged Traits & Habits)
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union, Callable

import numpy as np
from game_rng import GameRNG

# Concrete dependency implementations

@dataclass
class Home:
    """Resource storage and field references for an agent's dwelling."""

    water_storage: float = 0.0
    raw_inventory: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cooked_food: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    fiber: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    containers: Dict[str, Any] = field(default_factory=dict)
    fields: List[Any] = field(default_factory=list)

@dataclass
class Behavior:
    """Atomic action that an agent can attempt to perform."""

    name: str
    fn: Callable[["AgentF"], bool] | None
    impact: Dict = field(default_factory=dict)
    est_energy_cost: float = 0.0
    est_time_cost: float = 0.0
    task_id: str = ""
    primary_skill: str = ""


@dataclass
class Habit:
    """A learned sequence of behaviors executed under triggers."""

    name: str
    sequence: List[Union[Behavior, "Habit"]]
    trigger: Union[Dict, Callable[["AgentF"], bool]]
    score: float
    created_day: int
    last_used_day: int = field(init=False)

    def __post_init__(self) -> None:
        self.last_used_day = self.created_day

    def estimate_impact(self, agent: "AgentF") -> Dict[str, float]:
        total_impact = defaultdict(float)
        for item in self.sequence:
            if isinstance(item, Behavior):
                for k, v in item.impact.items():
                    delta_k = f"delta_{k}"
                    total_impact[delta_k] += v
                total_impact["delta_energy"] -= item.est_energy_cost
                total_impact["delta_time"] += item.est_time_cost
            elif isinstance(item, Habit):
                nested_impact = item.estimate_impact(agent)
                for k, v in nested_impact.items():
                    total_impact[k] += v
        return dict(total_impact)

    def execute(self, agent: "AgentF") -> bool:
        agent.base_logger.debug("habit_execute_start", habit_name=self.name)
        all_completed = True
        for item in self.sequence:
            completed = False
            if isinstance(item, Behavior):
                if item.fn:
                    try:
                        completed = item.fn(agent)
                        agent.daily_behavior_log.append(
                            (agent._capture_state_snapshot(), item.name)
                        )
                    except Exception as e:
                        agent.base_logger.error(
                            "behavior_execution_error",
                            behavior_name=item.name,
                            error=str(e),
                        )
                        completed = False
                else:
                    agent.base_logger.warning(
                        "behavior_missing_function", behavior_name=item.name
                    )
                    completed = False
            elif isinstance(item, Habit):
                completed = item.execute(agent)

            if not completed:
                all_completed = False
                agent.base_logger.debug(
                    "habit_execute_interrupted",
                    habit_name=self.name,
                    failed_item=item.name,
                )
                break

        self.last_used_day = agent.context.get("day", self.last_used_day)
        agent.base_logger.debug(
            "habit_execute_end", habit_name=self.name, completed=all_completed
        )
        return all_completed

    def is_triggered(self, agent: "AgentF") -> bool:
        if callable(self.trigger):
            try:
                return self.trigger(agent)
            except Exception as e:
                agent.base_logger.error(
                    "habit_trigger_callable_error", habit_name=self.name, error=str(e)
                )
                return False
        elif isinstance(self.trigger, dict):
            agent.base_logger.warning(
                "habit_trigger_dict_unhandled", habit_name=self.name
            )
            return False
        return False


@dataclass
class OutcomeResult:
    """Container for outcome evaluations."""

    evaluation: float = 0.0


@dataclass
class TraitProfile:
    """Trait modifiers influencing agent behaviour and recovery."""

    endurance: float = 1.0
    ingenuity: float = 1.0
    perception: float = 1.0
    will: float = 1.0
    resonance: float = 1.0


class FatigueSystem:
    """Tracks fatigue accumulation and recovery."""

    level: float = 0.0

    def __init__(self, endurance: float = 1.0) -> None:
        self.level = 0.0
        self._endurance_factor = max(0.1, 0.7 + 0.3 * endurance)

    def add_fatigue(self, amount: float, activity_intensity: float) -> None:
        gain = amount * max(0.1, activity_intensity) / self._endurance_factor
        self.level = min(100.0, self.level + gain)

    def recover(self, quality_rest: float, nutrition_bonus: float) -> None:
        recovery_amount = 5.0 + nutrition_bonus + (5.0 * quality_rest)
        self.level = max(0.0, self.level - recovery_amount)

    def get_performance_modifier(self) -> float:
        return max(0.1, 1.0 - (self.level / 125.0))


class IllnessSystem:
    """Models active illnesses and their effects on performance."""

    active_conditions: Dict[str, Dict[str, Any]]
    _endurance: float
    _will: float

    def __init__(self, endurance: float = 1.0, will: float = 1.0) -> None:
        self.active_conditions = {}
        self._endurance = endurance
        self._will = will

    def add_condition(self, name: str, severity: float, duration: int) -> None:
        if name not in self.active_conditions:
            self.active_conditions[name] = {"severity": severity, "duration": duration}

    def has_condition(self, condition_name: str) -> bool:
        return condition_name in self.active_conditions

    def get_total_severity(self) -> float:
        return sum(cond.get("severity", 0) for cond in self.active_conditions.values())

    def get_performance_modifiers(self) -> Dict[str, float]:
        total_severity = self.get_total_severity()
        severity_factor = total_severity / 10.0
        return {
            "efficiency": max(0.1, 1.0 - severity_factor * 0.5),
            "energy": max(0.5, 1.0 - severity_factor * 0.3),
        }

    def daily_update(self) -> List[str]:
        resolved: List[str] = []
        will_factor = 0.1 * (self._will - 1.0)
        endurance_factor = 0.05 * (self._endurance - 1.0)

        for name in list(self.active_conditions.keys()):
            condition = self.active_conditions[name]
            condition["duration"] -= 1
            if np.random.rand() < will_factor:
                condition["duration"] -= 1
            if np.random.rand() < endurance_factor:
                condition["severity"] = max(0.1, condition["severity"] * 0.9)

            if condition["duration"] <= 0:
                resolved.append(name)
                del self.active_conditions[name]
        return resolved


class ExperienceMemory:
    """Stores experiences for simple outcome prediction."""

    memories: Dict[str, List[Tuple[Dict, Dict, float]]]
    _ingenuity: float

    def __init__(self, ingenuity: float = 1.0) -> None:
        self.memories = defaultdict(list)
        self._ingenuity = ingenuity

    def add_memory(
        self,
        behavior_name: str,
        state_before: Dict,
        state_after: Dict,
        day: int,
        value: float,
    ) -> None:
        self.memories[behavior_name].append((state_before, state_after, value))
        if len(self.memories[behavior_name]) > 50:
            self.memories[behavior_name].pop(0)

    def predict_outcome(self, behavior_name: str, current_state: Dict) -> Dict | None:
        return None


class SelfConcept:
    """Tracks the agent's identity and behaviour alignment."""

    identity_aspects: Dict[str, float]
    behavior_identity_map: Dict[str, Dict[str, float]]
    _resonance: float

    def __init__(self, resonance: float = 1.0) -> None:
        self.identity_aspects = defaultdict(lambda: 0.5)
        self.behavior_identity_map = self._initialize_behavior_map()
        self._resonance = resonance

    def _initialize_behavior_map(self) -> Dict[str, Dict[str, float]]:
        return {
            "tend_field": {"farmer": 0.1, "provider": 0.05},
            "harvest_field": {"farmer": 0.15, "provider": 0.1},
            "fetch_water": {"provider": 0.02},
            "drink_tea": {"calm": 0.05},
            "use_ipecac": {"survivor": 0.1},
        }

    def get_identity_affinity(self, behavior_name: str) -> float:
        affinity = 0.0
        if behavior_name in self.behavior_identity_map:
            for aspect, weight in self.behavior_identity_map[behavior_name].items():
                affinity += self.identity_aspects[aspect] * weight
        return affinity

    def calculate_dissonance(self, behavior_name: str) -> float:
        dissonance = 0.0
        if behavior_name in self.behavior_identity_map:
            for aspect, weight in self.behavior_identity_map[behavior_name].items():
                if weight < 0 and self.identity_aspects[aspect] > 0.7:
                    dissonance += abs(weight) * (self.identity_aspects[aspect] - 0.5)
                elif weight > 0 and self.identity_aspects[aspect] < 0.3:
                    dissonance += abs(weight) * (0.5 - self.identity_aspects[aspect])
        return dissonance

    def update_from_behavior(
        self, behavior_name: str, completed: bool, perceived_value: float
    ) -> None:
        if not completed:
            return
        if behavior_name in self.behavior_identity_map:
            update_strength = 0.05 * self._resonance
            value_mod = 1.0 + max(-0.5, min(0.5, perceived_value * 0.1))
            for aspect, weight in self.behavior_identity_map[behavior_name].items():
                change = update_strength * weight * value_mod
                self.identity_aspects[aspect] = max(
                    0.0, min(1.0, self.identity_aspects[aspect] + change)
                )


class Calendar:
    day: int = 0


class Weather:
    pass


class Field:
    plots: Dict[str, np.ndarray]  # Define structure e.g., {'status': [], 'health': []}

    def __init__(self, size=10):
        # Example initialization
        self.plots = {
            "status": np.zeros(size, dtype=np.int8),
            "health": np.full(size, 100.0, dtype=np.float32),
            "crop_type": np.full(size, "", dtype=object),  # Store crop names
        }

    def tend_plots(self, indices: np.ndarray, health_increase: float):
        if indices.size > 0:
            self.plots["health"][indices] = np.minimum(
                100.0, self.plots["health"][indices] + health_increase
            )

    def harvest_ready_plots(
        self, weather: Weather, calendar: Calendar
    ) -> Dict[str, int]:
        # Find ready plots
        ready_indices = np.where(self.plots["status"] == PLOT_STATUS_MAP["ready"])[0]
        harvest_summary = defaultdict(int)
        if ready_indices.size > 0:
            for idx in ready_indices:
                crop_type = self.plots["crop_type"][idx]
                if crop_type:  # Ensure a crop was planted
                    # Yield based on health (example)
                    yield_amount = int(max(1, (self.plots["health"][idx] / 25.0)))
                    harvest_summary[crop_type] += yield_amount
                    # Reset plot status
                    self.plots["status"][idx] = 0  # Assuming 0 is 'empty' or similar
                    self.plots["health"][idx] = 50.0  # Reset health
                    self.plots["crop_type"][idx] = ""
        return dict(harvest_summary)


PLOT_STATUS_MAP = {"growing": 1, "ready": 2, "empty": 0}  # Add empty status
CROPS = {
    "peas": {"edible": True, "energy": 3.0, "nutrients": {"C": 0.1}},
    "mushrooms": {"edible": True},
    "flax": {"processable": True},
    "tea_leaves": {},
    "ipecac_root": {},
    "cannabis_bud": {},
}
WATER_CONTAINERS = {"pot": {"size": 2.0}}


# --- Helper ---
def flatten_behavior_names(item: Union[Habit, Behavior]) -> Set[str]:
    names = set()
    if isinstance(item, Behavior):
        names.add(item.name)
    elif isinstance(item, Habit):
        for sub_item in item.sequence:
            names.update(flatten_behavior_names(sub_item))  # Recurse
    return names


# === Agent Class V9.0 (Primary Definition) ===
class AgentF:
    """Enhanced agent with trait system integration (V9.0 Merged)."""

    def __init__(self, rng: GameRNG, home: Home, base_logger=None):
        self.rng = rng
        self.home = home
        self.energy = 16.0
        self.health = 100.0
        self.thirst = 0.0
        self.hunger = 0.0
        self.has_drunk_today = False
        self.hours = 0.0
        self.day_length = 24.0
        self.skills = defaultdict(lambda: 1.0)
        self.skills.update(
            {
                "farm": 1.0,
                "forage": 1.0,
                "prep_fiber": 1.0,
                "cook": 1.0,
                "living": 1.0,
                "first_aid": 1.0,
            }
        )
        self.nutrients = defaultdict(float)
        self.critical_nutrients = {"C": 3.0, "P": 3.0}
        self.carried_containers: List[str] = []
        self.context: Dict[str, Any] = {"day": 0}
        self.conditions: Set[str] = set()
        self.base_logger = base_logger if base_logger else self._get_dummy_logger()
        self.task_stats = defaultdict(
            lambda: {
                "uses": 0,
                "total_gain": 0.0,
                "total_energy": 0.0,
                "total_time": 0.0,
            }
        )

        trait_variance = 0.5  # Example variance - consider making this configurable
        self.traits = TraitProfile(
            endurance=max(
                0.1, min(2.0, 1.0 + rng.get_float(-trait_variance, trait_variance))
            ),
            ingenuity=max(
                0.1, min(2.0, 1.0 + rng.get_float(-trait_variance, trait_variance))
            ),
            perception=max(
                0.1, min(2.0, 1.0 + rng.get_float(-trait_variance, trait_variance))
            ),
            will=max(
                0.1, min(2.0, 1.0 + rng.get_float(-trait_variance, trait_variance))
            ),
            resonance=max(
                0.1, min(2.0, 1.0 + rng.get_float(-trait_variance, trait_variance))
            ),
        )
        self.fatigue = FatigueSystem(endurance=self.traits.endurance)
        self.illness = IllnessSystem(
            endurance=self.traits.endurance, will=self.traits.will
        )
        self.memory = ExperienceMemory(ingenuity=self.traits.ingenuity)
        self.self_concept = SelfConcept(resonance=self.traits.resonance)

        self.behaviors: Dict[str, Behavior] = self._initialize_behaviors()
        self.habits: List[Habit] = []
        self.daily_behavior_log: List[Tuple[Dict[str, Any] | None, str]] = []
        self.behavior_memory: deque[Tuple[Dict[str, Any] | None, str]] = deque(
            maxlen=250
        )
        self._behavior_dispatch_map: Dict[str, Callable[..., Any]] = (
            self._build_dispatch_map()
        )
        self.executed_habits_today: Dict[str, Tuple[bool, OutcomeResult | None]] = {}

        self.habit_reflect_min_count = max(2, int(5 - 2 * self.traits.ingenuity))
        self.habit_reflect_max_len = int(3 + self.traits.ingenuity)
        self.habit_prune_score_thr = 0.2
        self.habit_prune_inactive_days = int(20 + 10 * (2.0 - self.traits.resonance))

        self._seed_initial_habits()

    def _get_dummy_logger(self):
        class DummyLogger:
            def info(self, *args, **kwargs):
                print(f"INFO: {args} {kwargs}")

            def warning(self, *args, **kwargs):
                print(f"WARN: {args} {kwargs}")

            def error(self, *args, **kwargs):
                print(f"ERROR: {args} {kwargs}")

            def debug(self, *args, **kwargs):
                print(f"DEBUG: {args} {kwargs}")

        return DummyLogger()

    def _initialize_behaviors(self) -> Dict[str, Behavior]:
        # Using simplified impacts from above for consistency
        return {
            "fetch_water": Behavior(
                "fetch_water",
                self.fetch_water,
                impact={"fatigue": 5.0},
                est_energy_cost=1.2,
                est_time_cost=1.3,
                task_id="fetch_water",
                primary_skill="forage",
            ),
            "drink": Behavior(
                "drink",
                self.drink,
                impact={"fatigue": -1.0},
                est_energy_cost=0.1,
                est_time_cost=0.1,
                task_id="drink",
                primary_skill="living",
            ),
            "tend_field": Behavior(
                "tend_field",
                lambda a: a.tend_field(),
                impact={"fatigue": 7.0},
                est_energy_cost=2.0,
                est_time_cost=2.0,
                task_id="tend_field",
                primary_skill="farm",
            ),
            "harvest_field": Behavior(
                "harvest_field",
                lambda a: a.harvest_field(),
                impact={"fatigue": 8.0},
                est_energy_cost=3.0,
                est_time_cost=3.0,
                task_id="harvest_field",
                primary_skill="farm",
            ),
            "forage_mushrooms": Behavior(
                "forage_mushrooms",
                lambda a: a.forage_crop("mushrooms"),
                impact={"fatigue": 4.0},
                est_energy_cost=1.5,
                est_time_cost=1.0,
                task_id="forage_item",
                primary_skill="forage",
            ),
            "eat_peas": Behavior(
                "eat_peas",
                lambda a: a.eat_crop("peas"),
                impact={},
                est_energy_cost=0.0,
                est_time_cost=0.5,
                task_id="eat_food",
                primary_skill="living",
            ),
            "prepare_flax": Behavior(
                "prepare_flax",
                self.prepare_flax,
                impact={"fatigue": 3.0},
                est_energy_cost=1.0,
                est_time_cost=1.0,
                task_id="prep_fiber",
                primary_skill="prep_fiber",
            ),
            "boil_water": Behavior(
                "boil_water",
                self._behavior_boil_water,
                impact={"fatigue": 1.0},
                est_energy_cost=0.3,
                est_time_cost=0.2,
                task_id="boil_water",
                primary_skill="cook",
            ),
            "steep_leaves": Behavior(
                "steep_leaves",
                self._behavior_steep_leaves,
                impact={"fatigue": 0.5},
                est_energy_cost=0.1,
                est_time_cost=0.1,
                task_id="prep_drink",
                primary_skill="cook",
            ),
            "drink_tea": Behavior(
                "drink_tea",
                self._behavior_drink_tea,
                impact={"fatigue": -3.0},
                est_energy_cost=0.0,
                est_time_cost=0.1,
                task_id="drink",
                primary_skill="living",
            ),
            "use_ipecac": Behavior(
                "use_ipecac",
                self._behavior_use_ipecac,
                impact={"fatigue": 10.0},
                est_energy_cost=0.0,
                est_time_cost=0.3,
                task_id="use_medicine",
                primary_skill="first_aid",
            ),
            "use_cannabis": Behavior(
                "use_cannabis",
                self._behavior_use_cannabis,
                impact={"fatigue": -5.0},
                est_energy_cost=0.0,
                est_time_cost=0.5,
                task_id="use_drug",
                primary_skill="living",
            ),
        }

    def _build_dispatch_map(self) -> Dict[str, Callable[..., Any]]:
        return {n: b.fn for n, b in self.behaviors.items() if b.fn is not None}

    def _seed_initial_habits(self):
        def t_thirsty(a: "AgentF") -> bool:
            return a.thirst > 0.7

        def t_tea(a: "AgentF") -> bool:
            energy_threshold = 9.0 - 2.0 * (a.traits.endurance - 1.0)
            return (
                a.energy < energy_threshold
                and a.context.get("time_of_day") == 0
                and a.home.raw_inventory.get("tea_leaves", 0) > 0
                and a.thirst < 0.5
            )

        def t_poisoned(a: "AgentF") -> bool:
            return (
                a.illness.has_condition("poisoned")
                and a.home.raw_inventory.get("ipecac_root", 0) > 0
            )

        req = {
            "fetch_water",
            "drink",
            "boil_water",
            "steep_leaves",
            "drink_tea",
            "use_ipecac",
        }
        if req.issubset(self.behaviors.keys()):
            habits_to_add = [
                Habit(
                    "CritThirsty",
                    [self.behaviors["fetch_water"], self.behaviors["drink"]],
                    t_thirsty,
                    10.0,
                    0,
                ),
                Habit(
                    "MakeTea",
                    [
                        self.behaviors["fetch_water"],
                        self.behaviors["boil_water"],
                        self.behaviors["steep_leaves"],
                        self.behaviors["drink_tea"],
                    ],
                    t_tea,
                    5.0,
                    0,
                ),
                Habit(
                    "UseIpecacWhenPoisoned",
                    [self.behaviors["use_ipecac"]],
                    t_poisoned,
                    8.0,
                    0,
                ),
            ]
            self.habits.extend(habits_to_add)
        else:
            missing = req - self.behaviors.keys()
            self.base_logger.warning(
                "seed_habits_failed", reason="missing_behaviors", missing=list(missing)
            )

    def _behavior_boil_water(self) -> bool:
        b = self.behaviors["boil_water"]
        start_e, start_t = self.energy, self.hours
        success = False
        amount = 0
        water_needed = 1
        if self.home.water_storage >= water_needed:
            self.home.water_storage -= water_needed
            amount = 1
            self.hours += 0.2
            base_cost = 0.3
            ingenuity_mod = max(0.1, 1.0 - 0.2 * (self.traits.ingenuity - 1.0))
            energy_cost = base_cost * ingenuity_mod
            self.energy -= energy_cost
            self.fatigue.add_fatigue(amount=1.0, activity_intensity=0.5)
            self.home.raw_inventory["boiled_water"] = (
                self.home.raw_inventory.get("boiled_water", 0) + amount
            )
            success = True
        self._log_task(
            b.task_id,
            b.primary_skill,
            -(self.energy - start_e),
            self.hours - start_t,
            float(amount),
        )
        return success

    def prepare_flax(self) -> bool:
        return self._prepare_fiber_crop("flax", skill="prep_fiber")

    def _capture_state_snapshot(self) -> Dict[str, Any]:
        time_seg = self.context.get(
            "time_of_day", int(self.hours // (self.day_length / 3)) % 3
        )
        features = {
            "thirst_high": self.thirst > 0.7,
            "thirst_mod": 0.4 < self.thirst <= 0.7,
            "energy_low": self.energy < 8.0,
            "energy_crit": self.energy < 4.0,
            "time_morning": time_seg == 0,
            "time_afternoon": time_seg == 1,
            "health_low": self.health < 60.0,
            "health_crit": self.health < 30.0,
            "is_hungry": self.hunger > 0.6,
            "has_food": any(
                v > 0
                for k, v in self.home.raw_inventory.items()
                if CROPS.get(k, {}).get("edible")
            ),
            "has_water": self.home.water_storage > 0,
            "has_tea_leaves": self.home.raw_inventory.get("tea_leaves", 0) > 0,
            "has_ipecac": self.home.raw_inventory.get("ipecac_root", 0) > 0,
            "has_cannabis": self.home.raw_inventory.get("cannabis_bud", 0) > 0,
            "has_boiled_water": self.home.raw_inventory.get("boiled_water", 0) > 0,
            "has_prepared_tea": self.home.raw_inventory.get("prepared_tea", 0) > 0,
            "fatigue_high": self.fatigue.level > 70.0,
            "fatigue_mod": 40.0 < self.fatigue.level <= 70.0,
            "is_ill": bool(self.illness.active_conditions),
        }
        for c_name in self.illness.active_conditions:
            features[f"has_illness_{c_name}"] = True
        for c_name in self.conditions:
            features[f"has_condition_{c_name}"] = True
        return features

    def _estimate_habit_impact(self, habit: Habit) -> Dict[str, float]:
        base_impact = habit.estimate_impact(self)
        behavior_names = flatten_behavior_names(habit)
        if behavior_names and self.traits.ingenuity > 1.0:
            current_state = self._capture_state_snapshot()
            predicted_deltas = defaultdict(lambda: {"sum": 0.0, "count": 0})
            temp_state = current_state.copy()
            for behavior_name in behavior_names:  # Iterate over names
                predicted_changes = self.memory.predict_outcome(
                    behavior_name, temp_state
                )
                if predicted_changes:
                    for key, value in predicted_changes.items():
                        predicted_deltas[key]["sum"] += value
                        predicted_deltas[key]["count"] += 1
                        temp_state[key] = temp_state.get(key, 0) + value
            if predicted_deltas:
                memory_weight = max(
                    0, min(1, 0.2 + 0.5 * (self.traits.ingenuity - 1.0))
                )
                estimate_weight = 1.0 - memory_weight
                for key, data in predicted_deltas.items():
                    if data["count"] > 0:
                        avg_pred_delta = data["sum"] / data["count"]
                        delta_key = f"delta_{key}"
                        base_value = base_impact.get(delta_key, 0)  # Get base or 0
                        # Weighted average, handling cases where key only exists in one source
                        base_impact[delta_key] = (
                            base_value * estimate_weight
                            + avg_pred_delta * memory_weight
                        )

        # Add fatigue impact estimate if missing from memory prediction
        if "delta_fatigue" not in base_impact:
            est_fatigue = sum(
                b.impact.get("fatigue", 0)
                for b in habit.sequence
                if isinstance(b, Behavior)
            )
            if est_fatigue != 0:
                base_impact["delta_fatigue"] = est_fatigue

        # Adjust energy cost based on fatigue
        if "delta_energy" in base_impact and base_impact["delta_energy"] < 0:
            fatigue_factor = 1.0 + (self.fatigue.level / 100.0) * 0.5
            base_impact["delta_energy"] *= fatigue_factor
        return base_impact

    def _calculate_habit_utility(self, predicted_impact: Dict[str, float]) -> float:
        utility = 0.0
        W = {
            "thirst": 5.0,
            "energy": 2.5 * (1 + 0.5 * (2 - self.traits.endurance)),
            "health": 4.0,
            "hunger": 3.0,
            "time": -0.5,
            "e_cost": -1.0 * (1 + 0.3 * (2 - self.traits.endurance)),
            "food": 0.5,
            "water": 0.3,
            "fatigue": -0.1 * (self.fatigue.level / 50.0),
            "fatigue_recovery": 0.2 * (1 + (2 - self.traits.endurance)),
        }

        utility += (
            max(0, -predicted_impact.get("delta_thirst", 0)) * self.thirst * W["thirst"]
        )
        utility += (
            max(0, predicted_impact.get("delta_energy", 0))
            * (1 - self.energy / 16.0)
            * W["energy"]
        )
        utility += (
            max(0, predicted_impact.get("delta_health", 0))
            * (1 - self.health / 100.0)
            * W["health"]
        )
        utility += (
            max(0, -predicted_impact.get("delta_hunger", 0)) * self.hunger * W["hunger"]
        )
        food_gain = sum(
            v
            for k, v in predicted_impact.items()
            if k.startswith("delta_item:")
            and CROPS.get(k.split(":")[1], {}).get("edible")
            and v > 0
        )
        utility += food_gain * W["food"] * (1 + self.hunger * 2)
        water_gain = predicted_impact.get("delta_home:water_storage", 0)
        utility += max(0, water_gain) * W["water"] * (1 + self.thirst * 2)
        utility += predicted_impact.get("delta_time", 0) * W["time"]
        utility += (
            max(0, -predicted_impact.get("delta_energy", 0))
            * W["e_cost"]
            * (1 + (1 - self.energy / 16.0))
        )
        delta_fatigue = predicted_impact.get("delta_fatigue", 0)
        if delta_fatigue > 0:
            utility += delta_fatigue * W["fatigue"]
        elif delta_fatigue < 0:
            utility += abs(delta_fatigue) * W["fatigue_recovery"]

        if (
            self.illness.has_condition("poisoned")
            and predicted_impact.get("delta_condition_poisoned", 0) < 0
        ):
            utility += 50.0

        if self.traits.will > 1.0:
            long_term = sum(
                v
                for k, v in predicted_impact.items()
                if (
                    k.startswith("delta_skill")
                    or k.startswith("delta_item")
                    or k == "delta_home:water_storage"
                )
                and v > 0
            )
            if long_term > 0:
                utility += min(5.0, long_term * 0.3 * (self.traits.will - 0.5))
        return utility

    def _promote_composite_sequence(
        self,
        seq: Tuple[str, ...],
        trigger_features: Dict[str, Any],
        freq_score: float,
        current_day: int,
        name: str,
    ) -> bool:
        sequence_objects = []
        for b_name in seq:
            behavior_obj = self.behaviors.get(b_name)
            if behavior_obj is None:
                self.base_logger.warning(
                    "habit_promotion_failed",
                    reason="missing_behavior",
                    behavior_name=b_name,
                    habit_name=name,
                )
                return False
            sequence_objects.append(behavior_obj)
        if not sequence_objects:
            return False

        base_score = 1.0 + 0.05 * len(seq) + 0.02 * freq_score
        ingenuity_bonus = 0.1 * self.traits.ingenuity * freq_score
        resonance_factor = 0.8 + 0.4 * self.traits.resonance
        score = base_score * resonance_factor + ingenuity_bonus

        new_habit = Habit(name, sequence_objects, trigger_features, score, current_day)
        self.habits.append(new_habit)
        self.base_logger.info(
            "habit_promoted",
            habit_name=name,
            score=round(score, 2),
            trigger_count=len(trigger_features),
        )
        return True

    def decide_day_plan(self):
        self.executed_habits_today.clear()
        planned_behaviors = set()
        MAX_PLANNING_ITERATIONS = int(10 + 5 * self.traits.perception)

        for iteration in range(MAX_PLANNING_ITERATIONS):
            if self.hours >= self.day_length or self.energy <= 0.5:
                break
            potential_choices = []
            current_state_features = self._capture_state_snapshot()

            for habit in self.habits:
                if habit.name in self.executed_habits_today:
                    continue

                habit_triggered = False
                perception_bonus = 0.2 * (self.traits.perception - 1.0)
                if isinstance(habit.trigger, dict):
                    trigger_features = habit.trigger
                    total_features = len(trigger_features)
                    if total_features == 0:
                        matched_features = 0
                    else:
                        matched_features = sum(
                            1
                            for k, v in trigger_features.items()
                            if current_state_features.get(k) == v
                        )

                    if matched_features == total_features:
                        habit_triggered = True
                    elif (
                        total_features > 0
                        and matched_features > 0
                        and self.traits.perception > 1.2
                    ):  # Ensure total_features > 0
                        trigger_threshold = max(0.1, 0.7 - perception_bonus)
                        if (matched_features / total_features) >= trigger_threshold:
                            habit_triggered = True
                elif callable(habit.trigger):
                    try:
                        habit_triggered = habit.is_triggered(self)
                    except Exception as e:
                        self.base_logger.error(
                            "habit_trigger_error", habit_name=habit.name, error=str(e)
                        )
                        habit_triggered = False
                else:
                    self.base_logger.warning(
                        "invalid_habit_trigger_type",
                        habit_name=habit.name,
                        trigger_type=type(habit.trigger),
                    )
                    continue
                if not habit_triggered:
                    continue

                habit_b_names = set(flatten_behavior_names(habit))
                if not habit_b_names.isdisjoint(planned_behaviors):
                    continue

                predicted_impact = self._estimate_habit_impact(habit)
                est_energy_cost = max(0, -predicted_impact.get("delta_energy", 0))
                est_time_cost = max(0, predicted_impact.get("delta_time", 0))

                if (
                    self.energy > est_energy_cost
                    and (self.day_length - self.hours) > est_time_cost
                ):
                    utility = self._calculate_habit_utility(predicted_impact)
                    identity_alignment = 0.0
                    num_behaviors = len(habit_b_names)
                    if num_behaviors > 0:
                        for b_name in habit_b_names:
                            identity_alignment += (
                                self.self_concept.get_identity_affinity(b_name)
                            )
                        identity_alignment /= num_behaviors
                    identity_factor = 0.2 * self.traits.resonance * identity_alignment
                    adjusted_utility = utility + identity_factor
                    final_utility = adjusted_utility + habit.score * 0.01
                    potential_choices.append((final_utility, identity_alignment, habit))

            if not potential_choices:
                break

            potential_choices.sort(key=lambda x: x[0], reverse=True)
            best_utility_score, identity_alignment, best_habit = potential_choices[0]
            utility_threshold = 0.1 - 0.05 * (self.traits.will - 1.0)

            if best_habit and best_utility_score > utility_threshold:
                state_before = self._get_current_state_dict()
                dissonance = 0.0
                habit_b_names_d = flatten_behavior_names(best_habit)
                if habit_b_names_d:
                    # Avoid division by zero if len is 0 (shouldn't happen if habit_b_names is checked)
                    if len(habit_b_names_d) > 0:
                        for b_name in habit_b_names_d:
                            dissonance += self.self_concept.calculate_dissonance(b_name)
                        dissonance /= len(habit_b_names_d)
                        if dissonance > 0:
                            will_factor = 0.7 + 0.3 * self.traits.will
                            dissonance *= max(0, 2.0 - will_factor)
                        if dissonance > 0.5:
                            self.energy -= dissonance * 0.5

                completed = best_habit.execute(self)  # Executes sequence
                state_after = self._get_current_state_dict()
                outcome_res = self._evaluate_outcome(state_before, state_after)

                self.executed_habits_today[best_habit.name] = (completed, outcome_res)
                perceived_value = outcome_res.evaluation if outcome_res else 0
                habit_b_names_u = flatten_behavior_names(
                    best_habit
                )  # Recalculate or store
                num_behaviors_u = len(habit_b_names_u)
                if num_behaviors_u > 0:
                    value_per_b = perceived_value / num_behaviors_u
                    day = self.context.get("day", 0)
                    for b_name in habit_b_names_u:
                        self.self_concept.update_from_behavior(
                            b_name, completed, value_per_b
                        )
                        self.memory.add_memory(
                            b_name, state_before, state_after, day, value_per_b
                        )

                planned_behaviors.update(habit_b_names_u)
            else:
                break

        if self.hours < self.day_length - 6 and self.energy > 7:
            current_features = self._capture_state_snapshot()
            behaviors_done_in_habits = planned_behaviors
            action_taken = False
            if "forage_mushrooms" not in behaviors_done_in_habits and self.energy > 10:
                self.daily_behavior_log.append((current_features, "forage_mushrooms"))
                self.base_logger.debug(
                    "executing_fallback_action", action="forage_mushrooms"
                )
                self.forage_crop("mushrooms")
                action_taken = True

    def end_of_day_update(self):
        self.reflect_on_day()

        if not self.has_drunk_today:
            self.thirst = min(1.0, self.thirst + 0.3)
        self.has_drunk_today = False
        self.hours = 0.0
        for k in self.nutrients:
            self.nutrients[k] *= 0.95
        hunger_decay = 0.9 * (0.8 + 0.4 * self.traits.endurance)
        self.hunger = max(0.0, self.hunger * hunger_decay)
        self.update_health()

        quality_factor = 1.0
        if self.illness.active_conditions:
            quality_factor *= 0.7
        nutrition_bonus = (
            0.2
            if all(self.nutrients[n] >= v for n, v in self.critical_nutrients.items())
            else 0.0
        )
        self.fatigue.recover(
            quality_rest=quality_factor, nutrition_bonus=nutrition_bonus
        )

        resolved_conditions = self.illness.daily_update()
        if resolved_conditions:
            self.base_logger.info(
                "illness_resolved",
                conditions=resolved_conditions,
                agent_id=getattr(self, "id", "unknown"),
            )
            for condition in resolved_conditions:
                self.conditions.discard(condition)

        base_recovery = 8.0
        health_bonus = (self.health / 100.0) * 4.0 * (0.8 + 0.4 * self.traits.endurance)
        fatigue_penalty = 1.0 - (self.fatigue.level / 100.0) * 0.5
        illness_mods = self.illness.get_performance_modifiers()
        illness_energy_mod = illness_mods.get("energy", 1.0)
        total_recovery = (
            (base_recovery + health_bonus + nutrition_bonus)
            * fatigue_penalty
            * illness_energy_mod
        )
        self.energy = min(16.0, self.energy + total_recovery)

        self.conditions.discard("vomited")
        self.conditions.discard("euphoric")
        self.conditions.discard("hungry")

    def reflect_on_day(self):
        """Process daily behavior log to reinforce/prune habits."""
        self.base_logger.debug("reflect_on_day starting...")
        day = self.context.get("day", 0)
        # --- Habit Pruning ---
        inactive_threshold = day - self.habit_prune_inactive_days
        original_habit_count = len(self.habits)
        self.habits = [
            h
            for h in self.habits
            if h.score > self.habit_prune_score_thr
            or h.last_used_day >= inactive_threshold
        ]
        pruned_count = original_habit_count - len(self.habits)
        if pruned_count > 0:
            self.base_logger.info("habits_pruned", count=pruned_count)

        # --- Habit Score Decay ---
        for habit in self.habits:
            # Decay score slightly each day it wasn't used
            if habit.last_used_day < day:
                habit.score *= 0.98  # Daily decay factor

        # --- Learn New Habits from Behavior Sequence ---
        # Analyze self.daily_behavior_log for frequent sequences
        if not self.daily_behavior_log:
            return  # Nothing to learn from

        # Use behavior_memory which persists across days might be better?
        # Let's analyze today's log for now.
        behavior_names_today = [name for _, name in self.daily_behavior_log]
        if len(behavior_names_today) < self.habit_reflect_min_count:
            return  # Not enough actions

        sequences = defaultdict(lambda: {"count": 0, "triggers": []})
        for length in range(
            self.habit_reflect_min_count,
            min(self.habit_reflect_max_len + 1, len(behavior_names_today) + 1),
        ):
            for i in range(len(behavior_names_today) - length + 1):
                seq_tuple = tuple(behavior_names_today[i : i + length])
                # Find trigger state just before sequence started
                trigger_state, _ = self.daily_behavior_log[
                    i
                ]  # State before first action in seq
                sequences[seq_tuple]["count"] += 1
                if trigger_state:  # Store potential trigger features
                    sequences[seq_tuple]["triggers"].append(trigger_state)

        # Promote frequent sequences
        promoted_count = 0
        existing_habit_names = {h.name for h in self.habits}
        for seq, data in sequences.items():
            count = data["count"]
            if count < 2:
                continue  # Require at least 2 occurrences today? Or lower?

            # Create a simple name (can be improved)
            habit_name = f"Habit_{'_'.join(seq)}_{day}"
            if habit_name in existing_habit_names:
                continue  # Avoid duplicates per day

            # Determine common trigger features (simple approach: intersection of keys?)
            # This needs refinement - maybe use most common state?
            common_triggers = {}
            if data["triggers"]:
                # Find features present in *most* trigger states? Average values?
                # Simplistic: use features from the first trigger state for now
                common_triggers = data["triggers"][0]

            # Promote if score (based on frequency?) is high enough
            # Base promotion score on frequency today
            freq_score = count * length  # Simple score metric
            if freq_score > 3:  # Threshold for promotion
                if self._promote_composite_sequence(
                    seq, common_triggers, freq_score, day, habit_name
                ):
                    promoted_count += 1
                    existing_habit_names.add(
                        habit_name
                    )  # Add to set to avoid duplicates within loop

        if promoted_count > 0:
            self.base_logger.info("new_habits_learned", count=promoted_count)

        # Clear log for next day
        self.daily_behavior_log.clear()

    def _get_current_state_dict(self) -> Dict[str, Any]:
        # Return comprehensive state including inventory etc. for evaluation
        state = {
            "energy": self.energy,
            "health": self.health,
            "thirst": self.thirst,
            "hunger": self.hunger,
            "fatigue": self.fatigue.level,
            "hours": self.hours,
            "day": self.context.get("day", 0),
            **{f"skill_{k}": v for k, v in self.skills.items()},
            **{f"nutrient_{k}": v for k, v in self.nutrients.items()},
            **{f"item_{k}": v for k, v in self.home.raw_inventory.items()},
            "home_water": self.home.water_storage,
            **{
                f"illness_{k}": 1 for k in self.illness.active_conditions
            },  # Indicate presence
            **{f"condition_{k}": 1 for k in self.conditions},
        }
        return state

    def _evaluate_outcome(
        self, state_before: Dict, state_after: Dict
    ) -> OutcomeResult | None:
        if not state_before or not state_after:
            return None
        outcome = OutcomeResult()
        # Simple delta-based evaluation - tune weights
        eval_score = 0.0
        weights = {
            "health": 1.0,
            "energy": 0.5,
            "thirst": -0.8,
            "hunger": -0.7,
            "fatigue": -0.3,
        }
        for key, weight in weights.items():
            delta = state_after.get(key, 0) - state_before.get(key, 0)
            # Normalize delta? E.g., delta / typical_range?
            # Simple weighted sum for now
            eval_score += delta * weight

        # Consider resource changes?
        water_delta = state_after.get("home_water", 0) - state_before.get(
            "home_water", 0
        )
        eval_score += water_delta * 0.2  # Value gaining water

        # Food gain (crude sum over all items)
        food_after = sum(
            v
            for k, v in state_after.items()
            if k.startswith("item_")
            and CROPS.get(k.split("item_")[1], {}).get("edible")
        )
        food_before = sum(
            v
            for k, v in state_before.items()
            if k.startswith("item_")
            and CROPS.get(k.split("item_")[1], {}).get("edible")
        )
        eval_score += (food_after - food_before) * 0.3  # Value gaining food

        outcome.evaluation = eval_score
        return outcome

    def forage_crop(self, crop_name: str) -> bool:
        b_name = f"forage_{crop_name}"
        b = self.behaviors.get(b_name, self.behaviors.get("forage_item"))
        if not b:
            b = Behavior(b_name, None, task_id="forage_item", primary_skill="forage")
        start_e, start_t = self.energy, self.hours
        success = False
        gathered = 0
        energy_cost = 0
        if crop_name in CROPS:
            base_amount = self.rng.get_int(1, 4)
            perception_bonus = 1.0 + (self.traits.perception - 1.0)
            efficiency_bonus = self.action_efficiency("forage")
            gathered = int(base_amount * perception_bonus * efficiency_bonus)
            if gathered > 0:
                self.home.raw_inventory[crop_name] += gathered
                base_energy_cost = 1.5
                energy_cost = base_energy_cost / (0.7 + 0.3 * self.traits.endurance)
                self.energy -= energy_cost
                self.fatigue.add_fatigue(amount=4.0, activity_intensity=0.8)
                self.hours += 1.0
                skill_factor = 0.02 * (0.7 + 0.3 * self.traits.perception)
                self.skills["forage"] += skill_factor
                success = True
            else:
                energy_cost = 0
                self.hours += 1.0
                self.fatigue.add_fatigue(amount=1.0, activity_intensity=0.5)
        self._log_task(
            b.task_id,
            b.primary_skill,
            energy_cost if success else 0,
            self.hours - start_t,
            float(gathered),
        )
        return success

    def _prepare_fiber_crop(self, crop_name: str, skill: str = "prep_fiber") -> bool:
        b_name = f"prepare_{crop_name}"
        b = self.behaviors.get(b_name)
        if not b:
            b = Behavior(b_name, None, task_id="prep_fiber", primary_skill=skill)
        start_e, start_t = self.energy, self.hours
        success = False
        used = 0
        energy_cost = 0
        if (
            CROPS.get(crop_name, {}).get("processable")
            and self.home.raw_inventory.get(crop_name, 0) > 0
        ):
            base = 3.0
            ingenuity_factor = 0.7 + 0.5 * self.traits.ingenuity
            efficiency_factor = self.action_efficiency(skill)
            pot = int(base * efficiency_factor * ingenuity_factor)
            avail = self.home.raw_inventory[crop_name]
            used = min(avail, pot)
            if used > 0:
                self.home.raw_inventory[crop_name] -= used
                self.home.fiber[crop_name] = self.home.fiber.get(crop_name, 0) + used
                base_energy_cost = 1.0 * (used / base if base > 0 else 1.0)
                energy_cost = base_energy_cost / (0.7 + 0.3 * self.traits.endurance)
                self.energy -= energy_cost
                self.fatigue.add_fatigue(
                    amount=3.0, activity_intensity=used / base if base > 0 else 0.5
                )
                self.hours += 1.0
                skill_factor = 0.015 * (0.7 + 0.3 * self.traits.ingenuity)
                self.skills[skill] += skill_factor * (used / base if base > 0 else 1.0)
                success = True
            else:
                energy_cost = 0.2
                self.energy -= energy_cost
                self.hours += 0.5
                self.fatigue.add_fatigue(amount=0.5, activity_intensity=0.3)
        self._log_task(
            b.task_id,
            b.primary_skill,
            energy_cost if used > 0 else 0.2,
            self.hours - start_t,
            float(used),
        )
        return success

    def harvest_field(self, field_index: int = 0) -> bool:
        b = self.behaviors["harvest_field"]
        start_e, start_t = self.energy, self.hours
        success = False
        total_items = 0
        n_ready = 0
        energy_cost = 0
        if field_index < len(self.home.fields):
            field = self.home.fields[field_index]
            weather = self.context.get("weather")
            day = self.context.get("day", 0)
            if weather:
                cal = Calendar()
                cal.day = day
                n_ready = int(np.sum(field.plots["status"] == PLOT_STATUS_MAP["ready"]))
                if n_ready > 0:
                    if self.traits.perception > 1.2:
                        r_idx = np.where(
                            field.plots["status"] == PLOT_STATUS_MAP["ready"]
                        )[0]
                        if r_idx.size > 0:
                            perc_bonus = 5.0 * (self.traits.perception - 1.0)
                            field.plots["health"][r_idx] = np.minimum(
                                100.0, field.plots["health"][r_idx] + perc_bonus
                            )
                    summary = field.harvest_ready_plots(weather, cal)
                    if summary:
                        for c, a in summary.items():
                            self.home.raw_inventory[c] += a
                            total_items += a
                        base_energy = 1.0 * n_ready
                        energy_cost = base_energy / (0.7 + 0.3 * self.traits.endurance)
                        self.energy -= energy_cost
                        self.fatigue.add_fatigue(
                            amount=8.0,
                            activity_intensity=n_ready / 3.0 if n_ready > 0 else 0.5,
                        )
                        time_cost = 1.5 * n_ready
                        self.hours += time_cost
                        skill_factor = 0.03 * (0.7 + 0.3 * self.traits.ingenuity)
                        self.skills["farm"] += skill_factor * n_ready
                        success = True
                    else:
                        success = False
                        energy_cost = 0
                else:
                    success = True  # No plots ready is not a failure
        energy_spent = -(self.energy - start_e)  # Calculate actual spent energy
        self._log_task(
            b.task_id,
            b.primary_skill,
            energy_spent,
            self.hours - start_t,
            float(total_items),
        )
        return success

    def fetch_water(self) -> bool:
        b = self.behaviors["fetch_water"]
        start_e, start_t = self.energy, self.hours
        success = False
        amount = 0
        energy_cost = 0
        containers_to_use = (
            self.carried_containers
            if self.carried_containers
            else list(self.home.containers.keys())
        )
        if containers_to_use:
            cap = sum(
                WATER_CONTAINERS.get(c, {}).get("size", 0) for c in containers_to_use
            )
            amount = cap
            if amount > 0:
                self.home.water_storage += amount
                base_energy_cost = 1.0 + (cap * 0.2)
                base_time_cost = 1.0 + (cap * 0.3)
                energy_cost = base_energy_cost / (0.7 + 0.3 * self.traits.endurance)
                self.energy -= energy_cost
                self.fatigue.add_fatigue(
                    amount=5.0, activity_intensity=cap / 2.0 if cap > 0 else 0.5
                )
                self.hours += base_time_cost
                skill_gain = 0.01 * (0.7 + 0.3 * self.traits.perception)
                self.skills["forage"] += skill_gain
                success = True
        self._log_task(
            b.task_id,
            b.primary_skill,
            energy_cost if success else 0,
            self.hours - start_t,
            float(amount),
        )
        return success

    def _behavior_steep_leaves(self) -> bool:
        b = self.behaviors["steep_leaves"]
        start_e, start_t = self.energy, self.hours
        success = False
        amount = 0
        if (
            self.home.raw_inventory.get("boiled_water", 0) >= 1
            and self.home.raw_inventory.get("tea_leaves", 0) >= 1
        ):
            amount = 1
            self.hours += 0.1
            base_energy_cost = 0.1
            perception_modifier = max(0.1, 1.0 - 0.2 * (self.traits.perception - 1.0))
            energy_cost = base_energy_cost * perception_modifier
            self.energy -= energy_cost
            self.home.raw_inventory["prepared_tea"] = (
                self.home.raw_inventory.get("prepared_tea", 0) + amount
            )
            self.home.raw_inventory["boiled_water"] -= 1
            self.home.raw_inventory["tea_leaves"] -= 1
            self.fatigue.add_fatigue(amount=0.5, activity_intensity=0.2)
            success = True
        self._log_task(
            b.task_id,
            b.primary_skill,
            -(self.energy - start_e),
            self.hours - start_t,
            float(amount),
        )
        return success

    def _behavior_drink_tea(self) -> bool:
        b = self.behaviors["drink_tea"]
        start_e, start_t = self.energy, self.hours
        success = False
        amount = 0
        if self.home.raw_inventory.get("prepared_tea", 0) <= 0:
            self._log_task(b.task_id, b.primary_skill, 0, 0, 0)
            return False
        amount = 1
        self.home.raw_inventory["prepared_tea"] -= 1
        self.thirst = max(0.0, self.thirst - 0.3)
        base_energy_boost = 1.0
        resonance_modifier = 0.8 + 0.4 * self.traits.resonance
        self.energy = min(16.0, self.energy + base_energy_boost * resonance_modifier)
        self.fatigue.level = max(0, self.fatigue.level - 3.0)
        self.hours += 0.1
        success = True
        self._log_task(
            b.task_id,
            b.primary_skill,
            self.energy - start_e,
            self.hours - start_t,
            float(amount),
        )
        return success

    def _behavior_use_ipecac(self) -> bool:
        b = self.behaviors["use_ipecac"]
        start_e, start_t = self.energy, self.hours
        success = False
        amount = 0
        if self.home.raw_inventory.get("ipecac_root", 0) <= 0:
            self._log_task(b.task_id, b.primary_skill, 0, 0, 0)
            return False
        amount = 1
        self.home.raw_inventory["ipecac_root"] -= 1
        success = True
        self.hours += 0.3
        base_energy_cost = 1.5
        energy_cost = base_energy_cost / (0.8 + 0.2 * self.traits.will)
        self.energy -= energy_cost
        base_health_impact = -1.0
        perception_modifier = 1.0 - 0.5 * (self.traits.perception - 1.0)
        health_impact = base_health_impact * max(0.5, perception_modifier)
        self.health = max(0, self.health + health_impact)
        self.fatigue.add_fatigue(amount=10.0, activity_intensity=1.2)
        self.thirst = min(1.0, self.thirst + 0.2)
        self.conditions.add("vomited")
        cured_poison = False
        if self.illness.has_condition("poisoned"):
            self.illness.active_conditions.pop("poisoned", None)
            cured_poison = True
            skill_factor = 0.05 * (0.7 + 0.3 * self.traits.ingenuity)
            self.skills["first_aid"] = self.skills.get("first_aid", 1.0) + skill_factor
        outcome_value = 1.0 if cured_poison else 0.0
        self._log_task(
            b.task_id,
            b.primary_skill,
            -(self.energy - start_e),
            self.hours - start_t,
            outcome_value,
        )
        return success

    def _behavior_use_cannabis(self) -> bool:
        b = self.behaviors["use_cannabis"]
        start_e, start_t = self.energy, self.hours
        success = False
        amount = 0
        if self.home.raw_inventory.get("cannabis_bud", 0) <= 0:
            self._log_task(b.task_id, b.primary_skill, 0, 0, 0)
            return False
        amount = 1
        self.home.raw_inventory["cannabis_bud"] -= 1
        success = True
        self.hours += 0.5
        base_energy_effect = 0.5
        resonance_modifier = 0.8 + 0.4 * self.traits.resonance
        self.energy += base_energy_effect * resonance_modifier
        self.conditions.add("euphoric")
        self.conditions.add("hungry")
        base_hunger_increase = 0.6
        will_modifier = 1.0 - 0.2 * (self.traits.will - 1.0)
        self.hunger = min(1.0, self.hunger + base_hunger_increase * will_modifier)
        self.fatigue.level = max(0, self.fatigue.level - 5.0)
        self._log_task(
            b.task_id,
            b.primary_skill,
            self.energy - start_e,
            self.hours - start_t,
            float(amount),
        )
        return success

    def action_efficiency(self, skill: str) -> float:
        base_efficiency = max(0.1, self.energy / 16.0) * self.skills.get(
            skill, 1.0
        )  # Use .get
        fatigue_mod = self.fatigue.get_performance_modifier()
        illness_mods = self.illness.get_performance_modifiers()
        illness_efficiency_mod = illness_mods.get("efficiency", 1.0)
        trait_modifier = 1.0
        if skill in ["farm", "harvest"]:
            trait_modifier *= 0.8 + 0.4 * self.traits.endurance
        elif skill in ["forage", "hunt"]:
            trait_modifier *= 0.8 + 0.4 * self.traits.perception
        elif skill in ["cook", "prep_fiber", "craft"]:
            trait_modifier *= 0.8 + 0.4 * self.traits.ingenuity
        elif skill in ["first_aid", "medicine"]:
            trait_modifier *= (
                0.8 + 0.3 * self.traits.ingenuity + 0.2 * self.traits.perception
            )
        will_floor = 0.2 + 0.3 * self.traits.will
        combined = (
            base_efficiency * fatigue_mod * illness_efficiency_mod * trait_modifier
        )
        return max(will_floor, combined)

    def update_health(self):
        health_change = 0.0
        for nutrient, threshold in self.critical_nutrients.items():
            if self.nutrients[nutrient] < threshold:
                health_change -= 1.0
        if self.health < 100 and all(
            self.nutrients[n] >= v for n, v in self.critical_nutrients.items()
        ):
            recovery_bonus = 0.5 * (0.7 + 0.5 * self.traits.endurance)
            health_change += recovery_bonus
        if self.thirst > 0.8:
            health_change -= 2.0
        if self.hunger > 0.9:
            health_change -= 1.0
        illness_severity = self.illness.get_total_severity()
        if illness_severity > 0:
            health_change -= illness_severity * 1.5
        if self.fatigue.level > 80:
            health_change -= (self.fatigue.level - 80) / 10.0
        if health_change < 0:
            will_buffer = min(abs(health_change) * 0.3, 0.3 * self.traits.will)
            health_change += will_buffer
        self.health = max(0, min(100, self.health + health_change))
        if self.health < 50:
            self.energy *= 0.8 + 0.1 * self.traits.will

    def update_context(self, season: str, weather: "Weather", day: int):
        self.context["season"] = season
        self.context["weather"] = weather
        self.context["day"] = day
        self.context["time_of_day"] = int(self.hours // (self.day_length / 3)) % 3

    def _log_task(
        self,
        task_id: str,
        skill: str,
        energy_cost: float,
        time_cost: float,
        amount: float = 1.0,
    ):
        skill = skill or "unknown_skill"
        task_id = task_id or "unknown_task"
        key = f"{skill}:{task_id}"
        s = self.task_stats[key]
        s["uses"] += 1
        s["total_gain"] += amount
        s["total_energy"] += max(0, energy_cost)
        s["total_time"] += max(0, time_cost)

    def drink(self) -> bool:
        b = self.behaviors["drink"]
        start_e, start_t = self.energy, self.hours
        success = False
        amount = 0
        if self.home.water_storage > 0:
            amount = 1
            self.home.water_storage -= 1
            self.thirst = 0.0
            self.has_drunk_today = True
            self.hours += b.est_time_cost
            self.energy -= b.est_energy_cost
            success = True
            # Fatigue effect handled via impact dict now
            # self.fatigue.level = max(0, self.fatigue.level - b.impact.get('fatigue', 0))
        self._log_task(
            b.task_id,
            b.primary_skill or "living",
            -(self.energy - start_e),
            self.hours - start_t,
            float(amount),
        )
        return success

    def eat_crop(self, crop_name: str, amount: int = 1) -> bool:
        b_name = f"eat_{crop_name}"
        b = self.behaviors.get(b_name)
        data = CROPS.get(crop_name)
        if not data or not data.get("edible", False):
            self._log_task("eat_food", "living", 0, 0, 0)
            return False  # Use default task/skill if behavior missing
        if not b:  # Create placeholder if needed
            impact = {
                "energy": data.get("energy", 1.0) * amount,
                "hunger": -0.4 * amount,
                f"item:{crop_name}": -amount,
            }
            b = Behavior(
                b_name, None, impact=impact, task_id="eat_food", primary_skill="living"
            )

        start_e, start_t = self.energy, self.hours
        success = False
        eaten_amount = 0
        store_key = "cooked_food" if data.get("cooked") else "raw_inventory"
        store = getattr(self.home, store_key, None)
        if store is None:
            self._log_task(b.task_id, b.primary_skill, 0, 0, 0)
            return False

        available = store.get(crop_name, 0)
        if available >= amount:
            eaten_amount = amount
            store[crop_name] -= eaten_amount
            base_energy_gain = data.get("energy", 0) * eaten_amount
            base_hunger_reduction = 0.4 * eaten_amount
            energy_factor = 1.0
            hunger_factor = 1.0
            if self.illness.active_conditions:
                energy_factor = 0.6 + 0.4 * self.traits.will
                hunger_factor = 0.7 + 0.3 * self.traits.will
            self.energy = min(16.0, self.energy + base_energy_gain * energy_factor)
            self.hunger = max(0.0, self.hunger - base_hunger_reduction * hunger_factor)
            nutrient_factor = 0.9 + 0.2 * self.traits.perception
            for k, v in data.get("nutrients", {}).items():
                self.nutrients[k] += v * eaten_amount * nutrient_factor
            time_cost = 0.5 * eaten_amount
            self.hours += time_cost
            success = True
        self._log_task(
            b.task_id,
            b.primary_skill,
            self.energy - start_e,
            self.hours - start_t,
            float(eaten_amount),
        )
        return success

    def tend_field(self, field_index: int = 0, max_plots: int = 5) -> bool:
        b = self.behaviors["tend_field"]
        start_e, start_t = self.energy, self.hours
        success = False
        tended_count = 0
        energy_cost = 0
        if field_index < len(self.home.fields):
            field = self.home.fields[field_index]
            g_idx = np.where(field.plots["status"] == PLOT_STATUS_MAP["growing"])[0]
            if g_idx.size > 0:
                perception_bonus = int(max_plots * (self.traits.perception - 1.0))
                effective_max = max(1, max_plots + perception_bonus)
                n_tend = min(effective_max, g_idx.size)
                # Ensure rng.sample exists and works as expected
                idx_aff = (
                    self.rng.sample(list(g_idx), k=n_tend)
                    if hasattr(self.rng, "sample")
                    else g_idx[:n_tend]
                )
                idx_aff = np.array(idx_aff)  # Convert back to numpy array if needed
                if idx_aff.size > 0:
                    tended_count = idx_aff.size
                    base_effectiveness = 10.0
                    trait_modifier = (
                        0.7
                        + 0.15 * self.traits.endurance
                        + 0.15 * self.traits.perception
                    )
                    h_inc = (
                        base_effectiveness
                        * self.action_efficiency("farm")
                        * trait_modifier
                    )
                    field.tend_plots(idx_aff, h_inc)
                    relative_effort = (n_tend / max_plots) if max_plots > 0 else 1.0
                    energy_cost = b.est_energy_cost * (0.7 + 0.3 * relative_effort)
                    energy_cost /= 0.7 + 0.3 * self.traits.endurance
                    self.energy -= energy_cost
                    self.fatigue.add_fatigue(
                        amount=7.0,
                        activity_intensity=n_tend / 3.0 if n_tend > 0 else 0.5,
                    )
                    self.hours += b.est_time_cost * (0.8 + 0.4 * relative_effort)
                    skill_factor = 0.02 * (0.7 + 0.3 * self.traits.ingenuity)
                    self.skills["farm"] += skill_factor * n_tend
                    success = True
        self._log_task(
            b.task_id,
            b.primary_skill,
            energy_cost if success else 0,
            self.hours - start_t,
            float(tended_count),
        )
        return success


# --- END OF PRIMARY AgentF CLASS ---

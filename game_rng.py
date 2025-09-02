from __future__ import annotations

"""Integrated GameRNG module.

This module provides a deterministic random number generator used across the
project.  The implementation is intentionally lightweight compared to the full
standalone library but preserves the public API that the rest of the codebase
expects.  Several fixes from the upstream project are included:

* ``MetricsCollector.start`` correctly sets ``self.running`` before starting the
  thread.
* ``GameRNG.get_distribution`` always assigns ``dist_func`` and handles extra
  kwargs for "bell" and "triangle" distributions.
* ``weighted_choice`` safely evicts cached CDFs without referencing an
  undefined variable.
* ``noise_1d`` and ``noise_2d`` compute their input variables outside of the
  early return path.
* ``loot_table`` defines ``items`` and ``weights`` before checking for early
  exit conditions.
"""

import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


@dataclass
class MetricsCollector:
    """Collect simple metrics in a background thread."""

    collection_interval: float = 1.0
    metrics: Dict[str, int] = field(
        default_factory=lambda: {
            "weighted_choices": 0,
            "weighted_samples_ares": 0,
            "integers_generated": 0,
            "floats_generated": 0,
            "shuffles": 0,
            "samples": 0,
            "batch_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    )
    stats: Dict[str, float] = field(
        default_factory=lambda: {
            "start_time": time.time(),
            "last_collection_time": time.time(),
            "operations_per_second": 0.0,
            "cache_hit_rate": 0.0,
        }
    )
    updates_queue: List[tuple[str, int]] = field(default_factory=list)
    running: bool = False
    collection_thread: Optional[threading.Thread] = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def start(self) -> None:
        if self.collection_thread is not None:
            return
        self.running = True
        self.collection_thread = threading.Thread(target=self._loop, daemon=True)
        self.collection_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.collection_thread is not None:
            self.collection_thread.join(timeout=1.0)
            self.collection_thread = None

    def update(self, metric: str, value: int = 1) -> None:
        if metric in self.metrics:
            with self.lock:
                self.updates_queue.append((metric, value))

    def _loop(self) -> None:
        while self.running:
            time.sleep(self.collection_interval)
            self._process_updates()
            self._update_stats()

    def _process_updates(self) -> None:
        with self.lock:
            while self.updates_queue:
                metric, value = self.updates_queue.pop(0)
                self.metrics[metric] += value

    def _update_stats(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.stats["last_collection_time"]
            if elapsed > 0:
                ops = sum(
                    v
                    for k, v in self.metrics.items()
                    if k.endswith("_generated") or k in {"shuffles", "samples"}
                )
                self.stats["operations_per_second"] = ops / elapsed
                total_cache = self.metrics["cache_hits"] + self.metrics["cache_misses"]
                self.stats["cache_hit_rate"] = (
                    self.metrics["cache_hits"] / total_cache if total_cache else 0.0
                )
            self.stats["last_collection_time"] = now

    def get_metrics(self) -> Dict[str, Any]:
        self._process_updates()
        self._update_stats()
        return {"metrics": dict(self.metrics), "stats": dict(self.stats)}


# ---------------------------------------------------------------------------
# RNG implementation
# ---------------------------------------------------------------------------


class Distribution(Enum):
    BELL = auto()
    TRIANGLE = auto()
    POWER = auto()
    EXPONENTIAL = auto()


class GameRNG:
    def __init__(
        self,
        seed: Optional[int] = None,
        metrics: bool = False,
        noise_seed: Optional[int] = None,
    ) -> None:
        self.initial_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.rng = np.random.default_rng(self.initial_seed)
        self.noise_seed = noise_seed if noise_seed is not None else self.initial_seed
        self.metrics_enabled = metrics
        self.metrics = MetricsCollector() if metrics else None
        if self.metrics:
            self.metrics.start()
        self.weighted_choice_cache: Dict[Any, np.ndarray] = {}
        self.weighted_choice_cache_size = 100

        self.distributions: Dict[str, Callable[..., float]] = {
            "bell": self._bell_dist,
            "triangle": self._triangle_dist,
            "power": self._power_dist,
            "exponential": self._exp_dist,
        }

    # ------------------------------------------------------------------
    # basic random helpers
    # ------------------------------------------------------------------
    def get_int(self, a: int, b: int) -> int:
        if a > b:
            raise ValueError("a <= b")
        val = int(self.rng.integers(a, b + 1))
        if self.metrics:
            self.metrics.update("integers_generated")
        return val

    def get_ints(self, a: int, b: int, count: int) -> List[int]:
        return [self.get_int(a, b) for _ in range(count)]

    def get_randrange(
        self, start: int, stop: Optional[int] = None, step: int = 1
    ) -> int:
        if step == 0:
            raise ValueError("step must not be zero")
        if stop is None:
            stop = start
            start = 0
        width = stop - start
        if step > 0:
            if width <= 0:
                raise ValueError("empty range")
            n = (width + step - 1) // step
        else:
            if width >= 0:
                raise ValueError("empty range")
            n = (abs(width) + abs(step) - 1) // abs(step)
        idx = self.get_int(0, n - 1)
        return start + idx * step

    def get_float(self, a: float = 0.0, b: float = 1.0) -> float:
        if a > b:
            raise ValueError("a <= b")
        val = float(self.rng.random())
        if self.metrics:
            self.metrics.update("floats_generated")
        return a + (b - a) * val

    def get_floats(self, a: float, b: float, count: int) -> List[float]:
        return [self.get_float(a, b) for _ in range(count)]

    # ------------------------------------------------------------------
    # distributions
    # ------------------------------------------------------------------
    def _bell_dist(self) -> float:
        return (sum(self.get_float() for _ in range(4)) - 2.0) * 0.5

    def _triangle_dist(self) -> float:
        return (self.get_float() + self.get_float()) * 0.5

    def _power_dist(self, power: float = 2.0) -> float:
        if power <= 0:
            raise ValueError("power must be positive")
        return self.get_float() ** power

    def _exp_dist(self, lambd: float = 1.0) -> float:
        if lambd <= 0:
            raise ValueError("lambda must be positive")
        u = self.get_float(0.0, np.nextafter(1.0, 0.0))
        return -math.log(1.0 - u) / lambd

    def get_distribution(self, dist_type: str, **kwargs: Any) -> float:
        if dist_type not in self.distributions:
            raise ValueError(f"Unknown dist: {dist_type}")
        dist_func = self.distributions[dist_type]
        if dist_type == "power":
            result = dist_func(power=float(kwargs.get("power", 2.0)))
        elif dist_type == "exponential":
            result = dist_func(lambd=float(kwargs.get("lambd", 1.0)))
        elif dist_type in ("bell", "triangle"):
            if kwargs:
                print(f"Warn: Unused args for '{dist_type}': {kwargs.keys()}")
            result = dist_func()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Dist call logic missing: '{dist_type}'")
        return float(result)

    # ------------------------------------------------------------------
    # weighted helpers
    # ------------------------------------------------------------------
    def weighted_choice(
        self,
        items: Sequence[Any],
        weights: Sequence[float],
        cache_key: Any | None = None,
    ) -> Any:
        if len(items) != len(weights):
            raise ValueError("items/weights length mismatch")
        if not items:
            raise ValueError("items empty")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weight sum must be positive")

        cdf = None
        if cache_key is not None:
            cdf = self.weighted_choice_cache.get(cache_key)
            if self.metrics:
                self.metrics.update("cache_hits" if cdf is not None else "cache_misses")
        if cdf is None:
            cdf = np.cumsum(np.asarray(weights, dtype=float))
            cdf[-1] = total
            if cache_key is not None:
                if len(self.weighted_choice_cache) >= self.weighted_choice_cache_size:
                    cache_keys = list(self.weighted_choice_cache.keys())
                    key_to_delete = None
                    if cache_keys:
                        key_to_delete = random.choice(cache_keys)
                    if (
                        key_to_delete is not None
                        and key_to_delete in self.weighted_choice_cache
                    ):
                        del self.weighted_choice_cache[key_to_delete]
                self.weighted_choice_cache[cache_key] = cdf

        r = self.get_float(0.0, total)
        idx = int(np.searchsorted(cdf, r, side="left"))
        idx = min(idx, len(items) - 1)
        if self.metrics:
            self.metrics.update("weighted_choices")
        return items[idx]

    def weighted_sample_ares(
        self, items: Sequence[Any], weights: Sequence[float], k: int
    ) -> List[Any]:
        if len(items) != len(weights):
            raise ValueError("items/weights length mismatch")
        if k < 0 or k > len(items):
            raise ValueError("invalid k")
        if k == 0:
            return []
        weights = np.asarray(weights, dtype=float)
        positive = weights > 0
        if np.count_nonzero(positive) < k:
            raise ValueError("not enough positive weights")
        items_pos = [items[i] for i in range(len(items)) if positive[i]]
        weights_pos = weights[positive]
        rng_vals = self.rng.random(len(items_pos))
        keys = rng_vals ** (1.0 / weights_pos)
        idx = np.argsort(-keys)[:k]
        if self.metrics:
            self.metrics.update("weighted_samples_ares")
        return [items_pos[i] for i in idx]

    # ------------------------------------------------------------------
    # sequence utilities
    # ------------------------------------------------------------------
    def shuffle(self, seq: List[Any]) -> None:
        self.rng.shuffle(seq)
        if self.metrics:
            self.metrics.update("shuffles")

    def sample(
        self,
        items: Sequence[Any],
        k: int,
        replacement: bool = False,
        weights: Sequence[float] | None = None,
    ) -> List[Any]:
        if k < 0:
            raise ValueError("k >= 0")
        if k == 0:
            return []
        if not replacement and k > len(items):
            raise ValueError("k <= len(items) without replacement")
        if replacement:
            if weights is None:
                choices = self.rng.choice(items, size=k, replace=True)
            else:
                choices = [self.weighted_choice(items, weights) for _ in range(k)]
        else:
            if weights is None:
                choices = self.rng.choice(items, size=k, replace=False)
            else:
                choices = self.weighted_sample_ares(items, weights, k)
        if self.metrics:
            self.metrics.update("samples")
        return list(choices)

    # ------------------------------------------------------------------
    # misc helpers
    # ------------------------------------------------------------------
    def roll_dice(
        self, num_dice: int = 1, sides: int = 6, modifier: int = 0
    ) -> Dict[str, Any]:
        if sides < 1 or num_dice < 0:
            raise ValueError("invalid dice")
        rolls = self.get_ints(1, sides, num_dice) if num_dice > 0 else []
        total = sum(rolls) + modifier
        return {"total": total, "rolls": rolls, "modifier": modifier}

    def coin_flip(
        self, num_flips: int = 1, heads_probability: float = 0.5
    ) -> Union[str, List[str]]:
        if not 0.0 <= heads_probability <= 1.0:
            raise ValueError("probability out of range")
        results = [
            "heads" if self.get_float() < heads_probability else "tails"
            for _ in range(num_flips)
        ]
        return results[0] if num_flips == 1 else results

    def deck_of_cards(self, shuffled: bool = True) -> List[Dict[str, str]]:
        suits = ("hearts", "diamonds", "clubs", "spades")
        ranks = (
            "ace",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "jack",
            "queen",
            "king",
        )
        deck = [{"rank": r, "suit": s} for s in suits for r in ranks]
        if shuffled:
            self.shuffle(deck)
        return deck

    def loot_table(
        self, table: Dict[Any, float], count: int = 1, unique: bool = False
    ) -> List[Any]:
        if count < 0:
            raise ValueError("count >= 0")
        items = list(table.keys())
        weights = [float(w) for w in table.values()]
        if count == 0 or not items:
            return []
        return self.sample(items, count, replacement=not unique, weights=weights)

    def uuid(self) -> str:
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # noise (simple deterministic implementation, not full Perlin noise)
    # ------------------------------------------------------------------
    def _hash_seed(self, *vals: int) -> int:
        seed = self.noise_seed
        for v in vals:
            seed = (seed * 6364136223846793005 + v + 1) & 0xFFFFFFFFFFFFFFFF
        return seed

    def noise_1d(self, x: float, scale: float = 1.0, seed_offset: int = 0) -> float:
        if scale == 0:
            return 0.0
        input_x = int(x / scale)
        rng = np.random.default_rng(self._hash_seed(input_x, seed_offset))
        return float(rng.uniform(-1.0, 1.0))

    def noise_2d(
        self, x: float, y: float, scale: float = 1.0, seed_offset: int = 0
    ) -> float:
        if scale == 0:
            return 0.0
        input_x = int(x / scale)
        input_y = int(y / scale)
        rng = np.random.default_rng(self._hash_seed(input_x, input_y, seed_offset))
        return float(rng.uniform(-1.0, 1.0))

    # ------------------------------------------------------------------
    # state management
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "random_state": self.rng.bit_generator.state,
            "initial_seed": self.initial_seed,
            "noise_seed": self.noise_seed,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if "random_state" in state:
            self.rng.bit_generator.state = state["random_state"]
        if "initial_seed" in state:
            self.initial_seed = state["initial_seed"]
        if "noise_seed" in state:
            self.noise_seed = state["noise_seed"]

    def save_state_to_file(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f, indent=2)

    def load_state_from_file(self, filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.set_state(state)

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        if not self.metrics_enabled or not self.metrics:
            return None
        return self.metrics.get_metrics()

    def reset(
        self, seed: Optional[int] = None, noise_seed: Optional[int] = None
    ) -> None:
        self.initial_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.rng = np.random.default_rng(self.initial_seed)
        if noise_seed == "reset" or noise_seed is None:
            self.noise_seed = (
                self.initial_seed
                if noise_seed == "reset"
                else random.randint(0, 2**32 - 1)
            )
        else:
            self.noise_seed = noise_seed
        self.weighted_choice_cache.clear()
        if self.metrics_enabled and self.metrics:
            self.metrics.stop()
            self.metrics = MetricsCollector()
            self.metrics.start()


__all__ = ["GameRNG", "MetricsCollector"]

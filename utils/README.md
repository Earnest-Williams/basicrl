# utils/game_rng - Deterministic Random Number Generator

## Purpose

This module provides the `GameRNG` class, a high-performance, deterministic Random Number Generator (RNG) specifically designed for simulation and game development needs within the `simple_rl` project. It replaces the standard `random` module to ensure reproducibility, offer advanced features, and optimize performance for random number generation tasks.

This is a foundational component used by various parts of the project (e.g., `prototypes/Dungeon`, AI systems) to ensure that procedural generation and stochastic events are consistent and repeatable given the same initial seed and sequence of operations.

## Key Features

* **Determinism & Reproducibility:** Designed to produce the exact same sequence of random numbers given the same initial seed. This is crucial for debugging, testing, and consistent gameplay experiences.
* **State Management:** Supports saving and loading the complete internal state of the generator (`save_state_to_file`, `load_state_from_file`, `get_state`, `set_state`). This allows simulations to be paused and resumed perfectly, or specific scenarios to be replayed.
* **Multiple PRNG Algorithms:** Allows selection between different underlying Pseudo-Random Number Generators (PRNGs), currently supporting NumPy's PCG64 (default) and Xorshift128+ via Numba.
* **Performance:**
    * **Bit Buffering (`CircularBitBuffer`):** Generates random bits in larger chunks and serves requests from an efficient buffer, reducing the overhead of calling the underlying PRNG frequently for small requests (e.g., individual boolean flags).
    * **Numba Acceleration:** Includes Numba-accelerated implementations for Perlin noise generation (`noise_1d`, `noise_2d`).
* **Game-Specific Utilities:** Provides common functions needed in game development:
    * Dice rolling (`dice`).
    * Dealing cards from a standard deck (`deal_cards`).
    * Weighted sampling using A-Res algorithm (`weighted_sample_ares`).
    * Generating version 4 UUIDs (`uuid4`).
    * Standard distributions (`randint`, `choice`, `shuffle`, `uniform`, `normal`, etc.).
* **Thread Safety:** Uses `threading.RLock` to ensure safe usage across multiple threads if needed (though the primary usage pattern involves passing a single instance).
* **Metrics & Self-Tests:** Includes optional performance metrics collection (`MetricsCollector`) and internal self-tests to verify correctness.

## Usage Pattern

The intended usage is to instantiate `GameRNG` once at a high level (e.g., at the start of the main application or simulation) with a specific seed. This instance should then be passed down to any component or function that requires random numbers.

```python
# Example Usage (in a main script)
from utils.game_rng import GameRNG

# Initialize with a seed (e.g., from config or system time)
main_rng = GameRNG(seed=12345)

# Pass the same instance to different components
dungeon_map = generate_dungeon(..., rng=main_rng)
npc_behavior = update_npc(..., rng=main_rng)

# Save state if needed
main_rng.save_state_to_file("game_rng_state.json")

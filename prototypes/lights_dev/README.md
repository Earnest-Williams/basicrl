# simple_rl/lights_dev - Lighting, FOV, and Memory Simulation R&D

## Purpose

This directory serves as a dedicated research and development testbed for advanced visual and memory systems intended for the main `simple_rl` game. It focuses on implementing and iterating upon:

1.  **Field of View (FOV) / Line of Sight (LOS):** A shadow-casting algorithm (octant-based) inspired by Brogue, designed for performance using Numba.
2.  **Dynamic Lighting:** Simulation of colored light sources with inverse square falloff, supporting blending of multiple sources for True Color illumination. Future plans include animating light sources (e.g., flickering torches).
3.  **Memory Fade:** A novel system where the player's memory of explored areas fades over time based on a sigmoid decay function. The accuracy and retention of memory are planned to be influenced by agent traits and conditions (e.g., intelligence, illness, magic).

The components developed here are intended for eventual integration into the main game engine.

## Core Components

* **`main_game.py`:** The main simulation script for this testbed. It initializes the environment, manages the game loop, updates entity states (including light sources), calls the Numba-accelerated FOV/lighting/memory functions, and renders the output to the console using ANSI True Color codes. Includes optional `readchar` support for basic interaction or a profiling mode.
* **`dungeon_data.py`:** Defines the `Dungeon` **Numba jitclass**. This high-performance data structure holds the core grid arrays (`tiles`, `visible`, `memory_intensity`, `last_seen_time`) and is passed to Numba-accelerated functions.
* **`constants.py`:** Contains constants specific to this R&D environment, including rendering characters, True Color RGB values, memory fade parameters, and the `LIGHT_LEVEL_DATA` structure mapping light intensity to gameplay visibility checks.
* **`dungeon_generator.py`:** A **simple** procedural dungeon generator (creating U-shaped rooms) used specifically for creating basic test maps *for this lighting/FOV experiment*. It operates directly on the `Dungeon` jitclass.
* **`prototypes/Dungeon/` Directory:** Contains a *variant* of the main dungeon generation pipeline (`core.py`, `processor.py`, `shaper.py`, `run.sh`). This variant is used **solely for generating more complex test maps** within the `lights_dev` context. **Crucially, this version of `core.py` uses the standard `random` module**, not the project's `GameRNG`, meaning its output is non-deterministic unless explicitly seeded externally. It produces intermediate JSON files (`generated_cave_contextual.json`, `processed_cave_data.json`) and includes debug logging (`run_log.txt`).

## Key Mechanics & Implementation

* **FOV/LOS:** Octant-based recursive shadow casting implemented in Numba (`_compute_octant_for_boolean_array`) for performance. Relies on the `Dungeon.blocks_light` method.
* **Lighting:** Calculates light spread and intensity using an inverse square falloff. Blends RGB values from multiple sources. Implemented in Numba (`_compute_octant_for_color`).
* **Memory Fade:** Uses elapsed time (`Dungeon.current_time`, `Dungeon.last_seen_time`) and a sigmoid function (`_update_memory_fade_internal` in Numba) to decay `Dungeon.memory_intensity` for non-visible tiles. Intensity affects rendering character choice.
* **Rendering:** Outputs to console using ANSI True Color escape codes for blended lighting and specific characters for different visibility/memory states.

## Dependencies

* **Python:** 3.x
* **Core Libraries:** `numpy`, `numba`
* **Optional:** `readchar` (for interactive mode in `main_game.py`)
* **`prototypes/Dungeon/` Dependencies:** `numpy`, `scipy`, `scikit-image`, `polars`, `perlin-noise` (for some shaping features).

## Status & Integration

This component is under active development and refinement. The core algorithms for FOV, lighting, and memory are functional but require testing in more complex environments and further iteration (e.g., light animation, trait integration for memory). The systems developed here are planned for integration into the main simulation, potentially combined with perception systems from `simple_rl/pathfinding` under a unified orchestrator. The memory system might be separated due to its complexity and interactions with other game states.

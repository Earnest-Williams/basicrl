# simple_rl/pathfinding - Perception Simulation System

## Purpose

This component simulates non-visual perception modalities, primarily **noise propagation** and **scent tracking**, intended to provide input for agent AI systems (like the GOAP planner in `simple_rl/auto`). While named `pathfinding` (due to its conceptual origin in the game Sil, where perception was coupled with pathfinding), this module *does not* contain pathfinding algorithms itself. Instead, it generates the perception data (e.g., noise cost maps, scent trails) that pathfinding routines or decision-making logic can utilize.

The goal is to enable agents (particularly monsters or NPCs with heightened senses) to react to sounds and smells in the environment, allowing for behaviors like investigating disturbances or tracking targets by scent. Sound propagation data may also be relevant to the player character.

## Core Components

* **`perception_systems.py`:**
    * **Noise Propagation:** Implements `update_noise` which calculates the spread of sound using a cost-based flood fill (similar to Dijkstra/BFS) accelerated by Numba (`_propagate_noise_kernel`). It handles different terrain/feature interactions (e.g., how noise passes through doors) via `FlowType` enums and `FeatureType` enums. Outputs noise cost maps (`cave_cost`).
    * **Scent System:** Implements `update_smell` which simulates scent aging (`_age_scent_kernel`) and laying (`_lay_scent_kernel`), also using Numba kernels. Tracks scent intensity and the time it was last detected (`cave_when`).
    * **Monster Perception:** Provides `monster_perception` which checks if monsters detect the player based on noise, scent, or line-of-sight using `game.world.los.line_of_sight`. This check is parallelized across monsters using Joblib (`Parallel`, `delayed`) for efficiency. Monster data is expected in a Polars DataFrame.
    * **Placeholders:** Includes a placeholder function for `skill_check` that needs an actual implementation.
* **`prototypes/pathfinding/test.py`:**
    * Serves as a test harness and demonstration for the perception systems.
    * Initializes a sample map, player, and monsters (using Polars).
    * Runs a simple simulation loop calling `update_noise`, `update_smell`, and `monster_perception`.
    * **Demonstrates structured logging** using `structlog` with console and rotating file handlers, configured to output JSON logs. Includes sample command-line queries (`jq`, `zq`) for analyzing these logs. *Note: `structlog` usage here is experimental and not yet fully integrated project-wide.*
* **`../scripts/fix.sh`:** Standard code formatting script using `ruff`, `black`, etc.

## Dependencies

* **Python:** 3.x
* **Core Libraries:**
    * `numpy`: For grid data structures.
    * `numba`: For JIT-accelerating core propagation kernels.
    * `polars`: For handling monster data efficiently.
    * `joblib`: For parallelizing monster perception checks.
* **Testing/Logging (`test.py` specific):**
    * `structlog`: For structured logging demonstration.
    * `logging`: Standard Python logging library.

## Integration & Status

* The perception data (noise cost maps, scent maps) generated here is intended to be consumed by AI systems (primarily `simple_rl/auto`) to influence behavior and pathfinding decisions.
* The `FeatureType` enum defined here needs to be synchronized with the final feature set produced by the main `prototypes/Dungeon` generator once that is finalized.
* Line-of-sight checks rely on `game.world.los.line_of_sight`, a Bresenham implementation optionally accelerated with Numba.
* The core perception logic is functional but may require further tuning and integration with specific agent capabilities.

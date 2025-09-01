# prototypes/Dungeon - Procedural Cave System Generator

## Purpose

This component implements the primary procedural generation pipeline for the extensive cave systems featured in the game world. It is designed to create complex, multi-featured underground environments suitable for exploration and agent interaction, distinct from overland or settlement map generation.

## Workflow

The generation process follows a multi-stage pipeline orchestrated by `run.sh` (primarily for development and testing):

1.  **`core.py` (Core Generator):**
    * Generates the foundational 2D backbone layout of the cave network as a branching node graph.
    * Uses the project's central `GameRNG` instance for deterministic output.
    * Employs a probability-based growth algorithm with contextual triggers.
    * Flags nodes for specific geological or structural features (e.g., "big\_room:type", "cliff\_edge", "shaft\_opening") based on generation parameters and local context (density, angles, probability decay).
    * Uses `scipy.spatial.KDTree` for efficient convergence detection between branches.
    * Outputs the raw node graph structure (`generated_cave_contextual.json`).

2.  **`processor.py` (Interstitial Processor):**
    * Takes the raw node graph from `core.py`.
    * Calculates basic segment geometry (XY length, incline rate, depth change using `numpy`) and passes features through.
    * Currently serves primarily as a separation layer, allowing for future insertion of more complex intermediate processing steps (e.g., detailed geological feature calculation, flow analysis).
    * Outputs augmented node data (`processed_cave_data.json`).

3.  **`shaper.py` (Grid Shaper):**
    * Consumes the processed node data.
    * Rasterizes the cave network structure onto a 2D grid using `scikit-image` (lines, polygons, ellipses, noise-based shapes for caverns).
    * Implements flagged features (e.g., rendering different cavern types, cliffs, shafts).
    * Applies Cellular Automata (`scipy.signal.convolve2d` or Numba fallback) for smoothing and naturalization.
    * Calculates 3D information (depth, height) and stores it within the 2D grid representation.
    * Assigns Chamber IDs using `scipy.ndimage.label`.
    * Uses the `GameRNG` instance for procedural elements during shaping (e.g., cavern detail).
    * Outputs the final map as a high-performance Polars DataFrame, saved to an Apache Arrow (`.arrow`) file (`shaped_dungeon_map.arrow`).

4.  **Future: History & Lore Layer (Planned):**
    * A subsequent, planned stage (not yet implemented in these files) will layer historical structures onto the shaped map.
    * This includes elements reflecting different eras: deep graveyards, associated temple complexes, housing/storage networks, and defensive fortifications, adding narrative depth and environmental storytelling.

## Usage

During development, the pipeline is typically executed via the `run.sh` script:

```bash
./run.sh

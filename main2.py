# simple_rl/main.py
# Updated to instantiate and pass GameRNG

import argparse
import json
import logging
import sys
import time
import traceback
import warnings

from utils.logging_utils import setup_logging

import polars as pl  # Needed for potential final map check/info

# --- Import project modules ---
# Since main.py is in the root, Python should find these subdirectories
# Use try-except for robustness if structure changes or run differently
try:
    from Dungeon import core, processor, shaper
    from game_rng import GameRNG
except ImportError as e:
    print(
        f"Import Error: {e}. Please ensure the script is run from the project root "
        "or the necessary paths are in PYTHONPATH.",
        file=sys.stderr,
    )
    sys.exit(1)

# Suppress specific warnings if desired (e.g., NumbaPerformanceWarning)
warnings.filterwarnings(
    "ignore", message=".*Use Function.compile_options=.*", category=UserWarning
)

# --- Default Configuration ---
DEFAULT_SEED = int(time.time() * 1000)  # Time-based default
DEFAULT_MAX_NODES = 400
DEFAULT_MAX_DEPTH = 50
DEFAULT_CA_ITERATIONS = 8
DEFAULT_OUTPUT_FILE = "generated_dungeon.arrow"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate procedural dungeon map using GameRNG."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for RNG (default: time-based, currently {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=DEFAULT_MAX_NODES,
        help="Max nodes for core generator.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Max depth for core generator.",
    )
    parser.add_argument(
        "--ca-iterations",
        type=int,
        default=DEFAULT_CA_ITERATIONS,
        help="CA iterations for shaper.",
    )
    parser.add_argument(
        "--output-file", default=DEFAULT_OUTPUT_FILE, help="Output Arrow filename."
    )
    parser.add_argument(
        "--orjson",
        action="store_true",
        help="Use orjson for faster intermediate JSON (if needed for debug).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    # Add --noise-seed if you want separate control, otherwise GameRNG derives it
    # parser.add_argument("--noise-seed", type=int, default=None,
    # help="Specific seed for Perlin noise.")

    args = parser.parse_args()
    log_level = (
        logging.DEBUG
        if args.verbose
        else getattr(logging, args.log_level.upper(), logging.INFO)
    )
    setup_logging(log_level)

    start_time_main = time.time()
    print("--- Dungeon Generation Pipeline ---")
    print(f"Seed: {args.seed}")
    print(f"Max Nodes: {args.max_nodes}, Max Depth: {args.max_depth}")
    print(f"CA Iterations: {args.ca_iterations}")
    print(f"Output File: {args.output_file}")
    print("-" * 30)

    # --- Instantiate RNG ONCE ---
    try:
        # Note: GameRNG derives noise_seed from main seed if not provided explicitly
        rng_instance = GameRNG(
            seed=args.seed,
            metrics=False,  # Disable metrics unless profiling
            buffer_size=50000,  # Example larger buffer
            max_buffer_size=2000000,
        )
        print(
            f"GameRNG Initialized (Generator: {rng_instance.generator_type}, "
            f"Noise Seed: {rng_instance.noise_seed})"
        )
    except Exception as e:
        print(f"FATAL: Failed to initialize GameRNG: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Orchestration ---
    final_map: pl.DataFrame | None = None  # Type hint using PEP 604

    try:
        # 1. Core Generation
        print("\n--- Running Core Generator ---")
        core_start_time = time.time()
        generator = core.CaveGenerator(
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            rng=rng_instance,  # Pass the single RNG instance
        )
        generator.grow()
        core_end_time = time.time()
        print(
            f"Core: Generated {len(generator.nodes)} nodes "
            f"({core_end_time - core_start_time:.2f}s)"
        )

        # Extract data for processor (can use dict directly)
        # Use orjson if available for performance boost (especially large data)
        try:
            # Attempt to use orjson first if requested
            if args.orjson:
                import orjson

                json_serializer = orjson.dumps
                json_bytes = True
            else:
                raise ImportError  # Force fallback if orjson not requested

        except ImportError:
            # Fallback to standard json
            json_serializer = lambda d: json.dumps(d, indent=2).encode("utf-8")
            json_bytes = False
            if args.orjson:
                print("Core: orjson not found, falling back to standard json.")

        # Generate the dictionary representation
        raw_backbone_data_dict = {"nodes": [n.to_dict() for n in generator.nodes]}

        # 2. Processing
        print("\n--- Running Processor ---")
        proc_start_time = time.time()
        # Processor does not use RNG, only transforms data
        augmented_nodes, augmented_node_map = processor.process_backbone_graph(
            raw_backbone_data_dict
        )
        proc_end_time = time.time()
        if not augmented_nodes:
            raise RuntimeError("Processor returned empty node list.")
        print(
            f"Processor: Calculated geometry for {len(augmented_nodes)} nodes "
            f"({proc_end_time - proc_start_time:.2f}s)"
        )

        # 3. Shaping
        print("\n--- Running Shaper ---")
        shape_start_time = time.time()
        final_map = shaper.generate_shaped_cave(
            augmented_nodes,
            augmented_node_map,
            rng=rng_instance,  # Pass the SAME RNG instance
            ca_iterations=args.ca_iterations,
        )
        shape_end_time = time.time()
        print(f"Shaper: Completed ({shape_end_time - shape_start_time:.2f}s)")

        # 4. Save Final Output
        if final_map is not None and not final_map.is_empty():
            print("\n--- Saving Final Map ---")
            final_map.write_ipc(args.output_file)
            print(f"Saved map ({final_map.shape}) to {args.output_file}")
        elif final_map is not None:
            print("Warning: Final map generated by shaper is empty.")
        else:
            raise RuntimeError("Shaper failed to produce a map DataFrame.")

    except Exception:
        print("\n--- ERROR during pipeline execution ---")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # --- Final Summary ---
        end_time_main = time.time()
        print("-" * 30)
        if final_map is not None and not final_map.is_empty():
            print(f"Pipeline SUCCESSFUL. Map saved to: {args.output_file}")
            print(f"DataFrame shape: {final_map.shape}")
        elif final_map is not None:
            print("Pipeline WARNING: Generated empty map.")
        else:
            print("Pipeline FAILED.")
        print(f"Total execution time: {end_time_main - start_time_main:.2f} seconds")
        print("-" * 30)

        # Optionally print final RNG state (if metrics were enabled etc.)
        # if rng_instance.metrics_enabled:
        #     print("\n--- Final RNG Metrics ---")
        #     metrics_data = rng_instance.get_metrics()
        #     # Safely convert metrics to JSON string
        #     try:
        #         print(json.dumps(metrics_data, indent=2, default=str))
        #     except TypeError:
        #         print("Could not serialize RNG metrics to JSON.")

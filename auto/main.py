# auto/main.py
# Updated to pass seed and instantiate GameRNG within worker function

import argparse
import cProfile
import io
import logging
import pstats

# Removed 'random' import if it was implicitly used before
import sys
import time
import traceback  # Import traceback for error logging
from collections import Counter
from multiprocessing import Pool, cpu_count

from utils.logging_utils import setup_logging

# --- Use relative imports ---
# Use try-except for robustness
try:
    from game_rng import GameRNG  # Import GameRNG

    from .simulation import (
        GRID_SIZE,  # Relative import for use with -m
        MAX_TURNS,
        PASSIVE_HUNGER_PER_TURN,  # Keep relevant constants
        START_HEALTH,
        STARVATION_HEALTH_DAMAGE,
        ActionResult,
        AgentAI,
        World,
        enemy_act,
    )
except ImportError:
    print(
        "Error: Cannot perform relative imports. Run as module: python -m auto.main [...]",
        file=sys.stderr,
    )
    # Fallback for direct execution (less ideal)
    try:
        from game_rng import GameRNG  # Adjust path if needed

        from simulation import (
            GRID_SIZE,
            MAX_TURNS,
            PASSIVE_HUNGER_PER_TURN,
            START_HEALTH,
            STARVATION_HEALTH_DAMAGE,
            ActionResult,
            AgentAI,
            World,
            enemy_act,
        )
    except ImportError as e:
        print(f"Failed to import simulation components or GameRNG: {e}", file=sys.stderr)
        sys.exit(1)


# --- Configuration ---
DEFAULT_NUM_RUNS = 5
DEFAULT_NUM_WORKERS = max(1, cpu_count() // 2)
DEFAULT_SEED = int(time.time() * 1000)  # Use time-based seed if none provided


# --- Headless Simulation Function ---
# Modify function signature to accept seed
def run_single_headless(
    args: tuple[int, dict[str, float] | None, int],  # run_id, weights, seed
) -> tuple[int, dict[str, float], str]:
    """Runs a single headless simulation instance with Flee/Rest mechanics."""
    run_id, initial_weights_dict, run_seed = args  # Unpack seed
    # Instantiate GameRNG INSIDE the worker function using the passed seed
    # Ensures each process has its own deterministic RNG state
    try:
        # Use a performant generator like xorshift for headless runs if desired
        local_rng = GameRNG(seed=run_seed, generator="xorshift", metrics=False)
    except Exception as e:
        print(f"Error Run {run_id}: Failed to create GameRNG: {e}", file=sys.stderr)
        return 0, {}, "RNGCreationError"

    print(f"--- Starting Headless Run {run_id} (Seed: {run_seed}) ---")
    start_time = time.time()

    # Pass the local_rng instance to World and AgentAI
    world = World(size=GRID_SIZE, rng=local_rng)
    agent_ai = AgentAI(world=world, rng=local_rng)  # AgentAI needs RNG too

    if initial_weights_dict is not None:
        for key, value in initial_weights_dict.items():
            agent_ai.planner.action_weights[key] = value

    # World reset needs RNG for potential random placement/events during reset
    world.reset(rng=local_rng)
    agent = world.agent
    if not agent:
        print(f"Error Run {run_id}: Agent not created.", file=sys.stderr)
        return 0, dict(agent_ai.planner.action_weights), "AgentError"

    cause_of_death = "Unknown"

    while agent.health > 0 and world.turn < MAX_TURNS:
        world.turn += 1
        action_taken_this_turn: ActionResult | None = None

        # Agent Turn (passes world and agent, world has rng)
        action_taken_this_turn = agent_ai.act(agent)
        if agent.health <= 0:
            cause_of_death = "DiedAfterAction"
            break

        # Apply Passive Hunger & Starvation Damage
        new_hunger = max(0.0, agent.hunger - PASSIVE_HUNGER_PER_TURN)
        world.update_entity_hunger(agent.id, new_hunger)  # World handles update
        if agent.hunger <= 0:
            new_health = max(0.0, agent.health - STARVATION_HEALTH_DAMAGE)
            world.update_entity_health(agent.id, new_health)  # World handles update
            if new_health <= 0:
                cause_of_death = "Starvation"
                break
        # Re-check health after potential starvation
        if agent.health <= 0 and cause_of_death == "Unknown":
            cause_of_death = "Starvation"
            break

        # Enemy Turn
        agent_was_attacked_this_turn = False
        health_before_enemies = agent.health
        # Pass rng to enemy_act if it needs randomness
        current_enemy_ids = list(world.entities_by_kind["enemy"].keys())
        for enemy_id in current_enemy_ids:
            if enemy_id in world.entities:
                enemy = world.entities[enemy_id]
                enemy_act(enemy, world, local_rng)  # Pass RNG
                if agent.health <= 0:
                    if cause_of_death == "Unknown":
                        cause_of_death = "KilledByEnemy"
                    break
        if agent.health <= 0 and cause_of_death == "Unknown":
            cause_of_death = "KilledByEnemy"
            break
        if agent.health < health_before_enemies:
            agent_was_attacked_this_turn = True

        # World Events (Spawning needs RNG)
        world.spawn_random_enemy(rng=local_rng)  # Pass RNG

        # Resting Health Regen (No RNG needed here)
        agent_rested = action_taken_this_turn == "Wait" or action_taken_this_turn is None
        if agent_rested and not agent_was_attacked_this_turn:
            if agent.health < START_HEALTH:
                new_health = min(START_HEALTH, agent.health + REST_HEALTH_REGEN)
                if new_health > agent.health:
                    world.update_entity_health(agent.id, new_health)

    # Determine Final Outcome
    turns_survived = world.turn
    agent_survived = world.agent is not None and world.agent.health > 0
    if agent_survived and world.turn >= MAX_TURNS:
        final_outcome = "Survived (MaxTurns)"
    elif agent_survived:
        final_outcome = "Survived (Unknown)"
    else:
        final_outcome = cause_of_death if cause_of_death != "Unknown" else "Defeated (Unknown)"

    # Learning Step
    agent_ai.learn(turns_survived)
    final_weights = dict(agent_ai.planner.action_weights)

    end_time = time.time()
    print(
        f"--- Finished Headless Run {run_id} ({final_outcome}) --- Turns: {turns_survived}, "
        f"Time: {end_time - start_time:.2f}s ---"
    )

    return turns_survived, final_weights, final_outcome


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GOAP Simulation (GUI or Headless)")
    parser.add_argument("--mode", choices=["gui", "headless"], default="gui", help="Mode")
    parser.add_argument(
        "-n", "--num-runs", type=int, default=DEFAULT_NUM_RUNS, help="Number of runs"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of workers"
    )
    parser.add_argument(
        "--learn", choices=["independent", "shared"], default="independent", help="Learning mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Initial seed for generating run seeds (default: time-based)",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()
    log_level = (
        logging.DEBUG
        if args.verbose
        else getattr(logging, args.log_level.upper(), logging.INFO)
    )
    setup_logging(log_level)

    # --- GUI Mode ---
    if args.mode == "gui":
        print("Starting GUI Mode...")
        try:
            from PySide6.QtWidgets import QApplication

            # Use relative import for GUI components when run as module
            from .gui.main_window import MainWindow
        except ImportError:
            print(
                "Error: Could not import GUI components. Ensure PySide6 is installed.",
                file=sys.stderr,
            )
            print(
                "If running directly, ensure script is in the correct directory.", file=sys.stderr
            )
            sys.exit(1)

        try:
            app = QApplication(sys.argv)
            # GUI mode will likely use its own RNG instance internally or be passed one
            # For simplicity here, the GUI might instantiate its own based on args.seed
            # but it won't affect the headless runs.
            main_win = MainWindow()  # MainWindow needs update to accept/create RNG
            main_win.show()
            sys.exit(app.exec())
        except Exception as e:
            print(f"GUI Error: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    # --- Headless Mode ---
    elif args.mode == "headless":
        print(
            f"Starting Headless Mode: {args.num_runs} runs, {args.workers} workers, "
            f"{args.learn} learning. Base Seed: {args.seed}"
        )
        total_start_time = time.time()
        all_turns: list[int] = []
        all_weights: list[dict[str, float]] = []
        all_outcomes: list[str] = []
        profiler = None
        initial_weights_arg = None

        # Create a master RNG to generate seeds for each run
        master_rng = GameRNG(seed=args.seed)
        run_seeds = [master_rng.get_int(0, 2**32 - 1) for _ in range(args.num_runs)]

        if args.profile:
            print("Profiling enabled for the first run.")
            profiler = cProfile.Profile()
        if args.learn == "shared":
            print("Shared learning selected...")
            # Create temporary AI to get initial weights structure
            temp_ai = AgentAI(World(size=GRID_SIZE, rng=GameRNG()), rng=GameRNG())
            initial_weights_arg = dict(temp_ai.planner.action_weights)

        # Prepare arguments for each run (run_id, initial_weights, seed)
        run_args = [
            (i + 1, initial_weights_arg if args.learn == "shared" else None, run_seeds[i])
            for i in range(args.num_runs)
        ]

        results: list[tuple[int, dict[str, float], str]] = []
        num_actual_workers = min(args.workers, args.num_runs)

        try:
            if num_actual_workers > 1:
                print(f"Using multiprocessing Pool with {num_actual_workers} workers...")
                with Pool(processes=num_actual_workers) as pool:
                    if profiler:
                        profiler.enable()
                        first_result = run_single_headless(run_args[0])
                        profiler.disable()
                        results.append(first_result)
                        if len(run_args) > 1:
                            results.extend(pool.map(run_single_headless, run_args[1:]))
                    else:
                        results = pool.map(run_single_headless, run_args)
            else:
                print("Using sequential execution (1 worker)...")
                for i, run_arg_tuple in enumerate(run_args):
                    if i == 0 and profiler:
                        profiler.enable()
                        result = run_single_headless(run_arg_tuple)
                        profiler.disable()
                    else:
                        result = run_single_headless(run_arg_tuple)
                    results.append(result)
        except Exception as e:
            print(f"\n!!! Error during headless execution: {e} !!!", file=sys.stderr)
            traceback.print_exc()

        # Process results
        all_turns = [r[0] for r in results]
        all_weights = [r[1] for r in results]
        all_outcomes = [r[2] for r in results]

        print("\n=============== Headless Simulation Summary ===============")
        print(f"Total Runs Attempted: {args.num_runs}")
        print(f"Runs Completed: {len(results)}")
        print(f"Workers Used: {num_actual_workers}")
        print(f"Learning Mode: {args.learn}")

        if all_turns:
            avg_turns = sum(all_turns) / len(all_turns)
            max_turns_val = max(all_turns)  # Avoid conflict with MAX_TURNS constant
            min_turns_val = min(all_turns)
            print(f"Average Turns Survived: {avg_turns:.2f}")
            print(f"Max Turns Survived: {max_turns_val}")
            print(f"Min Turns Survived: {min_turns_val}")

            print("\nOutcome Summary:")
            outcome_counts = Counter(all_outcomes)
            for outcome, count in outcome_counts.most_common():
                print(f"  - {outcome}: {count} run(s)")
        else:
            print("No simulation runs completed successfully.")

        if all_weights:
            print("\nFinal Action Weights (from last completed run):")
            last_weights = dict(sorted(all_weights[-1].items()))
            for name, weight in last_weights.items():
                print(f"  - {name:<20}: {weight:.3f}")

        total_end_time = time.time()
        print(f"\nTotal Headless Execution Time: {total_end_time - total_start_time:.2f}s")

        if profiler:
            print("\n--- cProfile Results (First Run) ---")
            s = io.StringIO()
            # Sort by cumulative time
            stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            stats.print_stats(40)  # Print top 40 functions
            print(s.getvalue())
            print("------------------------------------")
    else:
        print(f"Error: Unknown mode '{args.mode}'", file=sys.stderr)
        sys.exit(1)

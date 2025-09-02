# worker.py
# REMOVED: import random
import time

import structlog

# Imports seem correct now
from PySide6.QtCore import QMutex, QMutexLocker, QObject, Signal, Slot

# Assuming World passes a GameRNG instance
from ...rng_utils.game_rng import GameRNG
from ..simulation import (
    ENEMY_SPAWN_CHANCE,
    MAX_TURNS,
    PASSIVE_HUNGER_PER_TURN,
    AgentAI,
    World,
    enemy_act,
)

log = structlog.get_logger(__name__)


class SimulationWorker(QObject):
    """
    Runs the GOAP simulation step-by-step in a separate thread,
    emitting signals to update the GUI and implementing frame skipping at high speeds.
    """

    # --- Signals ---
    world_updated = Signal(
        list, list, int, int, int
    )  # grid_list, entities_list, agent_health, turn, grid_size
    plan_updated = Signal(list)  # list of action names
    goal_updated = Signal(str)  # string representation of the goal
    actions_updated = Signal(dict)  # dict of action weights {name: weight}
    # turn number (emitted only when GUI updates)
    simulation_step_done = Signal(int)
    simulation_finished = Signal(str)  # End message (reason)
    status_update = Signal(str)  # General status messages

    # --- Initialization ---
    def __init__(self, world: World, agent_ai: AgentAI, parent=None):
        super().__init__(parent)
        self.world = world
        self.agent_ai = agent_ai
        self._running = False
        self._paused = False
        self._step = False
        self.mutex = QMutex()
        self.step_delay_ms = 200
        # Skip GUI updates if delay is this low or lower
        self.frame_skip_threshold_ms = 50
        self.simulation_step_counter = 0  # To track steps for skipping

        # Track last state to only emit goal/plan when changed
        self.last_emitted_goal_str = ""
        self.last_emitted_plan_list = []

    # --- Configuration Slot ---
    @Slot(int)
    def set_step_delay(self, delay_ms: int):
        """Sets the delay between simulation steps, enforcing a minimum."""
        self.step_delay_ms = max(10, delay_ms)

    # --- Main Simulation Loop ---
    @Slot()
    def run_simulation(self):
        """Starts and runs the main simulation loop."""
        with QMutexLocker(self.mutex):
            if self._running:
                return  # Prevent starting multiple times
            self._running = True
            self._paused = False
            self._step = False
            self.simulation_step_counter = 0  # Reset counter at start
            self.last_emitted_goal_str = ""
            self.last_emitted_plan_list = []

        self.status_update.emit("Simulation started.")
        try:
            # Ensure world has an rng instance before reset or access
            if not hasattr(self.world, "rng") or not isinstance(
                self.world.rng, GameRNG
            ):
                # Attempt to create one if missing? Or raise error?
                # For now, let's assume reset handles it or it's passed in world.__init__
                # If not, this will raise an AttributeError later, which is informative.
                pass

            self.world.reset()  # Reset world at the beginning of a run
            agent = self.world.agent
            if not agent:
                raise RuntimeError("Agent not created during world reset.")

            # --- Initial State Emission ---
            self.emit_world_state()
            self.emit_action_weights()
            initial_goal_str = "Goal: {}".format(
                self.agent_ai.current_goal if self.agent_ai.current_goal else "None"
            )
            initial_plan_list = [action.name for action in self.agent_ai.current_plan]
            self.goal_updated.emit(initial_goal_str)
            self.plan_updated.emit(initial_plan_list)
            self.last_emitted_goal_str = initial_goal_str
            self.last_emitted_plan_list = initial_plan_list
            # --- End Initial State ---

            while True:  # Main simulation loop
                with QMutexLocker(self.mutex):
                    if not self._running:
                        break  # Stop requested externally

                    # Handle pausing and stepping
                    if self._paused and not self._step:
                        self.mutex.unlock()
                        time.sleep(0.1)  # Pause sleep
                        self.mutex.lock()
                        continue  # Re-check flags

                    if self._step:
                        self._step = False
                        self._paused = True  # Implicitly pause after stepping

                # --- Check End Conditions ---
                if agent.health <= 0 or self.world.turn >= MAX_TURNS:
                    with QMutexLocker(self.mutex):
                        self._running = False  # Ensure flag is set
                    break  # End condition met

                # --- Start Turn ---
                self.world.turn += 1
                self.simulation_step_counter += 1
                turn_start_time = time.time()
                self.status_update.emit(f"Turn {self.world.turn} Running...")

                # --- Agent Turn ---
                prev_goal = self.agent_ai.current_goal
                prev_plan = list(self.agent_ai.current_plan)
                self.agent_ai.act(agent)  # Agent logic execution

                current_goal_str = "Goal: {}".format(
                    self.agent_ai.current_goal if self.agent_ai.current_goal else "None"
                )
                current_plan_list = [
                    action.name for action in self.agent_ai.current_plan
                ]
                goal_changed = current_goal_str != self.last_emitted_goal_str
                plan_changed = current_plan_list != self.last_emitted_plan_list

                if goal_changed:
                    self.goal_updated.emit(current_goal_str)
                    self.last_emitted_goal_str = current_goal_str
                if plan_changed:
                    self.plan_updated.emit(current_plan_list)
                    self.last_emitted_plan_list = current_plan_list
                # --- End Agent Turn ---

                if agent.health <= 0:
                    self.status_update.emit(
                        f"Agent died after its action (Turn {
                            self.world.turn})."
                    )
                    self.emit_world_state()  # Show final state before loop breaks
                    with QMutexLocker(self.mutex):
                        self._running = False
                    break

                # --- Hunger ---
                agent.health -= PASSIVE_HUNGER_PER_TURN
                if agent.health <= 0:
                    self.status_update.emit(f"Agent starved (Turn {self.world.turn}).")
                    self.emit_world_state()  # Show final state
                    with QMutexLocker(self.mutex):
                        self._running = False
                    break

                # --- Enemy Turn ---
                current_enemy_ids = list(self.world.entities_by_kind["enemy"].keys())
                for enemy_id in current_enemy_ids:
                    if enemy_id in self.world.entities:
                        enemy = self.world.entities[enemy_id]
                        enemy_act(enemy, self.world)  # Enemy logic execution
                    if agent.health <= 0:
                        self.status_update.emit(
                            f"Agent defeated by an enemy (Turn {
                                self.world.turn})."
                        )
                        break  # Exit inner enemy loop
                if agent.health <= 0:  # Check if agent died during enemy turns
                    self.emit_world_state()  # Show final state
                    with QMutexLocker(self.mutex):
                        self._running = False
                    break  # Exit main loop
                # --- End Enemy Turn ---

                # --- World Events ---
                # Use GameRNG from the world instance
                if not hasattr(self.world, "rng"):
                    raise AttributeError(
                        "World object does not have an 'rng' attribute for GameRNG."
                    )

                # *** CORRECTED LINE BELOW ***
                if (
                    self.world.rng.get_float() < ENEMY_SPAWN_CHANCE
                ):  # Use GameRNG instance
                    self.world.spawn_random_enemy()
                # --- End World Events ---

                # --- Determine if GUI update should be skipped (Frame Skipping) ---
                emit_gui_update_this_step = True
                if self.step_delay_ms <= self.frame_skip_threshold_ms:
                    if self.simulation_step_counter % 2 != 0:  # Skip odd steps
                        emit_gui_update_this_step = False

                # --- Emit State Updates (conditional) ---
                if emit_gui_update_this_step:
                    self.emit_world_state()
                    self.simulation_step_done.emit(self.world.turn)

                # --- Delay ---
                elapsed = time.time() - turn_start_time
                sleep_time_ms = self.step_delay_ms - (elapsed * 1000)
                if sleep_time_ms > 0:
                    time.sleep(sleep_time_ms / 1000.0)
                # --- End Turn Delay ---

            # --- End of Main Simulation Loop ---

        except Exception as e:
            self.status_update.emit(f"Simulation Error: {e}")
            log.error("Error in worker thread", error=e, exc_info=True)
            with QMutexLocker(self.mutex):
                self._running = False

        # --- Simulation Finished ---
        agent_survived = (
            hasattr(self.world, "agent")
            and self.world.agent is not None
            and self.world.agent.health > 0
        )
        if self.world.turn >= MAX_TURNS and agent_survived:
            end_reason = "Reached Max Turns"
        elif agent_survived:
            # Assume stopped externally if not max turns/defeated
            end_reason = "Survived (Stopped)"
        else:
            end_reason = "Defeated"

        self.status_update.emit(f"Simulation finished: {end_reason}")

        # Perform learning step
        if "agent" in locals() and agent is not None:  # Check if agent existed
            self.agent_ai.learn(self.world.turn)
            self.emit_action_weights()  # Show learned weights

        self.simulation_finished.emit(end_reason)  # Signal completion

    # --- Helper Emission Methods ---
    def emit_world_state(self):
        """Extracts serializable data from world and emits world_updated signal."""
        if (
            not hasattr(self, "world")
            or not hasattr(self.world, "entities")
            or not hasattr(self.world, "agent")
        ):
            return
        try:
            entities_list = [
                {"id": e.id, "x": e.x, "y": e.y, "kind": e.kind, "health": e.health}
                # Iterate over a copy
                for e in list(self.world.entities.values())
            ]
            agent_health = self.world.agent.health if self.world.agent else 0
            self.world_updated.emit(
                [],  # Sending empty list instead of grid_list
                entities_list,
                agent_health,
                self.world.turn,
                self.world.size,
            )
        except Exception as e:
            log.error("Error emitting world state", error=e)

    def emit_action_weights(self):
        """Emits the current action weights."""
        if hasattr(self, "agent_ai") and hasattr(self.agent_ai, "planner"):
            weights = dict(self.agent_ai.planner.action_weights)
            self.actions_updated.emit(weights)

    # --- Control Slots ---
    @Slot()
    def stop_simulation(self):
        """Signals the simulation loop to stop gracefully."""
        self.status_update.emit("Stopping simulation...")
        with QMutexLocker(self.mutex):
            if not self._running:
                return
            self._running = False
            self._paused = False
            self._step = False

    @Slot()
    def pause_simulation(self):
        """Pauses or resumes the simulation."""
        with QMutexLocker(self.mutex):
            if not self._running:
                return
            self._paused = not self._paused
            self.status_update.emit(
                "Simulation Paused" if self._paused else "Simulation Resumed"
            )
            if not self._paused:
                self._step = False

    @Slot()
    def step_simulation(self):
        """Executes exactly one simulation step if paused."""
        with QMutexLocker(self.mutex):
            if not self._running or not self._paused:
                return
            self._step = True
        self.status_update.emit("Executing one step...")

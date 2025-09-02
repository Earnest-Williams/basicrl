# Dungeon/core.py - Revised for main.py orchestration & GameRNG

import json
import math
import traceback

# Removed 'random' import
from dataclasses import dataclass, field

# Use modern type hints (list instead of List, etc.)
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

# Import GameRNG using relative path (assuming main.py is in parent dir)
try:
    # Adjust path relative to main.py's location
    from game_rng import GameRNG
except ImportError:
    # Fallback for running core.py directly for tests (requires PYTHONPATH)
    print(
        "Warning: Relative import failed. Ensure PYTHONPATH includes project root "
        "or run via main.py."
    )
    # Attempt absolute import for direct execution scenario
    try:
        from game_rng import GameRNG  # type: ignore # noqa
    except ImportError:
        print("FATAL: GameRNG not found via absolute path either.")
        raise

# --- Core Generation Constants ---
# (Keep all constants as they were)
DEFAULT_INITIAL_PROBABILITY = 100.0
DEPTH_METERS_PER_LEVEL_RANGE = (4.0, 6.0)
SEGMENT_LENGTH_RANGE = (25.0, 35.0)
BRANCH_CHECK_INTERVAL = 4
PROBABILITY_DECAY = 10.0
KDTREE_REBUILD_INTERVAL = 50
DEFAULT_BRANCH_MOMENTUM_BIAS_RATE = 0.2
DEFAULT_BRANCH_LEFT_RANGE = -10.0
DEFAULT_BRANCH_RIGHT_RANGE = 10.0
ANGLE_CLAMP_SINGLE = (-40.0, 40.0)
ANGLE_CLAMP_BRANCH = (-45.0, 45.0)
BRANCH_ANGLE_OFFSET_RANGE = (30.0, 60.0)
BRANCH_ANGLE_OFFSET_3WAY_RANGE = (20.0, 40.0)
LOOPS_ENABLED = False
BRANCH_SEGMENT_CONVERGENCE_MIN = 6
CONVERGENCE_R_MIN = 5.0
CONVERGENCE_R_MAX = 20.0
CONVERGENCE_ALPHA = 0.05
CONVERGENCE_ANGLE_THRESHOLD = 90.0
CONVERGENCE_TS_DEPTH_FACTOR = 0.1
PARALLEL_ESCAPE_DELTA_N = 5
PARALLEL_ESCAPE_DELTA_DEPTH = 10.0
PARALLEL_ESCAPE_MAX_ANGLE_DELTA = 10.0
CLIFF_FROM_LOW_PROB_CHANCE = 30.0
RESTART_FROM_CLIFF_CHANCE = 15.0
CLIFF_RESTART_DEPTH_LEVELS = (5, 10)
CLIFF_RESTART_NODE_COUNT = (3, 12)
CONVERGENCE_FEATURE_CHANCE = 40.0
CONVERGENCE_FEATURE_TYPE = "shaft_opening"
BIG_ROOM_CHANCE = 15.0
CAVERN_TYPES = [
    "ellipse",
    "rectangle",
    "multi_circle",
    "noisy_ellipse",
    "noise_blob",
]
WEIGHT_ADJUST_PERCENT = 0.20
LOW_PROB_ANGLE_THRESHOLD = 5.0
STRAIGHT_FEATURE_MOD_FACTOR = 1.5
TURN_FEATURE_MOD_FACTOR = 0.5
MID_DEPTH_FEATURE_PEAK_FACTOR = 0.5
MID_DEPTH_FEATURE_FACTOR = 1.5
DENSITY_CHECK_RADIUS_FACTOR = 1.5
DENSITY_LOW_THRESHOLD = 2
DENSITY_HIGH_THRESHOLD = 5
DENSE_AREA_FEATURE_MOD_FACTOR = 0.7
SPARSE_AREA_FEATURE_MOD_FACTOR = 1.2


@dataclass
class CaveStep:
    """Represents a step in the cave generation process for logging."""

    desc: str
    vars: Dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Converts step to a JSON-serializable dictionary."""
        # Convert potentially non-serializable types to string for logging
        serializable_vars = {k: str(v) for k, v in self.vars.items()}
        return {"desc": self.desc, "vars": serializable_vars}


@dataclass
class CaveNode:
    """Represents a node in the cave backbone graph."""

    id: int
    x: float
    y: float
    angle: float
    depth: int
    depth_m: float
    branch_segment_count: int
    probability_n: float
    last_angle_delta: float = 0.0
    parent: Optional["CaveNode"] = None
    children: list["CaveNode"] = field(default_factory=list)
    can_grow: bool = True
    feature: Optional[str] = None
    linked_node_id: Optional[int] = None

    def __post_init__(self):
        """Basic validation after initialization."""
        if not (self.depth >= 0 and self.branch_segment_count >= 1):
            raise ValueError("Invalid CaveNode parameters.")

    def to_dict(self) -> dict[str, Any]:
        """Converts node to a JSON-serializable dictionary."""
        data = {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "angle": round(self.angle, 2),
            "depth": self.depth,
            "depth_m": round(self.depth_m, 2),
            "branch_segment_count": self.branch_segment_count,
            "probability_n": round(self.probability_n, 2),
            "last_angle_delta": round(self.last_angle_delta, 2),
            "can_grow": self.can_grow,
            "parent_id": self.parent.id if self.parent else None,
            "children_ids": [c.id for c in self.children],
            "feature": self.feature,
            "linked_node_id": self.linked_node_id,
        }
        # Remove None values for cleaner output
        return {k: v for k, v in data.items() if v is not None}


class BranchConvergenceAnalyzer:  # Unchanged logic
    """Analyzes potential convergence between cave branches."""

    def __init__(self):
        pass

    def _distance(self, a: CaveNode, b: CaveNode) -> float:
        """Calculates Euclidean distance."""
        return np.hypot(a.x - b.x, a.y - b.y)

    def _adaptive_radius(self, a: CaveNode, b: CaveNode) -> float:
        """Calculates adaptive search radius based on depth."""
        avg_depth = (a.depth_m + b.depth_m) / 2.0
        radius = CONVERGENCE_ALPHA * avg_depth
        return min(CONVERGENCE_R_MAX, max(CONVERGENCE_R_MIN, radius))

    def _angle_difference(self, a1: float, a2: float) -> float:
        """Calculates the minimum angle difference (0-180 degrees)."""
        diff = abs(a1 - a2) % 360.0
        return min(diff, 360.0 - diff)

    def _temporal_threshold(self, a: CaveNode, b: CaveNode) -> int:
        """Calculates temporal threshold based on depth."""
        avg_depth = (a.depth_m + b.depth_m) / 2.0
        threshold = math.floor(float(CONVERGENCE_TS_DEPTH_FACTOR) * avg_depth)
        return max(2, threshold)

    def action_for_pair(self, new_node: CaveNode, other_node: CaveNode) -> str:
        """Determines action for a pair of nodes based on proximity and angle."""
        # Check distance first (optimization)
        if self._distance(new_node, other_node) > self._adaptive_radius(
            new_node, other_node
        ):
            return "continue"

        # Check angle difference
        if (
            self._angle_difference(new_node.angle, other_node.angle)
            >= CONVERGENCE_ANGLE_THRESHOLD
        ):
            return "continue"

        # Check parallel escape condition
        delta_n = abs(new_node.branch_segment_count - other_node.branch_segment_count)
        delta_depth = abs(new_node.depth_m - other_node.depth_m)
        if (
            delta_n > PARALLEL_ESCAPE_DELTA_N
            and delta_depth > PARALLEL_ESCAPE_DELTA_DEPTH
            and abs(new_node.last_angle_delta) < PARALLEL_ESCAPE_MAX_ANGLE_DELTA
            and abs(other_node.last_angle_delta) < PARALLEL_ESCAPE_MAX_ANGLE_DELTA
        ):
            return "continue"  # Likely parallel branches, don't converge

        # Check temporal threshold for convergence
        if delta_n <= self._temporal_threshold(new_node, other_node):
            return "converged"

        return "continue"  # Close, but not converged


class CaveGenerator:
    """Generates the cave backbone graph."""

    # Modified __init__ to accept GameRNG instance
    def __init__(
        self,
        max_nodes: int,
        max_depth: int,
        rng: GameRNG,
        initial_probability: float = DEFAULT_INITIAL_PROBABILITY,
    ):
        if max_nodes <= 0 or max_depth <= 0:
            raise ValueError("max_nodes and max_depth must be positive.")
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.initial_probability = float(initial_probability)
        self.rng = rng  # Store the passed RNG instance

        # --- Configurable Parameters (Constants moved here for instance control) ---
        self.branch_momentum_bias_rate = DEFAULT_BRANCH_MOMENTUM_BIAS_RATE
        self.branch_left_range = DEFAULT_BRANCH_LEFT_RANGE
        self.branch_right_range = DEFAULT_BRANCH_RIGHT_RANGE
        self.loops_enabled = LOOPS_ENABLED
        self.cliff_from_low_prob_chance = CLIFF_FROM_LOW_PROB_CHANCE
        self.restart_from_cliff_chance = RESTART_FROM_CLIFF_CHANCE
        self.cliff_restart_depth_levels = CLIFF_RESTART_DEPTH_LEVELS
        self.cliff_restart_node_count = CLIFF_RESTART_NODE_COUNT
        self.convergence_feature_chance = CONVERGENCE_FEATURE_CHANCE
        self.convergence_feature_type = CONVERGENCE_FEATURE_TYPE
        self.straight_feature_mod_factor = STRAIGHT_FEATURE_MOD_FACTOR
        self.turn_feature_mod_factor = TURN_FEATURE_MOD_FACTOR
        self.mid_depth_feature_peak_level = int(
            self.max_depth * MID_DEPTH_FEATURE_PEAK_FACTOR
        )
        self.mid_depth_feature_factor = MID_DEPTH_FEATURE_FACTOR
        self.dense_area_feature_mod_factor = DENSE_AREA_FEATURE_MOD_FACTOR
        self.sparse_area_feature_mod_factor = SPARSE_AREA_FEATURE_MOD_FACTOR
        self.density_check_radius_factor = DENSITY_CHECK_RADIUS_FACTOR
        self.big_room_chance = BIG_ROOM_CHANCE
        # --- End Configurable Parameters ---

        self.cavern_type_counts: dict[str, int] = {t: 0 for t in CAVERN_TYPES}
        self.nodes: list[CaveNode] = []
        # For logging/debugging generation steps
        self.steps: list[CaveStep] = []
        self.id_counter = 0
        self.analyzer = BranchConvergenceAnalyzer()
        self.active_nodes: list[CaveNode] = []
        self.kdtree: Optional[KDTree] = None
        self.kdtree_points: list[Tuple[float, float]] = []
        self.kdtree_node_ids: list[int] = []
        self.node_map: Dict[int, CaveNode] = {}
        self._nodes_since_kdtree = 0
        self._generation_complete = False

        # Initialize root node
        root = CaveNode(
            id=self.id_counter,
            x=0.0,
            y=0.0,
            angle=0.0,
            depth=0,
            depth_m=0.0,
            branch_segment_count=1,
            probability_n=self.initial_probability,
            last_angle_delta=0.0,
            can_grow=True,
        )
        self.id_counter += 1
        self.node_map[root.id] = root
        self.nodes.append(root)
        self.active_nodes.append(root)
        self.steps.append(CaveStep("Initialized root node", {"id": root.id}))

    def _adjust_counts(self):
        """Adjusts cavern type counts to encourage variety (inverse weighting)."""
        counts = self.cavern_type_counts
        if len(set(counts.values())) <= 1:
            return  # No adjustment needed if all counts are same or only one type used

        min_count = min(counts.values())
        max_count = max(counts.values())

        min_types = [t for t, c in counts.items() if c == min_count]
        max_types = [t for t, c in counts.items() if c == max_count]
        if not min_types or not max_types:
            return

        min_type = min_types[0]  # Handle ties simply
        max_type = max_types[0]

        if min_type == max_type:
            return  # Avoid adjusting the same type

        adj_min = max(1, math.ceil(min_count * WEIGHT_ADJUST_PERCENT))
        adj_max = max(1, math.ceil(max_count * WEIGHT_ADJUST_PERCENT))

        new_min_count = max(0, counts[min_type] - adj_min)
        new_max_count = counts[max_type] + adj_max

        self.cavern_type_counts[min_type] = new_min_count
        self.cavern_type_counts[max_type] = new_max_count

    def _get_weights(self) -> Tuple[list[str], list[float]]:
        """Calculates weights inversely proportional to counts."""
        counts = self.cavern_type_counts
        types = list(counts.keys())
        # Weight inversely proportional to count+1 (avoid division by zero)
        raw_weights = [1.0 / (counts[t] + 1.0) for t in types]
        total_weight = sum(raw_weights)
        if total_weight <= 1e-9:  # Avoid division by zero or instability
            # Fallback to uniform if total weight is negligible
            uniform_weight = 1.0 / len(types) if types else 1.0
            return types, [uniform_weight] * len(types)

        normalized_weights = [w / total_weight for w in raw_weights]
        return types, normalized_weights

    def _choose_and_assign_big_room_feature(self, parent: CaveNode, step_vars: Dict):
        """Performs weighted choice using GameRNG and assigns feature."""
        types, weights = self._get_weights()
        # USE self.rng for weighted choice
        chosen_type = self.rng.weighted_choice(types, weights)  # <- GAME RNG USED HERE

        parent.feature = f"big_room:{chosen_type}"
        self.cavern_type_counts[chosen_type] += 1

        step_vars["feature_assigned"] = parent.feature
        step_vars["weights"] = {
            t: round(w, 3) for t, w in zip(types, weights)
        }  # Log weights used
        self.steps.append(
            CaveStep(
                f"Assign Feature {parent.id}: {
                     parent.feature}",
                step_vars,
            )
        )

    def _rebuild_kdtree(self):  # Unchanged logic
        """Rebuilds the KDTree for efficient proximity searches."""
        # Filter nodes eligible for convergence checks
        filtered_nodes = [
            n
            for n in self.nodes
            if n.depth > 0 and n.branch_segment_count >= BRANCH_SEGMENT_CONVERGENCE_MIN
        ]
        self.kdtree_points = [(n.x, n.y) for n in filtered_nodes]
        self.kdtree_node_ids = [n.id for n in filtered_nodes]
        try:
            if self.kdtree_points:
                self.kdtree = KDTree(self.kdtree_points)
                self._nodes_since_kdtree = 0
            else:
                self.kdtree = None  # Handle case with no eligible points
        except Exception as e:
            print(f"Warning: KDTree rebuild failed: {e}")
            self.kdtree = None  # Ensure kdtree is None on failure

    def _add_node(
        self, parent: CaveNode, angle_delta: float, prob_n: float, segment_count: int
    ) -> Optional[CaveNode]:
        """Attempts to add a new node, checking constraints and convergence."""
        step_vars = {
            "parent_id": parent.id,
            "angle_delta": round(angle_delta, 2),
            "prob_n": round(prob_n, 2),
            "segment_count": segment_count,
        }
        if len(self.nodes) >= self.max_nodes:
            self.steps.append(CaveStep("Node add failed: Max nodes", step_vars))
            parent.can_grow = False
            return None

        final_angle = parent.angle + angle_delta
        # USE self.rng for random values
        length = self.rng.get_float(*SEGMENT_LENGTH_RANGE)  # <- GAME RNG USED HERE
        x = parent.x + math.cos(math.radians(final_angle)) * length
        y = parent.y + math.sin(math.radians(final_angle)) * length
        depth_increase_m = self.rng.get_float(
            *DEPTH_METERS_PER_LEVEL_RANGE
        )  # <- GAME RNG USED HERE
        new_depth_m = parent.depth_m + depth_increase_m

        new_node = CaveNode(
            id=self.id_counter,
            x=x,
            y=y,
            angle=final_angle,
            depth=parent.depth + 1,
            depth_m=new_depth_m,
            branch_segment_count=segment_count,
            probability_n=max(0.0, prob_n),
            last_angle_delta=angle_delta,
            parent=parent,
            can_grow=True,
        )

        step_vars.update(
            {
                "new_node_id": new_node.id,
                "x": round(x, 2),
                "y": round(y, 2),
                "final_angle": round(final_angle, 2),
                "depth_level": new_node.depth,
                "depth_m": round(new_depth_m, 2),
            }
        )

        # --- Convergence Check ---
        converged_node = None
        can_add_node = True
        feature_created = False
        if (
            new_node.branch_segment_count >= BRANCH_SEGMENT_CONVERGENCE_MIN
            and self.kdtree
        ):
            radius = self.analyzer._adaptive_radius(new_node, new_node)
            try:
                # Query KDTree for nearby nodes
                indices = self.kdtree.query_ball_point(
                    [new_node.x, new_node.y], r=radius
                )
            except Exception as e:
                print(f"Warning: KDTree query failed {new_node.id}: {e}")
                indices = []

            for i in indices:
                if 0 <= i < len(self.kdtree_node_ids):
                    candidate = self.node_map.get(self.kdtree_node_ids[i])
                    # Ensure candidate exists and is not the direct parent
                    if candidate and candidate is not parent:
                        action = self.analyzer.action_for_pair(new_node, candidate)
                        if action == "converged":
                            converged_node = candidate
                            step_vars.update({"converged_with": converged_node.id})

                            # Check for feature creation on convergence
                            # USE self.rng for random chance
                            if (
                                not parent.feature
                                and self.rng.get_float(0, 100)
                                < self.convergence_feature_chance
                            ):  # <- GAME RNG USED HERE
                                feature_type = self.convergence_feature_type
                                parent.feature = feature_type
                                parent.can_grow = False  # Stop parent growth
                                can_add_node = False  # Don't add the new node
                                feature_created = True
                                desc = (
                                    f"Converge {parent.id}-{candidate.id}. "
                                    f"Feature: {feature_type}"
                                )
                                self.steps.append(CaveStep(desc, step_vars))
                            else:
                                # Handle convergence without feature
                                if not self.loops_enabled:
                                    can_add_node = False
                                    desc = (
                                        "Terminate convergence (loops off/no "
                                        "feature/parent flagged)"
                                    )
                                    self.steps.append(CaveStep(desc, step_vars))
                                else:
                                    # Add node but mark it as non-growing (creates loop end)
                                    new_node.can_grow = False
                                    desc = (
                                        "Adding loop node (loops on/no feature/"
                                        "parent flagged)"
                                    )
                                    step_vars["loop_node_added_no_grow"] = True
                                    # Keep can_add_node as True, but node won't grow

                            break  # Stop checking neighbors once converged

        # --- Add Node (if allowed) ---
        if can_add_node:
            self.id_counter += 1
            self.node_map[new_node.id] = new_node
            parent.children.append(new_node)
            self.nodes.append(new_node)
            self._nodes_since_kdtree += 1
            # Add log entry based on outcome
            if converged_node and self.loops_enabled:
                self.steps.append(CaveStep(desc, step_vars))  # Log loop creation
            elif not converged_node:
                # Log normal add
                self.steps.append(CaveStep("Added node", step_vars))
            return new_node
        else:
            # Log why node wasn't added (already logged if feature created or terminated)
            if not feature_created and not converged_node:  # General failure case
                self.steps.append(
                    CaveStep("Node add failed (reason unspecified)", step_vars)
                )
            return None

    # Unchanged logic
    def _calculate_context_modifier(self, parent: CaveNode) -> float:
        """Calculates a multiplier for feature chances based on context."""
        # Angle modifier
        angle_mod = 1.0
        if abs(parent.last_angle_delta) < LOW_PROB_ANGLE_THRESHOLD:
            angle_mod = self.straight_feature_mod_factor
        elif abs(parent.last_angle_delta) > ANGLE_CLAMP_SINGLE[1] * 0.75:
            angle_mod = self.turn_feature_mod_factor

        # Depth modifier (peaks at mid-depth)
        depth_mod = 1.0
        if self.mid_depth_feature_peak_level > 0:
            depth_diff = abs(parent.depth - self.mid_depth_feature_peak_level)
            norm_depth_diff = depth_diff / max(
                1,
                max(
                    self.mid_depth_feature_peak_level,
                    self.max_depth - self.mid_depth_feature_peak_level,
                ),
            )
            # Quadratic falloff from peak
            depth_mod = 1.0 + (self.mid_depth_feature_factor - 1.0) * max(
                0, 1.0 - norm_depth_diff**2
            )

        # Density modifier (based on nearby nodes from KDTree)
        density_mod = 1.0
        if self.kdtree and parent.branch_segment_count > 1:
            check_radius = (
                (SEGMENT_LENGTH_RANGE[0] + SEGMENT_LENGTH_RANGE[1]) / 2.0
            ) * self.density_check_radius_factor
            try:
                num_neighbors = len(
                    self.kdtree.query_ball_point([parent.x, parent.y], r=check_radius)
                )
                if num_neighbors >= DENSITY_HIGH_THRESHOLD:
                    density_mod = self.dense_area_feature_mod_factor
                elif num_neighbors <= DENSITY_LOW_THRESHOLD:
                    density_mod = self.sparse_area_feature_mod_factor
            except Exception:
                pass  # Ignore KDTree query errors for context mod

        return angle_mod * depth_mod * density_mod

    def _perform_cliff_restart(self, parent: CaveNode):
        """Handles the logic for restarting growth below a cliff feature."""
        # USE self.rng for random values
        restart_depth_level_increase = self.rng.get_int(
            *self.cliff_restart_depth_levels
        )  # <- GAME RNG USED HERE
        restart_depth_level = parent.depth + restart_depth_level_increase
        num_restart_nodes = self.rng.get_int(
            *self.cliff_restart_node_count
        )  # <- GAME RNG USED HERE

        if restart_depth_level >= self.max_depth:
            self.steps.append(
                CaveStep(
                    "Cliff restart aborted: Max depth level", {"parent_id": parent.id}
                )
            )
            return

        # Calculate target depth in meters
        current_restart_depth_m = parent.depth_m
        for _ in range(restart_depth_level_increase):
            current_restart_depth_m += self.rng.get_float(
                *DEPTH_METERS_PER_LEVEL_RANGE
            )  # <- GAME RNG USED HERE

        self.steps.append(
            CaveStep(
                f"Attempting cliff restart below {parent.id}",
                {
                    "num_nodes": num_restart_nodes,
                    "target_depth_m": round(current_restart_depth_m, 2),
                },
            )
        )

        for _ in range(num_restart_nodes):
            if len(self.nodes) >= self.max_nodes:
                self.steps.append(
                    CaveStep(
                        "Cliff restart aborted: Max nodes", {"parent_id": parent.id}
                    )
                )
                break

            # Calculate position and angle for the new restart node
            initial_angle = parent.angle + self.rng.get_float(
                -45, 45
            )  # <- GAME RNG USED HERE
            random_component = self.rng.get_float(
                self.branch_left_range, self.branch_right_range
            )  # <- GAME RNG USED HERE
            angle_delta = max(
                ANGLE_CLAMP_SINGLE[0], min(ANGLE_CLAMP_SINGLE[1], random_component)
            )
            final_angle = initial_angle + angle_delta
            # Start slightly offset from parent
            length = (
                self.rng.get_float(*SEGMENT_LENGTH_RANGE) * 0.5
            )  # <- GAME RNG USED HERE
            x = parent.x + math.cos(math.radians(final_angle)) * length
            y = parent.y + math.sin(math.radians(final_angle)) * length

            restart_node = CaveNode(
                id=self.id_counter,
                x=x,
                y=y,
                angle=final_angle,
                depth=restart_depth_level,
                depth_m=current_restart_depth_m,
                branch_segment_count=1,
                probability_n=self.initial_probability * 0.75,  # Reduced start prob
                last_angle_delta=angle_delta,
                parent=None,  # No direct parent in the main tree
                can_grow=True,
                feature="restarted_below_cliff",
            )
            self.id_counter += 1
            self.node_map[restart_node.id] = restart_node
            self.nodes.append(restart_node)
            # Add to active nodes to grow
            self.active_nodes.append(restart_node)
            self.steps.append(
                CaveStep(
                    f"Added cliff restart node {restart_node.id}",
                    {"parent_cliff_node": parent.id},
                )
            )

    def grow(self):
        """Main generation loop that grows the cave network."""
        if self._generation_complete:
            return
        # Initial KDTree build if needed
        if not self.kdtree and len(self.nodes) > BRANCH_SEGMENT_CONVERGENCE_MIN:
            self._rebuild_kdtree()

        processed_nodes_in_cycle = 0
        # Loop while there are active nodes to process
        while self.active_nodes:
            processed_nodes_in_cycle += 1
            # Safety break for potential infinite loops
            if processed_nodes_in_cycle > self.max_nodes * 5:
                print("Warning: Potential infinite loop in grow(). Breaking.")
                break

            parent = self.active_nodes.pop()  # Get next node to process

            if not parent.can_grow:
                continue  # Skip nodes marked as unable to grow

            step_vars = {
                "parent_id": parent.id,
                "parent_depth": parent.depth,
                "parent_prob_n": round(parent.probability_n, 2),
            }
            # Calculate context modifier for feature chances
            combined_mod = self._calculate_context_modifier(parent)
            step_vars["context_mod"] = round(combined_mod, 2)

            terminate_reason = None
            feature_assigned_this_step = False

            # --- Termination / Feature Checks ---
            if parent.depth >= self.max_depth:
                terminate_reason = "max depth level"
            elif parent.probability_n <= 0:
                # USE self.rng for random chance
                if (
                    not parent.feature
                    and self.rng.get_float(0, 100)
                    < self.cliff_from_low_prob_chance * combined_mod
                ):  # <- GAME RNG USED HERE
                    parent.feature = "cliff_edge"
                    terminate_reason = "low probability -> cliff_edge"
                    feature_assigned_this_step = True
                    step_vars.update(
                        {
                            "final_action": "terminate_to_cliff",
                            "reason": terminate_reason,
                        }
                    )
                    self.steps.append(
                        CaveStep(
                            f"Assign Feature {parent.id}: {
                                parent.feature}",
                            step_vars,
                        )
                    )
                    # USE self.rng for random chance
                    if (
                        self.rng.get_float(0, 100) < self.restart_from_cliff_chance
                    ):  # <- GAME RNG
                        self._perform_cliff_restart(parent)
                else:
                    terminate_reason = "probability <= 0"
            # USE self.rng for random chance
            elif (
                not parent.feature
                and self.rng.get_float(0, 100) < self.big_room_chance * combined_mod
            ):  # <- GAME RNG USED HERE
                self._choose_and_assign_big_room_feature(parent, step_vars)
                feature_assigned_this_step = True
                # Node can still grow after becoming a big room feature point

            # --- End Feature Checks ---

            # Terminate if needed and no feature was just assigned
            if terminate_reason and not feature_assigned_this_step:
                parent.can_grow = False
                step_vars.update(
                    {"final_action": "terminate", "reason": terminate_reason}
                )
                self.steps.append(
                    CaveStep(
                        f"Terminate Node {parent.id}. Reason: {
                            terminate_reason}.",
                        step_vars,
                    )
                )
                continue

            # --- Branching Logic ---
            temp_n = parent.probability_n
            should_split = False
            num_branches = 1
            # Check if it's time to potentially branch
            if (
                parent.can_grow  # Redundant check? parent.can_grow was checked earlier
                and parent.branch_segment_count > 0
                and parent.branch_segment_count % BRANCH_CHECK_INTERVAL == 0
            ):
                # USE self.rng for random chance
                split_roll = self.rng.get_float(0.0, 100.0)  # <- GAME RNG USED HERE
                if split_roll <= temp_n:
                    temp_n = max(0.0, temp_n - PROBABILITY_DECAY)
                    branch_roll = self.rng.get_float(
                        0.0, 100.0
                    )  # <- GAME RNG USED HERE
                    if branch_roll <= temp_n:
                        should_split = True
                        num_branches = 2
                        temp_n = max(0.0, temp_n - PROBABILITY_DECAY)
                        three_way_roll = self.rng.get_float(
                            0.0, 100.0
                        )  # <- GAME RNG USED HERE
                        if three_way_roll <= temp_n:
                            num_branches = 3
                            temp_n = max(0.0, temp_n - PROBABILITY_DECAY)

            parent.probability_n = temp_n  # Update parent's probability
            step_vars.update(
                {
                    "did_split": should_split,
                    "final_num_branches": num_branches,
                    "final_n": round(temp_n, 2),
                }
            )

            # --- Perform Action (Add Node or Branch) ---
            nodes_added_this_step = []
            if not should_split:
                # Add a single continuing node
                step_vars["final_action"] = "add_single"
                momentum_bias = parent.last_angle_delta * self.branch_momentum_bias_rate
                # USE self.rng for random component
                random_component = self.rng.get_float(
                    self.branch_left_range, self.branch_right_range
                )  # <- GAME RNG USED HERE
                angle_delta = max(
                    ANGLE_CLAMP_SINGLE[0],
                    min(ANGLE_CLAMP_SINGLE[1], momentum_bias + random_component),
                )
                # Attempt to add the node
                new_node = self._add_node(
                    parent,
                    angle_delta,
                    parent.probability_n,
                    parent.branch_segment_count + 1,
                )
                if new_node:
                    nodes_added_this_step.append(new_node)
                else:
                    # If add_node failed (e.g., max nodes), mark parent to stop
                    parent.can_grow = False
            else:
                # Branching logic
                self._adjust_counts()  # Adjust weights before branching
                step_vars["final_action"] = f"branch_start_{num_branches}"
                available_slots = self.max_nodes - len(self.nodes)
                if available_slots < num_branches:
                    self.steps.append(CaveStep("Branch slots limited", step_vars))
                    num_branches = max(0, available_slots)

                # Calculate branch angles
                angles = []
                # USE self.rng for random offsets
                if num_branches == 2:
                    offset = self.rng.get_float(
                        *BRANCH_ANGLE_OFFSET_RANGE
                    )  # <- GAME RNG USED HERE
                    angles = [parent.angle - offset, parent.angle + offset]
                elif num_branches == 3:
                    offset1 = self.rng.get_float(
                        *BRANCH_ANGLE_OFFSET_3WAY_RANGE
                    )  # <- GAME RNG USED HERE
                    offset2 = self.rng.get_float(
                        *BRANCH_ANGLE_OFFSET_3WAY_RANGE
                    )  # <- GAME RNG USED HERE
                    angles = [
                        parent.angle - offset1,
                        parent.angle,
                        parent.angle + offset2,
                    ]

                parent.can_grow = False  # Parent stops growing after branching

                # Create branch nodes
                for i in range(num_branches):
                    try:
                        branch_base_angle = angles[i]
                        momentum_bias = (
                            parent.last_angle_delta * self.branch_momentum_bias_rate
                        )
                        # USE self.rng for random component
                        random_component = self.rng.get_float(
                            self.branch_left_range, self.branch_right_range
                        )  # <- GAME RNG USED HERE
                        angle_delta = max(
                            ANGLE_CLAMP_BRANCH[0],
                            min(
                                ANGLE_CLAMP_BRANCH[1], momentum_bias + random_component
                            ),
                        )
                        target_angle = branch_base_angle + angle_delta
                        # Calculate the delta relative to parent for _add_node
                        actual_delta_for_add_node = target_angle - parent.angle

                        new_node = self._add_node(
                            parent,
                            actual_delta_for_add_node,
                            parent.probability_n,  # Pass parent's potentially reduced prob
                            1,  # Start new branch segment count
                        )
                        if new_node:
                            # Ensure the angle is set correctly after potential modifications
                            new_node.angle = target_angle
                            nodes_added_this_step.append(new_node)
                    except Exception as e:
                        print(f"ERROR in branch loop {parent.id} (i={i}): {e}")
                        traceback.print_exc()

                if num_branches > 0:
                    self.steps.append(
                        CaveStep(
                            f"Branched ({num_branches}) from {
                                parent.id}",
                            step_vars,
                        )
                    )

            # Add successfully created new nodes to the active list for processing
            # Add in reverse to maintain depth-first-like exploration potentially
            for node in reversed(nodes_added_this_step):
                if node.can_grow:
                    self.active_nodes.append(node)

            # Rebuild KDTree periodically
            if self._nodes_since_kdtree >= KDTREE_REBUILD_INTERVAL:
                self._rebuild_kdtree()

        # --- End While Loop ---
        self._generation_complete = True
        print(f"Generation finished. Total nodes: {len(self.nodes)}")
        self.steps.append(
            CaveStep("Generation Complete", {"node_count": len(self.nodes)})
        )

    def to_json(self, include_steps=False) -> str:  # Unchanged
        """Serializes the generated cave data to JSON."""
        # Include generation settings used
        gen_settings = {
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "initial_probability": self.initial_probability,
            "DEPTH_METERS_PER_LEVEL_RANGE": DEPTH_METERS_PER_LEVEL_RANGE,
            "SEGMENT_LENGTH_RANGE": SEGMENT_LENGTH_RANGE,
            "LOOPS_ENABLED": self.loops_enabled,
            "CLIFF_FROM_LOW_PROB_CHANCE": self.cliff_from_low_prob_chance,
            "RESTART_FROM_CLIFF_CHANCE": self.restart_from_cliff_chance,
            "CONVERGENCE_FEATURE_CHANCE": self.convergence_feature_chance,
            "CONVERGENCE_FEATURE_TYPE": self.convergence_feature_type,
            "BIG_ROOM_CHANCE": self.big_room_chance,
            "CAVERN_TYPES": CAVERN_TYPES,
        }
        data = {
            "nodes": [n.to_dict() for n in self.nodes],
            "generation_settings": gen_settings,
        }
        if include_steps:
            max_steps_to_save = 5000  # Limit log size
            steps_to_save = self.steps[-max_steps_to_save:]
            data["steps"] = [s.to_dict() for s in steps_to_save]
            if len(self.steps) > max_steps_to_save:
                print(f"Warning: Saving only last {max_steps_to_save} steps.")
        return json.dumps(
            data, indent=2, default=str
        )  # default=str handles numpy types


# --- Example Usage (Modified for isolated testing) ---
if __name__ == "__main__":
    # This block should now only contain code for testing *core.py in isolation*
    # It should NOT run the full pipeline or expect arguments like
    # --rng-state-file
    print("Running core.py in isolation for testing...")
    try:
        import time

        test_start_time = time.time()
        # Instantiate a local RNG for testing purposes ONLY
        test_rng_seed = 1  # Fixed seed for test consistency
        test_rng = GameRNG(seed=test_rng_seed)
        print(f"Using Test RNG Seed: {test_rng_seed}")

        # Use smaller parameters for faster testing
        INSTANCE_MAX_NODES = 50
        INSTANCE_MAX_DEPTH = 10
        print(
            f"Test Params: Max Nodes={
                INSTANCE_MAX_NODES}, Max Depth={INSTANCE_MAX_DEPTH}"
        )

        test_gen = CaveGenerator(
            max_nodes=INSTANCE_MAX_NODES, max_depth=INSTANCE_MAX_DEPTH, rng=test_rng
        )
        test_gen.grow()
        print(f"Test generation complete: {len(test_gen.nodes)} nodes.")

        # Optionally save a small test JSON
        test_json = test_gen.to_json(include_steps=False)
        output_filename = "core_test_output.json"
        with open(output_filename, "w") as f:
            f.write(test_json)
        print(f"Test cave data saved to {output_filename}")

        # Print feature summary for test run
        cliff_edges = sum(1 for n in test_gen.nodes if n.feature == "cliff_edge")
        shaft_openings = sum(1 for n in test_gen.nodes if n.feature == "shaft_opening")
        restarts = sum(
            1 for n in test_gen.nodes if n.feature == "restarted_below_cliff"
        )
        big_rooms = sum(
            1 for n in test_gen.nodes if n.feature and n.feature.startswith("big_room")
        )
        print(
            f"Test Features generated: Cliffs={
                cliff_edges}, Shafts={shaft_openings}, "
            f"Restarts={restarts}, BigRooms={big_rooms}"
        )
        print(f"Test Cavern type counts: {test_gen.cavern_type_counts}")

        test_end_time = time.time()
        print(
            f"Total core.py isolation test time: {
                test_end_time - test_start_time:.2f} seconds"
        )

    except Exception as e:
        print(f"Error during isolated core test: {e}")
        traceback.print_exc()

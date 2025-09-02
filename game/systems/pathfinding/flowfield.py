# game/systems/pathfinding/flowfield.py
from typing import Final, List, Optional, Tuple

import heapq
import time

import numpy as np
import structlog  # Added

# --- Numba Acceleration ---
try:
    from numba import njit

    # For type hinting Numba dict if needed
    from numba.typed import Dict as NumbaDict

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    NumbaDict = dict  # Fallback type hint

    def njit(func=None, **options):
        if func:
            return func
        return lambda f: f


log = structlog.get_logger(__name__)

# --- Type Aliases ---
GridPosition = Tuple[int, int]  # (y, x) format

# --- Constants ---
DIRECTIONS_8: Final[np.ndarray] = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
    dtype=np.int8,
)
DIAGONAL_MOVE_COST: Final[float] = np.sqrt(2.0)
# Multiplier for height difference cost
DEFAULT_HEIGHT_COST_FACTOR: Final[float] = 0.5


# --- Numba Helper Functions ---
@njit(cache=True, fastmath=True)
def _calculate_light_intensity_scalar(
    dist_sq: np.float32,
    radius_sq: np.float32,
    falloff_power: np.float32,
    min_light_level: np.float32,
) -> np.float32:
    """Calculates light intensity based on distance squared."""
    if radius_sq < 0:
        return np.float32(1.0)
    if dist_sq > radius_sq:
        return np.float32(0.0)
    if dist_sq < 1e-6:
        return np.float32(1.0)
    if radius_sq <= 1e-6:
        return np.float32(0.0)

    dist = np.sqrt(dist_sq)
    radius = np.sqrt(radius_sq)
    falloff_ratio = dist / radius
    light_value = max(np.float32(0.0), np.float32(1.0 - falloff_ratio)) ** falloff_power
    intensity = max(light_value, min_light_level)
    return max(np.float32(0.0), min(np.float32(1.0), intensity))


# Use vectorize instead of relying on hasattr in Numba
_calculate_light_intensity_vectorized = np.vectorize(
    _calculate_light_intensity_scalar,
    otypes=[np.float32],
    excluded=["radius_sq", "falloff_power", "min_light_level"],
)


@njit(cache=True, fastmath=True)
def _calculate_flow_vectors_numba(
    integration_field: np.ndarray,
    passable_map: np.ndarray,  # Need this to avoid flowing into walls
    flow_x: np.ndarray,  # int8[:] - Modified in place
    flow_y: np.ndarray,  # int8[:] - Modified in place
):
    """
    Calculates the flow vectors based on the integration field.
    Points each cell towards the neighbour with the lowest integration cost.
    """
    height, width = integration_field.shape
    infinity = np.inf

    for y in range(height):
        for x in range(width):
            if not passable_map[y, x] or integration_field[y, x] == infinity:
                flow_x[y, x] = 0
                flow_y[y, x] = 0
                continue

            min_cost = integration_field[y, x]
            best_dx, best_dy = 0, 0

            for i in range(DIRECTIONS_8.shape[0]):
                dy, dx = DIRECTIONS_8[i]
                ny, nx = y + dy, x + dx

                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_cost = integration_field[ny, nx]
                    if neighbor_cost < min_cost:
                        min_cost = neighbor_cost
                        best_dx, best_dy = dx, dy

            flow_x[y, x] = best_dx
            flow_y[y, x] = best_dy


class FlowFieldPathfinder:
    """
    Manages creation of Integration and Flow Fields using NumPy.
    Includes height difference cost calculation.
    """

    def __init__(
        self,
        passable_map: np.ndarray,
        terrain_cost_map: np.ndarray,
        height_map: np.ndarray,  # ADDED
        max_traversable_step: int,  # ADDED
        height_cost_factor: float = DEFAULT_HEIGHT_COST_FACTOR,  # ADDED
    ):
        # (Validation logic largely unchanged - Source [source 1551-1553])
        if passable_map.shape != terrain_cost_map.shape:
            raise ValueError(
                "Passable map and terrain cost map must have the same shape."
            )
        if passable_map.shape != height_map.shape:  # ADDED height map shape check
            raise ValueError("Passable map and height map must have the same shape.")
        if not np.issubdtype(passable_map.dtype, np.bool_):
            raise TypeError("passable_map must be a boolean NumPy array.")
        if not np.issubdtype(terrain_cost_map.dtype, np.floating):
            raise TypeError("terrain_cost_map must be a floating-point NumPy array.")
        if not np.issubdtype(
            height_map.dtype, np.integer
        ):  # ADDED height map type check
            raise TypeError("height_map must be an integer NumPy array.")

        self.passable: np.ndarray = passable_map.copy()
        self.terrain_cost: np.ndarray = terrain_cost_map.copy()
        self.height_map: np.ndarray = height_map.copy()  # ADDED: Store height map
        self.max_traversable_step: int = max_traversable_step  # ADDED
        self.height_cost_factor: float = height_cost_factor  # ADDED

        # Ensure costs are positive
        self.terrain_cost[self.terrain_cost <= 0] = 1.0

        self.height, self.width = self.passable.shape

        # (Field initializations unchanged - Source [source 1554])
        self.integration_field: np.ndarray = np.full(
            self.passable.shape, np.inf, dtype=np.float32
        )
        self.flow_x: np.ndarray = np.zeros(self.passable.shape, dtype=np.int8)
        self.flow_y: np.ndarray = np.zeros(self.passable.shape, dtype=np.int8)
        self._last_sources: Optional[List[GridPosition]] = None

    def compute_field(self, stimulus_sources: List[GridPosition]) -> bool:
        # (Initial checks and source validation unchanged - Source [source 1555-1559])
        if not stimulus_sources:
            log.warning("No stimulus sources provided.")
            self.integration_field.fill(np.inf)
            self.flow_x.fill(0)
            self.flow_y.fill(0)
            self._last_sources = None
            return False

        if self._last_sources and set(stimulus_sources) == set(self._last_sources):
            log.info("Skipping field computation: sources unchanged.")
            return True

        log.info(
            f"Computing flow field from {
                len(stimulus_sources)} sources (with height cost)..."
        )
        start_time = time.time()

        self.integration_field.fill(np.inf)
        valid_sources = [
            s
            for s in stimulus_sources
            if 0 <= s[0] < self.height and 0 <= s[1] < self.width
        ]
        if not valid_sources:
            log.warning("All provided stimulus sources are out of bounds.")
            self._last_sources = list(stimulus_sources)
            return False

        pq = []  # Min-heap: [(cost, y, x)]
        for y_s, x_s in valid_sources:
            if self.passable[y_s, x_s]:
                cost = 0.0
                self.integration_field[y_s, x_s] = cost
                heapq.heappush(pq, (cost, y_s, x_s))
            else:
                log.warning(f"Source at ({y_s}, {x_s}) is on impassable terrain.")
        if not pq:
            log.warning("No valid stimulus sources on passable terrain.")
            self._last_sources = list(stimulus_sources)
            return False

            # --- Dijkstra Loop ---
        processed_count = 0
        while pq:
            cost, y, x = heapq.heappop(pq)
            processed_count += 1

            if cost > self.integration_field[y, x]:
                continue

            current_h = self.height_map[y, x]  # Get height of current cell

            # Explore neighbors
            for i in range(DIRECTIONS_8.shape[0]):
                dy, dx = DIRECTIONS_8[i]
                ny, nx = y + dy, x + dx

                # Check bounds
                if not (0 <= ny < self.height and 0 <= nx < self.width):
                    continue

                # --- MODIFIED: Check Passability AND Height Difference ---
                neighbor_h = self.height_map[ny, nx]
                delta_h = abs(neighbor_h - current_h)

                if self.passable[ny, nx] and delta_h <= self.max_traversable_step:
                    # --- Calculate Move Cost with Height Penalty ---
                    # Cost to enter neighbor
                    base_move_cost = self.terrain_cost[ny, nx]
                    diagonal_penalty = 0.0
                    if dy != 0 and dx != 0:  # Diagonal move
                        # Apply diagonal cost multiplier to base cost
                        diagonal_penalty = base_move_cost * (DIAGONAL_MOVE_COST - 1.0)

                    # Calculate height penalty (simple linear penalty for change)
                    height_penalty = delta_h * self.height_cost_factor
                    # Optional: Penalize uphill movement more?
                    # height_penalty = max(0, neighbor_h - current_h) * self.height_cost_factor

                    move_cost = base_move_cost + diagonal_penalty + height_penalty
                    new_cost = cost + move_cost
                    # --- End Cost Calculation ---

                    if new_cost < self.integration_field[ny, nx]:
                        self.integration_field[ny, nx] = new_cost
                        heapq.heappush(pq, (new_cost, ny, nx))
                # --- END MODIFICATION ---

        integration_time = time.time()
        log.info(
            f"  Integration field computed ({processed_count} nodes processed) in {
                integration_time - start_time:.4f}s"
        )

        # Calculate Flow Vectors (unchanged Numba call)
        self.flow_x.fill(0)
        self.flow_y.fill(0)
        _calculate_flow_vectors_numba(
            self.integration_field, self.passable, self.flow_x, self.flow_y
        )

        flow_time = time.time()
        log.info(
            f"  Flow vectors calculated in {
                 flow_time - integration_time:.4f}s"
        )
        log.info(
            f"Total field computation time: {
                 flow_time - start_time:.4f}s"
        )

        self._last_sources = list(stimulus_sources)
        return True

    # (get_flow_vector, get_flow_field, get_integration_field methods unchanged - Source [source 1568-1570])
    def get_flow_vector(self, y: int, x: int) -> GridPosition:
        """Returns the optimal flow vector (dx, dy) for a given cell."""
        if 0 <= y < self.height and 0 <= x < self.width:
            return (int(self.flow_x[y, x]), int(self.flow_y[y, x]))
        else:
            return (0, 0)

    def get_flow_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the complete flow field arrays (dx, dy components)."""
        return self.flow_x, self.flow_y

    def get_integration_field(self) -> np.ndarray:
        """Returns the computed integration field (cost to reach source)."""
        return self.integration_field


# --- Example Usage (Updated) ---
if __name__ == "__main__":

    # 1. Create Map Data (Includes height map now)
    map_h, map_w = 50, 70
    passable = np.ones((map_h, map_w), dtype=bool)
    costs = np.ones((map_h, map_w), dtype=np.float32)
    heights = np.zeros((map_h, map_w), dtype=np.int16)  # Base height 0

    # Add walls
    passable[map_h // 2, map_w // 4 : map_w * 3 // 4] = False
    passable[map_h // 4 : map_h * 3 // 4, map_w // 2] = False
    passable[10:15, 10:15] = False  # Box wall

    # Add varying heights
    heights[15:35, 15:35] = 2  # Plateau
    heights[20:30, 20:30] = 4  # Higher plateau
    heights[map_h // 2 + 5 : map_h // 2 + 15, map_w // 2 + 10 : map_w // 2 + 20] = (
        -2
    )  # Depression

    # Add high-cost terrain
    costs[40:48, 10:30] = 5.0  # Mud costs 5x

    # Set max step allowed (e.g., 1 meter = 2 height units)
    max_step = 2
    height_factor = 0.75  # Make height changes fairly costly

    log.info(f"Created map: {map_w}x{map_h} with height variations")
    log.info(
        f"Max traversable step: {
             max_step}, Height cost factor: {height_factor}"
    )

    # 2. Initialize Pathfinder (Pass height map and max step)
    pathfinder = FlowFieldPathfinder(
        passable,
        costs,
        heights,  # Pass height map
        max_step,  # Pass step limit
        height_cost_factor=height_factor,
    )

    # 3. Define Stimulus Source(s)
    sources: List[GridPosition] = [(map_h - 5, map_w - 5)]  # Bottom-right corner

    # Check source validity
    if not pathfinder.passable[sources[0]]:
        log.warning(f"Source {sources[0]} is initially impassable!")

    # 4. Compute the field
    success = pathfinder.compute_field(sources)

    # 5. Get Flow Vector / Cost for an Agent
    if success:
        agent_pos_y, agent_pos_x = 5, 5  # Top-left corner agent
        if pathfinder.passable[agent_pos_y, agent_pos_x]:
            flow_dx, flow_dy = pathfinder.get_flow_vector(agent_pos_y, agent_pos_x)
            log.info(
                f"Agent at ({agent_pos_y}, {agent_pos_x}) should move by: ({
                    flow_dx}, {flow_dy})"
            )

            cost_from_agent = pathfinder.get_integration_field()[
                agent_pos_y, agent_pos_x
            ]
            if cost_from_agent == np.inf:
                log.info("Cost from agent position to source: UNREACHABLE")
            else:
                log.info(
                    f"Cost from agent position to source: {
                         cost_from_agent:.2f}"
                )
        else:
            log.warning(
                f"Agent at ({agent_pos_y}, {
                        agent_pos_x}) is on impassable terrain."
            )

        # Test agent on plateau
        agent_on_plateau_y, agent_on_plateau_x = 25, 25
        if pathfinder.passable[agent_on_plateau_y, agent_on_plateau_x]:
            flow_dx_p, flow_dy_p = pathfinder.get_flow_vector(
                agent_on_plateau_y, agent_on_plateau_x
            )
            log.info(
                f"Agent at ({agent_on_plateau_y}, {agent_on_plateau_x}) [H={
                    heights[agent_on_plateau_y, agent_on_plateau_x]}] should move by: ({flow_dx_p}, {flow_dy_p})"
            )
            cost_from_plateau = pathfinder.get_integration_field()[
                agent_on_plateau_y, agent_on_plateau_x
            ]
            log.info(
                f"Cost from plateau agent to source: {
                     cost_from_plateau:.2f}"
            )

        # (Visualization code unchanged - Source [source 1574-1578])
        print("\nVisualizing Flow Field (Sample):")
        flow_x_field, flow_y_field = pathfinder.get_flow_field()
        viz_step = 3
        dir_symbols = {
            (-1, 0): "^",
            (-1, 1): "/",
            (0, 1): ">",
            (1, 1): "\\",
            (1, 0): "v",
            (1, -1): "/",
            (0, -1): "<",
            (-1, -1): "\\",
            (0, 0): "·",
        }
        for y in range(0, map_h, viz_step):
            row_str = ""
            for x in range(0, map_w, viz_step):
                if not passable[y, x]:
                    row_str += "###"
                elif (y, x) in sources:
                    row_str += " S "
                else:
                    dy = int(flow_y_field[y, x])
                    dx = int(flow_x_field[y, x])
                    symbol = dir_symbols.get((dy, dx), "?")
                    # Add space for alignment
                    row_str += f" {symbol} "
            print(row_str)

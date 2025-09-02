# auto/simulation.py
# Updated to use GameRNG instance

import heapq

# Removed 'random' import
import time
import typing  # Use typing instead of from typing import ...
import uuid
from collections import defaultdict, deque

import polars as pl
import structlog

log = structlog.get_logger(__name__)

# --- Use project GameRNG directly ---
try:
    from game_rng import GameRNG
except ImportError as e:  # pragma: no cover - fail fast if unavailable
    raise SystemExit(
        "GameRNG module is required for auto.simulation. Ensure game_rng.py is present"
    ) from e


# --- NumPy for potential future use ---
# Use Polars instead of direct numpy for entity state usually
# --- Polars DataFrame Library ---

# --- Numba JIT Compiler ---
try:
    from numba import njit
except ImportError:
    log.warning("Numba not found, distance calculation will be slower.")

    # Dummy decorator if Numba not available
    def njit(func=None, **options):  # type: ignore # noqa
        if func:
            return func
        else:
            return lambda f: f


# --- Type Hinting Aliases (Using modern types where possible) ---
Position = tuple[int, int]
EntityID = str
StateDict = dict[str, typing.Any]  # Keep Any for flexibility initially
# ActionPlan = deque["Action"] # Forward reference string needed if Action defined later
ActionPlan = deque["Action"]  # Use typing.Deque with forward reference

# Forward reference for Entity if used before definition
# If needed, otherwise define Entity first
Entity = typing.ForwardRef("Entity")
OptionalEntity = typing.Optional[Entity]
OptionalPosition = typing.Optional[Position]
OptionalPath = typing.Optional[list[Position]]
NearestResult = tuple[EntityID | None, float]
ActionResult = typing.Optional[str]
# Forward reference for Item if needed
Item = typing.ForwardRef("Item")  # If needed
OptionalItem = typing.Optional[Item]


# --- Item Definitions ---
class Item:
    """Base class for all items."""

    def __init__(self, name: str, kind: str, description: str = ""):
        self.id: str = str(uuid.uuid4())  # Use standard uuid4 string
        self.name = name
        self.kind = kind
        self.description = description

    def __repr__(self):
        return f"Item({self.name})"


class Consumable(Item):
    """Items that can be consumed for an effect."""

    def __init__(self, name: str, restores: str, amount: float, description: str = ""):
        super().__init__(name, "consumable", description)
        self.restores = restores
        self.amount = amount


class Weapon(Item):
    """Items that can be equipped to enhance attacks."""

    def __init__(self, name: str, damage: int, description: str = ""):
        super().__init__(name, "weapon", description)
        self.damage = damage


class Utility(Item):
    """Items with other uses (like torches)."""

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, "utility", description)


# Factory functions for items
def create_slime_mold() -> Consumable:
    return Consumable(
        "Slime Mold",
        restores="hunger",
        amount=40.0,
        description="A quivering, edible mold.",
    )


def create_health_potion() -> Consumable:
    return Consumable(
        "Health Potion",
        restores="health",
        amount=50.0,
        description="Restores some health.",
    )


def create_dagger() -> Weapon:
    return Weapon("Dagger", damage=5, description="A simple dagger.")


def create_torch() -> Utility:
    return Utility("Torch", description="Provides light (eventually).")


# --- Configuration Constants ---
GRID_SIZE: int = 15
MAX_TURNS: int = 200
START_HEALTH: float = 100.0
START_HUNGER: float = 100.0
SLIME_MOLD_HUNGER_RECOVERY: float = 40.0
HEALTH_POTION_RECOVERY: float = 50.0
BASE_AGENT_DAMAGE: int = 5
ENEMY_DMG: int = 15
SLIME_HEALTH: float = 15.0  # Use float for consistency
SLIME_DMG: int = 5
ENEMY_START_HEALTH_RANGE: tuple[int, int] = (40, 60)
ENEMY_FLEE_THRESHOLD: float = 0.25
ENEMY_SPAWN_CHANCE: float = 0.02  # Use rng.get_float() < CHANCE instead
# FOOD_RESPAWN_CHANCE: float = 0.10 # Not used currently? Remove if unused
PASSIVE_HUNGER_PER_TURN: float = 0.1
STARVATION_HEALTH_DAMAGE: float = 0.5
ATTACK_HUNGER_COST: float = 0.05
DEFEND_HUNGER_COST: float = 0.05
REST_HEALTH_REGEN: float = 0.05
LOW_HEALTH_FLEE_THRESHOLD: float = START_HEALTH * 0.4
ENEMY_NEARBY_FLEE_DISTANCE: int = 8
HEALTHY_THRESHOLD: float = START_HEALTH * 0.6
CRITICAL_HEALTH_THRESHOLD: float = START_HEALTH * 0.3
PLANNING_TIMEOUT: float = 0.1  # seconds
ACTION_WEIGHT_MIN: float = 0.1
ACTION_WEIGHT_MAX: float = 10.0
LEARNING_RATE_FACTOR: float = 0.15
LEARNING_SCORE_BASELINE: float = 0.6
AGENT_MAX_INVENTORY: int = 5


# --- Helper Functions ---
@njit(cache=True)  # Numba is good for simple math like this
def distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculates Manhattan distance."""
    return abs(x1 - x2) + abs(y1 - y2)


# --- Entity Class ---
class Entity:
    """Represents agents, enemies, or items on the ground."""

    def __init__(
        self,
        x: int,
        y: int,
        kind: str,
        rng: GameRNG,  # Add rng for potential random initialization
        health: Optional[float] = None,
        hunger: Optional[float] = None,
        target: OptionalEntity = None,
        item: OptionalItem = None,  # Use alias
        max_inventory: Optional[int] = None,  # Allow overriding default
    ):
        self.id: EntityID = str(uuid.uuid4())
        self.x: int = x
        self.y: int = y
        self.kind: str = kind
        self.target: OptionalEntity = target
        self.item: OptionalItem = item
        self.inventory: list[Item] = []  # Use modern type hint list[]

        # Set max inventory based on kind or override
        if max_inventory is not None:
            self.max_inventory: int = max_inventory
        else:
            self.max_inventory = AGENT_MAX_INVENTORY if kind == "agent" else 3

        # Use modern Optional hint
        self.equipped_weapon: Optional[Weapon] = None

        # --- Health Initialization ---
        if health is None:
            if kind == "agent":
                default_health = START_HEALTH
            elif kind == "enemy":
                # Use rng to determine starting health within range
                default_health = float(rng.get_int(*ENEMY_START_HEALTH_RANGE))
            elif kind == "slime":
                default_health = SLIME_HEALTH  # Already float
            else:  # Items etc.
                default_health = 0.0
            self.health: float = default_health
        else:
            self.health = float(health)

        # --- Hunger Initialization ---
        if hunger is None:
            # Assume relevant entities start full, others have no hunger concept
            self.hunger: float = (
                START_HUNGER if kind in ["agent", "enemy", "slime"] else 0.0
            )
        else:
            self.hunger = float(hunger)

    def get_position(self) -> Position:
        """Returns the entity's (x, y) position tuple."""
        return (self.x, self.y)

    def get_effective_damage(self) -> int:
        """Calculates damage based on equipment or base kind."""
        if self.equipped_weapon:
            return self.equipped_weapon.damage
        elif self.kind == "agent":
            return BASE_AGENT_DAMAGE
        elif self.kind == "enemy":
            return ENEMY_DMG
        elif self.kind == "slime":
            return SLIME_DMG
        return 0  # Items, etc., deal no damage

    def __repr__(self) -> str:
        """Provides a readable string representation of the entity."""
        details = []
        if self.kind in ["agent", "enemy", "slime"]:
            details.append(f"H:{self.health:.1f}")
            details.append(f"U:{self.hunger:.1f}")
        if self.item:
            details.append(f"Item:{self.item.name}")
        if self.equipped_weapon:
            details.append(f"W:{self.equipped_weapon.name}")
        if self.inventory:
            details.append(f"Inv:{len(self.inventory)}/{self.max_inventory}")
        detail_str = ", ".join(details)
        return (
            f"{self.kind.capitalize()}({self.id[:4]} @ ({self.x},{self.y})"
            + (f", {detail_str}" if detail_str else "")
            + ")"
        )


# --- World Class ---
class World:
    """Manages the simulation grid, entities, and game state."""

    def __init__(self, size: int, rng: GameRNG):  # Add rng to init
        if size <= 0:
            raise ValueError("Grid size must be positive.")
        self.size: int = size
        self.rng: GameRNG = rng  # Store RNG instance

        # --- State Attributes ---
        self.grid: list[list[OptionalEntity]] = [
            [None for _ in range(size)] for _ in range(size)
        ]
        self.entities: dict[EntityID, Entity] = {}
        self.entity_df: pl.DataFrame = self._create_empty_entity_df()
        self.entities_by_kind: defaultdict[str, dict[EntityID, Entity]] = defaultdict(
            dict
        )
        self.agent: OptionalEntity = None
        self.turn: int = 0
        self._free_tiles: set[Position] = set(
            (x, y) for x in range(size) for y in range(size)
        )

        # --- Constants accessible via instance if needed ---
        self.HUNGER_PER_TURN = PASSIVE_HUNGER_PER_TURN  # Example constant

    def _create_empty_entity_df(self) -> pl.DataFrame:
        """Creates an empty Polars DataFrame with the entity schema."""
        schema = {
            "id": pl.Utf8,
            "x": pl.Int32,
            "y": pl.Int32,
            "kind": pl.Utf8,
            "health": pl.Float32,
            "hunger": pl.Float32,
        }
        # Consider adding 'equipped_weapon_id', 'inventory_count' etc. if useful for queries
        return pl.DataFrame(schema=schema)

    def is_valid(self, x: int, y: int) -> bool:
        """Checks if coordinates are within the grid bounds."""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_entity_at(self, x: int, y: int) -> OptionalEntity:
        """Returns the entity at (x, y), or None if empty or out of bounds."""
        if not self.is_valid(x, y):
            return None
        # Grid access is typically fast for standard lists
        try:
            # Grid indexed [x][y] based on initialization
            return self.grid[x][y]
        except IndexError:  # Should not happen if is_valid works
            return None

    def get_entity_object(self, entity_id: EntityID) -> OptionalEntity:
        """Retrieves an entity object by its ID."""
        return self.entities.get(entity_id)

    def add_entity(self, entity: Entity) -> bool:
        """Adds an entity to the world if the position is valid and mostly empty."""
        pos = entity.get_position()
        if not self.is_valid(pos[0], pos[1]):
            log.warning("Cannot add entity outside grid", entity_id=entity.id, pos=pos)
            return False

        existing_entity = self.grid[pos[0]][pos[1]]

        # Allow placing items on floor, or agent/enemy displacing items
        can_place = False
        if existing_entity is None:
            can_place = True
        elif existing_entity.kind == "item" and entity.kind != "item":
            # Agent/Enemy overwrites item (item might be picked up later or destroyed)
            log.warning(
                "Entity displacing item",
                entity_kind=entity.kind,
                item_id=existing_entity.id,
                pos=pos,
            )
            self.remove_entity(existing_entity)  # Remove the item first
            can_place = True
        elif entity.kind == "item" and existing_entity.kind != "item":
            # Cannot place item on top of agent/enemy
            log.warning(
                "Cannot place item on occupied tile",
                item_id=entity.id,
                pos=pos,
                occupier=existing_entity.kind,
            )
            can_place = False
        elif entity.kind == "item" and existing_entity.kind == "item":
            # Allow stacking items? For now, disallow.
            log.warning(
                "Cannot stack item on existing item",
                item_id=entity.id,
                existing_item_id=existing_entity.id,
                pos=pos,
            )
            can_place = False
        else:  # Agent/Enemy trying to occupy same space
            can_place = False

        if not can_place:
            log.warning(
                "Position occupied; cannot add entity",
                pos=pos,
                entity_kind=entity.kind,
                entity_id=entity.id,
            )
            return False

        # Place entity
        self.grid[pos[0]][pos[1]] = entity
        self._free_tiles.discard(pos)  # Remove from free tiles
        self.entities[entity.id] = entity
        self.entities_by_kind[entity.kind][entity.id] = entity

        # Update DataFrame only for non-items (performance)
        if entity.kind != "item":
            # Use compatible dtypes for DataFrame construction
            new_entity_data = {
                "id": [entity.id],
                "x": [entity.x],
                "y": [entity.y],
                "kind": [entity.kind],
                "health": [entity.health],
                "hunger": [entity.hunger],
            }
            try:
                # Ensure schema matches, create new DF safely
                new_df = pl.DataFrame(
                    new_entity_data,
                    schema_overrides={
                        "id": pl.Utf8,
                        "x": pl.Int32,
                        "y": pl.Int32,
                        "kind": pl.Utf8,
                        "health": pl.Float32,
                        "hunger": pl.Float32,
                    },
                )
                # Check if entity_df is empty before concatenating
                if self.entity_df.is_empty():
                    self.entity_df = new_df
                else:
                    self.entity_df = pl.concat([self.entity_df, new_df], how="vertical")
            except Exception as e:
                log.error(
                    "Error updating Polars DataFrame", error=str(e), exc_info=True
                )
                # Potentially fallback or log error, but proceed with adding to dicts/grid

        # Assign agent reference if applicable
        if entity.kind == "agent":
            self.agent = entity

        return True

    def remove_entity(self, entity: Entity):
        """Removes an entity from the world state."""
        entity_id = entity.id
        if entity_id in self.entities:
            pos = self.entities[entity_id].get_position()
            del self.entities[entity_id]

            # Safely clear grid position
            if self.is_valid(pos[0], pos[1]) and self.grid[pos[0]][pos[1]] == entity:
                self.grid[pos[0]][pos[1]] = None
                self._free_tiles.add(pos)  # Add back to free tiles

            # Remove from kind dict
            self.entities_by_kind[entity.kind].pop(entity_id, None)

            # Remove from DataFrame if not item
            if entity.kind != "item":
                if not self.entity_df.is_empty() and "id" in self.entity_df.columns:
                    self.entity_df = self.entity_df.filter(pl.col("id") != entity_id)

            # Clear agent reference if it's the agent being removed
            if self.agent and entity_id == self.agent.id:
                self.agent = None

    def update_entity_health(self, entity_id: EntityID, new_health: float):
        """Updates entity health in object and DataFrame, clamping values."""
        entity_obj = self.get_entity_object(entity_id)
        if entity_obj is None:
            return

        # Clamp health between 0 and START_HEALTH (or a max_health attribute if exists)
        # Use START_HEALTH as a proxy for max health for now
        clamped_health = max(0.0, min(START_HEALTH, new_health))
        entity_obj.health = clamped_health

        # Update DataFrame if it's not an item
        if entity_obj.kind != "item":
            if not self.entity_df.is_empty() and "id" in self.entity_df.columns:
                self.entity_df = self.entity_df.with_columns(
                    pl.when(pl.col("id") == entity_id)
                    .then(pl.lit(clamped_health, dtype=pl.Float32))
                    .otherwise(pl.col("health"))
                    .alias("health")
                )

    def update_entity_hunger(self, entity_id: EntityID, new_hunger: float):
        """Updates entity hunger in object and DataFrame, clamping values."""
        entity_obj = self.get_entity_object(entity_id)
        if entity_obj is None:
            return

        # Clamp hunger between 0 and START_HUNGER
        clamped_hunger = max(0.0, min(START_HUNGER, new_hunger))
        entity_obj.hunger = clamped_hunger

        # Update DataFrame if relevant kind
        if entity_obj.kind in ["agent", "enemy", "slime"]:
            if not self.entity_df.is_empty() and "id" in self.entity_df.columns:
                self.entity_df = self.entity_df.with_columns(
                    pl.when(pl.col("id") == entity_id)
                    .then(pl.lit(clamped_hunger, dtype=pl.Float32))
                    .otherwise(pl.col("hunger"))
                    .alias("hunger")
                )

    def move_entity(self, entity: Entity, new_x: int, new_y: int) -> bool:
        """Moves an entity to a new position if valid and unoccupied (by non-items)."""
        if entity.kind == "item":
            # Items generally shouldn't move on their own
            return False

        entity_id = entity.id
        current_pos = entity.get_position()

        # Check validity of new position
        if not self.is_valid(new_x, new_y):
            return False

        # Check if target cell is occupied by something other than an item
        target_entity = self.grid[new_x][new_y]
        if target_entity is not None and target_entity.kind != "item":
            return False  # Blocked

        # Check if current position actually holds the entity (consistency check)
        if (
            not self.is_valid(current_pos[0], current_pos[1])
            or self.grid[current_pos[0]][current_pos[1]] != entity
        ):
            # This might indicate a state inconsistency
            log.warning(
                "Entity not found at expected position",
                entity_id=entity_id,
                expected_pos=current_pos,
            )
            # Allow move anyway if target is clear? Or return False? Let's be strict.
            return False

        # --- Perform Move ---
        # Clear old position
        self.grid[current_pos[0]][current_pos[1]] = None
        self._free_tiles.add(current_pos)

        # If target cell had an item, remove it (entity moves onto it)
        if target_entity is not None and target_entity.kind == "item":
            log.warning(
                "Entity moving onto item; item removed",
                entity_kind=entity.kind,
                item_id=target_entity.id,
                new_x=new_x,
                new_y=new_y,
            )
            self.remove_entity(target_entity)

        # Set new position
        self.grid[new_x][new_y] = entity
        self._free_tiles.discard((new_x, new_y))
        entity.x, entity.y = new_x, new_y

        # Update DataFrame
        if not self.entity_df.is_empty() and "id" in self.entity_df.columns:
            self.entity_df = self.entity_df.with_columns(
                pl.when(pl.col("id") == entity_id)
                .then(pl.lit(new_x, dtype=pl.Int32))
                .otherwise(pl.col("x"))
                .alias("x"),
                pl.when(pl.col("id") == entity_id)
                .then(pl.lit(new_y, dtype=pl.Int32))
                .otherwise(pl.col("y"))
                .alias("y"),
            )
        return True

    def get_entities_by_kind_df(self, kind: str) -> pl.DataFrame:
        """Efficiently gets a DataFrame subset for a specific entity kind."""
        if self.entity_df.is_empty():
            return self._create_empty_entity_df()  # Return empty DF with schema
        # Ensure 'kind' column exists before filtering
        if "kind" in self.entity_df.columns:
            return self.entity_df.filter(pl.col("kind") == kind)
        else:
            log.warning("'kind' column not found in entity_df")
            return self._create_empty_entity_df()

    def get_nearest_entity(  # Unchanged logic, uses Polars/dict lookup
        self,
        source_entity: Entity,
        kind: str,
        max_dist: int | None = None,
    ) -> NearestResult:
        """Finds the nearest entity of a specific kind using Manhattan distance."""
        sx, sy = source_entity.get_position()

        if kind != "item":  # Use Polars for agent/enemy/slime etc.
            target_df = self.get_entities_by_kind_df(kind)
            if target_df.is_empty():
                return None, float("inf")

            # Calculate Manhattan distances efficiently
            distances = (target_df["x"] - sx).abs() + (target_df["y"] - sy).abs()
            target_df_with_dist = target_df.with_columns(distances.alias("dist"))

            # Filter by max_dist if provided
            if max_dist is not None:
                target_df_filtered = target_df_with_dist.filter(
                    pl.col("dist") <= max_dist
                )
                if target_df_filtered.is_empty():
                    return None, float("inf")
                target_df_to_search = target_df_filtered
            else:
                target_df_to_search = target_df_with_dist

            # Find the minimum distance index
            min_dist_idx = target_df_to_search["dist"].arg_min()

            if min_dist_idx is not None:
                # Extract ID and distance using the index
                nearest_row = target_df_to_search.row(min_dist_idx, named=True)
                nearest_id = nearest_row["id"]
                min_distance = nearest_row["dist"]
                return nearest_id, float(min_distance)
            else:
                # Should not happen if df wasn't empty, but handle defensively
                return None, float("inf")

        else:  # Handle items using the dictionary (usually fewer items)
            min_d = float("inf")
            nearest_id = None
            # Safely iterate using .get()
            for entity_id_loop, entity_obj in self.entities_by_kind.get(
                kind, {}
            ).items():
                # Check if it's the source entity itself (e.g., finding nearest *other* item)
                if entity_obj.id == source_entity.id:
                    continue
                d = distance(sx, sy, entity_obj.x, entity_obj.y)
                if d < min_d and (max_dist is None or d <= max_dist):
                    min_d = d
                    nearest_id = entity_id_loop
            return nearest_id, min_d

    def _get_random_free_pos(
        self, rng: GameRNG
    ) -> OptionalPosition:  # Added rng parameter
        """Gets a random free position from the available set."""
        if not self._free_tiles:
            return None
        try:
            # Use rng.choice for deterministic selection from the set
            # Convert set to list for choice
            return rng.choice(list(self._free_tiles))
        except IndexError:  # Should not happen if set is not empty
            return None

    def populate_world(  # Added rng parameter
        self,
        rng: GameRNG,
        num_food: int,  # Currently unused, slime mold drops instead
        num_enemies: int,
        num_slimes: int = 3,
        num_items: int = 2,
    ):
        """Populates the world with agent, enemies, and items using RNG."""
        # Place Agent
        agent_start_pos = self._get_random_free_pos(rng)
        if agent_start_pos:
            agent = Entity(
                agent_start_pos[0],
                agent_start_pos[1],
                "agent",
                rng=rng,  # Pass rng
                health=START_HEALTH,
                hunger=START_HUNGER,
            )
            self.add_entity(agent)
        else:
            log.error("No space to place agent")
            # Consider raising an exception here?
            return

        # Place Enemies
        for _ in range(num_enemies):
            pos = self._get_random_free_pos(rng)
            if pos:
                enemy = Entity(
                    pos[0],
                    pos[1],
                    "enemy",
                    rng=rng,  # Pass rng
                    hunger=START_HUNGER,  # Health determined randomly in __init__
                )
                self.add_entity(enemy)

        # Place Slimes
        for _ in range(num_slimes):
            pos = self._get_random_free_pos(rng)
            if pos:
                slime = Entity(
                    pos[0],
                    pos[1],
                    "slime",
                    rng=rng,  # Pass rng
                    health=SLIME_HEALTH,
                    hunger=START_HUNGER,
                )
                self.add_entity(slime)

        # Place Items (Potions, Daggers)
        item_factories = [
            create_health_potion,
            create_dagger,
            create_torch,
        ]  # Example items
        for _ in range(num_items):
            pos = self._get_random_free_pos(rng)
            if pos:
                # Use rng.choice to select item factory
                item_factory = rng.choice(item_factories)
                item_obj = item_factory()
                item_entity = Entity(
                    pos[0], pos[1], "item", rng=rng, item=item_obj
                )  # Pass rng
                self.add_entity(item_entity)

    def reset(  # Added rng parameter
        self,
        rng: GameRNG,
        num_food: int = 0,
        num_enemies: int = 4,
        num_slimes: int = 3,
        num_items: int = 2,
    ):
        """Resets the world state and repopulates it using the provided RNG."""
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.entities = {}
        self.entity_df = self._create_empty_entity_df()
        self.entities_by_kind = defaultdict(dict)
        self.agent = None
        self.turn = 0
        self._free_tiles = set(
            (x, y) for x in range(self.size) for y in range(self.size)
        )
        # Populate using the provided RNG instance
        self.populate_world(rng, num_food, num_enemies, num_slimes, num_items)

    def spawn_random_enemy(self, rng: GameRNG):  # Added rng parameter
        """Spawns a random enemy (Enemy or Slime) at a free position using RNG."""
        pos = self._get_random_free_pos(rng)
        if pos:
            # Use rng.get_float for chance check
            if rng.get_float() < 0.4:  # 40% chance for Slime
                new_enemy = Entity(
                    pos[0],
                    pos[1],
                    "slime",
                    rng=rng,  # Pass rng
                    health=SLIME_HEALTH,
                    hunger=START_HUNGER,
                )
            else:  # 60% chance for Enemy
                new_enemy = Entity(
                    pos[0],
                    pos[1],
                    "enemy",
                    rng=rng,  # Pass rng
                    hunger=START_HUNGER,  # Health determined randomly in __init__
                )
            self.add_entity(new_enemy)

    def find_path(  # Unchanged logic, does not use random
        self, start_pos: Position, target_pos: Position, max_path_length: int = 20
    ) -> OptionalPath:
        """
        Implements A* pathfinding algorithm.
        Returns a list of positions or None if no path found.
        """
        if not (
            self.is_valid(start_pos[0], start_pos[1])
            and self.is_valid(target_pos[0], target_pos[1])
        ):
            return None
        if start_pos == target_pos:
            return []

        open_set: list[tuple[float, Position]] = []  # Use list as heap
        came_from: dict[Position, Position] = {}
        g_score: dict[Position, float] = defaultdict(lambda: float("inf"))
        f_score: dict[Position, float] = defaultdict(lambda: float("inf"))

        g_score[start_pos] = 0.0
        f_score[start_pos] = float(
            distance(start_pos[0], start_pos[1], target_pos[0], target_pos[1])
        )
        heapq.heappush(open_set, (f_score[start_pos], start_pos))

        path_nodes_explored = 0  # Limit search space

        while open_set and path_nodes_explored < max_path_length * 4:
            path_nodes_explored += 1
            current_f, current_pos = heapq.heappop(open_set)

            # Optimization: If popped node has higher f_score than already found, skip
            if current_f > f_score[current_pos]:
                continue

            if current_pos == target_pos:
                # Reconstruct path
                path: list[Position] = []
                temp = current_pos
                while temp in came_from:
                    path.append(temp)
                    temp = came_from[temp]
                # path.append(start_pos) # Add start if needed, usually excluded
                path.reverse()
                return path  # Path from step after start to target

            # Explore neighbors (4 directions)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if not self.is_valid(neighbor_pos[0], neighbor_pos[1]):
                    continue

                neighbor_entity = self.grid[neighbor_pos[0]][neighbor_pos[1]]
                # Check walkability (allow moving onto items)
                if neighbor_entity is not None and neighbor_entity.kind not in [
                    "item",
                    "food",
                ]:
                    continue

                # Cost to move to neighbor is 1
                tentative_g_score = g_score[current_pos] + 1.0

                if tentative_g_score < g_score[neighbor_pos]:
                    # Found a better path
                    came_from[neighbor_pos] = current_pos
                    g_score[neighbor_pos] = tentative_g_score
                    f_score[neighbor_pos] = tentative_g_score + float(
                        distance(
                            neighbor_pos[0],
                            neighbor_pos[1],
                            target_pos[0],
                            target_pos[1],
                        )
                    )
                    heapq.heappush(open_set, (f_score[neighbor_pos], neighbor_pos))

        return None  # No path found


# --- GOAP Action Class ---
# Type aliases using modern syntax
CostCalculator = typing.Callable[["World", Entity], float]
PreconditionsChecker = typing.Callable[[StateDict], bool]
EffectsApplier = typing.Callable[[StateDict], StateDict]
Executor = typing.Callable[["World", Entity], bool]


class Action:  # Unchanged logic
    """Represents a possible action in the GOAP system."""

    def __init__(
        self,
        name: str,
        cost_calculator: CostCalculator,
        preconditions_checker: PreconditionsChecker,
        effects_applier: EffectsApplier,
        executor: Executor,
    ):
        self.name: str = name
        self.calculate_cost: CostCalculator = cost_calculator
        self.check_preconditions: PreconditionsChecker = preconditions_checker
        self.apply_effects_to_state: EffectsApplier = effects_applier
        self.execute: Executor = executor

    def __repr__(self) -> str:
        return f"Action({self.name})"


# --- GOAP Planner ---
class GOAPPlanner:  # Unchanged logic
    """Finds a sequence of actions to achieve a goal state."""

    def __init__(self, available_actions: list[Action]):
        self.actions: list[Action] = available_actions
        # Action weights for learning
        self.action_weights: defaultdict[str, float] = defaultdict(lambda: 1.0)

    def plan(
        self, world: World, agent: Entity, goal_state: StateDict
    ) -> Optional[list[Action]]:
        """Plans a sequence of actions using A* search."""
        start_time = time.time()
        initial_state = self._get_world_state_representation(world, agent)

        # Priority queue: (f_score, counter, state, plan_list, cost_so_far)
        open_list: list[tuple[float, int, StateDict, list[Action], float]] = [
            (self._heuristic(initial_state, goal_state), 0, initial_state, [], 0.0)
        ]
        # Use frozenset of items for hashable state in closed set
        closed_set: set[frozenset[tuple[str, typing.Any]]] = set()
        counter = 1  # Tie-breaker for heapq

        while open_list:
            # Check timeout
            if time.time() - start_time > PLANNING_TIMEOUT:
                # print("Planning timeout!") # Optional debug
                return None

            # Get node with lowest f_score
            f_score, _, current_state, action_plan, cost_so_far = heapq.heappop(
                open_list
            )

            # Check if goal is satisfied
            if self._goal_satisfied(current_state, goal_state):
                return action_plan  # Found plan

            # Add current state to closed set (use hashable representation)
            state_tuple = frozenset(current_state.items())
            if state_tuple in closed_set:
                continue
            closed_set.add(state_tuple)

            # Explore neighbors (possible actions)
            for action in self.actions:
                if action.check_preconditions(current_state):
                    # Calculate cost, skip if infinite (impossible action)
                    action_base_cost = action.calculate_cost(world, agent)
                    if action_base_cost == float("inf"):
                        continue

                    # Apply learned weights
                    weighted_cost = action_base_cost * self.action_weights[action.name]

                    # Apply effects to get next state
                    next_state = action.apply_effects_to_state(current_state.copy())
                    next_state_tuple = frozenset(next_state.items())

                    # Skip if already explored
                    if next_state_tuple in closed_set:
                        continue

                    # Calculate new costs and scores
                    new_cost = cost_so_far + weighted_cost
                    h = self._heuristic(next_state, goal_state)
                    new_f_score = new_cost + h
                    new_plan = action_plan + [action]

                    # Add to open list
                    heapq.heappush(
                        open_list,
                        (new_f_score, counter, next_state, new_plan, new_cost),
                    )
                    counter += 1  # Increment tie-breaker

        return None  # No plan found

    def _get_world_state_representation(self, world: World, agent: Entity) -> StateDict:
        """Creates a dictionary representing the current world state relevant to the agent."""
        # Get nearest entities (ensure max_dist is reasonable if used)
        item_id, item_dist = world.get_nearest_entity(agent, "item")
        enemy_id, enemy_dist = world.get_nearest_entity(agent, "enemy")
        slime_id, slime_dist = world.get_nearest_entity(agent, "slime")

        # Determine nearest hostile
        nearest_hostile_id = None
        nearest_hostile_dist = float("inf")
        if enemy_id and enemy_dist < nearest_hostile_dist:
            nearest_hostile_dist = enemy_dist
            nearest_hostile_id = enemy_id
        if slime_id and slime_dist < nearest_hostile_dist:
            nearest_hostile_dist = slime_dist
            nearest_hostile_id = slime_id  # Overwrite if slime is closer

        # Check inventory contents more specifically
        inventory_count = len(agent.inventory)
        has_slime_mold = any(
            isinstance(item, Consumable) and item.name == "Slime Mold"
            for item in agent.inventory
        )
        has_health_potion = any(
            isinstance(item, Consumable) and item.name == "Health Potion"
            for item in agent.inventory
        )
        has_weapon_in_inv = any(isinstance(item, Weapon) for item in agent.inventory)

        state = {
            "agent_health": agent.health,
            "agent_hunger": agent.hunger,
            "is_starving": agent.hunger <= 0,
            "agent_pos": agent.get_position(),
            "is_healthy": agent.health > HEALTHY_THRESHOLD,
            "is_critically_injured": agent.health < CRITICAL_HEALTH_THRESHOLD,
            "can_find_item": item_id is not None,
            "nearest_item_dist": item_dist,
            "item_is_adjacent": item_dist <= 1.0,
            "can_find_enemy": nearest_hostile_id is not None,
            "nearest_enemy_dist": nearest_hostile_dist,
            "enemy_is_adjacent": nearest_hostile_dist <= 1.0,
            "inventory_count": inventory_count,
            "inventory_full": inventory_count >= agent.max_inventory,
            "has_slime_mold": has_slime_mold,
            "has_health_potion": has_health_potion,
            "has_weapon_in_inv": has_weapon_in_inv,
            "weapon_equipped": agent.equipped_weapon is not None,
            # Add other potentially relevant state:
            # "enemies_present": len(world.entities_by_kind.get("enemy", {})) + len(world.entities_by_kind.get("slime", {})),
            # "current_turn": world.turn, # Maybe useful for time-sensitive goals?
        }
        return state

    def _goal_satisfied(
        self, current_state: StateDict, goal_state: StateDict
    ) -> bool:  # Unchanged logic
        """Checks if the current state satisfies all conditions in the goal state."""
        if not goal_state:  # Empty goal is always satisfied
            return True
        for key, desired_value in goal_state.items():
            current_value = current_state.get(key)  # Use .get for safety

            # Determine if this specific condition is satisfied
            satisfied = False
            if isinstance(desired_value, bool):
                satisfied = current_value == desired_value
            elif callable(desired_value):
                # Ensure current_value is passed to callable, handle None if key missing
                satisfied = (
                    desired_value(current_value) if current_value is not None else False
                )
            elif (
                current_value is None
            ):  # Goal requires a key that isn't in current state
                satisfied = False
            else:  # Assume direct comparison for other types
                satisfied = current_value == desired_value

            # If *any* condition is not satisfied, the goal is not met
            if not satisfied:
                return False
        # If loop completes without returning False, all conditions were met
        return True

    def _heuristic(
        self, state: StateDict, goal_state: StateDict
    ) -> float:  # Unchanged logic
        """Estimates the cost from the current state to the goal state."""
        cost: float = 0.0
        # Basic cost: count unsatisfied goal conditions
        for key, desired_value in goal_state.items():
            current_value = state.get(key)
            is_satisfied = False
            if isinstance(desired_value, bool):
                is_satisfied = current_value == desired_value
            elif callable(desired_value):
                is_satisfied = (
                    desired_value(current_value) if current_value is not None else False
                )
            elif current_value is None:
                is_satisfied = False
            else:
                is_satisfied = current_value == desired_value
            if not is_satisfied:
                cost += 1.0  # Add cost for each unmet condition

        # --- More specific heuristic adjustments ---
        # Penalize being starving more heavily, reduce cost if food nearby/in inventory
        if goal_state.get("is_not_starving", False) and state.get("is_starving", True):
            cost += 2.0
            if state.get("has_slime_mold"):
                cost += 0.1  # Small cost if food in inventory
            elif state.get("can_find_item"):
                cost += state.get("nearest_item_dist", 0) / 2.0

        # Penalize low health, reduce cost if potion nearby/in inventory
        if goal_state.get("is_healthy", False) and not state.get("is_healthy", False):
            cost += 1.0
            if state.get("has_health_potion"):
                cost += 0.1
            elif state.get("can_find_item"):
                cost += state.get("nearest_item_dist", 0) / 3.0

        # Heuristic for fleeing: cost is proportional to how close the enemy still is
        if goal_state.get("flee_goal_achieved", False) and state.get(
            "can_find_enemy", False
        ):
            cost += max(
                0,
                ENEMY_NEARBY_FLEE_DISTANCE
                - state.get("nearest_enemy_dist", ENEMY_NEARBY_FLEE_DISTANCE),
            )

        # Heuristic for equipping weapon
        if goal_state.get("has_weapon_equipped", False) and not state.get(
            "weapon_equipped", False
        ):
            if state.get("has_weapon_in_inv"):
                cost += 0.5  # Small cost if in inventory
            elif state.get("can_find_item"):
                cost += (
                    state.get("nearest_item_dist", 0) / 4.0
                )  # Cost related to finding item

        return cost

    def update_weights(
        self, executed_action_names: list[str], success_metric: float
    ):  # Unchanged logic
        """Updates action weights based on the success of the executed plan."""
        if not executed_action_names:
            return

        # Calculate adjustment factor based on success (0 to 1) vs baseline
        learning_adjustment = (
            success_metric - LEARNING_SCORE_BASELINE
        ) * LEARNING_RATE_FACTOR
        updated_names = (
            set()
        )  # Update each action in the plan only once per learning step

        for name in executed_action_names:
            if name in updated_names:
                continue

            current_weight = self.action_weights[name]
            # Adjust weight: decrease for success (metric > baseline), increase for failure
            # Multiplicative update: new_weight = current * (1 - adjustment) -- careful with signs
            # Let's use additive adjustment factor for simplicity:
            # new_weight = current - adjustment_factor * current ? -> leads to zero
            # Let's stick to multiplicative change:
            # If success_metric > baseline, learning_adjustment > 0 -> weight decreases (cheaper)
            # If success_metric < baseline, learning_adjustment < 0 -> weight increases (costlier)
            # Using (1.0 - learning_adjustment) ensures inverse relationship
            new_weight = current_weight * (1.0 - learning_adjustment)

            # Clamp weight within bounds
            self.action_weights[name] = max(
                ACTION_WEIGHT_MIN, min(new_weight, ACTION_WEIGHT_MAX)
            )
            updated_names.add(name)


class AgentAI:
    """Manages the agent's GOAP planning and execution."""

    def __init__(self, world: World, rng: GameRNG):  # Add rng parameter
        self.world: World = world
        self.rng: GameRNG = rng  # Store rng instance
        self.actions: list[Action] = self._define_actions()
        self.planner: GOAPPlanner = GOAPPlanner(self.actions)
        self.current_plan: ActionPlan = deque()
        self.current_goal: Optional[StateDict] = None
        self.last_executed_plan_actions: list[str] = []
        self.last_action_name: ActionResult = None

    def _define_actions(self) -> list[Action]:
        """Defines all possible actions the agent can take."""
        actions_list: list[Action] = []

        # --- Action Helper Functions ---
        def _get_target_pos(w: World, a: Entity, kind: str) -> OptionalPosition:
            """Finds the position of the nearest entity of a given kind."""
            target_id, _ = w.get_nearest_entity(a, kind)
            if target_id:
                target_obj = w.get_entity_object(target_id)
                return target_obj.get_position() if target_obj else None
            return None

        def _execute_move_to(w: World, a: Entity, target_kind: str) -> bool:
            """Finds path to nearest target of kind and moves one step."""
            target_pos = _get_target_pos(w, a, target_kind)
            if not target_pos:
                return False
            path = w.find_path(a.get_position(), target_pos)
            if path and path:  # Check if path is not None and not empty
                # Check if next step is valid before moving
                next_x, next_y = path[0]
                if w.is_valid(next_x, next_y):
                    target_entity = w.grid[next_x][next_y]
                    if target_entity is None or target_entity.kind == "item":
                        return w.move_entity(a, next_x, next_y)
            return False  # No path or next step blocked

        # --- Cost Calculators ---
        def cost_simple(w: World, a: Entity) -> float:
            return 1.0

        def cost_attack(w: World, a: Entity) -> float:
            return 1.5

        def cost_explore(w: World, a: Entity) -> float:
            return 2.0

        def cost_flee(w: World, a: Entity) -> float:
            return 0.5  # Fleeing is priority

        def cost_wait(w: World, a: Entity) -> float:
            return 0.1  # Waiting is cheap

        def cost_pickup(w: World, a: Entity) -> float:
            return 0.8

        def cost_consume(w: World, a: Entity) -> float:
            return 0.5

        def cost_equip(w: World, a: Entity) -> float:
            return 0.6

        def cost_move_via_distance(w: World, a: Entity, kind: str) -> float:
            _, dist = w.get_nearest_entity(a, kind)
            # Add small base cost, return infinity if no target found
            return (dist + 0.1) if dist != float("inf") else float("inf")

        # --- Preconditions ---
        def pre_always_true(state: StateDict) -> bool:
            return True

        def pre_can_find_enemy(state: StateDict) -> bool:
            return state.get("can_find_enemy", False)

        def pre_enemy_adjacent(state: StateDict) -> bool:
            return state.get("enemy_is_adjacent", False)

        def pre_can_flee(state: StateDict) -> bool:
            return (
                state.get("can_find_enemy", False)
                and state.get("nearest_enemy_dist", float("inf"))
                < ENEMY_NEARBY_FLEE_DISTANCE
            )

        def pre_item_adjacent(state: StateDict) -> bool:
            return state.get("item_is_adjacent", False)

        def pre_inventory_not_full(state: StateDict) -> bool:
            return not state.get("inventory_full", True)

        def pre_has_slime_mold(state: StateDict) -> bool:
            return state.get("has_slime_mold", False)

        def pre_has_health_potion(state: StateDict) -> bool:
            return state.get("has_health_potion", False)

        def pre_has_weapon_in_inv(state: StateDict) -> bool:
            return state.get("has_weapon_in_inv", False)

        def pre_weapon_not_equipped(state: StateDict) -> bool:
            return not state.get("weapon_equipped", True)

        # --- Effects ---
        def effect_update_position(state: StateDict) -> StateDict:
            state["agent_pos"] = None  # Position becomes unknown after move
            # Reset adjacency flags
            state["enemy_is_adjacent"] = False
            state["item_is_adjacent"] = False
            # Increment distances (heuristic) - might be inaccurate, planner should handle
            # state["nearest_food_dist"] = state.get("nearest_food_dist", float("inf")) + 1
            state["nearest_enemy_dist"] = (
                state.get("nearest_enemy_dist", float("inf")) + 1
            )
            state["nearest_item_dist"] = (
                state.get("nearest_item_dist", float("inf")) + 1
            )
            return state

        def effect_attack_enemy(state: StateDict) -> StateDict:
            # Assume enemy might die or move away
            state["enemy_is_adjacent"] = False
            state["can_find_enemy"] = False  # Need to re-evaluate after attack
            state["nearest_enemy_dist"] = float("inf")
            return state

        def effect_explore(state: StateDict) -> StateDict:
            state = effect_update_position(state)
            # Mark that exploration happened
            state["explored_something"] = True
            return state

        def effect_flee(state: StateDict) -> StateDict:
            state = effect_update_position(state)  # Moving changes position
            state["enemy_is_adjacent"] = False  # Should no longer be adjacent
            state["nearest_enemy_dist"] = (
                state.get("nearest_enemy_dist", 0) + 5
            )  # Increase distance estimate
            state["flee_goal_achieved"] = True  # Mark flee goal met
            return state

        def effect_wait(state: StateDict) -> StateDict:
            state["is_resting"] = True  # Indicate resting state
            # Assuming wait helps restore health (can be refined)
            state["is_healthy"] = True  # Optimistically assume leads to health
            return state

        def effect_pickup_item(state: StateDict) -> StateDict:
            state["inventory_count"] = state.get("inventory_count", 0) + 1
            state["inventory_full"] = state["inventory_count"] >= AGENT_MAX_INVENTORY
            state["item_is_adjacent"] = False  # Item picked up
            state["can_find_item"] = False  # Need to re-evaluate nearby items
            state["nearest_item_dist"] = float("inf")
            # Assume any item picked up *could* be the needed type for planning
            state["has_slime_mold"] = True
            state["has_health_potion"] = True
            state["has_weapon_in_inv"] = True
            return state

        def effect_consume_slime_mold(state: StateDict) -> StateDict:
            state["agent_hunger"] = START_HUNGER  # Hunger fully restored
            state["is_starving"] = False
            state["inventory_count"] = state.get("inventory_count", 1) - 1
            state["inventory_full"] = False
            state["has_slime_mold"] = False  # Consumed the item
            return state

        def effect_consume_health_potion(state: StateDict) -> StateDict:
            state["agent_health"] = START_HEALTH  # Health fully restored
            state["is_healthy"] = True
            state["is_critically_injured"] = False
            state["inventory_count"] = state.get("inventory_count", 1) - 1
            state["inventory_full"] = False
            state["has_health_potion"] = False  # Consumed the item
            return state

        def effect_equip_weapon(state: StateDict) -> StateDict:
            state["weapon_equipped"] = True
            # Weapon moved from inv to equipped slot
            state["has_weapon_in_inv"] = False
            # Inventory count decreases as item leaves main inv
            state["inventory_count"] = state.get("inventory_count", 1) - 1
            state["inventory_full"] = False
            return state

        # --- Executors ---
        def execute_attack_adjacent_enemy(w: World, a: Entity) -> bool:
            """Attacks the nearest adjacent hostile."""
            # Find nearest adjacent hostile (enemy or slime)
            target_id, dist = w.get_nearest_entity(a, "enemy", max_dist=1)
            if not target_id:
                target_id, dist = w.get_nearest_entity(a, "slime", max_dist=1)

            if target_id:
                target_obj = w.get_entity_object(target_id)
                if target_obj:
                    damage_dealt = a.get_effective_damage()
                    new_target_health = max(0.0, target_obj.health - damage_dealt)
                    w.update_entity_health(target_id, new_target_health)

                    # Apply hunger cost to attacker
                    new_attacker_hunger = max(0.0, a.hunger - ATTACK_HUNGER_COST)
                    w.update_entity_hunger(a.id, new_attacker_hunger)

                    # Handle enemy death and potential item drops
                    if new_target_health <= 0:
                        target_pos = target_obj.get_position()
                        target_kind = target_obj.kind  # Store kind before removing
                        # Remove the defeated enemy
                        w.remove_entity(target_obj)
                        # Drop slime mold if slime was killed
                        if target_kind == "slime":
                            # Check if target position is now free before dropping
                            if w.grid[target_pos[0]][target_pos[1]] is None:
                                mold_item = create_slime_mold()
                                item_entity = Entity(
                                    target_pos[0],
                                    target_pos[1],
                                    "item",
                                    rng=w.rng,
                                    item=mold_item,
                                )
                                w.add_entity(item_entity)
                                print(f"Slime dropped {mold_item.name} at {target_pos}")
                            else:
                                print(
                                    f"Slime defeated at {target_pos}, but tile blocked for drop."
                                )
                        else:
                            print(
                                f"{target_kind.capitalize()
                                     } defeated at {target_pos}"
                            )
                    return True  # Attack happened
            return False  # No adjacent target found

        def execute_explore(w: World, a: Entity) -> bool:
            """Moves randomly to an adjacent walkable, empty cell."""
            possible_moves: list[Position] = []
            ax, ay = a.get_position()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ax + dx, ay + dy
                    # Check if valid and empty (or only item)
                    if w.is_valid(nx, ny):
                        target_entity = w.grid[nx][ny]
                        if target_entity is None or target_entity.kind == "item":
                            possible_moves.append((nx, ny))
            if possible_moves:
                # Use agent's RNG for choice
                nx, ny = w.rng.choice(possible_moves)  # <- GAME RNG USED HERE
                # move_entity handles displacing items
                return w.move_entity(a, nx, ny)
            return False  # No valid moves

        def execute_flee(w: World, a: Entity) -> bool:
            """Moves away from the nearest enemy."""
            # Find nearest hostile
            enemy_id, enemy_dist = w.get_nearest_entity(
                a, "enemy", max_dist=ENEMY_NEARBY_FLEE_DISTANCE
            )
            slime_id, slime_dist = w.get_nearest_entity(
                a, "slime", max_dist=ENEMY_NEARBY_FLEE_DISTANCE
            )
            nearest_hostile_id: Optional[EntityID] = None
            nearest_hostile_obj: OptionalEntity = None
            if enemy_id and enemy_dist < slime_dist:
                nearest_hostile_id = enemy_id
            elif slime_id:
                nearest_hostile_id = slime_id

            if not nearest_hostile_id:
                # No enemy nearby, explore instead
                return execute_explore(w, a)

            nearest_hostile_obj = w.get_entity_object(nearest_hostile_id)
            if not nearest_hostile_obj:
                return execute_explore(w, a)  # Should not happen

            agent_pos = a.get_position()
            enemy_pos = nearest_hostile_obj.get_position()
            best_flee_spot: OptionalPosition = None
            max_dist_from_enemy = -1.0
            current_dist = float(
                distance(agent_pos[0], agent_pos[1], enemy_pos[0], enemy_pos[1])
            )

            # Evaluate adjacent walkable cells
            possible_moves: list[tuple[Position, float]] = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
                    if w.is_valid(nx, ny):
                        target_entity = w.grid[nx][ny]
                        if target_entity is None or target_entity.kind == "item":
                            dist_from_enemy = float(
                                distance(nx, ny, enemy_pos[0], enemy_pos[1])
                            )
                            possible_moves.append(((nx, ny), dist_from_enemy))
                            if dist_from_enemy > max_dist_from_enemy:
                                max_dist_from_enemy = dist_from_enemy
                                best_flee_spot = (nx, ny)

            # Try to move to the spot furthest from the enemy
            if best_flee_spot and max_dist_from_enemy > current_dist:
                return w.move_entity(a, best_flee_spot[0], best_flee_spot[1])
            elif (
                possible_moves
            ):  # If no spot increases distance, pick the best available
                possible_moves.sort(key=lambda item: item[1], reverse=True)
                best_move = possible_moves[0][0]
                return w.move_entity(a, best_move[0], best_move[1])

            return False  # No valid moves found

        def execute_wait(w: World, a: Entity) -> bool:
            """Agent does nothing for a turn (potentially resting)."""
            # Resting logic (health regen) is handled in main loop now
            return True

        def execute_pickup_item(w: World, a: Entity) -> bool:
            """Picks up an item from the current or adjacent cell."""
            # Check current cell first
            current_cell_item = w.grid[a.x][a.y]
            target_item_entity = None
            if current_cell_item is not None and current_cell_item.kind == "item":
                target_item_entity = current_cell_item
            else:  # Check adjacent cells
                item_id, dist = w.get_nearest_entity(a, "item", max_dist=1)
                if item_id:
                    target_item_entity = w.get_entity_object(item_id)

            # Pickup if item found and inventory has space
            if (
                target_item_entity
                and target_item_entity.item
                and len(a.inventory) < a.max_inventory
            ):
                print(
                    f"Agent picking up {target_item_entity.item.name} at {target_item_entity.get_position()}"
                )
                a.inventory.append(target_item_entity.item)
                w.remove_entity(target_item_entity)
                return True
            # elif target_item_entity:
            # print(f"Agent sees {target_item_entity.item.name} but inventory full.")
            return False

        def execute_consume_slime_mold(w: World, a: Entity) -> bool:
            """Consumes the first Slime Mold found in inventory."""
            for i, item in enumerate(a.inventory):
                if isinstance(item, Consumable) and item.name == "Slime Mold":
                    print(f"Agent consuming {item.name}")
                    new_hunger = min(START_HUNGER, a.hunger + item.amount)
                    # Update world state
                    w.update_entity_hunger(a.id, new_hunger)
                    a.inventory.pop(i)  # Remove from inventory
                    return True
            return False  # Item not found

        def execute_consume_health_potion(w: World, a: Entity) -> bool:
            """Consumes the first Health Potion found in inventory."""
            for i, item in enumerate(a.inventory):
                if isinstance(item, Consumable) and item.name == "Health Potion":
                    print(f"Agent consuming {item.name}")
                    new_health = min(START_HEALTH, a.health + item.amount)
                    # Update world state
                    w.update_entity_health(a.id, new_health)
                    a.inventory.pop(i)
                    return True
            return False

        def execute_equip_weapon(w: World, a: Entity) -> bool:
            """Equips the first weapon found in inventory."""
            weapon_to_equip: Optional[Weapon] = None
            weapon_index: int = -1
            # Find first weapon in inventory
            for i, item in enumerate(a.inventory):
                if isinstance(item, Weapon):
                    weapon_to_equip = item
                    weapon_index = i
                    break

            if weapon_to_equip:
                # Check if something is already equipped
                if a.equipped_weapon:
                    # Try to move equipped weapon back to inventory
                    if (
                        len(a.inventory) - 1 < a.max_inventory
                    ):  # -1 because we remove the new weapon first
                        print(f"Agent unequipping {a.equipped_weapon.name}")
                        a.inventory.append(a.equipped_weapon)
                        a.equipped_weapon = None  # Clear equipped slot
                    else:
                        print(
                            f"Inventory full, cannot unequip {a.equipped_weapon.name} to equip {weapon_to_equip.name}"
                        )
                        return False  # Cannot equip if inventory full after unequipping

                # Equip the new weapon
                a.equipped_weapon = a.inventory.pop(weapon_index)
                print(f"Agent equipping {a.equipped_weapon.name}")
                return True
            return False  # No weapon found in inventory

        # --- Define Actions ---
        actions_list = [
            Action(
                "MoveToNearestItem",
                lambda w, a: cost_move_via_distance(w, a, "item"),
                lambda s: s.get("can_find_item", False)
                and not s.get("inventory_full", True),
                effect_update_position,
                lambda w, a: _execute_move_to(w, a, "item"),
            ),
            Action(
                "PickupItem",
                cost_pickup,
                lambda s: s.get("item_is_adjacent", False)
                and not s.get("inventory_full", True),
                effect_pickup_item,
                execute_pickup_item,
            ),
            Action(
                "ConsumeSlimeMold",
                cost_consume,
                pre_has_slime_mold,
                effect_consume_slime_mold,
                execute_consume_slime_mold,
            ),
            Action(
                "ConsumeHealthPotion",
                cost_consume,
                pre_has_health_potion,
                effect_consume_health_potion,
                execute_consume_health_potion,
            ),
            Action(
                "EquipWeapon",
                cost_equip,
                lambda s: s.get("has_weapon_in_inv", False)
                and not s.get("weapon_equipped", True),
                effect_equip_weapon,
                execute_equip_weapon,
            ),
            Action(
                "MoveToNearestEnemy",
                lambda w, a: cost_move_via_distance(w, a, "enemy"),
                pre_can_find_enemy,
                effect_update_position,
                lambda w, a: _execute_move_to(w, a, "enemy"),
            ),  # Enemy includes slimes via logic
            Action(
                "AttackAdjacentEnemy",
                cost_attack,
                pre_enemy_adjacent,
                effect_attack_enemy,
                execute_attack_adjacent_enemy,
            ),
            Action(
                "Explore",
                cost_explore,
                pre_always_true,
                effect_explore,
                execute_explore,
            ),
            Action("Flee", cost_flee, pre_can_flee, effect_flee, execute_flee),
            Action("Wait", cost_wait, pre_always_true, effect_wait, execute_wait),
        ]
        return actions_list

    def _select_goal(self, agent: Entity) -> StateDict:
        """Selects the most appropriate goal based on the agent's current state."""
        state = self.planner._get_world_state_representation(self.world, agent)

        # --- Goal Prioritization ---
        # 1. Critical Survival: Starving or Critically Injured
        if state["is_starving"]:
            if state["has_slime_mold"]:
                return {"is_not_starving": True}  # Eat if possible
            elif state["can_find_item"] and not state["inventory_full"]:
                return {"has_slime_mold": True}  # Find food item
            else:
                return {"explored_something": True}  # Explore desperately

        if state["is_critically_injured"]:
            if state["has_health_potion"]:
                return {"is_healthy": True}  # Heal if possible
            # Flee even if no potion if critically injured and enemy nearby
            if (
                state["can_find_enemy"]
                and state["nearest_enemy_dist"] < ENEMY_NEARBY_FLEE_DISTANCE
            ):
                return {"flee_goal_achieved": True}
            elif state["can_find_item"] and not state["inventory_full"]:
                return {"has_health_potion": True}  # Find potion
            else:
                # Default goal: try to rest/recover
                return {"is_healthy": True}

        # 2. Immediate Threat: Adjacent Enemy or Low Health Flee
        if state["enemy_is_adjacent"]:
            # Equip weapon first if available and not equipped
            if not state["weapon_equipped"] and state["has_weapon_in_inv"]:
                return {"has_weapon_equipped": True}
            # Goal is to make enemy not adjacent (attack/flee)
            return {"enemy_is_adjacent": False}

        if (
            agent.health < LOW_HEALTH_FLEE_THRESHOLD
            and state["can_find_enemy"]
            and state["nearest_enemy_dist"] < ENEMY_NEARBY_FLEE_DISTANCE
        ):
            return {"flee_goal_achieved": True}

        # 3. Preparation / Improvement: Heal, Satisfy Hunger, Equip Weapon
        if (
            agent.health < START_HEALTH
        ):  # Heal if below max and not critically injured/fleeing
            if state["has_health_potion"]:
                return {"is_healthy": True}
            # Maybe seek potion if moderately injured? Let's prioritize other things first.

        if agent.hunger < START_HUNGER * 0.5:  # Seek food if moderately hungry
            if state["has_slime_mold"]:
                return {"is_not_starving": True}
            elif state["can_find_item"] and not state["inventory_full"]:
                return {"has_slime_mold": True}

        # Equip weapon if one is available and enemies might be around (or just generally)
        if not state["weapon_equipped"] and state["has_weapon_in_inv"]:
            # Prioritize equipping if enemies known or potentially nearby
            if state["can_find_enemy"] or state["nearest_enemy_dist"] < float(
                "inf"
            ):  # Check if any enemy exists
                return {"has_weapon_equipped": True}
            # Optional: Equip preemptively even if no enemy seen? Less urgent.
            # return {"has_weapon_equipped": True}

        # 4. Engagement: Seek out enemy if healthy enough
        if state["can_find_enemy"] and agent.health > HEALTHY_THRESHOLD:
            # Goal: Engage the enemy (attack/move to)
            return {"enemy_is_adjacent": False}

        # 5. Resource Gathering: Find items if not full
        if state["can_find_item"] and not state["inventory_full"]:
            # Generic goal to trigger MoveToItem/Pickup
            return {"has_any_item": True}

        # 6. Default: Explore
        return {"explored_something": True}

    def _is_plan_still_valid(self, agent: Entity) -> bool:
        """Checks if the preconditions for the next action in the plan are met."""
        if not self.current_plan:
            return False
        next_action = self.current_plan[0]
        current_state_rep = self.planner._get_world_state_representation(
            self.world, agent
        )
        return next_action.check_preconditions(current_state_rep)

    def act(self, agent: Entity) -> ActionResult:
        """Determines the agent's action for the current turn."""
        self.last_action_name = None  # Reset last action
        current_state_rep = self.planner._get_world_state_representation(
            self.world, agent
        )

        # --- Plan Validation and Replanning ---
        needs_replan = False
        current_goal_satisfied = False

        if not self.current_plan:
            needs_replan = True
        elif not self._is_plan_still_valid(agent):
            # print(f"Plan invalidated: {self.current_plan[0].name} preconditions not met.") # Debug
            needs_replan = True
        elif self.current_goal:
            # Check if the current goal state is now satisfied by the world state
            # Use the planner's method for consistency
            current_goal_satisfied = self.planner._goal_satisfied(
                current_state_rep, self.current_goal
            )
            if current_goal_satisfied:
                # print(f"Goal '{self.current_goal}' satisfied.") # Debug
                needs_replan = True
        # --- End Plan Validation ---

        # --- Replan if necessary ---
        if needs_replan:
            self.current_goal = self._select_goal(agent)
            # print(f"Selected new goal: {self.current_goal}") # Debug
            new_plan_list = self.planner.plan(self.world, agent, self.current_goal)

            if new_plan_list:
                self.current_plan = deque(new_plan_list)
                self.last_executed_plan_actions = [
                    a.name for a in self.current_plan
                ]  # Store names for learning
                # print(f"Found plan: {[a.name for a in self.current_plan]}") # Debug
            else:  # Replanning failed - fallback behavior
                self.current_plan = deque()
                self.last_executed_plan_actions = []
                # print("Replanning failed. Executing fallback.") # Debug
                # --- Fallback Action ---
                # Prioritize basic survival or exploration
                wait_action = next((a for a in self.actions if a.name == "Wait"), None)
                explore_action = next(
                    (a for a in self.actions if a.name == "Explore"), None
                )

                if wait_action and agent.health < START_HEALTH * 0.9:  # Wait if injured
                    if wait_action.execute(self.world, agent):
                        self.last_action_name = wait_action.name
                elif explore_action:  # Explore if nothing else to do
                    if explore_action.execute(self.world, agent):
                        self.last_action_name = explore_action.name
                else:  # Absolute fallback: Random move (less ideal)
                    # Use execute_explore which includes random move logic
                    if execute_explore(self.world, agent):
                        self.last_action_name = "RandomMove"

                # Update last executed plan actions even for fallback
                if self.last_action_name:
                    self.last_executed_plan_actions = [self.last_action_name]
                return self.last_action_name
        # --- End Replanning ---

        # --- Execute Next Action in Plan ---
        if self.current_plan:
            action_to_execute = self.current_plan.popleft()
            # print(f"Executing planned action: {action_to_execute.name}") # Debug
            success = action_to_execute.execute(self.world, agent)
            if success:
                self.last_action_name = action_to_execute.name
            else:
                # Action failed, clear plan to force replan next turn
                # print(f"Action {action_to_execute.name} failed. Clearing plan.") # Debug
                self.current_plan = deque()
                self.last_action_name = None  # Indicate action failure
        else:
            # Should have replanned or fallen back, but handle case where plan is empty
            # print("No plan and no fallback executed.") # Debug
            self.last_action_name = None
        # --- End Action Execution ---

        return self.last_action_name

    def learn(self, turns_survived: int):
        """Updates action weights based on survival outcome."""
        # Calculate success metric (0 to 1)
        survival_score = turns_survived / MAX_TURNS
        health_score = 0.0
        agent_survived = self.world.agent is not None and self.world.agent.health > 0

        if agent_survived:
            # Reward based on final health and turns survived
            health_score = self.world.agent.health / START_HEALTH
            # Weighted average: survival duration more important?
            final_score = (survival_score * 0.7) + (health_score * 0.3)
        else:
            # Penalize based on how quickly the agent died
            final_score = survival_score * 0.5  # Max 0.5 if died

        # Update weights based on the *last attempted plan*
        if self.last_executed_plan_actions:
            # print(f"Learning: Score={final_score:.2f}, Actions={self.last_executed_plan_actions}") # Debug
            self.planner.update_weights(self.last_executed_plan_actions, final_score)
        # Reset last executed plan after learning
        self.last_executed_plan_actions = []


# --- Enemy AI ---
def enemy_act(enemy: Entity, world: World, rng: GameRNG):  # Added rng parameter
    """Basic enemy AI: move towards agent if nearby, attack if adjacent."""
    agent = world.agent
    if not agent or agent.health <= 0:
        return  # Agent is gone or dead

    enemy_pos = enemy.get_position()
    agent_pos = agent.get_position()
    dist_to_agent = distance(enemy_pos[0], enemy_pos[1], agent_pos[0], agent_pos[1])

    # --- Fleeing Logic ---
    # Use a base health reference (e.g., average start health or fixed value)
    max_enemy_health = (
        SLIME_HEALTH
        if enemy.kind == "slime"
        else (ENEMY_START_HEALTH_RANGE[0] + ENEMY_START_HEALTH_RANGE[1]) / 2.0
    )
    should_flee = enemy.health < (max_enemy_health * ENEMY_FLEE_THRESHOLD)

    if should_flee and dist_to_agent < 10:
        # Attempt to move away from the agent
        best_flee_spot: OptionalPosition = None
        max_dist_from_agent = -1.0
        current_dist = float(dist_to_agent)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = enemy_pos[0] + dx, enemy_pos[1] + dy
                if world.is_valid(nx, ny):
                    target_entity = world.grid[nx][ny]
                    if target_entity is None or target_entity.kind == "item":
                        dist_from_agent = float(
                            distance(nx, ny, agent_pos[0], agent_pos[1])
                        )
                        if dist_from_agent > max_dist_from_agent:
                            max_dist_from_agent = dist_from_agent
                            best_flee_spot = (nx, ny)

        if best_flee_spot and max_dist_from_agent > current_dist:
            world.move_entity(enemy, best_flee_spot[0], best_flee_spot[1])
            return  # Flee successful
        # If no better spot, might hold position or attack if adjacent

    # --- Attack Logic ---
    if dist_to_agent <= 1:
        damage = enemy.get_effective_damage()
        # Apply damage to agent
        new_agent_health = max(0.0, agent.health - damage)
        world.update_entity_health(agent.id, new_agent_health)
        # Apply hunger cost to agent for defending/being hit
        new_agent_hunger = max(0.0, agent.hunger - DEFEND_HUNGER_COST)
        world.update_entity_hunger(agent.id, new_agent_hunger)
        # print(f"{enemy.kind} attacks Agent for {damage} damage!") # Debug
        return  # Attacked

    # --- Pursue Logic ---
    else:
        # Use pathfinding to move towards agent
        path = world.find_path(enemy_pos, agent_pos)
        if path:  # Check if path exists and is not empty
            next_pos = path[0]
            # Check if next step is valid before moving
            if world.is_valid(next_pos[0], next_pos[1]):
                target_entity = world.grid[next_pos[0]][next_pos[1]]
                if target_entity is None or target_entity.kind == "item":
                    world.move_entity(enemy, next_pos[0], next_pos[1])
                    return  # Moved towards agent
        # If no path or blocked, do nothing (or random move?)
        # Consider adding random movement if idle and agent not visible?
        # if rng.get_float() < 0.2: # 20% chance to move randomly if idle
        #    # Simple random move logic (similar to explore)
        #    pass

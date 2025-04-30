# game/game_state.py
import numpy as np
import structlog
from typing import Tuple, Union, Dict, Any
import time
from game.world.game_map import GameMap

print(
    f">>> [{time.time():.4f}] Attempting import: game.entities.registry.EntityRegistry"
)  # <-- Add this line

from game.entities.registry import EntityRegistry

print(
    f"<<< [{time.time():.4f}] Successfully imported EntityRegistry"
)  # <-- Add this line

from game.items.registry import ItemRegistry

# --- NEW: Import GameRNG ---
from utils.game_rng import GameRNG

# --- End NEW ---

log = structlog.get_logger()


class GameState:
    def __init__(
        self,
        existing_map: GameMap,
        player_start_pos: Tuple[int, int],
        # Config values passed directly
        player_glyph: int,
        player_start_hp: int,
        player_fov_radius: int,
        item_templates: Dict[str, Any],
        effect_definitions: Dict[str, Any],
        # --- NEW: Add RNG seed parameter ---
        rng_seed: int | None,
        # --- End NEW ---
    ):
        log.info("Initializing GameState...")

        # (Input validation remains the same)
        if not isinstance(existing_map, GameMap):  # ...
            raise TypeError("GameState requires a valid GameMap instance.")
        if not player_start_pos or len(player_start_pos) != 2:  # ...
            raise ValueError(
                "GameState requires a valid player_start_pos tuple (x, y)."
            )

        self.game_map: GameMap = existing_map
        self._map_width: int = existing_map.width
        self._map_height: int = existing_map.height

        # --- NEW: Initialize GameRNG ---
        # Use provided seed, or generate one if None
        self.rng_instance: GameRNG = GameRNG(seed=rng_seed)
        log.debug("GameRNG initialized", seed=self.rng_instance.initial_seed)
        # --- End NEW ---

        self.entity_registry: EntityRegistry = EntityRegistry()
        log.debug("EntityRegistry initialized")

        self.item_registry: ItemRegistry = ItemRegistry(item_templates)
        self.effect_definitions: Dict[str, Any] = effect_definitions
        log.debug("ItemRegistry initialized", templates=len(item_templates))
        log.debug("Effect definitions stored", effects=len(effect_definitions))

        # (Player entity creation remains the same)
        player_start_x, player_start_y = player_start_pos
        self.player_id: int = self.entity_registry.create_entity(  # ...
            x=player_start_x,
            y=player_start_y,
            glyph=player_glyph,
            color_fg=(255, 255, 255),
            name="Player",
            blocks_movement=True,
            hp=player_start_hp,
            max_hp=player_start_hp,
        )
        log.debug(
            "Player entity created",
            player_id=self.player_id,  # ...
            pos=(player_start_x, player_start_y),
            hp=player_start_hp,
            glyph=player_glyph,
        )

        self.fov_radius = player_fov_radius
        self.message_log: list[tuple[str, tuple[int, int, int]]] = []
        self.turn_count: int = 0
        self.add_message("Welcome to BasicRL!", (0, 255, 0))

        log.info(
            "Game state initialized",
            map_size=f"{self._map_width}x{self._map_height}",
            player_id=self.player_id,
            # ... (rest of log info) ...
            item_templates_loaded=len(item_templates),
            effect_definitions_loaded=len(effect_definitions),
            rng_seed=self.rng_instance.initial_seed,  # Log the actual seed used
        )

        self.update_fov()  # Initial FOV calculation

    # (update_fov method remains the same)
    def update_fov(self) -> None:  # ...
        player_pos = self.player_position
        if player_pos:
            px, py = player_pos
            if not self.game_map.in_bounds(px, py):  # ... handle out of bounds ...
                self.game_map.visible[:] = False
                return
            self.game_map.compute_fov(px, py, self.fov_radius)
            if np.sum(self.game_map.visible) == 0:  # ... handle zero visible ...
                self.game_map.visible[py, px] = True
                self.game_map.explored[py, px] = True
        else:  # ... handle no player pos ...
            self.game_map.visible[:] = False

    # (Properties map_width, map_height, player_position remain the same)
    @property
    def map_width(self) -> int:
        return self._map_width

    @property
    def map_height(self) -> int:
        return self._map_height

    @property
    def player_position(self) -> Union[Tuple[int, int], None]:  # ...
        return self.entity_registry.get_position(self.player_id)

    # (add_message method remains the same)
    def add_message(
        self, text: str, color: tuple[int, int, int] = (255, 255, 255)
    ) -> None:  # ...
        self.message_log.append((text, color))
        log.debug("Message added", message=text, color=color)

    # (advance_turn method remains the same conceptually, but needs additions)
    def advance_turn(self) -> None:
        """Advances the game turn counter and performs turn-based updates."""
        self.turn_count += 1
        log.debug("Turn advanced", turn=self.turn_count)

        # --- TODO: Status Effect Duration Update ---
        # 1. Get all active entities.
        # 2. For each entity:
        #    a. Get current status_effects list.
        #    b. Create a new list, decrementing duration for non-permanent effects.
        #    c. Filter out effects where duration <= 0.
        #    d. Use set_entity_component to update the status_effects list.
        #    (Consider performance implications for many entities/statuses).
        # -----------------------------------------

        # TODO: Implement AI processing call here (e.g., call process_all_mob_ai)
        # TODO: Implement interaction system processing (for emergent effects)

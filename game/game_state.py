# game/game_state.py
from typing import Any, Dict, Literal, Tuple, Union  # Added Literal

import structlog
from game_rng import GameRNG  # Assuming this path is correct

from game.entities.registry import EntityRegistry
from game.items.registry import ItemRegistry

# Assuming these imports are correct relative to game_state.py
from game.world.game_map import GameMap
from game.systems.ai_system import dispatch_ai

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
        rng_seed: int | None,
    ):
        log.info("Initializing GameState...")

        if not isinstance(existing_map, GameMap):
            raise TypeError("GameState requires a valid GameMap instance.")
        if not player_start_pos or len(player_start_pos) != 2:
            raise ValueError(
                "GameState requires a valid player_start_pos tuple (x, y)."
            )

        self.game_map: GameMap = existing_map
        self._map_width: int = existing_map.width
        self._map_height: int = existing_map.height

        self.rng_instance: GameRNG = GameRNG(seed=rng_seed)
        log.debug("GameRNG initialized", seed=self.rng_instance.initial_seed)

        self.entity_registry: EntityRegistry = EntityRegistry()
        log.debug("EntityRegistry initialized")

        self.item_registry: ItemRegistry = ItemRegistry(item_templates)
        self.effect_definitions: Dict[str, Any] = effect_definitions
        log.debug("ItemRegistry initialized", templates=len(item_templates))
        log.debug("Effect definitions stored", effects=len(effect_definitions))

        player_start_x, player_start_y = player_start_pos
        # Ensure all necessary components are passed during creation if defaults changed
        self.player_id: int = self.entity_registry.create_entity(
            x=player_start_x,
            y=player_start_y,
            glyph=player_glyph,
            color_fg=(255, 255, 255),
            name="Player",
            blocks_movement=True,
            hp=player_start_hp,
            max_hp=player_start_hp,
            # Add defaults for mana/fullness etc. if needed by registry init
        )
        log.debug(
            "Player entity created",
            player_id=self.player_id,
            pos=(player_start_x, player_start_y),
            hp=player_start_hp,
            glyph=player_glyph,
        )

        self.fov_radius = player_fov_radius
        self.message_log: list[tuple[str, tuple[int, int, int]]] = []
        self.turn_count: int = 0

        # --- NEW: UI State ---
        self.ui_state: Literal["PLAYER_TURN", "INVENTORY_VIEW", "TARGETING"] = (
            "PLAYER_TURN"
        )
        # --- End NEW ---

        self.add_message("Welcome to BasicRL!", (0, 255, 0))

        log.info(
            "Game state initialized",
            map_size=f"{self._map_width}x{self._map_height}",
            player_id=self.player_id,
            item_templates_loaded=len(item_templates),
            effect_definitions_loaded=len(effect_definitions),
            rng_seed=self.rng_instance.initial_seed,
        )

        self.update_fov()  # Initial FOV calculation

    def update_fov(self) -> None:
        """Calculates Field of View based on player position."""
        player_pos = self.player_position
        if player_pos:
            px, py = player_pos
            if not self.game_map.in_bounds(px, py):
                log.warning("Player out of bounds, cannot compute FOV.", pos=(px, py))
                self.game_map.visible[:] = False  # Clear visibility if player is OOB
                return
            # Ensure origin height is passed correctly
            origin_height = int(self.game_map.height_map[py, px])
            self.game_map.compute_fov(
                px, py, self.fov_radius
            )  # compute_fov now handles explored
            # Post-check: Ensure origin is always visible if FOV somehow clears it
            if not self.game_map.visible[py, px]:
                log.warning(
                    "Origin tile became non-visible after FOV calculation, forcing visible.",
                    pos=(px, py),
                )
                self.game_map.visible[py, px] = True
                self.game_map.explored[py, px] = True  # Ensure explored too
        else:
            log.warning("Cannot update FOV: Player position not found.")
            self.game_map.visible[:] = False  # Clear visibility if no player

    @property
    def map_width(self) -> int:
        return self._map_width

    @property
    def map_height(self) -> int:
        return self._map_height

    @property
    def player_position(self) -> Union[Tuple[int, int], None]:
        """Gets the current player position from the EntityRegistry."""
        return self.entity_registry.get_position(self.player_id)

    def add_message(
        self, text: str, color: tuple[int, int, int] = (255, 255, 255)
    ) -> None:
        """Adds a message to the game log."""
        self.message_log.append((text, color))
        log.debug("Message added", message=text, color=color)
        # Optional: Trim log length if it gets too long
        # MAX_LOG_LENGTH = 100
        # if len(self.message_log) > MAX_LOG_LENGTH:
        #     self.message_log = self.message_log[-MAX_LOG_LENGTH:]

    def advance_turn(self) -> None:
        """Advances the game turn counter and performs turn-based updates."""
        self.turn_count += 1
        log.debug("Turn advanced", turn=self.turn_count)
        # --- Status Effect Duration Update ---
        # Iterate through active entities, decrementing status effect durations
        # and removing any expired effects.
        for row in self.entity_registry.entities_df.iter_rows(named=True):
            if not row.get("is_active", False):
                continue

            entity_id = row["entity_id"]
            status_effects = row.get("status_effects") or []
            if not status_effects:
                continue

            updated_effects: list[dict] = []
            for effect in status_effects:
                new_duration = effect.get("duration", 0) - 1
                if new_duration > 0:
                    updated_effects.append({**effect, "duration": new_duration})
                else:
                    log.debug(
                        "Status effect expired",
                        entity_id=entity_id,
                        effect=effect.get("id"),
                    )

            if updated_effects != status_effects:
                self.entity_registry.set_entity_component(
                    entity_id, "status_effects", updated_effects
                )

        # --- AI processing call ---
        log.debug("Invoking AI dispatcher")
        dispatch_ai(self)

        # --- Other turn-based updates ---
        # (e.g., hunger increase, light source fuel consumption)

    # --- NEW: State transition helper ---
    def change_ui_state(
        self, new_state: Literal["PLAYER_TURN", "INVENTORY_VIEW", "TARGETING"]
    ):
        # Basic state change, could add validation or hooks later
        if self.ui_state != new_state:
            log.debug(f"Changing UI state from {self.ui_state} to {new_state}")
            self.ui_state = new_state
        else:
            log.debug(f"UI state already {new_state}, no change.")

    # --- End NEW ---

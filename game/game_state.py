# game/game_state.py
from typing import Any, Callable, Dict, Literal, Set, Tuple, Union

import structlog
from game_rng import GameRNG  # Assuming this path is correct

from game.entities.registry import EntityRegistry
from game.entities.components import Position
from game.items.registry import ItemRegistry

# Assuming these imports are correct relative to game_state.py

from game.world.game_map import GameMap, LightSource
from game.systems.ai_system import dispatch_ai
from game.ai.perception import gather_perception
from simulation.zone_manager import ZoneManager

# Import sound system
try:
    from game.systems.sound import get_sound_manager, update_music_context
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    def get_sound_manager(): return None
    def update_music_context(context): pass


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
        entity_templates: Dict[str, Any] | None = None,
        effect_definitions: Dict[str, Any] | None = None,
        rng_seed: int | None = None,
        ai_config: Dict[str, Any] | None = None,
        memory_fade_config: Dict[str, Any] | None = None,
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

        # Store loaded entity templates in a simple registry
        from game.entities.template_registry import EntityTemplateRegistry

        self.entity_templates = EntityTemplateRegistry(entity_templates or {})

        self.item_registry: ItemRegistry = ItemRegistry(item_templates)
        self.effect_definitions: Dict[str, Any] = effect_definitions or {}
        self.ai_config: Dict[str, Any] = ai_config or {}
        mf_config = memory_fade_config or {}
        duration = mf_config.get("duration", 60.0)
        self.memory_fade_enabled: bool = mf_config.get("enabled", True)
        self.memory_fade_duration: float = duration
        self.memory_fade_midpoint: float = mf_config.get("midpoint", duration / 2.0)
        self.memory_fade_steepness: float = mf_config.get(
            "steepness", 6.0 / duration if duration else 0.0
        )
        tile_modifiers_cfg = mf_config.get("tile_modifiers", {})
        self.game_map.apply_memory_modifier_overrides(tile_modifiers_cfg)
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
        # Perception event queues processed by gather_perception
        self.noise_events: list[tuple[int, int, float]] = []
        self.scent_events: list[tuple[int, int, float]] = []
        # Track light sources (player has a default white light)
        self.light_sources: list[LightSource] = self.game_map.light_sources
        self.light_sources.append(
            LightSource(player_start_x, player_start_y, player_fov_radius, (255, 255, 255))
        )
        self.player_light_index: int = 0

        # Track simulation zones for coarse updates when entities are far away
        self.zone_manager: ZoneManager = ZoneManager(
            self._map_width, self._map_height
        )

        # --- NEW: UI State ---
        self.ui_state: Literal["PLAYER_TURN", "INVENTORY_VIEW", "TARGETING"] = (
            "PLAYER_TURN"
        )
        # --- End NEW ---
        
        # --- Sound System ---
        self.sound_manager = get_sound_manager() if SOUND_AVAILABLE else None
        if self.sound_manager:
            log.debug("Sound system initialized")

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
        
        # Initial sound context update
        if self.sound_manager:
            self._update_sound_context()

    def update_fov(self) -> None:
        """Calculates Field of View based on player position."""
        player_pos = self.player_position
        if player_pos:
            px, py = player_pos
            if not self.game_map.in_bounds(px, py):
                log.warning("Player out of bounds, cannot compute FOV.", pos=(px, py))
                self.game_map.visible[:] = False  # Clear visibility if player is OOB
                return
            self.game_map.compute_fov(
                px, py, self.fov_radius
            )  # compute_fov handles explored and origin height internally
            # Post-check: Ensure origin is always visible if FOV somehow clears it
            if not self.game_map.visible[py, px]:
                log.warning(
                    "Origin tile became non-visible after FOV calculation, forcing visible.",
                    pos=(px, py),
                )
                self.game_map.visible[py, px] = True
                self.game_map.explored[py, px] = True  # Ensure explored too
            # Update memory and last seen time for visible tiles
            self.game_map.memory_intensity[self.game_map.visible] = 1.0
            self.game_map.last_seen_time[self.game_map.visible] = self.turn_count
            # Fade memory for tiles no longer visible
            if self.memory_fade_enabled:
                self.game_map.update_memory_fade(
                    self.turn_count,
                    self.memory_fade_steepness,
                    self.memory_fade_midpoint,
                )
            # Keep player light source in sync with position
            try:
                self.light_sources[self.player_light_index].x = px
                self.light_sources[self.player_light_index].y = py
            except Exception:
                pass
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
    def player_position(self) -> Position | None:
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

    def schedule_low_detail_update(
        self, x: int, y: int, callback: Callable[["GameState"], None]
    ) -> None:
        """Queue a low-detail update for the zone containing ``(x, y)``.

        Systems can use this to defer expensive logic for entities that are
        far away from the player.  The callback will receive the ``GameState``
        instance when executed.
        """
        self.zone_manager.schedule_event(x, y, callback)

    def _process_status_effects_for_entity(self, entity_id: int) -> None:
        """Tick down status effects for a single entity."""
        status_effects = (
            self.entity_registry.get_entity_component(entity_id, "status_effects") or []
        )
        if not status_effects:
            return
        updated_effects: list[dict] = []
        for effect in status_effects:
            new_duration = effect.get("duration", 0) - 1
            if new_duration > 0:
                updated_effects.append({**effect, "duration": new_duration})
            else:
                effect_id = effect.get("id")
                log.debug(
                    "Status effect expired", entity_id=entity_id, effect=effect_id
                )
                entity_name = (
                    self.entity_registry.get_entity_component(entity_id, "name")
                    or f"Entity {entity_id}"
                )
                self.add_message(f"{entity_name}'s {effect_id} wears off.")
        if updated_effects != status_effects:
            self.entity_registry.set_entity_component(
                entity_id, "status_effects", updated_effects
            )

    def _process_zone(self, zone: Tuple[int, int]) -> None:
        """Aggregate update for all entities within ``zone``.

        This performs a very coarse simulation step used for areas that are far
        from the player.  Perception data is omitted for performance; AI
        adapters receive ``None`` for the perception argument.
        """
        for row in self.entity_registry.entities_df.iter_rows(named=True):
            if not row.get("is_active", False):
                continue
            if self.zone_manager.get_zone(row.get("x"), row.get("y")) != zone:
                continue
            entity_id = row["entity_id"]
            self._process_status_effects_for_entity(entity_id)
            if entity_id == self.player_id:
                continue
            ai_type = row.get("ai_type") or self.ai_config.get("default", "goap")
            adapter = get_adapter(ai_type)
            adapter(row, self, self.rng_instance, None)

    def advance_turn(self) -> None:
        """Advances the game turn counter and performs turn-based updates."""
        self.turn_count += 1
        log.debug("Turn advanced", turn=self.turn_count)
        player_pos = self.player_position
        if player_pos:
            px, py = player_pos
            self.scent_events.append((px, py, 5.0))

        active_zones: Set[Tuple[int, int]] = self.zone_manager.get_active_zones(
            player_pos
        )

        # Update nearby entities immediately and schedule distant zones for later
        for row in self.entity_registry.entities_df.iter_rows(named=True):
            if not row.get("is_active", False):
                continue
            entity_id = row["entity_id"]
            zone = self.zone_manager.get_zone(row.get("x"), row.get("y"))
            if zone in active_zones:
                self._process_status_effects_for_entity(entity_id)
            else:
                self.zone_manager.schedule_zone_event(zone, lambda gs, z=zone: gs._process_zone(z))

        # Recalculate FOV so perception and rendering use up-to-date visibility
        self.update_fov()

        # --- AI processing for nearby entities ---
        log.debug("Gathering perception data for AI")
        perception = gather_perception(self)

        log.debug("Processing AI-controlled entities")
        for row in self.entity_registry.entities_df.iter_rows(named=True):
            if not row.get("is_active", False):
                continue
            if row["entity_id"] == self.player_id:
                continue

            zone = self.zone_manager.get_zone(row.get("x"), row.get("y"))
            if zone not in active_zones:
                continue
            ai_type = row.get("ai_type") or self.ai_config.get("default", "goap")
            adapter = get_adapter(ai_type)
            adapter(row, self, self.rng_instance, perception)
            dispatch_ai(row, self, self.rng_instance, perception)


        # Process any queued low-detail zone updates
        self.zone_manager.process(self.turn_count, active_zones, self)

        # --- Other turn-based updates ---
        # (e.g., hunger increase, light source fuel consumption)
        
        # Update sound context for situational music
        if self.sound_manager:
            self._update_sound_context()

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

    # --- Sound System Integration ---
    def _update_sound_context(self) -> None:
        """Update the sound system with current game context for situational music."""
        if not self.sound_manager:
            return
            
        # Determine current game state
        game_state = "exploring"  # Default state
        
        # Check if player is in combat (nearby hostile entities)
        player_pos = self.player_position
        if player_pos:
            px, py = player_pos
            # Look for nearby hostile entities within FOV
            for row in self.entity_registry.entities_df.iter_rows(named=True):
                if not row.get("is_active", False) or row["entity_id"] == self.player_id:
                    continue
                    
                ex, ey = row.get("x", 0), row.get("y", 0)
                # Check if entity is within reasonable combat distance and visible
                distance_sq = (px - ex) ** 2 + (py - ey) ** 2
                if distance_sq <= (self.fov_radius + 2) ** 2:
                    if hasattr(self.game_map, 'visible') and self.game_map.visible[ey, ex]:
                        # Assume any visible nearby entity means combat
                        game_state = "combat"
                        break
        
        # Determine depth (if available)
        depth = getattr(self, 'current_depth', 1)
        
        # Check for special entity types nearby
        enemy_types = []
        if player_pos:
            px, py = player_pos
            for row in self.entity_registry.entities_df.iter_rows(named=True):
                if not row.get("is_active", False) or row["entity_id"] == self.player_id:
                    continue
                    
                ex, ey = row.get("x", 0), row.get("y", 0)
                distance_sq = (px - ex) ** 2 + (py - ey) ** 2
                if distance_sq <= self.fov_radius ** 2:
                    entity_name = row.get("name", "").lower()
                    if "boss" in entity_name or "dragon" in entity_name or "demon" in entity_name:
                        enemy_types.append("boss")
                    elif "elite" in entity_name or "champion" in entity_name:
                        enemy_types.append("elite")
        
        # Build context for sound system
        context = {
            "game_state": game_state,
            "depth": depth,
            "turn": self.turn_count,
            "player_hp_percent": 1.0,  # Default
            "ui_state": self.ui_state.lower()
        }
        
        # Add enemy type if any special enemies nearby
        if enemy_types:
            context["enemy_type"] = enemy_types
        
        # Get player HP percentage if available
        player_hp = self.entity_registry.get_entity_component(self.player_id, "hp")
        player_max_hp = self.entity_registry.get_entity_component(self.player_id, "max_hp")
        if player_hp is not None and player_max_hp is not None and player_max_hp > 0:
            context["player_hp_percent"] = player_hp / player_max_hp
        
        # Update background music based on context
        update_music_context(context)

    # --- End NEW ---

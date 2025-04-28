# game/game_state.py
import numpy as np
import structlog  # Added
from typing import Tuple, Union

from game.world.game_map import GameMap  # Keep existing imports
from game.entities.registry import EntityRegistry

log = structlog.get_logger()  # Added

# Define default values here, matching those in main.py's .get() calls
# (Defaults are now primarily handled in main.py's config loading)
# DEFAULT_PLAYER_GLYPH = 64
# DEFAULT_PLAYER_HP = 30
# DEFAULT_FOV_RADIUS = 8


class GameState:
    def __init__(
        self,
        existing_map: GameMap,
        player_start_pos: Tuple[int, int],
        # Config values passed directly now
        player_glyph: int,
        player_start_hp: int,
        player_fov_radius: int,
    ):
        log.info("Initializing GameState...")  # Added log

        if not isinstance(existing_map, GameMap):
            log.error("GameState init failed: Invalid GameMap instance provided.")
            raise TypeError("GameState requires a valid GameMap instance.")
        if not player_start_pos or len(player_start_pos) != 2:
            log.error(
                "GameState init failed: Invalid player_start_pos provided.",
                pos=player_start_pos,
            )
            raise ValueError(
                "GameState requires a valid player_start_pos tuple (x, y)."
            )

        self.game_map: GameMap = existing_map
        self._map_width: int = existing_map.width
        self._map_height: int = existing_map.height

        self.entity_registry: EntityRegistry = EntityRegistry()
        log.debug("EntityRegistry initialized")  # Added log

        # Unpack player start position
        player_start_x, player_start_y = player_start_pos

        # --- Use the passed config parameters ---
        self.player_id: int = self.entity_registry.create_entity(
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
            player_id=self.player_id,
            pos=(player_start_x, player_start_y),
            hp=player_start_hp,
            glyph=player_glyph,
        )

        self.fov_radius = player_fov_radius
        # --- End using parameters ---

        self.message_log: list[tuple[str, tuple[int, int, int]]] = []
        self.turn_count: int = 0
        self.add_message(
            "Welcome to BasicRL!", (0, 255, 0)
        )  # Log handled in add_message

        # Replaced print with log.info
        log.info(
            "Game state initialized",
            map_size=f"{self._map_width}x{self._map_height}",
            player_id=self.player_id,
            player_start_pos=(player_start_x, player_start_y),
            player_hp=player_start_hp,
            player_glyph=player_glyph,
            fov_radius=self.fov_radius,
        )

        self.update_fov()  # Initial FOV calculation

    def update_fov(self) -> None:
        """Updates the field of view based on player position."""
        player_pos = self.player_position
        if player_pos:
            px, py = player_pos
            log_context = {
                "player_id": self.player_id,
                "pos": player_pos,
                "radius": self.fov_radius,
            }

            if not self.game_map.in_bounds(px, py):
                # Replaced print with log.error
                log.error(
                    "Player position out of map bounds, cannot compute FOV",
                    **log_context,
                )
                self.game_map.visible[
                    :
                ] = False  # Clear visibility if player out of bounds
                return

            self.game_map.compute_fov(px, py, self.fov_radius)
            visible_count = np.sum(self.game_map.visible)

            # Keep emergency visibility check, use logging
            if visible_count == 0 and self.game_map.in_bounds(px, py):
                # Replaced print with log.warning
                log.warning(
                    "Zero tiles visible after FOV update! Forcing player pos visible.",
                    **log_context,
                )
                # Make player position visible (already checked in_bounds)
                self.game_map.visible[py, px] = True
                self.game_map.explored[py, px] = True
                visible_count = np.sum(self.game_map.visible)
                # Replaced print with log.info
                log.info(
                    "Emergency visibility set",
                    visible_count=visible_count,
                    **log_context,
                )
        else:
            # Replaced print with log.warning
            log.warning(
                "Cannot update FOV: Player position not found in registry",
                player_id=self.player_id,
            )
            self.game_map.visible[:] = False  # Clear visibility if no player pos

    @property
    def map_width(self) -> int:
        return self._map_width

    @property
    def map_height(self) -> int:
        return self._map_height

    @property
    def player_position(self) -> Union[Tuple[int, int], None]:
        """Gets player (x, y) position from the registry."""
        return self.entity_registry.get_position(self.player_id)

    def add_message(
        self, text: str, color: tuple[int, int, int] = (255, 255, 255)
    ) -> None:
        self.message_log.append((text, color))
        # Optional: Limit message log size
        # MAX_LOG_MESSAGES = 100
        # if len(self.message_log) > MAX_LOG_MESSAGES:
        #     self.message_log.pop(0)
        # Log the message added
        log.debug("Message added", message=text, color=color)

    def advance_turn(self) -> None:
        self.turn_count += 1
        # Placeholder for enemy turns, game logic updates, etc.
        # Log turn advance, potentially add more context later (e.g., active entity count)
        log.debug("Turn advanced", turn=self.turn_count)

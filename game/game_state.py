# game/game_state.py
from game.world.game_map import GameMap, TILE_ID_FLOOR
from game.entities.registry import EntityRegistry
import numpy as np
from typing import Tuple, Union

# Define default values here, matching those in main.py's .get() calls
DEFAULT_PLAYER_GLYPH = 64  # '@' symbol often used as default
DEFAULT_PLAYER_HP = 30
DEFAULT_FOV_RADIUS = 4


class GameState:
    def __init__(
        self,
        existing_map: GameMap,  # Require existing_map, remove width/height args
        player_start_pos: Tuple[int, int],
        # --- Add new config parameters with defaults ---
        player_glyph: int = DEFAULT_PLAYER_GLYPH,
        player_start_hp: int = DEFAULT_PLAYER_HP,
        player_fov_radius: int = DEFAULT_FOV_RADIUS,
        # --- End new parameters ---
    ):
        if not isinstance(existing_map, GameMap):
            raise TypeError("GameState requires a valid GameMap instance.")
        if not player_start_pos:
            raise ValueError("GameState requires a valid player_start_pos.")

        self.game_map: GameMap = existing_map
        self._map_width: int = existing_map.width
        self._map_height: int = existing_map.height

        self.entity_registry: EntityRegistry = EntityRegistry()

        # Unpack player start position
        player_start_x, player_start_y = player_start_pos

        # --- Use the passed config parameters ---
        self.player_id: int = self.entity_registry.create_entity(
            x=player_start_x,
            y=player_start_y,
            glyph=player_glyph,  # Use parameter
            color_fg=(255, 255, 255),  # Keep color for now, could be config too
            name="Player",
            blocks_movement=True,
            hp=player_start_hp,  # Use parameter
            max_hp=player_start_hp,  # Use parameter for max_hp too
        )

        self.fov_radius = player_fov_radius  # Use parameter
        # --- End using parameters ---

        self.message_log: list[tuple[str, tuple[int, int, int]]] = []
        self.turn_count: int = 0
        self.add_message("Welcome to BasicRL!", (0, 255, 0))

        print(
            f"Game state initialized. Player actual start: (x={player_start_x}, y={player_start_y}), "
            f"HP: {player_start_hp}, Glyph: {player_glyph}, FOV radius: {self.fov_radius}"
        )
        self.update_fov()  # Initial FOV calculation

    # No need for _find_initial_player_pos_fallback anymore if player_start_pos is required
    # def _find_initial_player_pos_fallback(self) -> tuple[int, int]: ...

    def update_fov(self) -> None:
        """Updates the field of view based on player position."""
        player_pos = self.player_position
        if player_pos:
            px, py = player_pos

            if not self.game_map.in_bounds(px, py):
                print(f"ERROR: Player position ({px}, {py}) is out of map bounds!")
                self.game_map.visible[:] = False
                return

            # Optional: Check if player is on a non-transparent tile (could happen?)
            # is_trans = self.game_map.is_transparent(px, py)
            # if not is_trans:
            #    print(f"WARNING: Player at ({px}, {py}) is not on a transparent tile!")

            self.game_map.compute_fov(px, py, self.fov_radius)  # Use self.fov_radius
            visible_count = np.sum(self.game_map.visible)

            # Keep emergency visibility check
            if visible_count == 0:
                print(
                    "WARNING: Zero tiles visible after FOV update! Making player pos visible."
                )
                if self.game_map.in_bounds(px, py):
                    self.game_map.visible[py, px] = True
                    self.game_map.explored[py, px] = True
                visible_count = np.sum(self.game_map.visible)
                print(f"After emergency visibility: {visible_count} tiles are visible")
        else:
            print("Warning: Cannot update FOV - player position not found in registry")
            self.game_map.visible[:] = False

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

    def advance_turn(self) -> None:
        self.turn_count += 1
        # Placeholder for enemy turns, game logic updates, etc.
        # print(f"--- Turn {self.turn_count} ---")

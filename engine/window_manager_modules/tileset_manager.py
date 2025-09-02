# engine/window_manager_modules/tileset_manager.py
"""
Manages loading, caching, and providing access to tileset data,
including Numba-compatible NumPy arrays.
"""
# Standard Imports
from pathlib import Path
from typing import Any
from typing import Dict as PyDict

# Third-party Imports
import numpy as np
import structlog
from numba import types as nb_types
from numba.typed import Dict as NumbaTypedDict
from PIL import Image

# Local Application Imports
# Use absolute paths from project root (basicrl)
from engine.tileset_loader import load_tiles

# Ensure GameMap and TILE_TYPES can be imported for schema validation/population
try:
    from game.world.game_map import TILE_TYPES
except ImportError:
    # Fallback if necessary, but ideally structure allows direct import
    try:
        from basicrl.game.world.game_map import TILE_TYPES
    except ImportError:
        TILE_TYPES = {}  # Fallback to empty if not found
        structlog.get_logger().error("Could not import TILE_TYPES for TilesetManager.")


log = structlog.get_logger(__name__)

SENTINEL_TILE_ARRAY_SHAPE = (0, 0, 4)
SENTINEL_TILE_ARRAY = np.empty(SENTINEL_TILE_ARRAY_SHAPE, dtype=np.uint8)


class TilesetManager:
    """Handles loading and caching of tile assets."""

    def __init__(
        self,
        initial_tileset_path: str,
        initial_tile_width: int,
        initial_tile_height: int,
        min_tile_size_cfg: int,
    ):
        log.info("Initializing TilesetManager...")
        self.min_tile_size: int = min_tile_size_cfg
        self.current_tileset_path: str = ""
        # PIL images {tile_index: Image}
        self.tiles: PyDict[int, Image.Image] = {}
        self.tile_width: int = 0
        self.tile_height: int = 0

        # --- Numba Cache and Render Data ---
        # Define the type for the Numba dictionary values (3D NumPy array)
        # Using nb_types.uint8[:,:,::1] specifies C-contiguous arrays which is common
        _array_value_type = nb_types.uint8[:, :, ::1]
        self.tile_arrays: NumbaTypedDict = NumbaTypedDict.empty(
            key_type=nb_types.int_, value_type=_array_value_type
        )  # Numba dict {tile_index: np.ndarray[h, w, 4]}
        self.max_defined_tile_id: int = -1
        self._tile_fg_colors: np.ndarray | None = None  # [max_id+1, 3] uint8
        self._tile_bg_colors: np.ndarray | None = None  # [max_id+1, 3] uint8
        # [max_id+1] uint16
        self._tile_indices_render: np.ndarray | None = None
        # --- End Numba Cache ---

        # Perform initial load
        self.load_new_tileset(
            initial_tileset_path, initial_tile_width, initial_tile_height
        )
        log.debug("TilesetManager initialized.")

    def load_new_tileset(self, folder: str, width: int, height: int) -> bool:
        """Loads a new tileset from the specified folder and dimensions."""
        target_abs_path_str = "unknown"
        try:
            clamped_width = max(self.min_tile_size, width)
            clamped_height = max(self.min_tile_size, height)

            # Resolve path correctly
            target_path_obj = Path(folder)
            if not target_path_obj.is_absolute():
                try:  # Assume relative to project root if possible
                    # Adjust based on actual file structure if needed
                    base_path = Path(__file__).parent.parent.parent
                except NameError:
                    base_path = Path(".")  # Fallback
                target_abs_path = (base_path / folder).resolve()
            else:
                target_abs_path = target_path_obj.resolve()
            target_abs_path_str = str(target_abs_path)

            if (
                clamped_width == self.tile_width
                and clamped_height == self.tile_height
                and target_abs_path_str == self.current_tileset_path
            ):
                log.info(
                    "Tileset unchanged, skipping reload.", path=target_abs_path_str
                )
                return True  # Considered successful as state is correct

            log.info(
                "Loading tileset",
                path=target_abs_path_str,
                w=clamped_width,
                h=clamped_height,
            )
            # Assuming load_tiles returns: Dict[int, Image.Image], bool
            loaded_tiles, _ = load_tiles(
                target_abs_path_str, clamped_width, clamped_height
            )

            self.current_tileset_path = target_abs_path_str
            # Ensure loaded_tiles is the correct type before assignment
            if isinstance(loaded_tiles, dict):
                self.tiles = loaded_tiles
            else:
                log.error(
                    "load_tiles did not return a dictionary.",
                    received_type=type(loaded_tiles),
                )
                self.tiles = {}  # Reset to empty on error
                # Consider raising an error or returning False earlier

            self.tile_width = clamped_width
            self.tile_height = clamped_height

            self._update_tile_array_cache()  # Update Numba cache

            log.info(
                "Tileset loaded successfully",
                path=target_abs_path_str,
                final_w=self.tile_width,
                final_h=self.tile_height,
                count=len(self.tiles),
            )
            return True

        except Exception as e:
            log.error(
                "Error loading tileset in TilesetManager",
                path_param=folder,
                abs_path=target_abs_path_str,
                error=str(e),
                exc_info=True,
            )
            # Decide whether to keep old state or reset
            # Resetting might be safer if load failed badly
            # self.tiles = {}; self.tile_width = 0; self.tile_height = 0;
            # self._update_tile_array_cache() # Ensure cache is cleared/reset too
            return False

    def _update_tile_array_cache(self) -> None:
        """Updates the Numba-compatible NumPy array cache and render data."""
        log.debug("Updating tile array cache (Numba typed.Dict & render data)...")
        pil_tile_count = 0
        numba_tile_count = 0
        # Ensure type consistency with __init__
        _array_value_type = nb_types.uint8[:, :, ::1]

        # Reset caches
        self.max_defined_tile_id = -1
        self._tile_fg_colors = None
        self._tile_bg_colors = None
        self._tile_indices_render = None
        temp_tile_arrays: NumbaTypedDict = NumbaTypedDict.empty(
            key_type=nb_types.int_, value_type=_array_value_type
        )

        # 1. Update render data cache (fg/bg colors, glyph indices) from TILE_TYPES
        if TILE_TYPES and isinstance(TILE_TYPES, dict):
            try:
                # Filter out non-integer keys just in case
                valid_tile_ids = [k for k in TILE_TYPES.keys() if isinstance(k, int)]
                if not valid_tile_ids:
                    log.warning("TILE_TYPES dictionary contains no integer keys.")
                    self.max_defined_tile_id = -1
                else:
                    self.max_defined_tile_id = max(valid_tile_ids)
            except Exception as e:
                log.error(
                    "Error finding max key in TILE_TYPES", error=str(e), data=TILE_TYPES
                )
                self.max_defined_tile_id = -1

            array_size = self.max_defined_tile_id + 1
            if array_size > 0:
                self._tile_fg_colors = np.zeros((array_size, 3), dtype=np.uint8)
                self._tile_bg_colors = np.zeros((array_size, 3), dtype=np.uint8)
                self._tile_indices_render = np.zeros(array_size, dtype=np.uint16)
                valid_ids_loaded = 0
                for tile_id_val, tile_type_data in TILE_TYPES.items():
                    if not isinstance(tile_id_val, int):
                        continue  # Skip non-int keys

                    if 0 <= tile_id_val <= self.max_defined_tile_id:
                        # Safely access attributes using getattr with defaults
                        fg_color = getattr(tile_type_data, "color_fg", (255, 0, 255))
                        bg_color = getattr(tile_type_data, "color_bg", (0, 0, 0))
                        # Default to glyph 0
                        tile_index = getattr(tile_type_data, "tile_index", 0)

                        if isinstance(fg_color, tuple) and len(fg_color) == 3:
                            self._tile_fg_colors[tile_id_val] = fg_color
                        if isinstance(bg_color, tuple) and len(bg_color) == 3:
                            self._tile_bg_colors[tile_id_val] = bg_color
                        self._tile_indices_render[tile_id_val] = int(tile_index)

                        valid_ids_loaded += 1
                log.debug(
                    "Tile render cache populated",
                    loaded_ids=valid_ids_loaded,
                    max_id=self.max_defined_tile_id,
                    array_size=array_size,
                    fg_shape=self._tile_fg_colors.shape,
                    bg_shape=self._tile_bg_colors.shape,
                    idx_shape=self._tile_indices_render.shape,
                )
            else:
                log.warning(
                    "max_defined_tile_id resulted in non-positive array size",
                    max_id=self.max_defined_tile_id,
                )
        else:
            log.warning("TILE_TYPES is empty or invalid, cannot populate render cache.")

        # 2. Update Numba array cache from loaded PIL tiles
        if not self.tiles or self.tile_width <= 0 or self.tile_height <= 0:
            log.warning("Cannot update Numba cache: Invalid tiles or dimensions.")
            self.tile_arrays = temp_tile_arrays  # Assign empty dict
            return

        pil_tile_count = len(self.tiles)
        for tile_index, img in self.tiles.items():
            if img is None:
                temp_tile_arrays[tile_index] = SENTINEL_TILE_ARRAY
                continue
            try:
                if img.size != (self.tile_width, self.tile_height):
                    img = img.resize(
                        (self.tile_width, self.tile_height), Image.Resampling.NEAREST
                    )
                if img.mode != "RGBA":
                    img = img.convert("RGBA")

                tile_np_array = np.array(img, dtype=np.uint8)
                # Ensure C-contiguity for Numba compatibility if needed
                if not tile_np_array.flags["C_CONTIGUOUS"]:
                    tile_np_array = np.ascontiguousarray(tile_np_array)

                # Check final shape AFTER potential conversion/resizing
                if tile_np_array.shape == (self.tile_height, self.tile_width, 4):
                    temp_tile_arrays[tile_index] = tile_np_array
                    numba_tile_count += 1
                else:
                    log.warning(
                        f"Tile {tile_index} final shape {tile_np_array.shape} != expected ({self.tile_height}, {
                            self.tile_width}, 4). Storing sentinel."
                    )
                    temp_tile_arrays[tile_index] = SENTINEL_TILE_ARRAY
            except Exception as e:
                log.warning(
                    f"Could not convert tile {tile_index} for Numba: {e}", exc_info=True
                )
                temp_tile_arrays[tile_index] = SENTINEL_TILE_ARRAY

        self.tile_arrays = temp_tile_arrays
        log.info(
            "TilesetManager cache updated",
            pil_count=pil_tile_count,
            numba_count=numba_tile_count,
        )

    def get_render_data(self) -> PyDict[str, Any]:
        """Returns the data needed by the renderer."""
        # Check if caches are valid
        cache_ready = (
            self._tile_fg_colors is not None
            and self._tile_bg_colors is not None
            and self._tile_indices_render is not None
            and self.max_defined_tile_id >= 0
            and self.tile_arrays is not None  # Check Numba dict itself
        )

        # *** ADDED LOGGING ***
        log.debug(
            "TilesetManager.get_render_data called.",
            cache_ready=cache_ready,
            max_id=self.max_defined_tile_id,
            fg_colors_valid=self._tile_fg_colors is not None,
            bg_colors_valid=self._tile_bg_colors is not None,
            indices_valid=self._tile_indices_render is not None,
            tile_arrays_items=(
                len(self.tile_arrays) if self.tile_arrays is not None else "None"
            ),
        )

        if not cache_ready:
            log.error("Render data cache is not ready in TilesetManager.")
            # Return empty/default data to avoid crashing renderer
            _array_type = nb_types.uint8[:, :, ::1]  # Match type
            return {
                "tile_arrays": NumbaTypedDict.empty(
                    key_type=nb_types.int_, value_type=_array_type
                ),
                "tile_fg_colors": np.zeros((1, 3), dtype=np.uint8),
                "tile_bg_colors": np.zeros((1, 3), dtype=np.uint8),
                "tile_indices_render": np.zeros(1, dtype=np.uint16),
                "max_defined_tile_id": -1,  # Indicate invalid cache
                "tile_w": self.tile_width,
                "tile_h": self.tile_height,
            }

        return {
            "tile_arrays": self.tile_arrays,
            "tile_fg_colors": self._tile_fg_colors,
            "tile_bg_colors": self._tile_bg_colors,
            "tile_indices_render": self._tile_indices_render,
            "max_defined_tile_id": self.max_defined_tile_id,
            "tile_w": self.tile_width,
            "tile_h": self.tile_height,
        }

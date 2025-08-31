# engine/window_manager_modules/ui_overlay_manager.py
"""
Manages the rendering of UI overlays loaded from a TOML configuration file.

Overlay Config Schema (``config/overlays.toml``):

```
[[overlay]]
id   = "debug"        # Unique identifier
type = "debug"        # Built-in overlay types: debug, height_key, inventory, image

# Image overlays can specify a path and pixel position
# [[overlay]]
# id       = "logo"
# type     = "image"
# path     = "assets/ui/logo.png"  # Relative to the config file
# position = [5, 5]
```

Mods can add or replace overlay definitions in this file and supply their own
tile graphics by referencing image paths.
"""
# Standard Imports
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing import Dict as PyDict
from typing import List, Tuple
import tomllib

# Third-party Imports
import polars as pl
import structlog
from PIL import Image, ImageDraw, ImageFont

# --- Type Checking Imports ---
if TYPE_CHECKING:
    # Use absolute paths relative to project root (basicrl)
    from engine.main_loop import MainLoop
    from engine.window_manager import (
        WindowManager,
    )  # Needs reference for config/fonts/state
    from game.game_state import GameState

log = structlog.get_logger(__name__)


class UIOverlayManager:
    """Handles rendering and state for UI overlays."""

    def __init__(
        self, window_manager_ref: "WindowManager", overlay_config_path: Path
    ) -> None:
        self.window_manager_ref: "WindowManager" = window_manager_ref
        self.overlay_config_path = overlay_config_path
        self._overlay_base_path = self.overlay_config_path.parent
        self.overlay_defs: List[PyDict[str, Any]] = self._load_overlay_definitions()
        self._image_cache: PyDict[str, Image.Image] = {}
        # Inventory state
        self.inventory_cursor: int = 0
        self.inventory_scroll_offset: int = 0  # For future scrolling
        # Map from display line index to (item_id | None, is_equipped_flag, is_attached)
        self._inventory_ui_map: PyDict[int, Tuple[int | None, bool, bool]] = {}
        log.debug("UIOverlayManager initialized.")

    def _load_overlay_definitions(self) -> List[PyDict[str, Any]]:
        """Loads overlay definitions from the TOML configuration file."""
        if not self.overlay_config_path.is_file():
            log.warning(
                "Overlay config file not found", path=str(self.overlay_config_path)
            )
            return []
        try:
            with self.overlay_config_path.open("rb") as f:
                data = tomllib.load(f)
            overlays = data.get("overlay", [])
            if not isinstance(overlays, list):
                log.warning("Overlay config missing 'overlay' list")
                return []
            return overlays
        except Exception as e:
            log.error(
                "Failed to load overlay config",
                path=str(self.overlay_config_path),
                error=e,
            )
            return []

    def reset_inventory_state(self) -> None:
        """Resets cursor and map when inventory is opened/closed."""
        self.inventory_cursor = 0
        self.inventory_scroll_offset = 0
        self._inventory_ui_map.clear()
        log.debug("Inventory UI state reset.")

    def render_overlays(
        self, base_image: Image.Image, gs: "GameState", main_loop_ref: "MainLoop"
    ) -> Image.Image:
        """Adds all relevant overlays to the base rendered image."""
        if not base_image:
            log.error("Received None base_image for overlays.")
            try:  # Attempt to get size from window manager for blank image
                w = self.window_manager_ref.label.width()
                h = self.window_manager_ref.label.height()
                return Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
            except Exception:
                return Image.new("RGBA", (100, 100), (0, 0, 0, 0))  # Fallback size
    
        img_copy = base_image.copy()  # Work on a copy
        draw = ImageDraw.Draw(img_copy)

        # Font Loading (Consider centralizing or caching in WindowManager/Config)
        font_cfg = self.window_manager_ref.app_config if hasattr(self.window_manager_ref, "app_config") else {}
        font_path = font_cfg.get("ui_font_path", "arial.ttf")
        font_size = int(font_cfg.get("ui_font_size", 10))
        try:
            text_font = ImageFont.truetype(font_path, font_size)
        except IOError:
            text_font = ImageFont.load_default()

        bg_rect_debug = (0, 0, 0, 0)
        for overlay in self.overlay_defs:
            if not overlay.get("enabled", True):
                continue
            otype = overlay.get("type")
            if otype == "debug":
                bg_rect_debug = self._render_debug_overlay(
                    draw, text_font, gs, main_loop_ref
                )
            elif otype == "height_key":
                if main_loop_ref.show_height_visualization:
                    self._render_height_key_overlay(
                        draw, text_font, bg_rect_debug, main_loop_ref
                    )
            elif otype == "inventory":
                if gs.ui_state == "INVENTORY_VIEW":
                    img_copy = self._render_inventory_overlay(
                        img_copy, draw, text_font, gs
                    )
            elif otype == "image":
                img_copy = self._render_image_overlay(img_copy, overlay)

        # 4. Message Log Overlay (Future)
        # self._render_message_log(draw, text_font, gs)

        return img_copy
        

    def _render_debug_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont,
        gs: "GameState",
        ml: "MainLoop",
    ) -> Tuple[int, int, int, int]:
        """Renders the debug text overlay. Returns bounding box of background."""
        try:
            wm = self.window_manager_ref  # Shortcut for readability
            turn = gs.turn_count
            player_pos = gs.player_position
            pos_str = f"({player_pos[0]},{player_pos[1]})" if player_pos else "N/A"

            entities_count_str = "?"
            try:  # Safely get entity count
                if gs.entity_registry:
                    entities_count_str = str(
                        gs.entity_registry.entities_df.filter(
                            pl.col("is_active")
                        ).height
                    )
            except Exception as e:
                log.warning("Could not get entity count for debug overlay", error=e)

            label_w = wm.label.width()
            label_h = wm.label.height()

            # --- CORRECTED: Get tile dimensions from TilesetManager via WindowManager ---
            if wm.tileset_manager:
                current_tile_w = wm.tileset_manager.tile_width
                current_tile_h = wm.tileset_manager.tile_height
            else:  # Fallback if manager isn't ready (shouldn't happen)
                current_tile_w = 1
                current_tile_h = 1
                log.error(
                    "TilesetManager not available on WindowManager for debug overlay."
                )
            # --- END CORRECTION ---

            vp_cols = max(1, label_w // current_tile_w) if current_tile_w > 0 else "?"
            vp_rows = max(1, label_h // current_tile_h) if current_tile_h > 0 else "?"

            debug_text = (
                f"T:{turn} P:{pos_str} E:{entities_count_str} "
                f"VP:{vp_cols}x{vp_rows} TR:{current_tile_w}x{current_tile_h} "  # Use fetched dimensions
                f"V:{'H' if ml.show_height_visualization else '-'} S:{gs.ui_state}"
            )

            text_x = 5
            text_y = 5
            text_color = (255, 255, 0, 255)  # Yellow
            bg_color = (0, 0, 0, 180)  # Semi-transparent black

            try:  # Calculate text bounding box
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((text_x, text_y), debug_text, font=font)
                elif hasattr(draw, "textlength"):
                    w = draw.textlength(debug_text, font=font)
                    h = (
                        font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
                        if hasattr(font, "getbbox")
                        else 12
                    )
                    bbox = (text_x, text_y, text_x + w, text_y + h)
                else:
                    w = len(debug_text) * 6
                    h = 12
                    bbox = (text_x, text_y, text_x + w, text_y + h)
            except Exception as e:
                log.error("Error calculating debug text bounding box", error=e)
                bbox = (text_x, text_y, text_x + 100, text_y + 15)  # Fallback bbox

            # Draw background and text
            bg_rect = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
            draw.rectangle(bg_rect, fill=bg_color)
            draw.text((text_x, text_y), debug_text, fill=text_color, font=font)
            return bg_rect  # Return background rect for positioning other elements

        except Exception as e_get_dbg:
            log.error("Error getting debug info for overlay", error=str(e_get_dbg))
            return (0, 0, 0, 0)  # Return empty rect on error

    def _render_height_key_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont,
        debug_bg_rect: Tuple[int, int, int, int],
        ml: "MainLoop",
    ) -> None:
        """Renders the height visualization key."""
        # This method remains the same as before, using ml (MainLoop ref) for config values
        try:
            ch_np = ml._cfg_height_color_high_np
            cm_np = ml._cfg_height_color_mid_np
            cl_np = ml._cfg_height_color_low_np
            ch = tuple(map(int, ch_np)) if ch_np is not None else (255, 255, 0)
            cm = tuple(map(int, cm_np)) if cm_np is not None else (0, 255, 0)
            cl = tuple(map(int, cl_np)) if cl_np is not None else (0, 128, 255)

            max_diff_units = ml._cfg_vis_max_diff
            max_diff_meters = max_diff_units / 2.0
            key_width = 15
            key_height = 100
            key_x = 10
            key_y = debug_bg_rect[3] + 5  # Position below debug text
            label_offset = 5
            line_color = (200, 200, 200, 255)

            for i in range(key_height):
                t_norm = (
                    (((key_height - 1 - i) / (key_height - 1)) * 2.0 - 1.0)
                    if key_height > 1
                    else 0.0
                )
                current_bar_color = (
                    self.window_manager_ref.lerp_color(cm, ch, t_norm)
                    if t_norm >= 0
                    else self.window_manager_ref.lerp_color(cm, cl, -t_norm)
                )
                draw.line(
                    [(key_x, key_y + i), (key_x + key_width - 1, key_y + i)],
                    fill=current_bar_color + (255,),
                )

            mid_y_pos = key_y + key_height // 2
            draw.line(
                [(key_x - 2, mid_y_pos), (key_x + key_width + 1, mid_y_pos)],
                fill=(255, 255, 255, 255),
                width=1,
            )
            text_label_x = key_x + key_width + label_offset
            draw.text(
                (text_label_x, key_y - 4),
                f"+{max_diff_meters:.1f}m",
                fill=line_color,
                font=font,
            )
            draw.text((text_label_x, mid_y_pos - 4), "0m", fill=line_color, font=font)
            draw.text(
                (text_label_x, key_y + key_height - 10),
                f"-{max_diff_meters:.1f}m",
                fill=line_color,
                font=font,
            )
        except Exception as e_key_draw:
            log.error(
                "Error drawing height key overlay", error=str(e_key_draw), exc_info=True
            )

    def _render_image_overlay(
        self, base_image: Image.Image, overlay: PyDict[str, Any]
    ) -> Image.Image:
        """Renders a static image overlay defined in the config."""
        path_str = overlay.get("path")
        if not path_str:
            return base_image
        img_path = Path(path_str)
        if not img_path.is_absolute():
            img_path = self._overlay_base_path / img_path
        try:
            cached = self._image_cache.get(str(img_path))
            if cached is None:
                cached = Image.open(img_path).convert("RGBA")
                self._image_cache[str(img_path)] = cached
            pos = overlay.get("position", [0, 0])
            base_image.paste(cached, tuple(pos), cached)
        except Exception as e:
            log.error(
                "Failed to render image overlay", path=str(img_path), error=e
            )
        return base_image

    def _render_inventory_overlay(
        self,
        base_image: Image.Image,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont,
        gs: "GameState",
    ) -> Image.Image:
        """Renders the inventory overlay. Reuses the passed draw object."""
        # (Implementation unchanged from previous step)
        player_id = gs.player_id
        item_reg = gs.item_registry
        entity_reg = gs.entity_registry
        if not item_reg or not entity_reg:
            log.error("Registries missing for inventory.")
            return base_image
        font_cfg = self.window_manager_ref.app_config if hasattr(self.window_manager_ref, "app_config") else {}
        font_path = font_cfg.get("ui_font_path", "arial.ttf")
        title_font = font
        line_height = 15
        panel_padding = 10
        title_height = 20
        panel_width = 350
        try:
            title_font = ImageFont.truetype(font_path, font.size + 2)
        except IOError:
            pass
        all_displayable_items = self._get_combined_inventory_list(gs)
        actual_item_count = sum(
            1
            for item_tuple in all_displayable_items
            if item_tuple[0] is not None and item_tuple[1] is not None
        )
        if actual_item_count == 0:
            panel_height = 60
            panel_x = max(10, (base_image.width - panel_width) // 2)
            panel_y = max(10, (base_image.height - panel_height) // 2)
            bg_color = (0, 0, 0, 200)
            draw.rectangle(
                [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
                fill=bg_color,
                outline=(150, 150, 150, 255),
            )
            draw.text(
                (panel_x + panel_padding, panel_y + panel_padding),
                "Inventory Empty",
                fill=(200, 200, 200, 255),
                font=font,
            )
            return base_image
        equipped_items_tuples = [
            item
            for item in all_displayable_items
            if item[0] is not None and item[2] and item[1] is not None
        ]
        inventory_items_tuples = [
            item
            for item in all_displayable_items
            if item[0] is not None and not item[2] and item[1] is not None
        ]

        has_equipped_items = bool(equipped_items_tuples)
        has_inventory_items = bool(inventory_items_tuples)

        num_display_lines = 0
        num_display_lines += 1  # Equipped header
        num_display_lines += len(equipped_items_tuples)
        for _, item_id, _ in equipped_items_tuples:
            try:
                num_display_lines += item_reg.get_attached_items(item_id).height
            except Exception:
                pass
        if not has_equipped_items:
            num_display_lines += 1  # Placeholder
        num_display_lines += 1  # Spacer + Inv Header
        num_display_lines += len(inventory_items_tuples)
        for _, item_id, _ in inventory_items_tuples:
            try:
                num_display_lines += item_reg.get_attached_items(item_id).height
            except Exception:
                pass
        if not has_inventory_items:
            num_display_lines += 1  # Placeholder
        panel_height = (
            title_height + (num_display_lines * line_height) + (2 * panel_padding)
        )
        panel_x = max(10, (base_image.width - panel_width) // 2)
        panel_y = max(10, (base_image.height - panel_height) // 2)
        bg_color = (0, 0, 0, 200)
        border_color = (150, 150, 150, 255)
        draw.rectangle(
            [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
            fill=bg_color,
            outline=border_color,
            width=1,
        )
        title_text = "INVENTORY"
        title_color = (255, 255, 255, 255)
        try:
            if hasattr(draw, "textbbox"):
                title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
                title_w = title_bbox[2] - title_bbox[0]
            elif hasattr(draw, "textlength"):
                title_w = draw.textlength(title_text, font=title_font)
            else:
                title_w = len(title_text) * 7
        except:
            title_w = len(title_text) * 7
        draw.text(
            (panel_x + (panel_width - title_w) // 2, panel_y + panel_padding),
            title_text,
            fill=title_color,
            font=title_font,
        )
        current_y = panel_y + panel_padding + title_height
        text_x = panel_x + panel_padding
        header_color = (200, 200, 50, 255)
        item_color = (220, 220, 220, 255)
        cursor_color = (0, 255, 255, 255)
        placeholder_color = (150, 150, 150, 255)
        list_index = 0
        self._inventory_ui_map.clear()
        draw.text((text_x, current_y), "-- Equipped --", fill=header_color, font=font)
        current_y += line_height
        self._inventory_ui_map[list_index] = (None, False, False)
        list_index += 1
        if equipped_items_tuples:
            for line, item_id, is_equipped in equipped_items_tuples:
                line_mod = line
                try:
                    if item_reg.get_item_component(item_id, "attachable_info"):
                        line_mod += " (attachable)"
                except Exception:
                    pass
                color = (
                    cursor_color if list_index == self.inventory_cursor else item_color
                )
                prefix = "> " if list_index == self.inventory_cursor else "  "
                draw.text((text_x, current_y), prefix + line_mod, fill=color, font=font)
                current_y += line_height
                self._inventory_ui_map[list_index] = (item_id, is_equipped, False)
                list_index += 1
                try:
                    attached_df = item_reg.get_attached_items(item_id)
                    if attached_df.height > 0:
                        for att in attached_df.iter_rows(named=True):
                            att_line = f"  {att.get('name', '?')} (attached)"
                            color = (
                                cursor_color
                                if list_index == self.inventory_cursor
                                else item_color
                            )
                            prefix = (
                                "> " if list_index == self.inventory_cursor else "  "
                            )
                            draw.text((text_x, current_y), prefix + att_line, fill=color, font=font)
                            current_y += line_height
                            self._inventory_ui_map[list_index] = (
                                att.get("item_id"),
                                False,
                                True,
                            )
                            list_index += 1
                except Exception as e:
                    log.error("Error listing attachments", error=e)
        else:
            draw.text(
                (text_x + 5, current_y),
                "(Nothing equipped)",
                fill=placeholder_color,
                font=font,
            )
            current_y += line_height
            self._inventory_ui_map[list_index] = (None, False, False)
            list_index += 1
        current_y += line_height // 2
        draw.text((text_x, current_y), "-- Inventory --", fill=header_color, font=font)
        current_y += line_height
        self._inventory_ui_map[list_index] = (None, False, False)
        list_index += 1
        if inventory_items_tuples:
            for line, item_id, is_equipped in inventory_items_tuples:
                line_mod = line
                try:
                    if item_reg.get_item_component(item_id, "attachable_info"):
                        line_mod += " (attachable)"
                except Exception:
                    pass
                color = (
                    cursor_color if list_index == self.inventory_cursor else item_color
                )
                prefix = "> " if list_index == self.inventory_cursor else "  "
                draw.text((text_x, current_y), prefix + line_mod, fill=color, font=font)
                current_y += line_height
                self._inventory_ui_map[list_index] = (item_id, is_equipped, False)
                list_index += 1
                try:
                    attached_df = item_reg.get_attached_items(item_id)
                    if attached_df.height > 0:
                        for att in attached_df.iter_rows(named=True):
                            att_line = f"  {att.get('name', '?')} (attached)"
                            color = (
                                cursor_color
                                if list_index == self.inventory_cursor
                                else item_color
                            )
                            prefix = (
                                "> " if list_index == self.inventory_cursor else "  "
                            )
                            draw.text((text_x, current_y), prefix + att_line, fill=color, font=font)
                            current_y += line_height
                            self._inventory_ui_map[list_index] = (
                                att.get("item_id"),
                                False,
                                True,
                            )
                            list_index += 1
                except Exception as e:
                    log.error("Error listing attachments", error=e)
        else:
            draw.text(
                (text_x + 5, current_y), "(Empty)", fill=placeholder_color, font=font
            )
            current_y += line_height
            self._inventory_ui_map[list_index] = (None, False, False)
            list_index += 1
        return base_image

    def _get_combined_inventory_list(
        self, gs: "GameState"
    ) -> List[Tuple[str | None, int | None, bool]]:
        """Generates the list of items for display, using GameState."""
        # (Implementation unchanged from previous step)
        combined_list: List[Tuple[str | None, int | None, bool]] = []
        player_id = gs.player_id
        item_reg = gs.item_registry
        entity_reg = gs.entity_registry
        if not item_reg or not entity_reg:
            return combined_list
        equipped_ids = entity_reg.get_equipped_ids(player_id)
        if equipped_ids:
            try:
                equipped_items_df = item_reg.items_df.filter(
                    pl.col("item_id").is_in(equipped_ids)
                    & (pl.col("location_type") == "equipped")
                    & pl.col("is_active")
                ).sort("equipped_slot")
                for item_data in equipped_items_df.iter_rows(named=True):
                    if isinstance(item_data, dict):
                        name = item_data.get("name", "?")
                        slot = item_data.get("equipped_slot", "?")
                        line = f"{name} ({slot})"
                        combined_list.append((line, item_data.get("item_id"), True))
            except Exception as e:
                log.error("Inv List: Equip fetch error", error=e)
        try:
            inv_items_df = item_reg.get_entity_inventory(player_id)
            active_inv_items_df = inv_items_df.filter(pl.col("is_active"))
            if active_inv_items_df.height > 0:
                for item_data in active_inv_items_df.sort("name").iter_rows(named=True):
                    if isinstance(item_data, dict):
                        name = item_data.get("name", "?")
                        qty = item_data.get("quantity", 1)
                        line = f"{name}" + (f" (x{qty})" if qty > 1 else "")
                        combined_list.append((line, item_data.get("item_id"), False))
        except Exception as e:
            log.error("Inv List: Inv fetch error", error=e)
        return combined_list

    # --- Inventory Interaction Logic ---
    def navigate(self, direction: str) -> None:
        """Handles inventory cursor movement."""
        # (Implementation unchanged from previous step)
        if not self._inventory_ui_map:
            log.debug("Inv map empty, cannot navigate.")
            return
        item_count = len(self._inventory_ui_map)
        if item_count <= 1:
            return
        current_idx = self.inventory_cursor
        start_idx = current_idx
        if direction == "down":
            while True:
                current_idx = (current_idx + 1) % item_count
                item_data = self._inventory_ui_map.get(current_idx)
                if item_data and item_data[0] is not None:
                    self.inventory_cursor = current_idx
                    break
                if current_idx == start_idx:
                    break
        elif direction == "up":
            while True:
                current_idx = (current_idx - 1 + item_count) % item_count
                item_data = self._inventory_ui_map.get(current_idx)
                if item_data and item_data[0] is not None:
                    self.inventory_cursor = current_idx
                    break
                if current_idx == start_idx:
                    break
        # Redraw will be triggered by InputHandler via WindowManager.update_frame()

    def get_action_for_key(self, action_type: str) -> PyDict[str, Any] | None:
        """Gets the game action dictionary for the selected inventory item."""
        # (Implementation unchanged from previous step)
        selected_data = self._inventory_ui_map.get(self.inventory_cursor)
        if not selected_data or selected_data[0] is None:
            return None
        item_id: int = selected_data[0]
        is_equipped: bool = selected_data[1]
        is_attached: bool = selected_data[2]
        action: PyDict[str, Any] | None = None
        gs = self.window_manager_ref.main_loop.game_state
        if action_type == "equip_unequip":
            action = {"type": "unequip" if is_equipped else "equip", "item_id": item_id}
        elif action_type == "use":
            if not is_equipped:
                action = {"type": "use", "item_id": item_id}
          +    elif action_type == "attach":
+        if is_equipped:
+            if gs:
+                gs.add_message("Cannot attach an equipped item.", (255, 100, 0))
+        elif is_attached:
+            if gs:
+                gs.add_message("Item is already attached.", (255, 100, 0))
+        else:
+            host_item_id = None
+            for _, data in self._inventory_ui_map.items():
                 if data[0] is not None and data[1]:
                     host_item_id = data[0]
                     break
             if host_item_id is None:
                 if gs:
                     gs.add_message("No equipped item to attach to.", (255, 100, 0))
             else:
                 action = {
                     "type": "attach",
                     "item_to_attach_id": item_id,
                     "target_host_item_id": host_item_id,
                 }  else:
                (
                    gs.add_message("Cannot use equipped items directly.", (255, 100, 0))
                    if gs
                    else None
                )
        elif action_type == "drop":
            action = {"type": "drop", "item_id": item_id}
        elif action_type == "attach":
            if is_equipped:
                if gs:
                    gs.add_message("Cannot attach an equipped item.", (255, 100, 0))
            elif is_attached:
                if gs:
                    gs.add_message("Item is already attached.", (255, 100, 0))
            else:
                # Find an equipped host to attach toelif action_type == "attach":
        if is_equipped:
            if gs:
                gs.add_message("Cannot attach an equipped item.", (255, 100, 0))
        elif is_attached:
            if gs:
                gs.add_message("Item is already attached.", (255, 100, 0))
        else:
            # Find an equipped host to attach to
            host_item_id = None
            for _, data in self._inventory_ui_map.items():
                if data[0] is not None and data[1]:  # data[1] == is_equipped
                    host_item_id = data[0]
                    break
            if host_item_id is None:
                if gs:
                    gs.add_message("No equipped item to attach to.", (255, 100, 0))
            else:
                action = {
                    "type": "attach",
                    "item_to_attach_id": item_id,
                    "target_host_item_id": host_item_id,
                }

    elif action_type == "detach":
        if is_attached:
            action = {"type": "detach", "item_to_detach_id": item_id}
        else:
            if gs:
                gs.add_message("Item is not attached.", (255, 100, 0))

    return action
                host_item_id = None
                for _, data in self._inventory_ui_map.items():
                    if data[0] is not None and data[1]:  # data[1] == is_equipped
                        host_item_id = data[0]
                        break
                if host_item_id is None:
                    if gs:
                        gs.add_message("No equipped item to attach to.", (255, 100, 0))
                else:
                    action = {
                        "type": "attach",
                        "item_to_attach_id": item_id,
                        "target_host_item_id": host_item_id,
                    }

        elif action_type == "detach":
            if is_attached:
                action = {"type": "detach", "item_to_detach_id": item_id}
            else:
                if gs:
                    gs.add_message("Item is not attached.", (255, 100, 0))

        return action

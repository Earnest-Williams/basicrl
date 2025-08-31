# engine/window_manager_modules/input_handler.py
"""
Handles processing of raw keyboard inputs, mapping them to game actions
or UI commands based on keybindings and game state.
"""
# Standard Imports
from typing import TYPE_CHECKING, Any
from typing import Dict as PyDict
from typing import List

# Third-party Imports
import structlog

# PySide6 Imports
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

# --- Type Checking Imports ---
if TYPE_CHECKING:
    # Use absolute paths relative to project root (basicrl)
    from engine.main_loop import MainLoop
    from engine.window_manager import (
        WindowManager,
    )  # Adjust if WindowManager moves later
    from game.game_state import GameState

log = structlog.get_logger(__name__)


class InputHandler:
    """
    Processes keyboard events and translates them into game actions or UI calls.
    """

    def __init__(
        self,
        keybindings_config: PyDict[str, Any],
        window_manager_ref: "WindowManager",  # WindowManager stays in engine/
    ):
        self.keybindings_config: PyDict[str, Any] = keybindings_config
        self.window_manager_ref: "WindowManager" = window_manager_ref
        log.debug("InputHandler initialized.")

    # --- Key Parsing Methods ---
    def _get_qt_key_enum(self, key_str: str | None) -> Qt.Key | None:
        """Converts a key string (e.g., "A", "F1", "KP_1") to a Qt.Key enum value."""
        if not key_str:
            return None
        qt_key = getattr(Qt.Key, f"Key_{key_str}", None)
        if qt_key:
            return qt_key
        if key_str.startswith("KP_"):
            kp_suffix = key_str[3:]
            qt_key_kp_full = getattr(Qt.Key, f"Key_{key_str}", None)
            if qt_key_kp_full:
                return qt_key_kp_full
            if kp_suffix.isdigit():
                qt_key = getattr(Qt.Key, f"Key_{kp_suffix}", None)
                if qt_key:
                    return qt_key
            qt_key_suffix_only = getattr(Qt.Key, f"Key_{kp_suffix}", None)
            if qt_key_suffix_only:
                return qt_key_suffix_only
        if len(key_str) == 1:
            qt_key_upper = getattr(Qt.Key, f"Key_{key_str.upper()}", None)
            if qt_key_upper:
                return qt_key_upper
        key_str_lower = key_str.lower()
        common_map = {
            "up": Qt.Key.Key_Up,
            "down": Qt.Key.Key_Down,
            "left": Qt.Key.Key_Left,
            "right": Qt.Key.Key_Right,
            "home": Qt.Key.Key_Home,
            "end": Qt.Key.Key_End,
            "pageup": Qt.Key.Key_PageUp,
            "pagedown": Qt.Key.Key_PageDown,
            "space": Qt.Key.Key_Space,
            "period": Qt.Key.Key_Period,
            "return": Qt.Key.Key_Return,
            "enter": Qt.Key.Key_Enter,
            "escape": Qt.Key.Key_Escape,
            "tab": Qt.Key.Key_Tab,
            "backspace": Qt.Key.Key_Backspace,
            "delete": Qt.Key.Key_Delete,
        }
        if key_str_lower in common_map:
            return common_map[key_str_lower]
        log.warning("Could not map key string to Qt.Key", key_str=key_str)
        return None

    def _get_qt_modifier_enum(self, mods_list: List[str]) -> Qt.KeyboardModifier:
        """Converts a list of modifier strings (e.g., ["Ctrl", "Shift"]) to Qt.KeyboardModifiers."""
        modifier = Qt.KeyboardModifier.NoModifier
        if not mods_list:
            return modifier
        for mod_str in mods_list:
            mod_lower = mod_str.lower()
            if mod_lower == "ctrl" or mod_lower == "control":
                modifier |= Qt.KeyboardModifier.ControlModifier
            elif mod_lower == "shift":
                modifier |= Qt.KeyboardModifier.ShiftModifier
            elif mod_lower == "alt":
                modifier |= Qt.KeyboardModifier.AltModifier
        return modifier

    def _get_action_for_key(
        self,
        key_code: int,
        modifiers: Qt.KeyboardModifier,
        active_keybinding_sets: List[str],
    ) -> PyDict[str, Any] | None:
        """Finds the action dictionary corresponding to a key press and active binding sets."""
        bindings = self.keybindings_config.get("bindings", {})
        for set_name in active_keybinding_sets:
            binding_set = bindings.get(set_name)
            if not binding_set or not isinstance(binding_set, dict):
                continue
            for action_name, binding_data in binding_set.items():
                if not isinstance(binding_data, dict):
                    continue
                bound_key_str: str | None = binding_data.get("key")
                bound_key_enum: Qt.Key | None = self._get_qt_key_enum(bound_key_str)
                bound_mods_list: List[str] = binding_data.get("mods", [])
                required_mod_enum: Qt.KeyboardModifier = self._get_qt_modifier_enum(
                    bound_mods_list
                )
                if key_code == bound_key_enum and modifiers == required_mod_enum:
                    action_type: str | None = binding_data.get("action_type")
                    if action_type == "move":
                        return {
                            "type": "move",
                            "dx": binding_data.get("dx", 0),
                            "dy": binding_data.get("dy", 0),
                        }
                    elif action_type == "action":
                        # Use the action_name from the keybinding as the 'type'
                        return {"type": action_name}
                    elif action_type == "ui":
                        return {"type": "ui", "ui_action": action_name}
                    else:
                        log.warning(
                            "Unknown action_type in keybinding",
                            action_name=action_name,
                            type=action_type,
                        )
                        return None
        return None

    # --- End Key Parsing Methods ---

    def process_key_event(
        self,
        event: QKeyEvent,
        game_state: "GameState",
        main_loop_ref: "MainLoop",
        active_keybinding_sets: List[str],
    ) -> bool:
        """
        Processes a QKeyEvent.
        Returns True if the key was handled, False otherwise.
        Calls methods on window_manager_ref for UI interactions.
        """
        key = event.key()
        mods = event.modifiers()
        log.debug(
            "Processing key event",
            key_text=event.text(),
            key_enum_val=key,
            mods_val=mods,
            ui_state=game_state.ui_state,
        )

        action_to_dispatch: PyDict[str, Any] | None = None
        key_handled = False

        # Universal Escape handling
        if key == Qt.Key.Key_Escape:
            if game_state.ui_state != "PLAYER_TURN":
                log.info(
                    "Escape pressed, returning to player turn.",
                    from_state=game_state.ui_state,
                )
                self.window_manager_ref.ui_return_to_player_turn()
                key_handled = True
            else:
                quit_action = self._get_action_for_key(
                    key, mods, ["common"] # Only check common for quit on Esc
                )
                if quit_action and quit_action.get("ui_action") == "quit_game_alt":
                    log.info("Quit key (Escape) pressed in PLAYER_TURN.")
                    self.window_manager_ref.ui_quit_game()
                    key_handled = True
            if key_handled:
                return True

        # Universal Help handling
        help_action_lookup = self._get_action_for_key(key, mods, active_keybinding_sets)
        if help_action_lookup and help_action_lookup.get("ui_action") == "show_help":
            self.window_manager_ref.ui_show_help_dialog()
            return True

        # State-dependent input processing
        match game_state.ui_state:
            case "PLAYER_TURN":
                looked_up_action = self._get_action_for_key(
                    key, mods, active_keybinding_sets
                )
                if looked_up_action:
                    action_name = looked_up_action.get("type") # e.g., 'move', 'wait', 'pickup', 'ui'
                    ui_action_name = looked_up_action.get("ui_action") # Only present if type is 'ui'

                    # Explicitly handle known action types
                    if action_name == "move":
                        action_to_dispatch = looked_up_action
                        key_handled = True
                    elif action_name in ["wait", "pickup", "wait_alt", "wait_alt2"]:
                        # *** MODIFIED: Simplify action assignment ***
                        canonical_action_name = action_name
                        if action_name.startswith("wait"):
                            canonical_action_name = "wait"

                        if canonical_action_name == action_name:
                            # No change needed, use the original dict
                            action_to_dispatch = looked_up_action
                        else:
                            # Create a simple dict with the canonical name
                            action_to_dispatch = {"type": canonical_action_name}
                        # *** END MODIFICATION ***
                        key_handled = True
                    elif action_name == "ui":
                        # Handle UI-specific actions triggered from player turn
                        if ui_action_name == "inventory":
                            self.window_manager_ref.ui_open_inventory_view()
                            key_handled = True
                        elif ui_action_name == "toggle_height_vis":
                            self.window_manager_ref.ui_toggle_height_visualization()
                            key_handled = True
                        # Add other UI actions...
                        else:
                            log.warning(
                                "Unhandled UI action from PLAYER_TURN",
                                ui_action=ui_action_name,
                            )
                            key_handled = False
                    else:
                        # Log unknown action *names* here
                        log.warning(
                            "Unknown action name received in PLAYER_TURN",
                            action_name=action_name,
                            action_data=looked_up_action,
                        )
                        key_handled = False

            case "INVENTORY_VIEW":
                nav_map = {
                    Qt.Key.Key_Down: "down",
                    Qt.Key.Key_J: "down",
                    Qt.Key.Key_S: "down",
                    Qt.Key.Key_Up: "up",
                    Qt.Key.Key_K: "up",
                    Qt.Key.Key_W: "up",
                }
                action_key_map = {
                    Qt.Key.Key_E: "equip_unequip",
                    Qt.Key.Key_U: "use",
                    Qt.Key.Key_D: "drop",
                    # TODO: Add Attach/Detach keys here
                }
                if key in nav_map:
                    self.window_manager_ref.ui_overlay_manager.navigate(nav_map[key])
                    key_handled = True
                elif key in action_key_map:
                    action_to_dispatch = (
                        self.window_manager_ref.ui_overlay_manager.get_action_for_key(
                            action_key_map[key]
                        )
                    )
                    key_handled = True
                else:
                    key_handled = False

            case "TARGETING":
                log.debug("Key in TARGETING state - NI")
                key_handled = False
            case _:
                log.warning(
                    "Key pressed in unknown UI state", state=game_state.ui_state
                )
                key_handled = False

        # Dispatch game action if generated
        if action_to_dispatch:
            log.debug("Dispatching action to main loop", action=action_to_dispatch)
            turn_taken = main_loop_ref.handle_action(action_to_dispatch)
            if game_state.ui_state == "INVENTORY_VIEW":
                self.window_manager_ref.update_frame()

        elif key_handled and game_state.ui_state != "PLAYER_TURN":
            self.window_manager_ref.update_frame()

        return key_handled

# tile_mapper_utils.py

from PySide6.QtCore import Qt


# --- Helper to parse modifier strings ---
def parse_modifier(modifier_str: str | None) -> Qt.KeyboardModifier:
    """Parses a modifier string (e.g., 'Ctrl+Shift') into Qt.KeyboardModifiers."""
    modifier = Qt.KeyboardModifier.NoModifier
    if not modifier_str or modifier_str.lower() == "none":
        return modifier
    # Normalize and split
    parts = [
        part.strip().lower()
        for part in modifier_str.replace("control", "ctrl").split("+")
    ]
    if "ctrl" in parts:
        modifier |= Qt.KeyboardModifier.ControlModifier
    if "shift" in parts:
        modifier |= Qt.KeyboardModifier.ShiftModifier
    if "alt" in parts:
        modifier |= Qt.KeyboardModifier.AltModifier
    # Add Meta/Command key if needed in the future
    # if "meta" in parts or "cmd" in parts or "command" in parts:
    #     modifier |= Qt.KeyboardModifier.MetaModifier
    return modifier


# --- Helper to get Qt.Key from string ---
def parse_key(key_str: str | None) -> Qt.Key | None:
    """Parses a key string (e.g., 'F1', '1', 'A', 'Enter') into Qt.Key enum."""
    if not key_str:
        return None

    key_str_upper = key_str.upper()
    # Handle function keys (F1-F35)
    if key_str_upper.startswith("F") and key_str_upper[1:].isdigit():
        f_num = int(key_str_upper[1:])
        if 1 <= f_num <= 35:
            return getattr(Qt.Key, f"Key_F{f_num}", None)

    # Handle single digits (0-9)
    if key_str.isdigit() and len(key_str) == 1:
        return getattr(Qt.Key, f"Key_{key_str}", None)

    # Handle single letters (A-Z)
    if len(key_str) == 1 and key_str.isalpha():
        return getattr(Qt.Key, f"Key_{key_str_upper}", None)

    # Handle common key names (case-insensitive)
    key_map = {
        "esc": Qt.Key.Key_Escape,
        "escape": Qt.Key.Key_Escape,
        "tab": Qt.Key.Key_Tab,
        "enter": Qt.Key.Key_Enter,
        "return": Qt.Key.Key_Return,
        "space": Qt.Key.Key_Space,
        "spacebar": Qt.Key.Key_Space,
        "backspace": Qt.Key.Key_Backspace,
        "delete": Qt.Key.Key_Delete,
        "del": Qt.Key.Key_Delete,
        "insert": Qt.Key.Key_Insert,
        "ins": Qt.Key.Key_Insert,
        "home": Qt.Key.Key_Home,
        "end": Qt.Key.Key_End,
        "pageup": Qt.Key.Key_PageUp,
        "pgup": Qt.Key.Key_PageUp,
        "pagedown": Qt.Key.Key_PageDown,
        "pgdn": Qt.Key.Key_PageDown,
        "up": Qt.Key.Key_Up,
        "uparrow": Qt.Key.Key_Up,
        "down": Qt.Key.Key_Down,
        "downarrow": Qt.Key.Key_Down,
        "left": Qt.Key.Key_Left,
        "leftarrow": Qt.Key.Key_Left,
        "right": Qt.Key.Key_Right,
        "rightarrow": Qt.Key.Key_Right,
        # Add more mappings as needed (e.g., +, -, =, etc.)
        "+": Qt.Key.Key_Plus,
        "plus": Qt.Key.Key_Plus,
        "-": Qt.Key.Key_Minus,
        "minus": Qt.Key.Key_Minus,
        "=": Qt.Key.Key_Equal,
        "equals": Qt.Key.Key_Equal,
    }
    return key_map.get(key_str.lower())

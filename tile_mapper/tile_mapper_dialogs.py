# tile_mapper_dialogs.py

import re

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

# Import constants/functions from config if needed by dialogs (e.g., for default tile checks)
# Example: from .tile_mapper_config import CONFIG_FILE (adjust path as needed)
from tile_mapper_config import CONFIG_FILE  # Assuming flat structure for now


# --- Edit Tile Dialog ---
class EditTileDialog(QDialog):
    """Dialog for adding or editing a tile type."""

    def __init__(
        self,
        tile_char=None,
        tile_data=None,
        existing_chars=None,
        parent=None,
        app_config=None,
    ):
        """Requires app_config dict to check current default tile."""
        super().__init__(parent)
        self.setWindowTitle("Edit Tile" if tile_char else "Add New Tile")
        self.existing_chars = existing_chars if existing_chars is not None else set()
        self.original_char = tile_char
        self.is_editing = tile_char is not None
        self.app_config = app_config if app_config else {}  # Store config reference

        # Widgets
        self.char_edit = QLineEdit(tile_char if tile_char else "")
        self.char_edit.setMaxLength(1)
        self.desc_edit = QLineEdit(
            tile_data.get("description", "") if tile_data else ""
        )
        self.color_button = QPushButton()
        self.color_button.setFixedSize(40, 25)
        self.color_button.setFlat(False)

        initial_color = QColor(200, 200, 200)  # Default gray
        if tile_data:
            color_qt = tile_data.get("color_qt")
            if isinstance(color_qt, QColor):
                initial_color = color_qt
            elif (
                isinstance(tile_data.get("color"), list)
                and len(tile_data["color"]) == 3
            ):
                try:
                    initial_color = QColor(*tile_data["color"])
                except (TypeError, ValueError):
                    pass

        self.set_button_color(initial_color)
        self.color_button.clicked.connect(self.pick_color)

        # Layout
        form_layout = QFormLayout()
        form_layout.addRow("Character:", self.char_edit)
        form_layout.addRow("Description:", self.desc_edit)
        form_layout.addRow("Color:", self.color_button)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        self.setMinimumWidth(300)

    def set_button_color(self, color):
        """Sets the background color and stores the QColor."""
        self.current_color = color
        self.color_button.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid black;"
        )

    def pick_color(self):
        """Opens a color dialog to choose a new color."""
        new_color = QColorDialog.getColor(self.current_color, self, "Select Tile Color")
        if new_color.isValid():
            self.set_button_color(new_color)

    def validate_and_accept(self):
        """Validates input before accepting the dialog."""
        char = self.char_edit.text()
        if not char:
            QMessageBox.warning(
                self, "Validation Error", "Tile character cannot be empty."
            )
            return
        if len(char) > 1:
            QMessageBox.warning(
                self, "Validation Error", "Tile character must be a single character."
            )
            return
        if char != self.original_char and char in self.existing_chars:
            QMessageBox.warning(
                self, "Validation Error", f"Character '{char}' already used."
            )
            return

        # Use stored app_config reference
        current_default = self.app_config.get("default_tile", ".")
        if (
            self.is_editing
            and self.original_char == current_default
            and char != current_default
        ):
            reply = QMessageBox.warning(
                self,
                "Confirm Default Tile Change",
                f"Changing character ('{self.original_char}') designated as 'default_tile'.\n\n"
                f"Update 'default_tile' in '{CONFIG_FILE}' manually if needed.\n\n"
                f"Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.accept()

    def get_tile_data(self):
        """Returns the entered tile data."""
        return {
            "char": self.char_edit.text(),
            "data": {
                "description": self.desc_edit.text(),
                "color_qt": self.current_color,
                "color": [
                    self.current_color.red(),
                    self.current_color.green(),
                    self.current_color.blue(),
                ],
            },
        }


# --- Map Selection Dialog ---
class MapSelectionDialog(QDialog):
    """Dialog for selecting maps from a multi-map file."""

    def __init__(self, map_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Maps to Load")
        self.map_keys = map_keys
        self.selected_keys = []
        self.tiling_mode = "Single"
        self.grid_rows = 1
        self.grid_cols = 1

        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.addItems(self.map_keys)
        layout.addWidget(QLabel("Available Maps:"))
        layout.addWidget(self.list_widget)

        self.selection_input = QLineEdit()
        self.selection_input.setPlaceholderText("Or enter e.g., 1, 3-5, lobby, all")
        layout.addWidget(QLabel("Selection String:"))
        layout.addWidget(self.selection_input)

        # Tiling Options
        tiling_layout = QHBoxLayout()
        tiling_layout.addWidget(QLabel("Tiling:"))
        self.tiling_combo = QComboBox()
        self.tiling_combo.addItems(["Single", "Vertical", "Horizontal", "Grid"])
        self.tiling_combo.currentTextChanged.connect(self.update_grid_options)
        tiling_layout.addWidget(self.tiling_combo)
        self.rows_label = QLabel("Rows:")
        self.rows_spin = QSpinBox()
        self.rows_spin.setMinimum(1)
        self.rows_spin.setValue(2)
        self.cols_label = QLabel("Cols:")
        self.cols_spin = QSpinBox()
        self.cols_spin.setMinimum(1)
        self.cols_spin.setValue(2)
        tiling_layout.addWidget(self.rows_label)
        tiling_layout.addWidget(self.rows_spin)
        tiling_layout.addWidget(self.cols_label)
        tiling_layout.addWidget(self.cols_spin)
        tiling_layout.addStretch()
        layout.addLayout(tiling_layout)
        self.update_grid_options("Single")  # Initial state

        # Dialog Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.process_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setMinimumWidth(400)

    def update_grid_options(self, tiling_mode):
        is_grid = tiling_mode == "Grid"
        self.rows_label.setVisible(is_grid)
        self.rows_spin.setVisible(is_grid)
        self.cols_label.setVisible(is_grid)
        self.cols_spin.setVisible(is_grid)

    def parse_selection_string(self, selection_str):
        """Parses a selection string like '1, 3-5, lobby, all'."""
        selected = set()
        parts = selection_str.split(",")
        all_keys_set = set(self.map_keys)

        if "all" in [p.strip().lower() for p in parts]:
            return self.map_keys

        for part in parts:
            part = part.strip()
            if not part:
                continue

            range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
            if range_match:
                try:
                    start, end = int(range_match.group(1)), int(range_match.group(2))
                    if start <= end:
                        for key in self.map_keys:
                            try:
                                if start <= int(key) <= end:
                                    selected.add(key)
                            except ValueError:
                                continue
                    else:
                        print(f"Warning: Invalid range '{part}'.")
                except ValueError:
                    print(f"Warning: Could not parse range '{part}'.")
            elif part in all_keys_set:
                selected.add(part)
            else:
                print(f"Warning: Unknown map key or invalid format '{part}'.")

        return sorted(
            list(selected),
            key=lambda k: (
                self.map_keys.index(k) if k in self.map_keys else float("inf")
            ),
        )

    def process_selection(self):
        """Processes list selection and input string, validates."""
        input_str = self.selection_input.text().strip()
        list_selected_items = self.list_widget.selectedItems()

        self.selected_keys = []
        if input_str:
            self.selected_keys = self.parse_selection_string(input_str)
        elif list_selected_items:
            self.selected_keys = [item.text() for item in list_selected_items]

        if not self.selected_keys:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select maps or enter a valid selection string.",
            )
            return

        self.tiling_mode = self.tiling_combo.currentText()
        num_selected = len(self.selected_keys)

        if num_selected <= 1 and self.tiling_mode != "Single":
            print("Info: Tiling mode reset to 'Single' for single map selection.")
            self.tiling_mode = "Single"
            self.tiling_combo.setCurrentText("Single")
        elif num_selected > 1 and self.tiling_mode == "Single":
            QMessageBox.warning(
                self,
                "Invalid Tiling",
                "Please choose Vertical, Horizontal, or Grid for multiple maps.",
            )
            return

        self.grid_rows = self.rows_spin.value() if self.tiling_mode == "Grid" else 1
        self.grid_cols = self.cols_spin.value() if self.tiling_mode == "Grid" else 1

        if (
            self.tiling_mode == "Grid"
            and self.grid_rows * self.grid_cols < num_selected
        ):
            QMessageBox.warning(
                self,
                "Grid Too Small",
                f"Grid ({self.grid_rows}x{self.grid_cols}) too small for {num_selected} maps.",
            )
            return

        self.accept()

    def get_selection(self):
        """Returns the selected map keys and tiling options."""
        return self.selected_keys, self.tiling_mode, self.grid_rows, self.grid_cols

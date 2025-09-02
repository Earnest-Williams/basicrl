# gui_widgets.py
#!/usr/bin/env python3

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont  # Keep QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListWidget,
    QScrollArea,  # Added QScrollArea
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# --- ASCII Grid View ---
class AsciiGridView(QWidget):
    """Widget to display the game world grid using ASCII characters and HTML colors."""

    def __init__(self, grid_size: int, parent=None):
        super().__init__(parent)
        self.grid_size = grid_size
        self.entities = {}  # Store entities for rendering {id: {'x', 'y', 'kind', 'health'}}
        self.agent_health = 0  # Store separately if needed for display elsewhere

        # Character and Color Mapping
        self.char_map = {
            "agent": "@",
            "enemy": "E",
            "food": "*",
            "empty": ".",  # Character for empty cells
        }
        self.color_map = {
            "agent": "blue",
            "enemy": "red",
            "food": "lime",  # Brighter green
            "empty": "#555555",  # Dark gray for empty cells
            "default": "gray",  # Fallback color
        }

        # Main layout for this widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Use full space

        # Use QLabel to render HTML content
        self.display_label = QLabel()
        self.display_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # Important for preformatted text
        self.display_label.setWordWrap(False)

        # Set Monospace Font
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        font.setPointSize(10)  # Adjust size as needed
        self.display_label.setFont(font)

        # Ensure label size adjusts (might need scroll area if large)
        self.display_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Add ScrollArea in case grid is very large
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.display_label)
        layout.addWidget(scroll_area)

        self.setLayout(layout)
        self.setMinimumSize(300, 300)  # Adjust minimum size if needed

        # Initial empty render
        self._render_grid()

    def _render_grid(self):
        """Renders the current entity state into an HTML string using efficient joining."""
        # Profile point: Entire grid rendering process
        if self.grid_size <= 0:
            self.display_label.setText("<pre>Grid not initialized</pre>")
            return

        # Create a 2D list representing the characters/colors (creation is
        # usually fast)
        char_grid = [
            [(self.char_map["empty"], self.color_map["empty"])
             for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]

        # Place entities onto the grid (iteration depends on entity count)
        for entity_data in self.entities.values():  # Iterate directly over values
            x, y = entity_data["x"], entity_data["y"]
            kind = entity_data["kind"]
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:  # Bounds check
                char = self.char_map.get(kind, "?")
                color = self.color_map.get(kind, self.color_map["default"])
                char_grid[y][x] = (
                    char,
                    color,
                )  # Grid is typically [row][col] -> [y][x]

        # --- OPTIMIZATION: Efficient String Building ---
        # Build the HTML string using lists and join(), which is generally faster
        # than repeated string concatenation with '+='.
        html_lines = []
        for r in range(self.grid_size):
            line_parts = []  # Create a list to hold parts of the current line
            for c in range(self.grid_size):
                char, color = char_grid[r][c]
                # Append the formatted HTML segment for this cell to the list
                line_parts.append(f"<font color='{color}'>{char}</font>")
            # Join all parts of the line with a space separator
            html_lines.append(" ".join(line_parts))

        # Join all lines with HTML line breaks '<br>' and wrap in <pre> tags
        full_html = f"<pre>{'<br>'.join(html_lines)}</pre>"
        # --- END OPTIMIZATION ---

        # Setting QLabel text (can have some overhead depending on
        # complexity/size)
        self.display_label.setText(full_html)

    @Slot(list, list, int, int, int)  # Matching worker signal
    def update_world(self, grid_list_ignored, entities_list, agent_health, turn, grid_size):
        """Updates the view with new world state."""
        # Update internal state (fast)
        if grid_size != self.grid_size:
            self.grid_size = grid_size
            # Might need further adjustments if size changes

        self.entities = {e["id"]: e for e in entities_list}
        self.agent_health = agent_health

        # Trigger re-render (cost depends on _render_grid)
        self._render_grid()  # Profile this call indirectly by profiling _render_grid


# --- Action View Widget ---
# (No changes needed here, assumed efficient enough for now)
class ActionViewWidget(QWidget):
    """Displays the list of actions and their weights."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Action Name", "Weight"])
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)  # Read-only
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)  # Nice touch

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Learned Action Weights:</b>"))
        layout.addWidget(self.table)
        self.setLayout(layout)

    @Slot(dict)  # Matching worker signal {name: weight}
    def update_actions(self, weights: dict):
        """Updates the table with current action weights."""
        # QTableWidget updates are generally efficient for moderate numbers of
        # rows
        self.table.setRowCount(len(weights))
        row = 0
        # Sort by name for consistent display
        for name, weight in sorted(weights.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{weight:.3f}"))
            row += 1
        # self.table.resizeColumnsToContents() # Adjust columns if needed


# --- Planning View Widget ---
# (No changes needed here, assumed efficient enough for now)
class PlanningViewWidget(QWidget):
    """Displays the current goal and action plan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.goal_label = QLabel("Goal: None")
        self.plan_list = QListWidget()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Current Plan:</b>"))
        layout.addWidget(self.goal_label)
        layout.addWidget(self.plan_list)
        self.setLayout(layout)

    @Slot(str)  # Matching worker signal
    def update_goal(self, goal_str: str):
        """Updates the displayed goal."""
        # QLabel update is fast
        self.goal_label.setText(goal_str)

    @Slot(list)  # Matching worker signal
    def update_plan(self, plan_list: list):
        """Updates the displayed action plan."""
        # QListWidget updates are generally efficient for moderate numbers of
        # items
        self.plan_list.clear()
        if plan_list:
            self.plan_list.addItems(plan_list)
        else:
            self.plan_list.addItem("---")  # Indicate empty plan

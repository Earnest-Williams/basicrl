# main_window.py
import logging  # Import logging library

# Import QTimer if needed later, not now
from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import QDockWidget, QLabel, QMainWindow, QPushButton, QSpinBox, QStatusBar

from ..simulation import GRID_SIZE, AgentAI, World  # Import simulation components
from .gui_widgets import (
    ActionViewWidget,  # Import AsciiGridView instead
    AsciiGridView,
    PlanningViewWidget,
)
from .worker import SimulationWorker

# --- Logging Setup ---
# Configure logging to show debug messages, timestamps, and level names
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)
# Optionally, reduce logging level for libraries if too verbose
# logging.getLogger("PySide6").setLevel(logging.WARNING)
# --- End Logging Setup ---


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        logging.info("Initializing MainWindow...")
        self.setWindowTitle("GOAP Simulation Viewer")
        self.setGeometry(100, 100, 1200, 700)  # x, y, width, height

        # --- Simulation Objects ---
        # These hold the state, worker modifies them
        self.world = World(size=GRID_SIZE)
        self.agent_ai = AgentAI(self.world)

        # --- Simulation Thread Management ---
        self.simulation_thread: QThread | None = None
        self.worker: SimulationWorker | None = None

        # --- UI Widgets ---
        logging.debug("Creating UI Widgets...")
        # Instantiate the new ASCII view
        self.game_view = AsciiGridView(self.world.size)
        self.action_view = ActionViewWidget()
        self.planning_view = PlanningViewWidget()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- Layout ---
        logging.debug("Setting up layout and docks...")
        self.setCentralWidget(self.game_view)

        action_dock = QDockWidget("Actions", self)
        action_dock.setWidget(self.action_view)
        action_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, action_dock)

        planning_dock = QDockWidget("Planning", self)
        planning_dock.setWidget(self.planning_view)
        planning_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, planning_dock)

        # --- Toolbar / Controls ---
        logging.debug("Creating toolbar controls...")
        toolbar = self.addToolBar("Controls")
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.step_button = QPushButton("Step")
        self.stop_button = QPushButton("Stop")
        speed_label = QLabel(" Speed (ms/step):")
        self.speed_slider = QSpinBox()
        # Enforce a reasonable minimum speed to prevent potential GUI overload
        # issues
        self.speed_slider.setRange(50, 2000)  # Min 50ms, Max 2000ms
        self.speed_slider.setValue(200)  # Default speed
        self.speed_slider.setSingleStep(50)

        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.pause_button)
        toolbar.addWidget(self.step_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addSeparator()
        toolbar.addWidget(speed_label)
        toolbar.addWidget(self.speed_slider)

        # Initial button states
        self.pause_button.setEnabled(False)
        self.step_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        # --- Connect Signals and Slots for Controls ---
        logging.debug("Connecting control signals...")
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.step_button.clicked.connect(self.step_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.speed_slider.valueChanged.connect(self.update_speed)

        # --- Initial State Display ---
        # Display initial weights before simulation starts
        self.action_view.update_actions(dict(self.agent_ai.planner.action_weights))
        logging.info("MainWindow initialization complete.")

    @Slot()
    def start_simulation(self):
        """Creates worker and thread, starts the simulation."""
        logging.info("Start simulation requested.")
        if self.simulation_thread and self.simulation_thread.isRunning():
            logging.warning("Simulation already running, ignoring request.")
            return

        logging.debug("Resetting world state for new run.")
        self.world.reset()
        # Re-create AI or ensure its state is clean if needed. Re-using is fine for learning.
        # self.agent_ai = AgentAI(self.world) # If needed

        logging.debug("Creating SimulationWorker and QThread...")
        self.worker = SimulationWorker(self.world, self.agent_ai)
        # Parent thread to main window context
        self.simulation_thread = QThread(self)
        self.worker.moveToThread(self.simulation_thread)
        logging.debug(
            f"Worker {self.worker} moved to thread {
                self.simulation_thread}"
        )

        # --- Connect Worker Signals to GUI Slots ---
        logging.debug("Connecting worker signals to GUI slots...")
        self.worker.world_updated.connect(self.game_view.update_world)
        self.worker.actions_updated.connect(self.action_view.update_actions)
        self.worker.goal_updated.connect(self.planning_view.update_goal)
        self.worker.plan_updated.connect(self.planning_view.update_plan)
        self.worker.status_update.connect(self.status_bar.showMessage)
        # Connect simulation_finished to the main window's handler slot first
        self.worker.simulation_finished.connect(self.on_simulation_finished)
        # self.worker.simulation_step_done.connect(lambda turn:
        # logging.debug(f"Worker completed turn {turn}")) # Optional debug log

        # --- Connect Thread Signals and **REVISED** Cleanup ---
        logging.debug("Connecting thread start/finish signals and cleanup...")
        self.simulation_thread.started.connect(self.worker.run_simulation)

        # 1. When worker signals it's finished, tell the thread to quit its event loop.
        # Use QueuedConnection to ensure signal processed by thread's event
        # loop.
        self.worker.simulation_finished.connect(
            self.simulation_thread.quit, Qt.ConnectionType.QueuedConnection
        )
        logging.debug("Connected worker.simulation_finished -> simulation_thread.quit")

        # 2. When the thread's event loop has *actually* finished, THEN
        # schedule deletion.
        self.simulation_thread.finished.connect(
            self.worker.deleteLater
        )  # Delete worker AFTER thread finishes
        self.simulation_thread.finished.connect(
            self.simulation_thread.deleteLater
        )  # Delete thread AFTER thread finishes
        logging.debug("Connected simulation_thread.finished -> worker.deleteLater")
        logging.debug("Connected simulation_thread.finished -> simulation_thread.deleteLater")
        # --- End Revised Cleanup ---

        logging.info("Starting simulation thread...")
        self.simulation_thread.start()

        # Update UI button states
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self.step_button.setEnabled(False)  # Enabled only when paused
        self.stop_button.setEnabled(True)
        self.update_speed()  # Ensure worker gets initial speed value

    @Slot()
    def pause_simulation(self):
        """Sends pause/resume signal to worker."""
        logging.debug("Pause/Resume button clicked.")
        if self.worker:
            # Determine target state based on current button text
            is_currently_paused = self.pause_button.text() == "Resume"
            target_pause_state = not is_currently_paused
            self.worker.pause_simulation()  # Worker toggles its internal state
            # Update button based on the NEW state
            self.pause_button.setText("Resume" if target_pause_state else "Pause")
            self.step_button.setEnabled(target_pause_state)
            logging.info(
                f"Simulation {
                    'paused' if target_pause_state else 'resumed'}."
            )
        else:
            logging.warning("Pause clicked but worker does not exist.")

    @Slot()
    def step_simulation(self):
        """Sends step signal to worker."""
        logging.debug("Step button clicked.")
        if self.worker:
            self.worker.step_simulation()
            # Keep Step button enabled after stepping, as worker should
            # re-pause
            self.step_button.setEnabled(True)
        else:
            logging.warning("Step clicked but worker does not exist.")

    @Slot()
    def stop_simulation(self):
        """Sends stop signal to worker."""
        logging.info("Stop simulation requested.")
        if self.worker:
            self.worker.stop_simulation()
            # UI state will be updated by on_simulation_finished when worker
            # confirms stop
        else:
            logging.warning("Stop clicked but worker/thread does not exist.")
            # Ensure buttons are in a safe state if somehow stop is clicked
            # without worker
            self._reset_button_states()

    @Slot(str)
    def on_simulation_finished(self, reason: str):
        """Cleans up and resets UI state when simulation ends or is stopped."""
        logging.info(f"Simulation finished signal received. Reason: {reason}")
        self.status_bar.showMessage(
            f"Simulation Finished: {
                reason}",
            5000,
        )  # Show for 5 secs

        # Reset button states
        self._reset_button_states()

        # Clear references to allow garbage collection (deleteLater handles actual deletion)
        # Important: Do not manually delete worker/thread here, rely on
        # deleteLater
        self.worker = None
        self.simulation_thread = None
        logging.debug("Worker and Thread references cleared in main window.")

    def _reset_button_states(self):
        """Helper to reset control buttons to idle state."""
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.step_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        logging.debug("Control buttons reset to idle state.")

    @Slot(int)
    def update_speed(self, value: int = -1):
        """Sends new speed value (ms delay) to worker."""
        if value == -1:  # Handle initial call from start_simulation
            value = self.speed_slider.value()
        logging.debug(f"Updating simulation speed delay to {value} ms.")
        if self.worker:
            self.worker.set_step_delay(value)
        # No need to log warning if worker doesn't exist yet, speed will be set
        # on start

    def closeEvent(self, event):
        """Ensure thread is stopped cleanly on window close."""
        logging.info("Close event received. Cleaning up simulation...")
        if self.simulation_thread and self.simulation_thread.isRunning():
            logging.debug("Simulation thread is running, attempting graceful stop.")
            # Request stop
            self.stop_simulation()
            # Wait a short time for the thread to finish.
            # This WILL block the GUI momentarily, but increases chance of clean exit.
            # Adjust timeout as needed.
            if not self.simulation_thread.wait(1500):  # Wait max 1.5 seconds
                logging.warning("Simulation thread did not finish cleanly after 1.5s.")
            else:
                logging.debug("Simulation thread finished.")
        else:
            logging.debug("No active simulation thread to stop.")
        event.accept()  # Accept the close event
        logging.info("Exiting application.")

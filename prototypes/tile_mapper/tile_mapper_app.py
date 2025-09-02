# tile_mapper_app.py

import sys
import traceback

import structlog
from PySide6.QtWidgets import QApplication, QMessageBox

# Import the main window class
# Assuming flat structure, adjust path if needed (e.g., from .tile_mapper_gui import ...)
from tile_mapper_gui import MainWindow

log = structlog.get_logger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    # Optional High DPI settings (consider platform specifics)
    # ... (Add DPI code here if needed, before QApplication) ...

    app = QApplication(sys.argv)

    # --- Catch startup errors ---
    main_window = None
    try:
        log.info("Initializing MainWindow")
        main_window = MainWindow()  # Config is loaded inside MainWindow.__init__
        log.info("Showing MainWindow")
        main_window.show()
        log.info("Starting application event loop")
        exit_code = app.exec()
        log.info("Application finished", exit_code=exit_code)
        sys.exit(exit_code)

    except Exception:
        log.exception("Application initialization failed")

        # Attempt to show error message box
        try:
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.setWindowTitle("Application Startup Error")
            error_msg.setText(
                f"Failed to start Tile Mapper:\n\n{
                    e}\n\nSee console/stderr for details."
            )
            # Limit detail length for QMessageBox
            tb_text = traceback.format_exc()
            if len(tb_text) > 4000:
                tb_text = tb_text[:4000] + "\n... (traceback truncated)"
            error_msg.setDetailedText(tb_text)
            error_msg.exec()
        except Exception as box_e:
            log.error("Could not display error message box", error=str(box_e))

        sys.exit(1)  # Exit with error code

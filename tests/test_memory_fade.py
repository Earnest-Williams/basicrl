import numpy as np
import pytest
import pathlib
import sys

# Import lights_dev modules directly via path since the directory isn't a package
lights_dev_path = pathlib.Path(__file__).resolve().parent.parent / "lights_dev"
sys.path.append(str(lights_dev_path))
import main_game  # type: ignore  # pylint: disable=import-error
import constants  # type: ignore  # pylint: disable=import-error


def test_memory_fade_bounds():
    height, width = 1, 1
    current_time = np.float32(100.0)

    last_seen = np.full((height, width), current_time, dtype=np.float32)
    memory_intensity = np.ones((height, width), dtype=np.float32)
    visible = np.zeros((height, width), dtype=bool)

    # Immediately after being seen, intensity should remain at 1.0
    main_game._update_memory_fade_internal(
        current_time, last_seen, memory_intensity, visible, height, width
    )
    assert memory_intensity[0, 0] == pytest.approx(1.0)

    # After MEMORY_DURATION seconds, intensity should decay to ~0
    last_seen[0, 0] = current_time - constants.MEMORY_DURATION
    memory_intensity[0, 0] = 1.0
    main_game._update_memory_fade_internal(
        current_time, last_seen, memory_intensity, visible, height, width
    )
    assert memory_intensity[0, 0] == pytest.approx(0.0, abs=1e-6)

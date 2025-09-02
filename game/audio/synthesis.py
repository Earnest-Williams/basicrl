"""Procedural audio synthesis generators for BasicRL.

This module provides simple sound generators for common in-game effects.
The generators produce temporary WAV files that can be played by the
existing sound system.  NumPy is used to generate waveforms and the
standard ``wave`` module writes them to disk.

The generators are intentionally lightweight and do not require an audio
backend to be present at import time.  They return the path to a temporary
WAV file which can then be fed to any audio backend that supports file
playback.
"""

from __future__ import annotations

from pathlib import Path
import math
import tempfile
import wave
from typing import Callable, Dict, Any

import numpy as np

SAMPLE_RATE = 44100  # Hz


def _write_wave(data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write ``data`` to a temporary WAV file and return its path."""
    # Normalize to 16-bit range
    max_val = np.max(np.abs(data)) or 1.0
    audio = (data / max_val * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return Path(tmp.name)


def generate_footstep(duration: float = 0.2, frequency: float = 150.0) -> Path:
    """Generate a simple footstep sound.

    The sound is modelled as filtered noise with an exponential decay.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    noise = np.random.normal(0.0, 1.0, t.shape)
    envelope = np.exp(-t * 20.0)
    waveform = np.sin(2 * math.pi * frequency * t) * envelope * 0.3 + noise * 0.2
    return _write_wave(waveform)


def generate_magic(duration: float = 0.5, start_freq: float = 880.0) -> Path:
    """Generate a simple magical chime sound."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    freq_sweep = np.linspace(start_freq, start_freq * 2, t.size)
    waveform = np.sin(2 * math.pi * freq_sweep * t) * np.exp(-t * 3.0)
    return _write_wave(waveform)


# Registry of available generators
GENERATORS: Dict[str, Callable[..., Path]] = {
    "footsteps": generate_footstep,
    "magic": generate_magic,
}


def generate_sound(generator: str, settings: Dict[str, Any]) -> Path:
    """Generate a procedural sound using the named generator.

    Parameters
    ----------
    generator:
        Name of the generator function to use.
    settings:
        Parameters passed to the generator.
    """
    func = GENERATORS.get(generator)
    if not func:
        raise ValueError(f"Unknown sound generator: {generator}")
    return func(**settings)

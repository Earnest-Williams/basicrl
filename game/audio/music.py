"""Adaptive background music generator.

This module provides a very small `MusicGenerator` class used by the
:class:`~game.systems.sound.SoundManager` to create background music on the
fly.  The generator synthesizes a simple motif whose ``tempo``, ``harmony``
and ``intensity`` can be customised.  The implementation is intentionally
minimal – it only relies on the Python standard library so that tests can run
in restricted environments.

Example
-------
>>> gen = MusicGenerator()
>>> path = gen.generate(tempo=120, harmony="major", intensity=0.5)
>>> path.suffix
'.wav'
"""

from __future__ import annotations

from pathlib import Path
import math
import struct
import tempfile
import wave

SAMPLE_RATE = 22050  # Hz – lower rate keeps files small for tests


class MusicGenerator:
    """Create short musical motifs.

    The generated audio is a monophonic line that alternates between the root
    and third of the requested harmony.  ``tempo`` controls the speed in beats
    per minute while ``intensity`` controls the amplitude of the waveform.
    """

    def __init__(self) -> None:
        pass

    def generate(
        self,
        tempo: float = 120.0,
        harmony: str = "major",
        intensity: float = 0.5,
        duration: float = 4.0,
    ) -> Path:
        """Generate a WAV file based on the supplied parameters.

        Parameters
        ----------
        tempo:
            Beats per minute of the motif.
        harmony:
            Either ``"major"`` or ``"minor"``.  Determines the third used.
        intensity:
            Output amplitude in the range ``0.0``–``1.0``.
        duration:
            Length of the generated clip in seconds.
        """
        beat_len = 60.0 / max(tempo, 1.0)
        total_samples = int(SAMPLE_RATE * duration)
        # Frequencies for a root note of A4 (440Hz)
        root = 440.0
        third = root * (2 ** ((4 if harmony == "major" else 3) / 12))
        waveform = bytearray()
        for i in range(total_samples):
            t = i / SAMPLE_RATE
            beat_phase = (t % beat_len) / beat_len
            freq = root if beat_phase < 0.5 else third
            sample = intensity * math.sin(2 * math.pi * freq * t)
            waveform.extend(struct.pack("<h", int(sample * 32767)))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(waveform))
        return Path(tmp.name)

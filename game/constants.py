from enum import IntEnum


class FlowType(IntEnum):
    """Flow field types for noise and smell propagation."""

    PASS_DOORS = 0  # Monsters that can open/bash doors
    NO_DOORS = 1  # Monsters blocked by closed doors
    REAL_NOISE = 2  # Actual noise for stealth/perception (dampened by doors)
    MONSTER_NOISE = 3  # Noise originating from monsters


class FeatureType(IntEnum):
    """Map feature identifiers used across the game."""

    FLOOR = 0
    WALL = 1
    CLOSED_DOOR = 2
    OPEN_DOOR = 3
    SECRET_DOOR = 4


MAX_FLOWS: int = len(FlowType)

__all__ = ["FlowType", "FeatureType", "MAX_FLOWS"]

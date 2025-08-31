"""Lightweight action schema used by AI adapters.

This module defines a small ActionType enum and helper functions to validate action dicts.
AI adapters should return action dictionaries that conform to this schema. The schema is
intentionally small and decoupled from the full game internals to simplify testing and
parallelization.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class ActionType(str, Enum):
    MOVE = "move"
    ATTACK = "attack"
    USE = "use"
    WAIT = "wait"
    CUSTOM = "custom"


def make_move_action(actor: int, dx: int, dy: int, priority: int = 0) -> Dict[str, Any]:
    return {"actor": actor, "type": ActionType.MOVE.value, "priority": priority, "payload": {"dx": dx, "dy": dy}}


def validate_action(action: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    if "actor" not in action or not isinstance(action["actor"], int):
        return False
    if "type" not in action or not isinstance(action["type"], str):
        return False
    if "payload" in action and not isinstance(action["payload"], dict):
        return False
    return True
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Position:
    """Spatial position on the map."""
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class Renderable:
    """Rendering information for an entity."""
    glyph: int
    color_fg: Tuple[int, int, int]
    name: str
    blocks_movement: bool = True


@dataclass
class CombatStats:
    """Core combat related statistics."""
    hp: int = 0
    max_hp: int = 0
    mana: float = 0.0
    max_mana: float = 0.0
    fullness: float = 0.0
    max_fullness: float = 0.0


@dataclass
class Inventory:
    """Container for items carried by an entity."""
    capacity: int
    items: List[int] = field(default_factory=list)

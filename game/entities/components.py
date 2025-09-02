from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
    art_ranks: Dict[str, int] = field(default_factory=dict)
    substance_ranks: Dict[str, int] = field(default_factory=dict)


@dataclass
class Inventory:
    """Container for items carried by an entity."""

    capacity: int
    items: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class SealTags:
    """Tags that can be consumed to unlock seals or similar mechanics."""

    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FontSources:
    """Sources providing fonts or glyph sets for entities."""

    sources: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class VentTargets:
    """Targets that vents or releases can be applied to."""

    targets: List[str] = field(default_factory=list)

"""Core data models for the magic subsystem.

This module defines a very small framework used by tests to exercise the
``Work.calculate_effect_level`` method.  The system models a traditional fantasy
magic setup where a *Work* combines an ``Art`` (what the caster wishes to do)
and a ``Substance`` (what the magic is applied to).  ``Bounds`` limit the effect,
``Flow`` represents raw power and adjunct Works may provide additional support.

The real project will eventually expand on these ideas, but for the purposes of
unit tests we keep the implementation intentionally compact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class Art(Enum):
    """Fundamental magical arts.

    The exact meaning of each art is intentionally vague; in tests we only care
    that different arts exist and can be combined with ranks.
    """

    CREATE = auto()
    PERCEIVE = auto()
    TRANSFORM = auto()
    DESTROY = auto()
    CONTROL = auto()


class Substance(Enum):
    """Base substances that magic can affect."""

    AIR = auto()
    EARTH = auto()
    FIRE = auto()
    WATER = auto()
    SPIRIT = auto()


@dataclass(frozen=True)
class Bounds:
    """Limitations applied to a magical work.

    Each bound increases the complexity of the effect.  For simplicity bounds
    are represented as numeric ranks that are summed when computing the total
    power of a work.
    """

    range: int = 0
    duration: int = 0
    target: int = 0

    def total(self) -> int:
        """Return the combined rank of all bounds."""

        return self.range + self.duration + self.target


@dataclass(frozen=True)
class Balances:
    """Represents costs or counterweights for a work.

    The balancing system is outside the scope of the current tests but the
    dataclass is provided to mirror the intended architecture.
    """

    cost: int = 0
    risk: int = 0


@dataclass(frozen=True)
class Flow:
    """Raw power channelled into a work."""

    strength: int = 0

    def total(self) -> int:
        """Return the total contribution from flow."""

        return self.strength


@dataclass
class Seals:
    """Restrictions or conditions placed on a work.

    Not currently used in any computations but included for completeness.
    """

    description: str = ""
    power: int = 0


@dataclass
class Work:
    """A magical working combining arts, substances and other modifiers."""

    art: Art
    art_rank: int
    substance: Substance
    substance_rank: int
    bounds: Bounds = field(default_factory=Bounds)
    adjuncts: List["Work"] = field(default_factory=list)
    flow: Flow = field(default_factory=Flow)

    def calculate_effect_level(self) -> int:
        """Calculate the overall effect level of the work.

        The calculation is intentionally straightforward: the ranks of the
        primary art and substance are summed together.  Any adjunct works
        contribute their own art and substance ranks.  Finally, the totals from
        bounds and flow are added.  This mirrors a typical tabletop RPG style
        magic system where range/duration/target modifiers and raw power adjust
        the base potency of a spell.
        """

        level = self.art_rank + self.substance_rank

        # Adjunct works contribute additional art and substance ranks.
        for adjunct in self.adjuncts:
            level += adjunct.art_rank + adjunct.substance_rank

        # Bounds and flow provide further modifiers.
        level += self.bounds.total()
        level += self.flow.total()
        return level

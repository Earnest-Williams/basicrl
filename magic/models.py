"""Core data models for the magic subsystem + ledger-grammar AST.

- The *engine* types (Art, Substance, Bounds, Balances, Flow, Seals, Work)
  are the tested core used by `Work.calculate_effect_level`.

- The *ledger AST* types (ArtClause, BoundsClause, BalancesClause, FlowClause,
  SealsClause, ProvisionsClause, IntentClause, SeatClause, TendingClause,
  WorkDecl) model parsed, stringly-typed clauses from the ledger grammar.

- `compile_ledger_work` provides a tolerant, best-effort conversion from a
  `WorkDecl` to an engine `Work`. Unknown or missing fields default to 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import re


# =========================
# Engine (tested) data model
# =========================


class Art(Enum):
    """Fundamental magical arts."""

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
    """Limitations applied to a magical work."""

    range: int = 0
    duration: int = 0
    target: int = 0

    def total(self) -> int:
        return self.range + self.duration + self.target


@dataclass(frozen=True)
class Balances:
    """Represents costs or counterweights for a work."""

    cost: int = 0
    risk: int = 0


@dataclass(frozen=True)
class Flow:
    """Raw power channelled into a work."""

    strength: int = 0

    def total(self) -> int:
        return self.strength


@dataclass(frozen=True)
class Seals:
    """Restrictions or conditions placed on a work."""

    description: str = ""
    power: int = 0


@dataclass(frozen=True)
class Work:
    """A magical working combining arts, substances and other modifiers."""

    art: Art
    art_rank: int
    substance: Substance
    substance_rank: int
    bounds: Bounds = field(default_factory=Bounds)
    adjuncts: List["Work"] = field(default_factory=list)
    flow: Flow = field(default_factory=Flow)
    balances: Balances = field(default_factory=Balances)
    seals: Seals = field(default_factory=Seals)
    provisions: str = ""
    intent: str = ""
    seat: str = ""
    tending: str = ""

    def calculate_effect_level(self) -> int:
        """Compute overall effect level."""
        level = self.art_rank + self.substance_rank

        for adjunct in self.adjuncts:
            level += adjunct.art_rank + adjunct.substance_rank

        level += self.bounds.total()
        level += self.flow.total()
        level += self.seals.power
        level -= self.balances.cost
        level -= self.balances.risk
        return level


# =========================
# Ledger grammar AST (parser side)
# =========================


@dataclass
class ArtClause:
    """Represents the ART clause as parsed text."""

    value: str


@dataclass
class BoundsClause:
    """Represents the BOUNDS clause as parsed text."""

    value: str


@dataclass
class BalancesClause:
    """Represents the BALANCES clause as parsed text."""

    value: str


@dataclass
class FlowClause:
    """Represents the FLOW clause as parsed text."""

    value: str


@dataclass
class SealsClause:
    """Represents the SEALS clause as parsed text."""

    value: str


@dataclass
class ProvisionsClause:
    """Represents the PROVISIONS clause as parsed text."""

    value: str


@dataclass
class IntentClause:
    """Represents the INTENT clause as parsed text."""

    value: str


@dataclass
class SeatClause:
    """Represents the optional SEAT clause as parsed text."""

    value: str


@dataclass
class TendingClause:
    """Represents the optional TENDING clause as parsed text."""

    value: str


@dataclass
class WorkDecl:
    """Complete representation of a ledger work declaration (AST)."""

    art: ArtClause
    bounds: BoundsClause
    balances: BalancesClause
    flow: FlowClause
    seals: SealsClause
    provisions: ProvisionsClause
    intent: IntentClause
    seat: Optional[SeatClause] = None
    tending: Optional[TendingClause] = None


# ==========================================
# Tolerant compiler from AST -> engine model
# ==========================================

_ART_ALIASES = {
    "create": Art.CREATE,
    "perceive": Art.PERCEIVE,
    "transform": Art.TRANSFORM,
    "destroy": Art.DESTROY,
    "control": Art.CONTROL,
}

_SUBSTANCE_ALIASES = {
    "air": Substance.AIR,
    "earth": Substance.EARTH,
    "fire": Substance.FIRE,
    "water": Substance.WATER,
    "spirit": Substance.SPIRIT,
}

_INT_RE = re.compile(
    r"(?P<key>range|duration|target|strength|power|cost|risk)\s*=\s*(-?\d+)", re.I
)


def _kv_ints(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in _INT_RE.finditer(s or ""):
        key = m.group("key").lower()
        out[key] = int(m.group(2))
    return out


def _parse_art_and_ranks(text: str) -> tuple[Art, int, Substance, int]:
    """
    Accepts strings like:
      'create(3) on fire(2)'
      'Perceive/Spirit rank art=2 substance=1'
    Very forgiving: picks first known art and first known substance; ranks default to 0.
    """
    t = (text or "").strip()

    # Find an Art token
    art = None
    for name, enum_val in _ART_ALIASES.items():
        if re.search(rf"\b{name}\b", t, re.I):
            art = enum_val
            break
    art = art or Art.CREATE  # default

    # Rank in parentheses after art, e.g., "create(3)"
    art_rank = 0
    m = re.search(rf"{art.name}\s*\(\s*(-?\d+)\s*\)", t, re.I)
    if m:
        art_rank = int(m.group(1))
    else:
        # or "art=3"
        m = re.search(r"art\s*=\s*(-?\d+)", t, re.I)
        if m:
            art_rank = int(m.group(1))

    # Substance
    substance = None
    for name, enum_val in _SUBSTANCE_ALIASES.items():
        if re.search(rf"\b{name}\b", t, re.I):
            substance = enum_val
            break
    substance = substance or Substance.AIR  # default

    # Rank in parentheses after substance, e.g., "fire(2)" or "substance=2"
    sub_rank = 0
    m = re.search(rf"{substance.name}\s*\(\s*(-?\d+)\s*\)", t, re.I)
    if m:
        sub_rank = int(m.group(1))
    else:
        m = re.search(r"(substance|form)\s*=\s*(-?\d+)", t, re.I)
        if m:
            sub_rank = int(m.group(2))

    return art, art_rank, substance, sub_rank


def _parse_bounds(text: str) -> Bounds:
    kv = _kv_ints(text or "")
    return Bounds(
        range=kv.get("range", 0),
        duration=kv.get("duration", 0),
        target=kv.get("target", 0),
    )


def _parse_balances(text: str) -> Balances:
    kv = _kv_ints(text or "")
    return Balances(cost=kv.get("cost", 0), risk=kv.get("risk", 0))


def _parse_flow(text: str) -> Flow:
    kv = _kv_ints(text or "")
    return Flow(strength=kv.get("strength", 0))


def _parse_seals(text: str) -> Seals:
    kv = _kv_ints(text or "")
    # Description = everything with the numeric key/vals removed, compacted
    desc = re.sub(_INT_RE, "", text or "").strip()
    desc = re.sub(r"\s+", " ", desc)
    return Seals(description=desc, power=kv.get("power", 0))


def compile_ledger_work(decl: WorkDecl) -> Work:
    """
    Best-effort conversion from a parsed ledger `WorkDecl` to an engine `Work`.
    Unknown or absent numeric fields default to zero. Non-numeric fields are
    preserved verbatim but generally do not affect effect level.
    """
    art, art_rank, substance, sub_rank = _parse_art_and_ranks(decl.art.value)
    bounds = _parse_bounds(decl.bounds.value)
    balances = _parse_balances(decl.balances.value)
    flow = _parse_flow(decl.flow.value)
    seals = _parse_seals(decl.seals.value)
    provisions = (decl.provisions.value or "").strip()
    intent = (decl.intent.value or "").strip()
    seat = (decl.seat.value or "").strip() if decl.seat else ""
    tending = (decl.tending.value or "").strip() if decl.tending else ""

    return Work(
        art=art,
        art_rank=art_rank,
        substance=substance,
        substance_rank=sub_rank,
        bounds=bounds,
        adjuncts=[],
        flow=flow,
        balances=balances,
        seals=seals,
        provisions=provisions,
        intent=intent,
        seat=seat,
        tending=tending,
    )


__all__ = [
    # Engine
    "Art",
    "Substance",
    "Bounds",
    "Balances",
    "Flow",
    "Seals",
    "Work",
    # Ledger AST
    "ArtClause",
    "BoundsClause",
    "BalancesClause",
    "FlowClause",
    "SealsClause",
    "ProvisionsClause",
    "IntentClause",
    "SeatClause",
    "TendingClause",
    "WorkDecl",
    # Compiler
    "compile_ledger_work",
]

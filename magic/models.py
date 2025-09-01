from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Art:
    """Represents the ART clause."""
    value: str


@dataclass
class Bounds:
    """Represents the BOUNDS clause."""
    value: str


@dataclass
class Balances:
    """Represents the BALANCES clause."""
    value: str


@dataclass
class Flow:
    """Represents the FLOW clause."""
    value: str


@dataclass
class Seals:
    """Represents the SEALS clause."""
    value: str


@dataclass
class Provisions:
    """Represents the PROVISIONS clause."""
    value: str


@dataclass
class Intent:
    """Represents the INTENT clause."""
    value: str


@dataclass
class Seat:
    """Represents the optional SEAT clause."""
    value: str


@dataclass
class Tending:
    """Represents the optional TENDING clause."""
    value: str


@dataclass
class Work:
    """Complete representation of a ledger work declaration."""

    art: Art
    bounds: Bounds
    balances: Balances
    flow: Flow
    seals: Seals
    provisions: Provisions
    intent: Intent
    seat: Optional[Seat] = None
    tending: Optional[Tending] = None

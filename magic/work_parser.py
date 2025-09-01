from __future__ import annotations

import re
from typing import List, Tuple

from .models import (
    Art,
    Balances,
    Bounds,
    Flow,
    Intent,
    Provisions,
    Seals,
    Seat,
    Tending,
    Work,
)

Token = Tuple[str, str]

TOKEN_RE = re.compile(
    r"\s*(ART|BOUNDS|BALANCES|FLOW|SEALS|PROVISIONS|INTENT|SEAT|TENDING)\b",
    re.IGNORECASE,
)


def tokenize(source: str) -> List[Token]:
    """Tokenize source string into clause keywords and values."""
    tokens: List[Token] = []
    pos = 0
    while pos < len(source):
        match = TOKEN_RE.match(source, pos)
        if match:
            tokens.append(("KEYWORD", match.group(1).upper()))
            pos = match.end()
            continue
        next_match = TOKEN_RE.search(source, pos)
        if next_match:
            value = source[pos:next_match.start()].strip()
            if value:
                tokens.append(("VALUE", value))
            pos = next_match.start()
        else:
            value = source[pos:].strip()
            if value:
                tokens.append(("VALUE", value))
            break
    return tokens


def parse(source: str) -> Work:
    """Parse a ledger work declaration into structured dataclasses.

    Parameters
    ----------
    source:
        Text of the work declaration following the ledger grammar.

    Returns
    -------
    Work
        Parsed representation of the work.
    """

    tokens = tokenize(source)
    index = 0

    def expect_keyword(name: str) -> None:
        nonlocal index
        if index >= len(tokens) or tokens[index] != ("KEYWORD", name):
            raise ValueError(f"Expected clause {name}")
        index += 1

    def read_value() -> str:
        nonlocal index
        if index >= len(tokens) or tokens[index][0] != "VALUE":
            raise ValueError("Expected value following clause keyword")
        value = tokens[index][1]
        index += 1
        return value

    expect_keyword("ART")
    art = Art(read_value())
    expect_keyword("BOUNDS")
    bounds = Bounds(read_value())
    expect_keyword("BALANCES")
    balances = Balances(read_value())
    expect_keyword("FLOW")
    flow = Flow(read_value())
    expect_keyword("SEALS")
    seals = Seals(read_value())
    expect_keyword("PROVISIONS")
    provisions = Provisions(read_value())
    expect_keyword("INTENT")
    intent = Intent(read_value())

    seat = None
    tending = None
    if index < len(tokens):
        if tokens[index] == ("KEYWORD", "SEAT"):
            index += 1
            seat = Seat(read_value())
        elif tokens[index] == ("KEYWORD", "TENDING"):
            index += 1
            tending = Tending(read_value())
        else:
            raise ValueError(f"Unexpected clause {tokens[index][1]}")

    if index != len(tokens):
        raise ValueError("Unexpected trailing tokens")

    return Work(
        art=art,
        bounds=bounds,
        balances=balances,
        flow=flow,
        seals=seals,
        provisions=provisions,
        intent=intent,
        seat=seat,
        tending=tending,
    )

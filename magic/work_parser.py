from __future__ import annotations

import re
from typing import List, Tuple

from .models import (
    ArtClause,
    BalancesClause,
    BoundsClause,
    FlowClause,
    IntentClause,
    ProvisionsClause,
    SealsClause,
    SeatClause,
    TendingClause,
    WorkDecl,
    compile_ledger_work,
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
    """Parse a ledger work declaration and compile it into a ``Work``.

    The function first constructs a :class:`~magic.models.WorkDecl` AST composed of
    clause dataclasses (:class:`ArtClause`, :class:`BoundsClause`, etc.).  The
    resulting declaration is then converted into an engine-level
    :class:`~magic.models.Work` using :func:`compile_ledger_work`.

    Parameters
    ----------
    source:
        Text of the work declaration following the ledger grammar.

    Returns
    -------
    Work
        Executable representation of the work compiled from the declaration.
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
    art = ArtClause(read_value())
    expect_keyword("BOUNDS")
    bounds = BoundsClause(read_value())
    expect_keyword("BALANCES")
    balances = BalancesClause(read_value())
    expect_keyword("FLOW")
    flow = FlowClause(read_value())
    expect_keyword("SEALS")
    seals = SealsClause(read_value())
    expect_keyword("PROVISIONS")
    provisions = ProvisionsClause(read_value())
    expect_keyword("INTENT")
    intent = IntentClause(read_value())

    seat: SeatClause | None = None
    tending: TendingClause | None = None
    if index < len(tokens):
        if tokens[index] == ("KEYWORD", "SEAT"):
            index += 1
            seat = SeatClause(read_value())
        elif tokens[index] == ("KEYWORD", "TENDING"):
            index += 1
            tending = TendingClause(read_value())
        else:
            raise ValueError(f"Unexpected clause {tokens[index][1]}")

    if index != len(tokens):
        raise ValueError("Unexpected trailing tokens")

    decl = WorkDecl(
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

    work = compile_ledger_work(decl)

    # Preserve optional clauses for callers that care about them (e.g., scripting)
    if seat is not None:
        setattr(work, "seat", seat)
    if tending is not None:
        setattr(work, "tending", tending)

    return work

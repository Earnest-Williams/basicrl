from magic.work_parser import tokenize, parse
from magic.models import Art, Substance


def test_tokenize_basic():
    source = "ART fire BOUNDS range=1"
    assert tokenize(source) == [
        ("KEYWORD", "ART"),
        ("VALUE", "fire"),
        ("KEYWORD", "BOUNDS"),
        ("VALUE", "range=1"),
    ]


def test_parse_with_optional_clauses_any_order():
    source = (
        "ART create(2) on fire(1) "
        "BOUNDS range=1 "
        "BALANCES cost=0 "
        "FLOW strength=3 "
        "SEALS none "
        "PROVISIONS none "
        "INTENT test "
        "TENDING gentle "
        "SEAT stone"
    )
    work = parse(source)
    assert work.art is Art.CREATE
    assert work.substance is Substance.FIRE
    assert work.art_rank == 2
    assert work.substance_rank == 1
    assert getattr(work, "tending").value == "gentle"
    assert getattr(work, "seat").value == "stone"

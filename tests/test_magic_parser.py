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
    assert work.balances.cost == 0
    assert work.balances.risk == 0
    assert work.flow.strength == 3
    assert work.seals.description == "none"
    assert work.seals.power == 0
    assert work.provisions == "none"
    assert work.intent == "test"
    assert work.tending == "gentle"
    assert work.seat == "stone"


def test_parse_preserves_fields_and_effect_level():
    source = (
        "ART transform(1) on water(2) "
        "BOUNDS range=2 duration=1 target=1 "
        "BALANCES cost=2 risk=1 "
        "FLOW strength=4 "
        "SEALS vow power=3 "
        "PROVISIONS herb "
        "INTENT cleanse "
        "SEAT tree "
        "TENDING careful"
    )
    work = parse(source)
    assert work.balances == type(work.balances)(cost=2, risk=1)
    assert work.seals.power == 3
    assert work.seals.description == "vow"
    assert work.provisions == "herb"
    assert work.intent == "cleanse"
    assert work.seat == "tree"
    assert work.tending == "careful"
    expected = (
        work.art_rank
        + work.substance_rank
        + work.bounds.total()
        + work.flow.total()
        + work.seals.power
        - work.balances.cost
        - work.balances.risk
    )
    assert work.calculate_effect_level() == expected

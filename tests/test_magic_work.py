import pytest

from magic.models import Art, Substance, Bounds, Flow, Work, Balances, Seals


def test_calculate_effect_level_with_adjuncts_bounds_and_flow():
    bounds = Bounds(range=2, duration=1, target=1)
    flow = Flow(strength=3)
    adjunct = Work(art=Art.DESTROY, art_rank=1, substance=Substance.EARTH, substance_rank=2)
    work = Work(
        art=Art.CREATE,
        art_rank=5,
        substance=Substance.FIRE,
        substance_rank=4,
        bounds=bounds,
        adjuncts=[adjunct],
        flow=flow,
    )
    # Base art+substance ranks + adjunct ranks + bounds total + flow strength
    expected = (5 + 4) + (1 + 2) + bounds.total() + flow.total()
    assert work.calculate_effect_level() == expected


def test_calculate_effect_level_with_balances_and_seals():
    work = Work(
        art=Art.TRANSFORM,
        art_rank=3,
        substance=Substance.WATER,
        substance_rank=2,
        balances=Balances(cost=2, risk=1),
        seals=Seals(description="oath", power=4),
    )
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

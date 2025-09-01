import pytest

from magic.models import Art, Substance, Bounds, Flow, Work


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

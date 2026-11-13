import pytest

from transport_phenomena_experiments import diffusivity as di
from transport_phenomena_experiments.constants import ATM


def test_fuller_ethanol_air_25C():
    assert di.d_fuller("ethanol", "air", 298.15, ATM) == pytest.approx(1.234e-5, rel=2e-3)


def test_fuller_and_chapman_enskog_agree_for_ethanol():
    f = di.d_fuller("ethanol", "air", 298.15, ATM)
    ce = di.d_chapman_enskog("ethanol", "air", 298.15, ATM)
    assert abs(f / ce - 1) < 0.06


def test_chapman_enskog_is_poor_for_polar_water():
    # Lennard-Jones Chapman-Enskog ignores water's strong polarity and comes out
    # ~15 % below Fuller; the models therefore use Fuller for water.
    f = di.d_fuller("water", "air", 298.15, ATM)
    ce = di.d_chapman_enskog("water", "air", 298.15, ATM)
    assert 0.10 < f / ce - 1 < 0.25


def test_pressure_and_temperature_scaling():
    d1 = di.d_fuller("ethanol", "air", 298.15, ATM)
    assert di.d_fuller("ethanol", "air", 298.15, 2 * ATM) == pytest.approx(d1 / 2)
    assert di.scale_diffusivity(d1, 298.15, 318.15, 1.75) == pytest.approx(
        di.d_fuller("ethanol", "air", 318.15, ATM))

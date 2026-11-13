import numpy as np
import pytest

from transport_phenomena_experiments import properties as pr
from transport_phenomena_experiments.constants import ATM, c_to_k


def test_air_prandtl_matches_table():
    a = pr.air(pr._AIR_T, ATM)
    np.testing.assert_allclose(a.Pr, pr._AIR_PR_TABLE, rtol=0.01)


def test_air_kinematic_viscosity_scales_with_pressure():
    a1, a2 = pr.air(300.0, ATM), pr.air(300.0, 2 * ATM)
    assert a2.nu == pytest.approx(a1.nu / 2, rel=1e-12)


def test_air_out_of_range_raises():
    with pytest.raises(ValueError):
        pr.air(150.0, ATM)


def test_normal_boiling_points():
    # ethanol nbp 78.29 degC, water 100 degC: P_sat = 1 atm within 0.5 %
    assert pr.ETHANOL.psat(c_to_k(78.29)) == pytest.approx(ATM, rel=5e-3)
    assert pr.WATER.psat(c_to_k(100.0)) == pytest.approx(ATM, rel=5e-3)


def test_ethanol_vapour_pressure_25C():
    # 7.87 kPa at 25 degC (NIST-based reference value, within 1 %)
    assert pr.ETHANOL.psat(298.15) == pytest.approx(7.87e3, rel=0.01)


def test_watson_reproduces_reference_and_decreases():
    assert pr.ETHANOL.dh_vap_molar(298.15) == pytest.approx(42.3e3)
    assert pr.ETHANOL.dh_vap_molar(320.0) < pr.ETHANOL.dh_vap_molar(290.0)


def test_dew_point_inverts_relative_humidity():
    T = c_to_k(25.0)
    Td = pr.dew_point(0.6, T, ATM)
    assert pr.WATER.psat(Td) == pytest.approx(0.6 * pr.WATER.psat(T), rel=1e-9)
    assert 16.0 < Td - 273.15 < 17.5


def test_antoine_out_of_range_raises():
    with pytest.raises(ValueError):
        pr.ETHANOL.psat(c_to_k(95.0))

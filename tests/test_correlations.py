import numpy as np
import pytest

from transport_phenomena_experiments import correlations as co
from transport_phenomena_experiments.constants import ATM, SIGMA


def test_radiation_coefficient_is_exact_linearization():
    eps, Ts, Tsur = 0.9, 360.0, 295.0
    q = eps * SIGMA * (Ts**4 - Tsur**4)
    assert co.h_radiation(eps, Ts, Tsur) * (Ts - Tsur) == pytest.approx(q, rel=1e-12)


def test_hilpert_and_churchill_bernstein_agree_moderately():
    for Re in (10.0, 200.0, 2000.0):
        a, b = co.nu_hilpert(Re, 0.707), float(co.nu_churchill_bernstein(Re, 0.707))
        assert abs(a / b - 1) < 0.25


def test_hilpert_nearly_continuous_at_range_boundaries():
    for Re in (4.0, 40.0, 4000.0, 40000.0):
        lo, hi = co.nu_hilpert(Re * (1 - 1e-9), 0.7), co.nu_hilpert(Re * (1 + 1e-9), 0.7)
        assert abs(lo / hi - 1) < 0.05


def test_flat_plate_laminar_value():
    assert co.sh_flat_plate(1.0e4, 1.0) == pytest.approx(66.4)


def test_chilton_colburn_reduces_to_reynolds_analogy_for_le_1():
    assert co.h_from_kc_chilton_colburn(0.01, 1.2, 1000.0, 1.0) == pytest.approx(12.0)


def test_free_convection_cylinder_order_of_magnitude():
    h = co.h_free_horizontal_cylinder(4e-3, np.array([373.15]), 298.15, ATM)
    assert 8.0 < h[0] < 20.0


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        co.nu_hilpert(0.1, 0.7)
    with pytest.raises(ValueError):
        co.nu_horizontal_plate(100.0, 1.0, unstable=False)
    with pytest.raises(ValueError):
        co.sh_flat_plate(1e4, 0.5)

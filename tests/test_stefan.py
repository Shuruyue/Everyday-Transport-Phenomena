import numpy as np
import pytest

from transport_phenomena_experiments.constants import ATM, R
from transport_phenomena_experiments.diffusivity import d_fuller
from transport_phenomena_experiments.mass import stefan
from transport_phenomena_experiments.properties import ETHANOL

T = 298.15
D = float(d_fuller("ethanol", "air", T, ATM))
C = ATM / (R * T)
C_L = ETHANOL.rho(T) / ETHANOL.M
Y_S = ETHANOL.psat(T) / ATM


def test_moving_boundary_confirms_pseudo_steady_law():
    t = np.linspace(0.0, 48 * 3600, 97)
    Z, N_mouth = stefan.moving_boundary(t, 0.03, D, C, C_L, Y_S, 0.0, n_cells=40)
    K = stefan.stefan_slope(C, C_L, D, Y_S, 0.0)
    Zq = np.sqrt(0.03**2 + K * t)
    late = t > 3600
    # The only lasting difference is a constant offset of a few micrometres,
    # about (c / c_L) z0 y_s / 3: vapour stored in the initially vapour-free column
    # (y_s/2 on average) minus the permeation time-lag deficit (y_s/6). It is far
    # below any level reading (~0.1 mm).
    offset = Z[late] - Zq[late]
    assert np.all(np.abs(offset) < 5e-6)
    assert np.ptp(offset) < 0.3e-6
    # ... and the slope of Z^2 vs t is the pseudo-steady K to 0.2 %
    slope = np.polyfit(t[late], Z[late] ** 2, 1)[0]
    assert slope == pytest.approx(K, rel=2e-3)
    # flux leaving the mouth equals Stefan's law at the current depth (0.5 %)
    N_law = stefan.stefan_flux(C, D, Z[late], Y_S, 0.0)
    np.testing.assert_allclose(N_mouth[late], N_law, rtol=5e-3)


def test_fit_recovers_diffusivity_from_noisy_levels():
    rng = np.random.default_rng(20260106)
    t = np.linspace(0.0, 36 * 3600, 37)
    K = stefan.stefan_slope(C, C_L, D, Y_S, 0.0)
    z = np.sqrt(0.03**2 + K * t) + rng.normal(0.0, 2e-5, t.size)  # 0.02 mm noise
    fit = stefan.fit_stefan_tube(t, z, 0.03, C, C_L, Y_S, 0.0)
    assert abs(fit.D - D) < 2 * fit.D_ci95
    assert fit.D == pytest.approx(D, rel=0.03)


def test_level_from_mass_roundtrip():
    A = np.pi * 0.006**2
    z = stefan.level_from_mass(np.array([10.0e-3, 9.9e-3]), 10.0e-3, 0.03, 789.0, A)
    assert z[1] - z[0] == pytest.approx(0.1e-3 / (789.0 * A))

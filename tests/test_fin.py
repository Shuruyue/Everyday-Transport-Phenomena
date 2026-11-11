"""Verification of the finite-volume rod model against the analytical fin solutions."""

import numpy as np
import pytest

from etp.heat import fin
from etp.heat.fin_fv import Rod, SurfaceLoss, uniform_section
from etp.properties import MATERIALS

MAT = MATERIALS["AISI_304"]
D, L, H = 4e-3, 0.15, 15.0
T_INF, T_B = 298.15, 373.15
AC, P = fin.pin_section(D)
M = fin.fin_parameter(H, P, MAT.k, AC)
LOSS = SurfaceLoss(lambda T: np.full_like(T, H), T_INF, 0.0, T_INF)


def rod(n, tip="adiabatic", k=MAT.k):
    a, p = uniform_section(AC, P)
    return Rod(L, n, a, p, k, MAT.rho * MAT.cp, tip)


def observed_orders(errors):
    return np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))


def test_steady_second_order_convergence():
    errs, qerrs = [], []
    q_exact = fin.q_base_adiabatic_tip(H, P, MAT.k, AC, L, T_B - T_INF)
    for n in (20, 40, 80, 160):
        r = rod(n)
        T = r.steady(T_B, LOSS)
        exact = T_INF + (T_B - T_INF) * fin.theta_adiabatic_tip(r.x, L, M)
        errs.append(np.abs(T - exact).max())
        qerrs.append(abs(r.base_heat_rate(T, LOSS) / q_exact - 1))
    assert np.all(np.abs(observed_orders(errs) - 2) < 0.1)
    assert np.all(np.abs(observed_orders(qerrs) - 2) < 0.1)
    assert errs[-1] < 2e-3


def test_convective_tip_matches_analytic():
    r = rod(200, tip="convective")
    T = r.steady(T_B, LOSS)
    exact = T_INF + (T_B - T_INF) * fin.theta_convective_tip(r.x, L, M, H, MAT.k)
    assert np.abs(T - exact).max() < 5e-3


def test_corrected_length_approximates_convective_tip():
    q_conv = fin.q_base_convective_tip(H, P, MAT.k, AC, L, 75.0, H)
    Lc = fin.corrected_length(L, AC, P)
    q_corr = fin.q_base_adiabatic_tip(H, P, MAT.k, AC, Lc, 75.0)
    assert q_corr == pytest.approx(q_conv, rel=1e-3)


def test_transient_matches_series_solution():
    t = np.array([0.0, 10.0, 30.0, 90.0, 300.0, 900.0])
    x = np.linspace(0, L, 9)
    exact = fin.theta_transient_adiabatic_tip(x, t[1:], L, M, MAT.alpha, n_terms=600)
    errs = []
    for n in (20, 40, 80):
        r = rod(n)
        T0 = np.full(n + 1, T_INF)
        T0[0] = T_B
        Tn = r.transient(t, lambda _: T_B, T0, LOSS, rtol=1e-10, atol=1e-10)[1:]
        num = (np.array([np.interp(x, r.x, row) for row in Tn]) - T_INF) / (T_B - T_INF)
        errs.append(np.abs(num - exact).max())
    assert errs[-1] < 2e-3
    assert np.all(observed_orders(errs) > 1.7)


def test_transient_approaches_steady_state():
    r = rod(60)
    T0 = np.full(61, T_INF)
    T0[0] = T_B
    tau = fin.slowest_time_constant(L, M, MAT.alpha)
    Tn = r.transient(np.array([0.0, 20 * tau]), lambda _: T_B, T0, LOSS)
    np.testing.assert_allclose(Tn[-1], r.steady(T_B, LOSS), atol=1e-5)


def test_energy_conservation_with_nonlinear_loss():
    from etp.correlations import h_free_horizontal_cylinder

    loss = SurfaceLoss(lambda T: h_free_horizontal_cylinder(D, T, T_INF, 101325.0), T_INF,
                       0.3, T_INF)
    r = rod(60)
    T0 = np.full(61, T_INF)
    T0[0] = T_B
    t = np.linspace(0, 600, 2401)
    Tn = r.transient(t, lambda _: T_B, T0, loss, rtol=1e-10, atol=1e-10)
    stored, net = r.energy_residual(t, Tn, loss)
    assert stored == pytest.approx(net, rel=2e-3)


def test_series_truncation_guard():
    with pytest.raises(ValueError):
        fin.theta_transient_adiabatic_tip([0.05], [0.01], L, M, MAT.alpha, n_terms=5)

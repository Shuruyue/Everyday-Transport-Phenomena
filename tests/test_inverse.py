"""Parameter recovery from synthetic data with known truth (fixed seeds).

If a fit cannot recover parameters it generated itself, it cannot be trusted
on measurements either.
"""

import numpy as np
import pytest

from etp.constants import ATM, c_to_k
from etp.correlations import h_forced_cylinder, h_free_horizontal_cylinder
from etp.diffusivity import d_fuller
from etp.heat import fin, inverse
from etp.heat.fin_fv import Rod, SurfaceLoss, uniform_section
from etp.heat.lumped import cool_lumped
from etp.mass import evaporation as ev
from etp.mass.activity import Ideal
from etp.mass.inverse import fit_evaporation, mass_rate
from etp.properties import ETHANOL, MATERIALS

MAT = MATERIALS["AISI_304"]
D, L = 4e-3, 0.15
AC, P = fin.pin_section(D)
T_INF, T_B = 298.15, 373.15


def within(fit, name, truth, k=2.5):
    return abs(fit[name] - truth) <= k * fit.ci(name)


def test_steady_profile_recovers_m():
    rng = np.random.default_rng(1)
    m_true = 30.0
    x = np.array([0.01, 0.02, 0.04, 0.07, 0.12])
    Lc = fin.corrected_length(L, AC, P)
    T = T_INF + 75 * fin.theta_adiabatic_tip(x, Lc, m_true) + rng.normal(0, 0.1, x.size)
    res, h_over_k = inverse.fit_steady_profile(x, T, T_B, T_INF, AC, P, L, m0=20.0)
    assert within(res, "m", m_true)
    assert h_over_k == pytest.approx(m_true**2 * AC / P, rel=0.05)


def test_transient_rod_recovers_h_and_k():
    rng = np.random.default_rng(2)
    h_true, k_true = 15.0, MAT.k
    a, p = uniform_section(AC, P)
    rod = Rod(L, 40, a, p, k_true, MAT.rho * MAT.cp, "adiabatic")
    t = np.arange(0.0, 600.0 + 1e-9, 10.0)
    x_s = np.array([0.01, 0.02, 0.04, 0.07])
    T0 = np.full(41, T_INF)
    T0[0] = T_B
    loss = SurfaceLoss(lambda T: np.full_like(T, h_true), T_INF, 0.0, T_INF)
    Tn = rod.transient(t, lambda _: T_B, T0, loss)
    data = np.array([np.interp(x_s, rod.x, r) for r in Tn]) + rng.normal(0, 0.1, (t.size, 4))
    res, _ = inverse.fit_transient_rod(t, x_s, data, rod, lambda _: T_B, T0, T_INF,
                                       {"kind": "constant"}, p0=[8.0, 25.0], n_cells=40)
    assert within(res, "h_eff", h_true) and within(res, "k", k_true)
    assert abs(res.corr[0, 1]) < 0.99  # the transient separates h from k


def test_cooling_fits():
    rng = np.random.default_rng(3)
    C = MAT.rho * MAT.cp * AC * 0.22  # J/K, 22 cm stainless rod
    A = P * 0.22 + 2 * AC  # m^2, lateral surface plus both ends
    h_corr = lambda T: h_free_horizontal_cylinder(D, T, T_INF, ATM)  # noqa: E731
    loss = SurfaceLoss(lambda T: 1.25 * h_corr(T), T_INF, 0.3, T_INF)
    t = np.arange(0.0, 1500.0, 10.0)
    T = cool_lumped(t, 368.0, C, A, loss) + rng.normal(0, 0.1, t.size)
    res, _ = inverse.fit_cooling_physical(t, T, C, A, h_corr, T_INF, T_INF,
                                          fit=("T_i", "C_h"), fixed={"eps": 0.3})
    assert within(res, "C_h", 1.25) and within(res, "T_i", 368.0)
    exp = inverse.fit_cooling_exponential(t, T, T_INF, tau0=200.0)
    # the single exponential cannot follow the nonlinear h: residuals are not white
    assert exp.s > 1.5 * res.s


def test_velocity_inversion_roundtrip():
    h = h_forced_cylinder(D, 2.3, 330.0, T_INF, ATM, "churchill_bernstein")
    U = inverse.velocity_from_h(h, D, 330.0, T_INF, ATM, "churchill_bernstein", (0.05, 20.0))
    assert U == pytest.approx(2.3, rel=1e-8)


def test_evaporation_fit_recovers_multiplier_and_bottom_conductance():
    rng = np.random.default_rng(4)
    amb = ev.Ambient(c_to_k(25.0), c_to_k(25.0), ATM, 0.0)
    diff = ev.Diffusivities(lambda T: float(d_fuller("ethanol", "air", T, ATM)),
                            lambda T: float(d_fuller("water", "air", T, ATM)))
    dish = ev.Dish(0.045, 0.002, 20.0, 0.03, 0.95)
    true_model = ev.ForcedFlatPlate(U=1.5, length=0.09, multiplier=1.3)
    n0 = 12e-3 / ETHANOL.M
    t = np.arange(0.0, 3600.0, 120.0)
    r = ev.simulate_evaporation(t, n0, 0.0, amb.T_inf, dish, amb, true_model, diff, Ideal(),
                                water_exchange=False, isothermal=False,
                                recess_model="stagnant_layer", stop_fraction=1e-3)
    m = r.mass + rng.normal(0, 0.01e-3, t.size)
    Ts = r.T_s + rng.normal(0, 0.2, t.size)
    start = ev.Dish(0.045, 0.002, 20.0, 0.1, 0.95)
    res, _ = fit_evaporation(t, m, Ts, 0.01e-3, 0.2, n0, 0.0, start, amb,
                             ev.ForcedFlatPlate(1.5, 0.09, 1.0), diff, Ideal(), False,
                             "stagnant_layer", fit=("multiplier", "UA_bottom"), p0=[1.0, 0.1])
    assert within(res, "multiplier", 1.3) and within(res, "UA_bottom", 0.03)
    rate, ci, _ = mass_rate(t, m, 0.0, 1200.0)
    assert rate > 0 and ci < 0.1 * rate

import numpy as np
import pytest
from scipy import integrate

from etp.heat import lumped
from etp.heat.fin_fv import SurfaceLoss

C, A, H, T_I, T_INF = 2.0, 2e-3, 20.0, 373.15, 298.15


def test_ode_reproduces_exponential_for_constant_h():
    t = np.linspace(0, 600, 301)
    loss = SurfaceLoss(lambda T: np.full_like(T, H), T_INF, 0.0, T_INF)
    T = lumped.cool_lumped(t, T_I, C, A, loss)
    exact = lumped.lumped_analytic(t, T_I, T_INF, lumped.time_constant(C, H, A))
    np.testing.assert_allclose(T, exact, atol=1e-7)


def test_instantaneous_h_recovers_constant_h():
    t = np.linspace(0, 400, 401)
    T = lumped.lumped_analytic(t, T_I, T_INF, lumped.time_constant(C, H, A))
    h = lumped.instantaneous_h(t, T, T_INF, C, A, window=21, polyorder=3)
    np.testing.assert_allclose(h, H, rtol=1e-3)


def test_energy_released_equals_heat_lost():
    t = np.linspace(0, 3000, 6001)
    loss = SurfaceLoss(lambda T: 5.0 * np.abs(T - T_INF) ** 0.25 + 2.0, T_INF, 0.9, T_INF)
    T = lumped.cool_lumped(t, T_I, C, A, loss)
    q = A * loss.flux(T)
    assert integrate.simpson(q, x=t) == pytest.approx(C * (T[0] - T[-1]), rel=1e-6)


def test_time_to_reach():
    tau = 100.0
    t = lumped.time_to_reach(T_INF + 2.0, T_I, T_INF, tau)
    assert lumped.lumped_analytic(t, T_I, T_INF, tau) == pytest.approx(T_INF + 2.0)

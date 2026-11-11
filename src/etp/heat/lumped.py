"""Lumped-capacitance cooling of the rod (TP2 Lecture 5).

    C dT/dt = -A_s q''(T),   C = m c  (thermal capacitance, J/K)

Valid when Bi = h (V/A_s) / k < 0.1. With constant h and no radiation the
solution is the exponential theta/theta_i = exp(-t/tau), tau = C / (h A_s).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from ..fitting import savgol_derivative


def biot(h, V, A_s, k):
    """Bi = h L_c / k with L_c = V / A_s."""
    return h * (V / A_s) / k


def time_constant(C, h, A_s):
    """tau = C / (h A_s) [s]."""
    return C / (h * A_s)


def lumped_analytic(t, T_i, T_inf, tau):
    """T(t) [K] for constant h, no separate radiation term."""
    return T_inf + (T_i - T_inf) * np.exp(-np.asarray(t, dtype=float) / tau)


def time_to_reach(T_target, T_i, T_inf, tau):
    """Time [s] for the exponential to reach T_target."""
    if not (T_inf < T_target < T_i or T_i < T_target < T_inf):
        raise ValueError("T_target must lie between T_i and T_inf")
    return tau * np.log((T_i - T_inf) / (T_target - T_inf))


def cool_lumped(t_eval, T_i, C, A_s, loss, rtol=1e-10, atol=1e-10):
    """Integrate C dT/dt = -A_s q''(T) with a SurfaceLoss (nonlinear h, radiation)."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = integrate.solve_ivp(lambda t, T: -A_s * loss.flux(T) / C, (t_eval[0], t_eval[-1]),
                              [T_i], method="Radau", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0]


def instantaneous_h(t, T, T_inf, C, A_s, window, polyorder):
    """Total (convection + radiation) coefficient from the cooling curve:

        h(t) = -C (dT/dt) / (A_s (T - T_inf))

    dT/dt from a Savitzky-Golay filter whose window and order are explicit
    analysis choices and must be reported with the result.
    """
    dTdt = savgol_derivative(t, T, window, polyorder)
    theta = np.asarray(T, dtype=float) - T_inf
    return -C * dTdt / (A_s * theta)

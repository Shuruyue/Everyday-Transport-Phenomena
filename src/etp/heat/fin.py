"""Analytical fin (extended-surface) solutions for a rod of uniform cross-section.

Governing equation, theta = T - T_inf, constant h, k, A_c, P:

    d^2 theta / dx^2 - m^2 theta = 0,      m^2 = h P / (k A_c)

Base x = 0 held at theta_b (water surface), tip at x = L.

These solutions serve two purposes: (1) the model the course expects in the
report, (2) the exact reference that the finite-volume solver in
``fin_fv.py`` is verified against.
"""

from __future__ import annotations

import numpy as np


def pin_section(D):
    """Cross-sectional area [m^2] and perimeter [m] of a circular rod of diameter D [m]."""
    return np.pi * D**2 / 4.0, np.pi * D


def rectangular_section(width, thickness):
    """Cross-sectional area [m^2] and perimeter [m] of a rectangular rod."""
    return width * thickness, 2.0 * (width + thickness)


def fin_parameter(h, P, k, Ac):
    """m = sqrt(h P / (k A_c)) [1/m]."""
    return np.sqrt(h * P / (k * Ac))


def corrected_length(L, Ac, P):
    """L_c = L + A_c / P: replaces the convective tip by an adiabatic one (L + D/4 for a pin)."""
    return L + Ac / P


def theta_adiabatic_tip(x, L, m):
    """theta / theta_b for an adiabatic tip at x = L."""
    x = np.asarray(x, dtype=float)
    return np.cosh(m * (L - x)) / np.cosh(m * L)


def theta_convective_tip(x, L, m, h_tip, k):
    """theta / theta_b for a convective tip, -k dtheta/dx = h_tip theta at x = L."""
    x = np.asarray(x, dtype=float)
    a = h_tip / (m * k)
    return (np.cosh(m * (L - x)) + a * np.sinh(m * (L - x))) / (np.cosh(m * L) + a * np.sinh(m * L))


def q_base_convective_tip(h, P, k, Ac, L, theta_b, h_tip):
    """Heat rate entering the fin at the base [W], convective tip."""
    m = fin_parameter(h, P, k, Ac)
    M = np.sqrt(h * P * k * Ac) * theta_b
    a = h_tip / (m * k)
    return M * (np.sinh(m * L) + a * np.cosh(m * L)) / (np.cosh(m * L) + a * np.sinh(m * L))


def q_base_adiabatic_tip(h, P, k, Ac, L, theta_b):
    """Heat rate entering the fin at the base [W], adiabatic tip at L."""
    m = fin_parameter(h, P, k, Ac)
    return np.sqrt(h * P * k * Ac) * theta_b * np.tanh(m * L)


def theta_transient_adiabatic_tip(x, t, L, m, alpha, n_terms, tol=1e-10):
    """theta / theta_b for a rod initially at T_inf whose base is stepped to T_b at t = 0.

    Solution (derived in docs/derivations.md):
        theta/theta_b = cosh m(L-x)/cosh mL
                        - sum_n  2 lam_n / (L (lam_n^2 + m^2)) sin(lam_n x) exp(-alpha (lam_n^2 + m^2) t)
        lam_n = (2n - 1) pi / (2 L)

    Returns an array of shape (len(t), len(x)). Raises if n_terms is too small for
    the earliest time requested (series truncation would be visible).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t <= 0):
        raise ValueError("series solution needs t > 0 (the t = 0 state is theta = 0)")
    n = np.arange(1, n_terms + 1)
    lam = (2 * n - 1) * np.pi / (2 * L)
    B = 2 * lam / (L * (lam**2 + m**2))
    decay = np.exp(-alpha * np.outer(t, lam**2 + m**2))  # (nt, n)
    last = B[-1] * decay[:, -1].max()
    if last > tol:
        raise ValueError(f"n_terms = {n_terms} too small: last term {last:.2e} > {tol:.0e} "
                         f"at t = {t.min():g} s")
    series = (decay * B) @ np.sin(np.outer(lam, x))  # (nt, nx)
    return theta_adiabatic_tip(x, L, m)[None, :] - series


def slowest_time_constant(L, m, alpha):
    """1 / (alpha (m^2 + (pi / 2L)^2)) [s]: the e-folding time of the approach to steady state."""
    return 1.0 / (alpha * (m**2 + (np.pi / (2 * L)) ** 2))

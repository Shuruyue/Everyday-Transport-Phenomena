"""Convection correlations (heat and mass) and radiation coefficient.

References
----------
[I]  Incropera et al., Fundamentals of Heat and Mass Transfer, 7th ed.
[L67] TP2 Lecture 6-7 (Hilpert, Zukauskas, Churchill-Bernstein for cylinders).
[L1113] TP2 Lecture 11-13 (flat-plate Sherwood correlations, Chilton-Colburn).

Each correlation raises ``ValueError`` outside its stated range of validity.
The heat and mass forms of a boundary-layer correlation are the same function
with (Nu, Pr) replaced by (Sh, Sc); the analogy is used, not re-typed.
"""

from __future__ import annotations

import numpy as np

from .constants import G, SIGMA
from .properties import air


def _check_range(name, value, lo, hi):
    v = np.asarray(value, dtype=float)
    if np.any(v < lo) or np.any(v > hi):
        raise ValueError(f"{name} = {v} outside correlation range [{lo:g}, {hi:g}]")


# ---------------------------------------------------------------------------
# Cylinder in cross flow
# ---------------------------------------------------------------------------
_HILPERT = [  # (Re_lo, Re_hi, C, m), [I] Table 7.2
    (0.4, 4.0, 0.989, 0.330),
    (4.0, 40.0, 0.911, 0.385),
    (40.0, 4.0e3, 0.683, 0.466),
    (4.0e3, 4.0e4, 0.193, 0.618),
    (4.0e4, 4.0e5, 0.027, 0.805),
]


def nu_hilpert(Re, Pr):
    """Hilpert: Nu_D = C Re^m Pr^(1/3), film properties, Pr >= 0.7 [L67]."""
    Re = float(Re)
    _check_range("Re_D (Hilpert)", Re, 0.4, 4.0e5)
    for lo, hi, C, m in _HILPERT:
        if lo <= Re <= hi:
            return C * Re**m * Pr ** (1.0 / 3.0)
    raise AssertionError("unreachable")


_ZUKAUSKAS = [  # (Re_lo, Re_hi, C, m), [I] Table 7.4
    (1.0, 40.0, 0.75, 0.4),
    (40.0, 1.0e3, 0.51, 0.5),
    (1.0e3, 2.0e5, 0.26, 0.6),
    (2.0e5, 1.0e6, 0.076, 0.7),
]


def nu_zukauskas(Re, Pr_inf, Pr_s):
    """Zukauskas: Nu_D = C Re^m Pr^n (Pr/Pr_s)^(1/4), properties at T_inf [L67]."""
    Re = float(Re)
    _check_range("Re_D (Zukauskas)", Re, 1.0, 1.0e6)
    _check_range("Pr (Zukauskas)", Pr_inf, 0.7, 500.0)
    n = 0.37 if Pr_inf <= 10 else 0.36
    for lo, hi, C, m in _ZUKAUSKAS:
        if lo <= Re <= hi:
            return C * Re**m * Pr_inf**n * (Pr_inf / Pr_s) ** 0.25
    raise AssertionError("unreachable")


def nu_churchill_bernstein(Re, Pr):
    """Churchill-Bernstein, all Re with Re Pr >= 0.2, film properties [L67]."""
    Re = np.asarray(Re, dtype=float)
    _check_range("Re*Pr (Churchill-Bernstein)", Re * Pr, 0.2, np.inf)
    return 0.3 + (0.62 * Re**0.5 * Pr ** (1 / 3)) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25 * (
        1 + (Re / 282000.0) ** (5 / 8)
    ) ** (4 / 5)


def nu_churchill_chu_horizontal_cylinder(Ra, Pr):
    """Free convection from a long horizontal cylinder, Ra_D <= 1e12 [I] Eq. 9.34."""
    Ra = np.asarray(Ra, dtype=float)
    _check_range("Ra_D (Churchill-Chu cylinder)", Ra, 0.0, 1.0e12)
    return (0.60 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2


# ---------------------------------------------------------------------------
# Horizontal surface (free convection). Used for the open dish, as heat or mass.
# ---------------------------------------------------------------------------
def nu_horizontal_plate(Ra, Pr, unstable):
    """Free convection from a horizontal plate, L = A_s / P [I] Eqs. 9.30-9.32.

    unstable=True : buoyant fluid below heavy fluid (hot surface facing up, or a
                    surface releasing light gas such as water vapour).
    unstable=False: stable stratification (cold surface facing up, or a surface
                    releasing heavy vapour such as ethanol).
    Same function gives Sh with (Ra_mass, Sc).
    """
    Ra = float(Ra)
    if unstable:
        if 1.0e4 <= Ra <= 1.0e7:
            _check_range("Pr/Sc", Pr, 0.7, np.inf)
            return 0.54 * Ra**0.25
        if 1.0e7 < Ra <= 1.0e11:
            return 0.15 * Ra ** (1 / 3)
        raise ValueError(f"Ra_L = {Ra:g} outside 1e4-1e11 (unstable horizontal plate)")
    _check_range("Ra_L (stable horizontal plate)", Ra, 1.0e4, 1.0e9)
    _check_range("Pr/Sc", Pr, 0.7, np.inf)
    return 0.52 * Ra**0.2


# ---------------------------------------------------------------------------
# Flat plate, parallel flow (course forms) [L1113]
# ---------------------------------------------------------------------------
RE_TRANSITION_PLATE = 3.0e5


def sh_flat_plate(Re_L, Sc):
    """Average Sherwood number over a flat plate of length L [L1113].

    Laminar  (Re_L < 3e5): Sh = 0.664 Re^(1/2) Sc^(1/3)
    Turbulent(Re_L > 3e5): Sh = 0.036 Re^0.8 Sc^(1/3), 0.6 < Sc < 2500
    The heat-transfer form is the same with (Nu, Pr).
    """
    _check_range("Sc/Pr (flat plate)", Sc, 0.6, 2500.0)
    if Re_L < RE_TRANSITION_PLATE:
        return 0.664 * Re_L**0.5 * Sc ** (1 / 3)
    return 0.036 * Re_L**0.8 * Sc ** (1 / 3)


# ---------------------------------------------------------------------------
# Radiation and the heat-mass analogy
# ---------------------------------------------------------------------------
def h_radiation(eps, T_s, T_sur):
    """Linearized radiation coefficient: q'' = h_rad (T_s - T_sur) exactly."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"emissivity must be in [0, 1], got {eps}")
    T_s = np.asarray(T_s, dtype=float)
    return eps * SIGMA * (T_s + T_sur) * (T_s**2 + T_sur**2)


def h_from_kc_chilton_colburn(kc, rho, cp, Le):
    """Chilton-Colburn, j_H = j_D  =>  h = k_c rho c_p Le^(2/3), Le = alpha / D_AB."""
    return kc * rho * cp * Le ** (2.0 / 3.0)


# ---------------------------------------------------------------------------
# Convenience wrappers with film properties of air
# ---------------------------------------------------------------------------
def h_free_horizontal_cylinder(D, T_s, T_inf, P):
    """Free-convection h [W/(m^2 K)] for a horizontal cylinder of diameter D [m].

    Vectorized over T_s. Film properties, beta = 1/T_f (ideal gas).
    """
    T_s = np.asarray(T_s, dtype=float)
    T_f = 0.5 * (T_s + T_inf)
    a = air(T_f, P)
    Ra = G * np.abs(T_s - T_inf) / T_f * D**3 / (a.nu * a.alpha)
    return nu_churchill_chu_horizontal_cylinder(Ra, a.Pr) * a.k / D


def h_forced_cylinder(D, U, T_s, T_inf, P, correlation):
    """Forced-convection h [W/(m^2 K)] for a cylinder in cross flow at speed U [m/s].

    correlation: "churchill_bernstein" or "hilpert" (film properties).
    """
    T_f = 0.5 * (T_s + T_inf)
    a = air(T_f, P)
    Re = U * D / float(a.nu)
    Pr = float(a.Pr)
    if correlation == "churchill_bernstein":
        Nu = float(nu_churchill_bernstein(Re, Pr))
    elif correlation == "hilpert":
        Nu = nu_hilpert(Re, Pr)
    else:
        raise ValueError(f"unknown correlation {correlation!r}")
    return Nu * float(a.k) / D

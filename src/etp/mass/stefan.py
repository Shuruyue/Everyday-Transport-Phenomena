"""Stefan tube: diffusion of a vapour through stagnant air (TP2 Lecture 11-13).

Steady flux through a gas column of height z (Stefan's law):

    N_A = (c D_AB / z) ln[(1 - y_inf) / (1 - y_s)]

Pseudo-steady liquid recession, c_L dz/dt = N_A, integrates to

    z^2 - z0^2 = K t,     K = 2 (c / c_L) D_AB ln[(1 - y_inf) / (1 - y_s)]

with c = P / (R T) the gas and c_L = rho_L / M_A the liquid molar concentration.
A linear fit of z^2 - z0^2 against t gives D_AB.

``moving_boundary`` solves the full transient problem (unsteady gas-phase
diffusion with Stefan flow and a moving interface) so the pseudo-steady
assumption can be checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate

from ..fitting import fit_linear


def log_factor(y_s, y_inf):
    """ln[(1 - y_inf)/(1 - y_s)], written with log1p for accuracy at small y."""
    return np.log1p((y_s - y_inf) / (1.0 - y_s))


def stefan_flux(c, D, z, y_s, y_inf):
    """Molar flux N_A [mol/(m^2 s)] through a stagnant gas column of height z [m]."""
    return c * D / z * log_factor(y_s, y_inf)


def stefan_slope(c, c_L, D, y_s, y_inf):
    """K [m^2/s] in z^2 - z0^2 = K t."""
    return 2.0 * (c / c_L) * D * log_factor(y_s, y_inf)


def level_from_mass(m, m0, z0, rho_L, A):
    """Interface depth below the tube mouth z [m] from the tube mass m [kg]."""
    return z0 + (m0 - np.asarray(m, dtype=float)) / (rho_L * A)


@dataclass(frozen=True)
class StefanFit:
    D: float  # m^2/s
    D_ci95: float  # from the slope only; property uncertainty via Monte Carlo
    K: float  # m^2/s
    K_ci95: float
    intercept: float  # m^2, should be ~0 if z0 was read correctly
    intercept_ci95: float
    r2: float
    residuals: np.ndarray


def fit_stefan_tube(t, z, z0, c, c_L, y_s, y_inf):
    """D_AB from the slope of z^2 - z0^2 vs t (intercept kept free and reported)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(z, dtype=float) ** 2 - z0**2
    lf = fit_linear(t, y, through_origin=False)
    scale = 1.0 / (2.0 * (c / c_L) * log_factor(y_s, y_inf))
    return StefanFit(D=lf.slope * scale, D_ci95=lf.slope_ci95 * scale, K=lf.slope,
                     K_ci95=lf.slope_ci95, intercept=lf.intercept,
                     intercept_ci95=lf.intercept_ci95, r2=lf.r2, residuals=lf.residuals)


def moving_boundary(t_eval, Z0, D, c, c_L, y_s, y_inf, n_cells, rtol=1e-9, atol=1e-12):
    """Transient gas-phase diffusion in a Stefan tube with a receding interface.

    Lab frame: interface at depth Z(t) below the mouth. Landau transform
    zeta = (distance above interface) / Z in [0, 1]. With constant c (isothermal,
    isobaric ideal gas) total continuity makes the molar flux N uniform in z,
    and the interface jump conditions give (derivation in docs/derivations.md):

        dZ/dt = u = -c D y_zeta(0) / (Z c_L (1 - y_s))
        N     = (c_L - c) u
        y_t   = D / Z^2 y_zz - [N / (c Z) + (1 - zeta) u / Z] y_zeta

    Initial gas: y = y_inf (tube just filled). Returns (Z(t), N_A at the mouth).
    """
    n = n_cells
    zeta = np.linspace(0.0, 1.0, n + 1)
    dz = zeta[1] - zeta[0]
    y0 = np.full(n - 1, y_inf, dtype=float)

    def unpack(s):
        y = np.empty(n + 1)
        y[0], y[-1] = y_s, y_inf
        y[1:-1] = s[:-1]
        return y, s[-1]

    def rhs(t, s):
        y, Z = unpack(s)
        y_z0 = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dz)
        u = -c * D * y_z0 / (Z * c_L * (1.0 - y_s))
        N = (c_L - c) * u
        y_z = (y[2:] - y[:-2]) / (2 * dz)
        y_zz = (y[2:] - 2 * y[1:-1] + y[:-2]) / dz**2
        dy = D / Z**2 * y_zz - (N / (c * Z) + (1.0 - zeta[1:-1]) * u / Z) * y_z
        return np.append(dy, u)

    sol = integrate.solve_ivp(rhs, (0.0, t_eval[-1]), np.append(y0, Z0), method="BDF",
                              t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    Z = sol.y[-1]
    N_mouth = []
    for s in sol.y.T:
        y, Zi = unpack(s)
        y_z1 = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dz)
        y_z0 = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dz)
        u = -c * D * y_z0 / (Zi * c_L * (1.0 - y_s))
        N = (c_L - c) * u
        N_mouth.append(y[-1] * N - c * D * y_z1 / Zi)
    return Z, np.array(N_mouth)

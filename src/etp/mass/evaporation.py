"""Evaporation of ethanol (or aqueous ethanol) from an open dish.

One model covers every Week 2 condition, so the conditions differ only where the
physics differs (scientific-coding rule: shared code for shared parts):

  * convective mass transfer, film form with Stefan-flow correction
        N_i = k_c,i c_f (y_i,s - y_i,inf) phi,
        phi = ln[(1 - y_inf,tot)/(1 - y_s,tot)] / (y_s,tot - y_inf,tot)
    exact for one diffusing species (pure ethanol); for ethanol + water both
    diffusing through air it is the usual film-model approximation.
  * k_c from a correlation chosen by the condition (forced flat plate, free
    convection over a horizontal surface, or the pure-diffusion disk limit),
    evaluated for ethanol; water uses k_c,W = k_c,E (D_W/D_E)^n with the
    exponent n that the same correlation implies;
  * an optional recess resistance once the liquid sits below the rim,
        1/k_eff = 1/k_c + h_r / D,      1/h_eff = 1/h + h_r / k_air
    ("stagnant_layer"). It treats the gas above the liquid as stagnant, so it is
    an UPPER BOUND on the extra resistance: air flowing over a shallow dish
    ventilates the recess, and even in still air the 2D diffusion solution
    (diffusion2d.py) gives a smaller effect than the 1D layer. Use a shallow dish
    filled close to the rim so this term stays small;
  * the surface energy balance (evaporative cooling), h from Chilton-Colburn:
        (sum n_i M_i c_p,i + C_dish) dT_s/dt = (h_eff A + UA_b)(T_inf - T_s)
             + eps sigma A (T_sur^4 - T_s^4) - A sum N_i dH_vap,i
  * liquid phase well mixed; vapour-liquid equilibrium at the surface,
        y_i,s = x_i gamma_i P_sat,i(T_s) / P   (NRTL for ethanol-water).

Switches: ``isothermal`` holds T_s = T_inf (the textbook assumption, kept as the
comparison case); ``water_exchange`` lets water vapour from humid air condense
into, or evaporate from, the liquid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, optimize

from ..constants import G, R, SIGMA, T0
from ..correlations import h_from_kc_chilton_colburn, nu_horizontal_plate, sh_flat_plate
from ..properties import ETHANOL, M_AIR, WATER, air
from .stefan import log_factor


@dataclass(frozen=True)
class Ambient:
    T_inf: float  # K
    T_sur: float  # K, radiating surroundings
    P: float  # Pa
    RH: float  # relative humidity, fraction

    @property
    def y_water(self):
        return self.RH * WATER.psat(self.T_inf) / self.P

    @property
    def rho(self):
        """Humid-air density [kg/m^3]."""
        yw = self.y_water
        return self.P * (yw * WATER.M + (1 - yw) * M_AIR) / (R * self.T_inf)


@dataclass(frozen=True)
class Dish:
    radius: float  # m, radius of the liquid surface
    recess0: float  # m, initial depth of the liquid surface below the rim
    C_dish: float  # J/K, heat capacity of the dish moving with the liquid temperature
    UA_bottom: float  # W/K, conductance liquid -> ambient through dish wall and table
    eps: float  # emissivity of the liquid surface

    @property
    def area(self):
        return np.pi * self.radius**2

    @property
    def L_char(self):
        """A / P = R / 2 for a disk (free-convection length scale)."""
        return self.radius / 2.0


@dataclass(frozen=True)
class Diffusivities:
    ethanol: Callable[[float], float]  # D_ethanol-air(T) [m^2/s]
    water: Callable[[float], float]  # D_water-air(T) [m^2/s]


# ---------------------------------------------------------------- k_c models
@dataclass(frozen=True)
class ForcedFlatPlate:
    """Parallel flow at U over the dish, plate length L = dish diameter (course correlation)."""

    U: float  # m/s
    length: float  # m, in the flow direction
    multiplier: float  # 1 = correlation as published; fitted value = data / correlation

    def kc(self, T_s, rho_s, amb, D):
        """(k_c [m/s], n) with k_c ~ D^n implied by the correlation (Sh ~ Sc^(1/3))."""
        a = air(0.5 * (T_s + amb.T_inf), amb.P)
        Re = self.U * self.length / float(a.nu)
        Sh = sh_flat_plate(Re, float(a.nu) / D)
        return self.multiplier * Sh * D / self.length, 2.0 / 3.0


@dataclass(frozen=True)
class FreeConvectionSurface:
    """Free convection over an upward-facing horizontal surface, driven by the gas
    density difference (composition and temperature). Ethanol vapour is heavier
    than air and the surface is cold, so the layer is usually STABLE.

    below_range: what to do when Ra_L < 1e4, below the correlation's data
      "raise"       - stop (default choice for fitting measured runs)
      "extrapolate" - continue the power law below 1e4, floored at the pure-
                      diffusion disk value Sh_L = 2/pi. Needed when the layer
                      passes through neutral buoyancy (aqueous ethanol losing its
                      ethanol); results in that range are model extrapolations.
    """

    L_char: float  # m, A / P
    multiplier: float
    below_range: str

    def kc(self, T_s, rho_s, amb, D):
        a = air(0.5 * (T_s + amb.T_inf), amb.P)
        rho_inf = amb.rho
        d_rho = rho_s - rho_inf
        Ra = G * abs(d_rho) / (0.5 * (rho_s + rho_inf)) * self.L_char**3 / (float(a.nu) * D)
        unstable = bool(d_rho < 0)
        if Ra >= 1.0e4 or self.below_range == "raise":
            Sh = nu_horizontal_plate(Ra, float(a.nu) / D, unstable=unstable)
            # Sh ~ Ra^b with Ra ~ 1/D  =>  k_c ~ D^(1 - b)
            b = 1 / 5 if not unstable else (1 / 4 if Ra <= 1.0e7 else 1 / 3)
        elif self.below_range == "extrapolate":
            Sh_pl, b = (0.54 * Ra**0.25, 1 / 4) if unstable else (0.52 * Ra**0.2, 1 / 5)
            Sh_diff = 2.0 / np.pi  # disk into still air, based on L = R/2
            Sh, b = (Sh_pl, b) if Sh_pl > Sh_diff else (Sh_diff, 0.0)
        else:
            raise ValueError(f"below_range must be 'raise' or 'extrapolate', got {self.below_range!r}")
        return self.multiplier * Sh * D / self.L_char, 1.0 - b


@dataclass(frozen=True)
class DiffusionLimitDisk:
    """Steady diffusion from a disk into still, unbounded air: m_dot = 4 R D Delta c.
    k_c = 4 D / (pi R); the lower bound when there is no convection at all."""

    radius: float

    def kc(self, T_s, rho_s, amb, D):
        return 4.0 * D / (np.pi * self.radius), 1.0


# ------------------------------------------------------------- surface state
@dataclass(frozen=True)
class SurfaceState:
    x_E: float
    y_E: float
    y_W: float
    N_E: float  # mol/(m^2 s), positive = evaporation
    N_W: float
    h_eff: float  # W/(m^2 K)
    kc_E: float  # m/s, flush-surface value (before recess)
    recess: float  # m


def surface_state(n_E, n_W, T_s, V0, dish, amb, model, diff, activity, water_exchange,
                  recess_model):
    """Fluxes and coefficients at the liquid surface for the current state."""
    if not water_exchange:
        n_W = 0.0  # pure-ethanol model: n_W is not a state (checked at the start)
    # Newton iterates of the implicit integrator may step slightly below zero for a
    # species that is running out; fluxes are evaluated at the clipped amount and
    # the ACCEPTED solution is checked for negativity in simulate_evaporation.
    n_E, n_W = max(n_E, 0.0), max(n_W, 0.0)
    n_tot = n_E + n_W
    if n_tot > 0.0:
        x_E = n_E / n_tot
    else:  # a Newton iterate past complete drying; composition of the last liquid
        x_E = 1.0 if not water_exchange else 0.0
    g_E, g_W = activity.gamma(x_E, T_s)
    y_E = x_E * g_E * ETHANOL.psat(T_s) / amb.P
    y_W = (1.0 - x_E) * g_W * WATER.psat(T_s) / amb.P
    y_W_inf = amb.y_water if water_exchange else 0.0
    if not water_exchange:
        y_W = 0.0
    if y_E + y_W >= 1.0:
        raise ValueError("surface vapour mole fraction >= 1: liquid would boil")

    rho_s = amb.P * (y_E * ETHANOL.M + y_W * WATER.M + (1 - y_E - y_W) * M_AIR) / (R * T_s)
    T_f = 0.5 * (T_s + amb.T_inf)
    D_E, D_W = diff.ethanol(T_f), diff.water(T_f)
    # The correlation is evaluated for ethanol only (Sc = 1.26, inside every
    # correlation's range). Water (Sc = 0.61, below the Pr >= 0.7 bound of the
    # free-convection correlations) is scaled with the D-exponent the same
    # correlation implies, instead of evaluating it out of range.
    kc_E, n_D = model.kc(T_s, rho_s, amb, D_E)
    kc_W = kc_E * (D_W / D_E) ** n_D

    a = air(T_f, amb.P)
    h = h_from_kc_chilton_colburn(kc_E, float(a.rho), float(a.cp), float(a.alpha) / D_E)

    V = n_E * ETHANOL.M / ETHANOL.rho(T_s) + n_W * WATER.M / WATER.rho(T_s)  # ideal mixing
    if recess_model == "stagnant_layer":
        h_r = dish.recess0 + max(V0 - V, 0.0) / dish.area
    elif recess_model == "none":
        h_r = 0.0
    else:
        raise ValueError(f"recess_model must be 'none' or 'stagnant_layer', got {recess_model!r}")
    kE_eff = 1.0 / (1.0 / kc_E + h_r / D_E)
    kW_eff = 1.0 / (1.0 / kc_W + h_r / D_W)
    h_eff = 1.0 / (1.0 / h + h_r / float(a.k))

    ys_tot, yi_tot = y_E + y_W, y_W_inf
    d = ys_tot - yi_tot
    phi = log_factor(ys_tot, yi_tot) / d if abs(d) > 1e-12 else 1.0 / (1.0 - ys_tot)
    c_f = amb.P / (R * T_f)
    N_E = kE_eff * c_f * y_E * phi
    N_W = kW_eff * c_f * (y_W - y_W_inf) * phi if water_exchange else 0.0
    return SurfaceState(x_E, y_E, y_W, N_E, N_W, h_eff, kc_E, h_r)


@dataclass(frozen=True)
class EvaporationResult:
    t: np.ndarray  # s
    n_E: np.ndarray  # mol
    n_W: np.ndarray  # mol
    T_s: np.ndarray  # K
    mass: np.ndarray  # kg of liquid
    mdot: np.ndarray  # kg/s, net loss rate
    x_E: np.ndarray
    recess: np.ndarray  # m
    stopped_at: float | None  # s, time the stop fraction was reached (None if not)


def simulate_evaporation(t_eval, n_E0, n_W0, T_s0, dish, amb, model, diff, activity,
                         water_exchange, isothermal, recess_model, stop_fraction):
    """Integrate liquid amounts and surface temperature.

    Stops early when the liquid mass falls to stop_fraction * initial mass.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    m0 = n_E0 * ETHANOL.M + n_W0 * WATER.M
    V0 = n_E0 * ETHANOL.M / ETHANOL.rho(T_s0) + n_W0 * WATER.M / WATER.rho(T_s0)
    A = dish.area
    if isothermal and T_s0 != amb.T_inf:
        raise ValueError("isothermal model requires T_s0 = T_inf")
    if not water_exchange and n_W0 != 0.0:
        raise ValueError("water_exchange=False requires a pure-ethanol liquid (n_W0 = 0)")

    def state(y):
        return surface_state(y[0], y[1], y[2], V0, dish, amb, model, diff, activity,
                             water_exchange, recess_model)

    def rhs(t, y):
        n_E, n_W, T_s = y
        s = state(y)
        dn_E, dn_W = -A * s.N_E, -A * s.N_W
        if isothermal:
            return [dn_E, dn_W, 0.0]
        C = n_E * ETHANOL.M * ETHANOL.cp + n_W * WATER.M * WATER.cp + dish.C_dish
        q = ((s.h_eff * A + dish.UA_bottom) * (amb.T_inf - T_s)
             + dish.eps * SIGMA * A * (amb.T_sur**4 - T_s**4)
             - A * (s.N_E * ETHANOL.dh_vap_molar(T_s) + s.N_W * WATER.dh_vap_molar(T_s)))
        return [dn_E, dn_W, q / C]

    def done(t, y):
        return y[0] * ETHANOL.M + y[1] * WATER.M - stop_fraction * m0

    done.terminal = True
    done.direction = -1
    sol = integrate.solve_ivp(rhs, (t_eval[0], t_eval[-1]), [n_E0, n_W0, T_s0], method="Radau",
                              t_eval=t_eval, events=done, rtol=1e-8, atol=[1e-10, 1e-10, 1e-8])
    if sol.status == -1:
        raise RuntimeError(sol.message)
    n_E, n_W, T_s = sol.y
    tol = 1e-8 * (n_E0 + n_W0)
    if n_E.min() < -tol or n_W.min() < -tol:
        raise RuntimeError("integration produced a negative liquid amount; tighten tolerances")
    n_E, n_W = np.maximum(n_E, 0.0), np.maximum(n_W, 0.0)
    states = [state(y) for y in sol.y.T]
    mdot = np.array([A * (s.N_E * ETHANOL.M + s.N_W * WATER.M) for s in states])
    stopped = float(sol.t_events[0][0]) if sol.t_events[0].size else None
    return EvaporationResult(sol.t, n_E, n_W, T_s, n_E * ETHANOL.M + n_W * WATER.M, mdot,
                             np.array([s.x_E for s in states]),
                             np.array([s.recess for s in states]), stopped)


def wet_bulb_temperature(dish, amb, model, diff, activity_pure, recess_model):
    """Steady surface temperature of evaporating PURE ethanol (dT_s/dt = 0) at the
    initial recess depth dish.recess0. With UA_bottom = 0 and eps = 0 this is the
    psychrometric wet-bulb temperature of ethanol in dry air."""
    n_dummy = 1.0  # surface fluxes do not depend on the amount of a pure liquid
    V0 = n_dummy * ETHANOL.M / ETHANOL.rho(amb.T_inf)
    A = dish.area

    def f(T_s):
        s = surface_state(n_dummy, 0.0, T_s, V0, dish, amb, model, diff, activity_pure,
                          water_exchange=False, recess_model=recess_model)
        return ((s.h_eff * A + dish.UA_bottom) * (amb.T_inf - T_s)
                + dish.eps * SIGMA * A * (amb.T_sur**4 - T_s**4)
                - A * s.N_E * ETHANOL.dh_vap_molar(T_s))

    # lower bracket at 0.5 degC: the liquid-density fit starts at 0 degC; if the
    # wet-bulb temperature were lower, brentq raises (no sign change) instead of
    # extrapolating.
    return optimize.brentq(f, T0 + 0.5, amb.T_inf, xtol=1e-9)


def kc_from_mass_rate(mdot, A, T_s, amb):
    """Effective k_c [m/s] of pure ethanol from a measured loss rate mdot [kg/s] and
    the measured surface temperature T_s [K] (dry-ambient ethanol, y_inf = 0)."""
    y_s = ETHANOL.psat(T_s) / amb.P
    c_f = amb.P / (R * 0.5 * (T_s + amb.T_inf))
    return mdot / (ETHANOL.M * A * c_f * log_factor(y_s, 0.0))

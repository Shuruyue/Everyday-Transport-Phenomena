"""Turn config entries (lab units) into model objects (SI). All conversions are here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import c_to_k
from .diffusivity import d_fuller
from .heat import fin
from .heat.fin_fv import uniform_section
from .mass.evaporation import Ambient, Diffusivities
from .properties import MATERIALS

CM, MM, G = 1e-2, 1e-3, 1e-3  # m per cm, m per mm, kg per g


@dataclass(frozen=True)
class RodSpec:
    name: str
    material: object  # properties.Solid
    Ac: float  # m^2
    P: float  # m
    total_length: float  # m
    mass: float  # kg
    eps: float
    eps_range: tuple[float, float]
    k_rel_unc: float
    diameter: float  # m, hydraulic-equivalent diameter used by cylinder correlations

    @property
    def C(self):
        """Thermal capacitance of the whole rod [J/K] from the measured mass."""
        return self.mass * self.material.cp

    @property
    def A_surface(self):
        """Total surface [m^2]: lateral plus both ends."""
        return self.P * self.total_length + 2 * self.Ac

    @property
    def density_check(self):
        """Measured density / handbook density (should be ~1)."""
        return self.mass / (self.Ac * self.total_length) / self.material.rho

    def section_functions(self):
        return uniform_section(self.Ac, self.P)


def rod_spec(cfg_rod):
    mat = MATERIALS[cfg_rod["material"]]
    if cfg_rod["section"] == "circle":
        D = cfg_rod["diameter_mm"] * MM
        Ac, P = fin.pin_section(D)
    elif cfg_rod["section"] == "rectangle":
        w, th = cfg_rod["width_mm"] * MM, cfg_rod["thickness_mm"] * MM
        Ac, P = fin.rectangular_section(w, th)
        D = 4 * Ac / P
    else:
        raise ValueError(f"section must be 'circle' or 'rectangle', got {cfg_rod['section']!r}")
    lo, hi = cfg_rod["emissivity_range"]
    return RodSpec(name=cfg_rod["name"], material=mat, Ac=Ac, P=P,
                   total_length=cfg_rod["total_length_cm"] * CM, mass=cfg_rod["mass_g"] * G,
                   eps=float(cfg_rod["emissivity"]), eps_range=(float(lo), float(hi)),
                   k_rel_unc=float(cfg_rod["k_uncertainty_rel"]), diameter=D)


def environment(cfg_env):
    """(T_inf [K], T_sur [K], P [Pa]) from the [environment] table."""
    return c_to_k(cfg_env["T_inf_C"]), c_to_k(cfg_env["T_sur_C"]), cfg_env["P_kPa"] * 1e3


def ambient(cfg_env):
    T_inf, T_sur, P = environment(cfg_env)
    return Ambient(T_inf=T_inf, T_sur=T_sur, P=P, RH=cfg_env["RH_pct"] / 100.0)


def diffusivities(cfg_diff, P):
    """D(T) callables. source = "fuller", or "measured" with D_ethanol_m2_s at T_ref_C
    (e.g. your Stefan-tube value, measured at the same room pressure), scaled with
    exponent 1.75."""
    D_w = lambda T: float(d_fuller("water", "air", T, P))  # noqa: E731
    if cfg_diff["source"] == "fuller":
        return Diffusivities(lambda T: float(d_fuller("ethanol", "air", T, P)), D_w)
    if cfg_diff["source"] == "measured":
        D_ref, T_ref = cfg_diff["D_ethanol_m2_s"], c_to_k(cfg_diff["T_ref_C"])
        return Diffusivities(lambda T: D_ref * (T / T_ref) ** 1.75, D_w)
    raise ValueError(f"diffusivity source must be 'fuller' or 'measured', got {cfg_diff['source']!r}")


def by_name(items, name):
    found = [it for it in items if it["name"] == name]
    if len(found) != 1:
        raise KeyError(f"expected exactly one entry named {name!r}, found {len(found)}")
    return found[0]


def as_array(x):
    return np.asarray(x, dtype=float)

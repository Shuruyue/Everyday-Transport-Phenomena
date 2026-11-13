"""Liquid-phase activity coefficients for ethanol(1)-water(2).

NRTL with the Aspen Plus / Dortmund Data Bank parameters tabulated in Beneke,
Peters, Glasser & Hildebrandt, Understanding Distillation Using Column Profile
Maps (Wiley 2013), App. B -- the same parameters and the same formula as
Shuruyue/Ethanol-Water-VLE (report2026/calc/vle_core.py), where they reproduce
the Kamihama et al. (2012) 1 atm data to 0.07 K.

Caveat: those parameters were regressed near 351-373 K; evaporating 75 %
ethanol sits at ~280-300 K, so the b_ij/T temperature dependence is an
extrapolation. The report should say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NRTL:
    """tau_ij = a_ij + b_ij / T,  G_ij = exp(-alpha tau_ij)."""

    a12: float
    b12: float  # K
    a21: float
    b21: float  # K
    alpha: float

    def gamma(self, x1, T):
        x1 = np.asarray(x1, dtype=float)
        if np.any(x1 < 0) or np.any(x1 > 1):
            raise ValueError(f"x1 outside [0, 1]: {x1}")
        x2 = 1.0 - x1
        t12, t21 = self.a12 + self.b12 / T, self.a21 + self.b21 / T
        G12, G21 = np.exp(-self.alpha * t12), np.exp(-self.alpha * t21)
        D1, D2 = x1 + x2 * G21, x2 + x1 * G12
        ln1 = x2**2 * (t21 * (G21 / D1) ** 2 + t12 * G12 / D2**2)
        ln2 = x1**2 * (t12 * (G12 / D2) ** 2 + t21 * G21 / D1**2)
        return np.exp(ln1), np.exp(ln2)


ETHANOL_WATER_NRTL = NRTL(a12=-0.801, b12=246.18, a21=3.458, b21=-586.08, alpha=0.3)


@dataclass(frozen=True)
class Ideal:
    """Raoult's law, gamma = 1."""

    def gamma(self, x1, T):
        x1 = np.asarray(x1, dtype=float)
        return np.ones_like(x1), np.ones_like(x1)

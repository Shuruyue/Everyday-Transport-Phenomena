"""Binary gas diffusion coefficients.

Fuller-Schettler-Giddings in the form and with the diffusion volumes of the
course (TP2 Lecture 11-13, after Geankoplis, Transport Processes and Separation
Process Principles):

    D_AB = 1.00e-7 T^1.75 (1/M_A + 1/M_B)^(1/2) / (P [(sum v)_A^(1/3) + (sum v)_B^(1/3)]^2)

with D in m^2/s, T in K, P in atm, M in g/mol.

Chapman-Enskog (Bird, Stewart & Lightfoot Eq. 17.3-12, Neufeld collision
integral) is the independent check.
"""

from __future__ import annotations

import numpy as np

from .constants import ATM

# Atomic and molecular diffusion volumes, course table (Geankoplis Table 6.2-2)
ATOMIC_VOLUME = {"C": 16.5, "H": 1.98, "O": 5.48, "N": 5.69}
MOLECULE_VOLUME = {"air": 20.1, "H2O": 12.7, "N2": 17.9, "O2": 16.6, "CO2": 26.9}

# Molar masses in g/mol for these formulas
MOLAR_MASS_G = {"ethanol": 46.068, "water": 18.015, "air": 28.97}

# Lennard-Jones parameters (sigma [Angstrom], eps/k [K]); Poling, Prausnitz &
# O'Connell, The Properties of Gases and Liquids, 5th ed., Appendix B.
LENNARD_JONES = {"ethanol": (4.530, 362.6), "water": (2.641, 809.1), "air": (3.711, 78.6)}


def diffusion_volume(formula_atoms):
    """Sum of atomic diffusion volumes, e.g. {'C': 2, 'H': 6, 'O': 1} for ethanol."""
    return sum(ATOMIC_VOLUME[el] * n for el, n in formula_atoms.items())


SUM_V = {"ethanol": diffusion_volume({"C": 2, "H": 6, "O": 1}),
         "water": MOLECULE_VOLUME["H2O"], "air": MOLECULE_VOLUME["air"]}


def d_fuller(species_a, species_b, T, P):
    """Fuller D_AB [m^2/s] at T [K], P [Pa]."""
    Ma, Mb = MOLAR_MASS_G[species_a], MOLAR_MASS_G[species_b]
    va, vb = SUM_V[species_a], SUM_V[species_b]
    P_atm = P / ATM
    return 1.00e-7 * np.asarray(T, dtype=float) ** 1.75 * np.sqrt(1 / Ma + 1 / Mb) / (
        P_atm * (va ** (1 / 3) + vb ** (1 / 3)) ** 2
    )


def _omega_d_neufeld(T_star):
    """Collision integral for diffusion, Neufeld et al. (1972)."""
    return (1.06036 / T_star**0.15610 + 0.19300 / np.exp(0.47635 * T_star)
            + 1.03587 / np.exp(1.52996 * T_star) + 1.76474 / np.exp(3.89411 * T_star))


def d_chapman_enskog(species_a, species_b, T, P):
    """Chapman-Enskog D_AB [m^2/s] with Lennard-Jones combining rules."""
    sa, ea = LENNARD_JONES[species_a]
    sb, eb = LENNARD_JONES[species_b]
    sigma_ab = 0.5 * (sa + sb)
    eps_ab = np.sqrt(ea * eb)
    T = np.asarray(T, dtype=float)
    omega = _omega_d_neufeld(T / eps_ab)
    Ma, Mb = MOLAR_MASS_G[species_a], MOLAR_MASS_G[species_b]
    d_cm2 = 0.0018583 * np.sqrt(T**3 * (1 / Ma + 1 / Mb)) / ((P / ATM) * sigma_ab**2 * omega)
    return d_cm2 * 1e-4


def scale_diffusivity(D, T_from, T_to, exponent):
    """Scale D between temperatures at fixed pressure: D ~ T^exponent.

    Fuller implies exponent 1.75; the exponent is a modelling choice, so it is
    a required argument.
    """
    return D * (T_to / T_from) ** exponent

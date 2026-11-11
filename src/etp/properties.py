"""Thermophysical properties of air, ethanol, water and the rod materials.

SI units: T in K, P in Pa, densities in kg/m^3, heat capacities in J/(kg K),
enthalpies in J/kg unless a name says per mole.

Every correlation raises ``ValueError`` outside the range it was fitted or
tabulated over. Nothing here extrapolates silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MMHG, R, T0

# ---------------------------------------------------------------------------
# Dry air at 1 atm
# Incropera, DeWitt, Bergman, Lavine, Fundamentals of Heat and Mass Transfer,
# 7th ed., Table A.4. mu, k and cp are interpolated piecewise as power laws
# (linear in log-log), which follows their ~T^0.7-0.8 dependence far better
# than linear interpolation over 50 K steps. Density comes from the ideal-gas
# law at the actual pressure, so nu, alpha and Pr are computed consistently
# from (rho, mu, k, cp). Note: the tabulated rho (1.1614 kg/m^3 at 300 K) and
# hence nu and alpha in Table A.4 correspond to 1 bar, not 1 atm; at 1 atm nu
# is 1.3 % lower than the table value.
# ---------------------------------------------------------------------------
M_AIR = 28.97e-3  # kg/mol
_AIR_T = np.array([200.0, 250.0, 300.0, 350.0, 400.0, 450.0])  # K
_AIR_MU = np.array([132.5, 159.6, 184.6, 208.2, 230.1, 250.7]) * 1e-7  # Pa s
_AIR_K = np.array([18.1, 22.3, 26.3, 30.0, 33.8, 37.3]) * 1e-3  # W/(m K)
_AIR_CP = np.array([1007.0, 1006.0, 1007.0, 1009.0, 1014.0, 1021.0])  # J/(kg K)
_AIR_PR_TABLE = np.array([0.737, 0.720, 0.707, 0.700, 0.690, 0.686])  # for tests


def _loglog_interp(T, xp, fp):
    T = np.asarray(T, dtype=float)
    if np.any(T < xp[0]) or np.any(T > xp[-1]):
        raise ValueError(f"T = {T} K outside the air table range {xp[0]}-{xp[-1]} K")
    return np.exp(np.interp(np.log(T), np.log(xp), np.log(fp)))


@dataclass(frozen=True)
class AirProperties:
    T: np.ndarray  # K
    rho: np.ndarray  # kg/m^3
    mu: np.ndarray  # Pa s
    nu: np.ndarray  # m^2/s
    k: np.ndarray  # W/(m K)
    cp: np.ndarray  # J/(kg K)
    alpha: np.ndarray  # m^2/s
    Pr: np.ndarray  # -


def air(T, P):
    """Dry-air properties at temperature T [K] and pressure P [Pa].

    mu, k and cp are pressure independent at these conditions; rho follows the
    ideal-gas law, so nu and alpha scale as 1/P.
    """
    T = np.asarray(T, dtype=float)
    if P <= 0:
        raise ValueError(f"P must be positive, got {P}")
    rho = P * M_AIR / (R * T)
    mu = _loglog_interp(T, _AIR_T, _AIR_MU)
    k = _loglog_interp(T, _AIR_T, _AIR_K)
    cp = _loglog_interp(T, _AIR_T, _AIR_CP)
    return AirProperties(T=T, rho=rho, mu=mu, nu=mu / rho, k=k, cp=cp,
                         alpha=k / (rho * cp), Pr=cp * mu / k)


def gas_mixture_density(y_a, M_a, T, P):
    """Density [kg/m^3] of an ideal-gas mixture of species A (mole fraction y_a,
    molar mass M_a [kg/mol]) in air."""
    y_a = np.asarray(y_a, dtype=float)
    if np.any(y_a < 0) or np.any(y_a > 1):
        raise ValueError(f"mole fraction outside [0, 1]: {y_a}")
    return P * (y_a * M_a + (1.0 - y_a) * M_AIR) / (R * T)


# ---------------------------------------------------------------------------
# Antoine equation, log10(P/mmHg) = A - B / (T/degC + C)
# ---------------------------------------------------------------------------
def _antoine(T, A, B, C, t_min_c, t_max_c, name):
    t_c = np.asarray(T, dtype=float) - T0
    if np.any(t_c < t_min_c) or np.any(t_c > t_max_c):
        raise ValueError(f"{name} Antoine set valid {t_min_c} to {t_max_c} degC, got {t_c} degC")
    return 10.0 ** (A - B / (t_c + C)) * MMHG  # Pa


@dataclass(frozen=True)
class Liquid:
    """Pure-liquid data. Temperature-dependent quantities are methods."""

    name: str
    M: float  # kg/mol
    Tc: float  # K, critical temperature (Watson correlation)
    dh_vap_ref: float  # J/mol at T_ref
    T_ref: float  # K
    antoine: tuple[float, float, float, float, float]  # A, B, C, t_min_c, t_max_c
    rho_coeffs: tuple[float, float]  # rho = a + b * (T/degC), kg/m^3
    rho_range_c: tuple[float, float]
    cp: float  # J/(kg K), taken constant over 0-40 degC
    source: str

    def psat(self, T):
        """Saturation pressure [Pa]."""
        A, B, C, lo, hi = self.antoine
        return _antoine(T, A, B, C, lo, hi, self.name)

    def rho(self, T):
        """Liquid density [kg/m^3]."""
        t_c = np.asarray(T, dtype=float) - T0
        lo, hi = self.rho_range_c
        if np.any(t_c < lo) or np.any(t_c > hi):
            raise ValueError(f"{self.name} density fit valid {lo}-{hi} degC, got {t_c}")
        a, b = self.rho_coeffs
        return a + b * t_c

    def dh_vap_molar(self, T):
        """Enthalpy of vaporization [J/mol], Watson correlation (exponent 0.38)."""
        T = np.asarray(T, dtype=float)
        if np.any(T >= self.Tc):
            raise ValueError("T above the critical temperature")
        return self.dh_vap_ref * ((1.0 - T / self.Tc) / (1.0 - self.T_ref / self.Tc)) ** 0.38

    def dh_vap(self, T):
        """Enthalpy of vaporization [J/kg]."""
        return self.dh_vap_molar(T) / self.M


ETHANOL = Liquid(
    name="ethanol",
    M=46.068e-3,
    Tc=513.9,
    dh_vap_ref=42.3e3,
    T_ref=298.15,
    # Common tabulated set (mmHg, degC), range -57 to 80 degC. Same set and
    # range note as Shuruyue/Ethanol-Water-VLE data/parameter_db.json. The NIST
    # Ambrose & Sprake set starts at 292.8 K, too warm for an evaporating
    # surface that cools to ~5 degC.
    antoine=(8.20417, 1642.89, 230.300, -57.0, 80.0),
    # Linear fit to CRC Handbook densities (806.3 at 0 degC, 789.4 at 20 degC,
    # 781.0 at 30 degC), within 0.1 % over 0-40 degC.
    rho_coeffs=(806.3, -0.843),
    rho_range_c=(0.0, 40.0),
    # 112.3 J/(mol K) at 298 K (NIST WebBook); about 5 % lower at 0 degC.
    cp=2440.0,
    source="Antoine: common tabulated set; dH_vap: NIST WebBook 42.3 kJ/mol at 298 K, "
    "Tc = 513.9 K (Poling et al.); rho: CRC Handbook fit; cp: NIST WebBook",
)

WATER = Liquid(
    name="water",
    M=18.015e-3,
    Tc=647.1,
    dh_vap_ref=43.99e3,
    T_ref=298.15,
    antoine=(8.07131, 1730.63, 233.426, -20.0, 100.0),
    # 999.8 at 0 degC to 995.7 at 30 degC; a constant 998 is within 0.3 %.
    rho_coeffs=(998.0, 0.0),
    rho_range_c=(0.0, 40.0),
    cp=4181.0,
    source="Antoine: common tabulated set (mmHg, degC); dH_vap: NIST WebBook 43.99 kJ/mol "
    "at 298 K; Tc = 647.1 K",
)


def water_mole_fraction_from_rh(rh, T, P):
    """Water-vapour mole fraction in air at relative humidity rh (0-1)."""
    if not 0.0 <= rh <= 1.0:
        raise ValueError(f"relative humidity must be a fraction in [0, 1], got {rh}")
    return rh * WATER.psat(T) / P


def dew_point(rh, T, P):
    """Dew-point temperature [K] of air at T [K] and relative humidity rh."""
    from scipy.optimize import brentq

    p_w = rh * WATER.psat(T)
    return brentq(lambda Td: WATER.psat(Td) - p_w, T0 - 20.0 + 1e-9, T)


# ---------------------------------------------------------------------------
# Rod materials. Values at ~300 K. Choose the key in the config after the
# magnet test (304 is barely magnetic, 430 is clearly magnetic).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Solid:
    name: str
    rho: float  # kg/m^3
    cp: float  # J/(kg K)
    k: float  # W/(m K)
    source: str

    @property
    def alpha(self):
        return self.k / (self.rho * self.cp)


MATERIALS = {
    "AISI_304": Solid("AISI 304 stainless steel", 7900.0, 477.0, 14.9,
                      "Incropera 7th ed. Table A.1, 300 K"),
    "AISI_430": Solid("AISI 430 ferritic stainless steel", 7740.0, 460.0, 26.1,
                      "typical datasheet values (ASM / AK Steel 430), k at 100 degC"),
    "aluminum_pure": Solid("pure aluminum", 2702.0, 903.0, 237.0,
                           "Incropera 7th ed. Table A.1, 300 K"),
    "aluminum_6061_T6": Solid("aluminum alloy 6061-T6", 2700.0, 896.0, 167.0,
                              "ASM Handbook typical values"),
    "copper_pure": Solid("pure copper", 8933.0, 385.0, 401.0,
                         "Incropera 7th ed. Table A.1, 300 K"),
}

"""Physical constants (CODATA values via scipy.constants)."""

from scipy import constants as _c

R = _c.R  # J/(mol K), molar gas constant
SIGMA = _c.Stefan_Boltzmann  # W/(m^2 K^4)
G = _c.g  # m/s^2, standard gravity
ATM = _c.atm  # Pa
T0 = _c.zero_Celsius  # K, 0 degC
MMHG = _c.mmHg  # Pa per mmHg


def c_to_k(t_c):
    """degC -> K."""
    return t_c + T0


def k_to_c(t_k):
    """K -> degC."""
    return t_k - T0

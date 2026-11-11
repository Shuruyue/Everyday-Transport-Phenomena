import numpy as np
import pytest
from scipy import optimize

from etp.constants import ATM, R, c_to_k
from etp.diffusivity import d_fuller
from etp.mass import evaporation as ev
from etp.mass.activity import ETHANOL_WATER_NRTL, Ideal
from etp.mass.stefan import log_factor
from etp.properties import ETHANOL, WATER

AMB_DRY = ev.Ambient(c_to_k(25.0), c_to_k(25.0), ATM, 0.0)
AMB_HUMID = ev.Ambient(c_to_k(25.0), c_to_k(25.0), ATM, 0.7)
DIFF = ev.Diffusivities(lambda T: float(d_fuller("ethanol", "air", T, ATM)),
                        lambda T: float(d_fuller("water", "air", T, ATM)))
DISH = ev.Dish(radius=0.045, recess0=0.0, C_dish=2.0, UA_bottom=0.0, eps=0.0)


def n_ethanol(grams):
    return grams * 1e-3 / ETHANOL.M


def test_isothermal_constant_kc_gives_linear_mass_loss():
    model = ev.DiffusionLimitDisk(DISH.radius)
    t = np.linspace(0, 3600, 61)
    r = ev.simulate_evaporation(t, n_ethanol(10.0), 0.0, AMB_DRY.T_inf, DISH, AMB_DRY, model,
                                DIFF, Ideal(), water_exchange=False, isothermal=True,
                                recess_model="none", stop_fraction=0.01)
    T = AMB_DRY.T_inf
    D = DIFF.ethanol(T)
    y_s = ETHANOL.psat(T) / ATM
    rate = ETHANOL.M * DISH.area * (4 * D / (np.pi * DISH.radius)) * ATM / (R * T) * log_factor(y_s, 0)
    np.testing.assert_allclose(r.mass, 10e-3 - rate * t, rtol=1e-7)


def test_surface_relaxes_to_wet_bulb_temperature():
    model = ev.ForcedFlatPlate(U=2.0, length=0.09, multiplier=1.0)
    Tw = ev.wet_bulb_temperature(DISH, AMB_DRY, model, DIFF, Ideal(), "none")
    t = np.linspace(0, 900, 91)
    r = ev.simulate_evaporation(t, n_ethanol(5.0), 0.0, AMB_DRY.T_inf, DISH, AMB_DRY, model, DIFF,
                                Ideal(), water_exchange=False, isothermal=False,
                                recess_model="none", stop_fraction=0.01)
    assert r.T_s[-1] == pytest.approx(Tw, abs=0.02)
    assert 3.0 < Tw - 273.15 < 9.0  # ethanol wet bulb in dry 25 degC air, 2 m/s


def test_ethanol_vapour_layer_is_stable():
    model = ev.FreeConvectionSurface(L_char=0.0225, multiplier=1.0, below_range="raise")
    y_s = ETHANOL.psat(c_to_k(12.0)) / ATM
    rho_s = ATM * (y_s * ETHANOL.M + (1 - y_s) * 28.97e-3) / (R * c_to_k(12.0))
    kc, n = model.kc(c_to_k(12.0), rho_s, AMB_DRY, 1.2e-5)
    assert rho_s > AMB_DRY.rho and n == pytest.approx(0.8)  # stable branch: Sh ~ Ra^(1/5)


def test_humid_air_condenses_into_cold_ethanol():
    model = ev.ForcedFlatPlate(U=2.0, length=0.09, multiplier=1.0)
    t = np.linspace(0, 1200, 41)
    kw = dict(water_exchange=True, isothermal=False, recess_model="none", stop_fraction=0.01)
    wet = ev.simulate_evaporation(t, n_ethanol(10.0), 0.0, AMB_HUMID.T_inf, DISH, AMB_HUMID,
                                  model, DIFF, ETHANOL_WATER_NRTL, **kw)
    dry = ev.simulate_evaporation(t, n_ethanol(10.0), 0.0, AMB_DRY.T_inf, DISH, AMB_DRY,
                                  model, DIFF, ETHANOL_WATER_NRTL, **kw)
    assert wet.n_W[-1] > 0.0 and np.all(np.diff(wet.n_W) >= 0)
    assert abs(dry.n_W[-1]) < 1e-15  # dry air: no water (round-off only)


def test_nrtl_azeotrope_matches_ethanol_water_vle_repo():
    # Shuruyue/Ethanol-Water-VLE (src/, same Antoine sets): x1 = 0.897, 78.17 degC
    def bubble(x1):
        g1, g2 = ETHANOL_WATER_NRTL.gamma(x1, 351.0)
        T = optimize.brentq(lambda T: x1 * ETHANOL_WATER_NRTL.gamma(x1, T)[0] * ETHANOL.psat(T)
                            + (1 - x1) * ETHANOL_WATER_NRTL.gamma(x1, T)[1] * WATER.psat(T) - ATM,
                            340.0, 353.0)
        g1, _ = ETHANOL_WATER_NRTL.gamma(x1, T)
        return T, x1 * g1 * ETHANOL.psat(T) / ATM

    x_az = optimize.brentq(lambda x: bubble(x)[1] - x, 0.80, 0.97)
    assert x_az == pytest.approx(0.897, abs=0.004)
    assert bubble(x_az)[0] - 273.15 == pytest.approx(78.17, abs=0.1)

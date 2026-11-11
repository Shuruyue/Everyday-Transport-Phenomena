"""Inverse problems for Week 2: loss rates, k_c and model parameters from data."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..fitting import fit_linear, fit_nonlinear
from .evaporation import simulate_evaporation


def mass_rate(t, m, t_start, t_end):
    """Loss rate -dm/dt [kg/s] from a straight-line fit of m(t) over [t_start, t_end].

    The window is an analysis choice (early, nearly constant-rate period) and must
    be reported. Returns (rate, rate_ci95, LinearFit).
    """
    t = np.asarray(t, dtype=float)
    m = np.asarray(m, dtype=float)
    sel = (t >= t_start) & (t <= t_end)
    if sel.sum() < 4:
        raise ValueError(f"only {sel.sum()} points in the window [{t_start}, {t_end}] s")
    lf = fit_linear(t[sel], m[sel], through_origin=False)
    return -lf.slope, lf.slope_ci95, lf


def fit_evaporation(t, m, T_s, sigma_m, sigma_T, n_E0, n_W0, dish, amb, model, diff, activity,
                    water_exchange, recess_model, fit, p0):
    """Fit model parameters to the measured liquid mass m(t) [kg] and, if given,
    surface temperature T_s(t) [K], weighted by their measurement standard
    deviations sigma_m [kg] and sigma_T [K].

    fit: subset of ("multiplier", "UA_bottom"); multiplier scales the k_c
    correlation (1 = correlation exact), UA_bottom is the liquid-to-ambient
    conductance through the dish bottom [W/K]. Parameters not fitted keep the
    values already in ``model`` and ``dish``.
    """
    names = tuple(fit)
    if not set(names) <= {"multiplier", "UA_bottom"}:
        raise ValueError(f"unknown fit parameters {names}")
    t = np.asarray(t, dtype=float)
    m = np.asarray(m, dtype=float)
    use_T = T_s is not None

    def run(p):
        d = dict(zip(names, p))
        mdl = replace(model, multiplier=d["multiplier"]) if "multiplier" in d else model
        dsh = replace(dish, UA_bottom=d["UA_bottom"]) if "UA_bottom" in d else dish
        r = simulate_evaporation(t - t[0], n_E0, n_W0, T_s[0] if use_T else amb.T_inf, dsh, amb,
                                 mdl, diff, activity, water_exchange, isothermal=False,
                                 recess_model=recess_model, stop_fraction=1e-3)
        mm, TT = r.mass, r.T_s
        if mm.size < t.size:  # liquid ran out before the last data point
            mm = np.concatenate((mm, np.full(t.size - mm.size, mm[-1])))
            TT = np.concatenate((TT, np.full(t.size - TT.size, TT[-1])))
        return mm, TT

    def resid(p):
        mm, TT = run(p)
        r = (mm - m) / sigma_m
        return np.concatenate((r, (TT - T_s) / sigma_T)) if use_T else r

    lo = {"multiplier": 0.05, "UA_bottom": 0.0}
    hi = {"multiplier": 20.0, "UA_bottom": 10.0}
    scale = {"multiplier": 0.1, "UA_bottom": 0.01}
    res = fit_nonlinear(resid, p0, names, bounds=([lo[n] for n in names], [hi[n] for n in names]),
                        x_scale=[scale[n] for n in names], diff_step=1e-4)
    return res, run(res.values)

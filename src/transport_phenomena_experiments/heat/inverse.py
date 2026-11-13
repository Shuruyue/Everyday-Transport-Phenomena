"""Inverse problems for Week 1: recover h, k (and their uncertainty) from data."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..correlations import h_forced_cylinder
from ..fitting import fit_nonlinear
from . import fin
from .fin_fv import Rod, SurfaceLoss
from .lumped import cool_lumped, lumped_analytic


def fit_steady_profile(x, T, T_b, T_inf, Ac, P, L, m0):
    """Fit the fin parameter m to a steady temperature profile.

    Model: corrected-length adiabatic tip, theta = theta_b cosh m(L_c - x)/cosh m L_c.
    Only the combination m^2 = hP/(kA_c) is identifiable from a steady profile;
    h follows only after k is fixed. Returns (FitResult, h_over_k) with
    h/k = m^2 A_c / P [1/m].
    """
    x = np.asarray(x, dtype=float)
    theta = np.asarray(T, dtype=float) - T_inf
    theta_b = T_b - T_inf
    Lc = fin.corrected_length(L, Ac, P)

    def resid(p):
        return theta_b * fin.theta_adiabatic_tip(x, Lc, p[0]) - theta

    res = fit_nonlinear(resid, [m0], ["m"], bounds=([1e-3], [1e4]), x_scale=[m0])
    return res, res["m"] ** 2 * Ac / P


def fit_transient_rod(t, x_sensors, T_data, rod_template, T_b_fn, T_init, T_inf, h_model, p0,
                      n_cells):
    """Fit the lateral heat-loss model and k to transient sensor data with the
    finite-volume rod. rho*c of rod_template is held fixed; the transient
    carries alpha = k/(rho c), which is what separates h from k.

    h_model:
      {"kind": "constant"}                         -> parameters [h_eff, k]
          one constant coefficient lumping convection and radiation
      {"kind": "correlation", "h_corr": f(T), "eps": e, "T_sur": Ts}
                                                   -> parameters [C_h, k]
          h_conv = C_h * h_corr(T) plus radiation with emissivity e
    Comparing the k from both variants shows the model-form uncertainty, which
    the statistical CI of either fit does not contain.

    T_data: shape (len(t), len(x_sensors)) in K.
    """
    t = np.asarray(t, dtype=float)
    T_data = np.asarray(T_data, dtype=float)
    assert T_data.shape == (t.size, len(x_sensors))
    kind = h_model["kind"]
    if kind == "constant":
        names = ("h_eff", "k")
    elif kind == "correlation":
        names = ("C_h", "k")
    else:
        raise ValueError(f"h_model kind must be 'constant' or 'correlation', got {kind!r}")

    def loss_for(p0_):
        if kind == "constant":
            return SurfaceLoss(lambda T: np.full_like(T, p0_), T_inf, 0.0, T_inf)
        return SurfaceLoss(lambda T: p0_ * h_model["h_corr"](T), T_inf, h_model["eps"],
                           h_model["T_sur"])

    def model(p):
        rod = Rod(L=rod_template.L, n_cells=n_cells, Ac=rod_template.Ac, P=rod_template.P,
                  k=p[1], rho_c=rod_template.rho_c, tip=rod_template.tip)
        T_nodes = rod.transient(t, T_b_fn, np.interp(rod.x, rod_template.x, T_init), loss_for(p[0]),
                                rtol=1e-9, atol=1e-9)
        return np.array([np.interp(x_sensors, rod.x, Tn) for Tn in T_nodes])

    def resid(p):
        return (model(p) - T_data).ravel()

    res = fit_nonlinear(resid, p0, names, bounds=([1e-3, 0.1], [1e3, 1e3]), x_scale=p0,
                        diff_step=1e-4)
    return res, model(res.values)


def fit_cooling_exponential(t, T, T_inf, tau0):
    """Fit T = T_inf + (T_i - T_inf) exp(-t / tau): parameters [T_i, tau]."""
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)

    def resid(p):
        return lumped_analytic(t, p[0], T_inf, p[1]) - T

    return fit_nonlinear(resid, [T[0], tau0], ["T_i", "tau"],
                         bounds=([T_inf, 1e-3], [T_inf + 500, 1e6]), x_scale=[1.0, tau0])


def fit_cooling_physical(t, T, C, A_s, h_corr, T_inf, T_sur, fit, fixed):
    """Fit the nonlinear lumped model with h_conv = C_h * h_corr(T) plus radiation.

    fit:   names to estimate, a subset of ("T_i", "C_h", "eps")
    fixed: values for the rest, e.g. {"eps": 0.95}
    C_h = 1 means the correlation is exact; the fitted C_h is the
    experiment/correlation ratio with its confidence interval.
    """
    names = tuple(fit)
    allowed = {"T_i", "C_h", "eps"}
    if not set(names) <= allowed or set(names) & set(fixed) or set(names) | set(fixed) != allowed:
        raise ValueError(f"fit {names} and fixed {sorted(fixed)} must partition {allowed}")
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)

    def unpack(p):
        d = dict(fixed)
        d.update(zip(names, p))
        return d

    def model(p):
        d = unpack(p)
        loss = SurfaceLoss(h_conv=lambda Ts: d["C_h"] * h_corr(Ts), T_inf=T_inf, eps=d["eps"],
                           T_sur=T_sur)
        return cool_lumped(t - t[0], d["T_i"], C, A_s, loss)

    p0 = {"T_i": T[0], "C_h": 1.0, "eps": 0.5}
    lo = {"T_i": T_inf, "C_h": 0.05, "eps": 0.0}
    hi = {"T_i": T_inf + 500, "C_h": 20.0, "eps": 1.0}
    scale = {"T_i": 1.0, "C_h": 0.1, "eps": 0.1}
    res = fit_nonlinear(lambda p: model(p) - T, [p0[n] for n in names], names,
                        bounds=([lo[n] for n in names], [hi[n] for n in names]),
                        x_scale=[scale[n] for n in names], diff_step=1e-4)
    return res, model(res.values)


def velocity_from_h(h_conv, D, T_s, T_inf, P, correlation, U_bracket):
    """Invert a cross-flow correlation: the air speed U [m/s] that gives h_conv.

    h_conv must be the CONVECTIVE part only (radiation removed).
    """
    f = lambda U: h_forced_cylinder(D, U, T_s, T_inf, P, correlation) - h_conv  # noqa: E731
    lo, hi = U_bracket
    if f(lo) * f(hi) > 0:
        raise ValueError(f"h_conv = {h_conv:.3g} not reachable for U in {U_bracket} m/s")
    return optimize.brentq(f, lo, hi, xtol=1e-10)

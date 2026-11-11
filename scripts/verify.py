"""Verification figures: every solver against an exact solution or a conservation law.

    python scripts/verify.py            -> figures/verification/*.png + verification.json

These figures use no experimental data and can go straight into the report
appendix ("how do we know the code is right").
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from etp.constants import ATM, R, c_to_k
from etp.diffusivity import d_fuller
from etp.heat import fin
from etp.heat.fin_fv import Rod, SurfaceLoss, uniform_section
from etp.mass import evaporation as ev
from etp.mass import stefan
from etp.mass.activity import Ideal
from etp.mass.diffusion2d import series_resistance_estimate, solve_dish
from etp.plotting import COLORS, save
from etp.properties import ETHANOL, MATERIALS

OUT = Path("figures/verification")
summary = {}

# ------------------------------------------------------------- fin solver
mat = MATERIALS["AISI_304"]
D, L, h = 4e-3, 0.15, 15.0
T_inf, T_b = 298.15, 373.15
Ac, P = fin.pin_section(D)
m = fin.fin_parameter(h, P, mat.k, Ac)
a, p = uniform_section(Ac, P)
loss = SurfaceLoss(lambda T: np.full_like(T, h), T_inf, 0.0, T_inf)
Ns = np.array([10, 20, 40, 80, 160, 320])
err_T, err_q, err_tr = [], [], []
x_chk = np.linspace(0, L, 9)
t_chk = np.array([10.0, 30.0, 90.0, 300.0])
exact_tr = fin.theta_transient_adiabatic_tip(x_chk, t_chk, L, m, mat.alpha, n_terms=800)
q_ex = fin.q_base_adiabatic_tip(h, P, mat.k, Ac, L, T_b - T_inf)
for N in Ns:
    rod = Rod(L, int(N), a, p, mat.k, mat.rho * mat.cp, "adiabatic")
    T = rod.steady(T_b, loss)
    err_T.append(np.abs(T - (T_inf + 75 * fin.theta_adiabatic_tip(rod.x, L, m))).max())
    err_q.append(abs(rod.base_heat_rate(T, loss) / q_ex - 1))
    if N <= 160:
        T0 = np.full(N + 1, T_inf)
        T0[0] = T_b
        Tn = rod.transient(np.concatenate(([0.0], t_chk)), lambda _: T_b, T0, loss,
                           rtol=1e-11, atol=1e-11)[1:]
        num = (np.array([np.interp(x_chk, rod.x, r) for r in Tn]) - T_inf) / 75
        err_tr.append(np.abs(num - exact_tr).max())
order = lambda e: np.polyfit(np.log(Ns[: len(e)]), np.log(e), 1)[0]  # noqa: E731
summary["fin_steady_order"] = -order(err_T)
summary["fin_heat_rate_order"] = -order(err_q)
summary["fin_transient_order"] = -order(err_tr)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(L / Ns, err_T, "o-", label=f"steady T, order {summary['fin_steady_order']:.2f}")
ax[0].loglog(L / Ns, np.array(err_q) * 75, "s-",
             label=f"base heat rate (x75 K), order {summary['fin_heat_rate_order']:.2f}")
ax[0].loglog(L / Ns[: len(err_tr)], np.array(err_tr) * 75, "^-",
             label=f"transient T, order {summary['fin_transient_order']:.2f}")
ref = L / Ns
ax[0].loglog(ref, err_T[0] * (ref / ref[0]) ** 2, "k:", label="slope 2")
ax[0].set(xlabel="grid spacing dx [m]", ylabel="max error [K]",
          title="Finite-volume rod vs exact fin solutions")
ax[0].legend(fontsize=8)
rod = Rod(L, 80, a, p, mat.k, mat.rho * mat.cp, "adiabatic")
T0 = np.full(81, T_inf)
T0[0] = T_b
tt = np.array([0.0, 10, 30, 90, 300, 1200])
Tn = rod.transient(tt, lambda _: T_b, T0, loss)
xx = np.linspace(0, L, 300)
ex = fin.theta_transient_adiabatic_tip(xx, tt[1:], L, m, mat.alpha, n_terms=800)
for i, ti in enumerate(tt[1:]):
    ax[1].plot(xx * 100, ex[i], color=plt.cm.viridis(i / 5), label=f"t = {ti:g} s")
    ax[1].plot(rod.x[::4] * 100, (Tn[i + 1, ::4] - T_inf) / 75, "o", ms=3,
               color=plt.cm.viridis(i / 5))
ax[1].set(xlabel="x above the water surface [cm]", ylabel=r"$\theta/\theta_b$",
          title="Transient: series solution (lines) vs FV (dots)")
ax[1].legend(fontsize=8)
save(fig, OUT / "fin_solver.png", synthetic=False)

# ---------------------------------------------------------- Stefan tube
T = 298.15
D_e = float(d_fuller("ethanol", "air", T, ATM))
c, cL, ys = ATM / (R * T), ETHANOL.rho(T) / ETHANOL.M, ETHANOL.psat(T) / ATM
t = np.linspace(0, 48 * 3600, 289)
Z, Nm = stefan.moving_boundary(t, 0.03, D_e, c, cL, ys, 0.0, n_cells=60)
K = stefan.stefan_slope(c, cL, D_e, ys, 0.0)
Zq = np.sqrt(0.03**2 + K * t)
summary["stefan_offset_um"] = float(np.median((Z - Zq)[t > 3600]) * 1e6)
summary["stefan_offset_estimate_um"] = float(c / cL * 0.03 * ys / 3 * 1e6)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(t / 3600, (Zq - 0.03) * 1e3, color=COLORS["model"], lw=3, alpha=0.4,
           label="pseudo-steady: $z^2 - z_0^2 = Kt$")
ax[0].plot(t / 3600, (Z - 0.03) * 1e3, "k--", label="transient PDE, moving interface")
ax[0].set(xlabel="t [h]", ylabel="level drop $z - z_0$ [mm]", title="Stefan tube, ethanol, 25 °C")
ax[0].legend(fontsize=8)
ax[1].plot(t[1:] / 3600, (Z - Zq)[1:] * 1e6)
ax[1].axhline(summary["stefan_offset_estimate_um"], color="k", ls=":",
              label="$(c/c_L) z_0 y_s/3$: vapour stored in the column\nminus the diffusion time-lag deficit")
ax[1].set(xlabel="t [h]", ylabel="transient minus pseudo-steady [µm]",
          title="Difference: ~2 µm out of a 12 mm drop")
ax[1].legend(fontsize=8)
save(fig, OUT / "stefan_tube.png", synthetic=False)

# --------------------------------------------------- 2D dish diffusion
h0s = [0.04, 0.02, 0.01, 0.005, 0.0025]
Qs = [solve_dish(1.0, 0.0, h0, 1.08, 60.0).Q_norm for h0 in h0s]
summary["disk_Q_norm_finest"] = Qs[-1]
hr = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
Qr = np.array([solve_dish(1.0, x, 0.005, 1.08, 60.0).Q_norm for x in hr])
summary["recess_factor_2D"] = dict(zip(map(float, hr), map(float, Qr / Qr[0])))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].semilogx(h0s, Qs, "o-", label="FV, graded grid")
ax[0].axhline(4.0, color="k", ls=":", label="exact (Weber): 4")
ax[0].set(xlabel="smallest cell / R", ylabel=r"$\dot m / (D_{AB} R \Delta c)$",
          title="Disk on a plane: convergence")
ax[0].set_xticks(h0s, [f"{v:g}" for v in h0s])
ax[0].xaxis.set_minor_formatter(plt.NullFormatter())
ax[0].legend(fontsize=8)
ax[1].plot(hr, Qr / Qr[0], "o-", label="2D diffusion")
ax[1].plot(hr, series_resistance_estimate(1.0, hr) / 4.0, "s--", label="1D stagnant-layer estimate")
ax[1].set(xlabel="recess depth $h_r / R$", ylabel="rate / flush rate",
          title="Liquid below the rim: slower")
ax[1].legend(fontsize=8)
res = solve_dish(1.0, 0.2, 0.01, 1.08, 60.0)
rc = 0.5 * (res.r_faces[:-1] + res.r_faces[1:])
zc = 0.5 * (res.z_faces[:-1] + res.z_faces[1:])
sel_r, sel_z = rc < 2.5, zc < 2.0
cs = ax[2].contourf(rc[sel_r], zc[sel_z], res.c[np.ix_(sel_r, sel_z)].T, levels=np.linspace(0, 1, 11),
                    cmap="Blues")
ax[2].plot([0, 1, 1, 2.5], [-0.2, -0.2, 0, 0], "k-", lw=1.5)
fig.colorbar(cs, ax=ax[2], label=r"$c / c_s$")
ax[2].set(xlabel="r / R", ylabel="z / R", title="Vapour field, $h_r = 0.2R$", aspect="equal")
ax[2].grid(False)
save(fig, OUT / "dish_diffusion.png", synthetic=False)

# ------------------------------------------- evaporation: wet bulb + energy
amb = ev.Ambient(c_to_k(25.0), c_to_k(25.0), ATM, 0.0)
diff = ev.Diffusivities(lambda T: float(d_fuller("ethanol", "air", T, ATM)),
                        lambda T: float(d_fuller("water", "air", T, ATM)))
dish = ev.Dish(0.045, 0.0, 2.0, 0.0, 0.0)
fig, ax = plt.subplots(figsize=(6, 4))
for U, col in ((1.0, COLORS["model"]), (3.0, COLORS["alt"])):
    model = ev.ForcedFlatPlate(U, 0.09, 1.0)
    Tw = ev.wet_bulb_temperature(dish, amb, model, diff, Ideal(), "none")
    tt = np.linspace(0, 1200, 241)
    r = ev.simulate_evaporation(tt, 8e-3 / ETHANOL.M, 0.0, amb.T_inf, dish, amb, model, diff,
                                Ideal(), False, False, "none", 1e-3)
    ax.plot(r.t / 60, r.T_s - 273.15, color=col, label=f"U = {U:g} m/s, ODE")
    summary[f"wet_bulb_C_U{U:g}"] = float(Tw - 273.15)
    summary[f"T_s_end_minus_wet_bulb_K_U{U:g}"] = float(r.T_s[-1] - Tw)
ax.axhline(Tw - 273.15, color="k", ls=":",
           label=f"wet-bulb root {Tw - 273.15:.2f} °C (same for any U\nwithout radiation or dish conduction)")
ax.set(xlabel="t [min]", ylabel="liquid temperature [°C]",
       title="Ethanol in dry 25 °C air: ODE relaxes to the wet-bulb root")
ax.legend(fontsize=8)
save(fig, OUT / "evaporative_cooling.png", synthetic=False)

(OUT / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for k, v in summary.items():
    print(f"{k}: {v}")

"""Week 1 analysis: rod heated in boiling water (fin) and cooled in air (lumped).

    python scripts/week1_analysis.py config/week1.toml
    python scripts/week1_analysis.py config/synthetic/week1.toml   # pipeline test

Writes figures, week1_results.json and summary.md to [meta].output_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from etp import build
from etp.constants import c_to_k
from etp.correlations import h_forced_cylinder, h_free_horizontal_cylinder, h_radiation
from etp.heat import fin, inverse
from etp.heat.fin_fv import Rod, SurfaceLoss
from etp.heat.lumped import biot, instantaneous_h
from etp.io import check_time_increasing, load_config, read_table
from etp.plotting import COLORS, save

CP_REL_UNC = 0.03  # handbook specific heat vs the actual alloy
D_ABS_UNC_MM = 0.05  # caliper reading


def heating_run(run, spec, env, an, rng, syn, out, truth):
    T_inf, T_sur, P = env
    n_s = len(run["sensor_positions_cm"])
    cols = ["t_s", "T_base_C"] + [f"T{i + 1}_C" for i in range(n_s)]
    d = read_table(run["csv"], cols)
    t = d["t_s"]
    check_time_increasing(t, run["csv"])
    Tb_series = c_to_k(d["T_base_C"])
    Ts = np.column_stack([c_to_k(d[f"T{i + 1}_C"]) for i in range(n_s)])
    x_s = np.array(run["sensor_positions_cm"]) * build.CM
    L = run["exposed_length_cm"] * build.CM
    k = spec.material.k
    a, p = spec.section_functions()

    # ---- steady profile: fin parameter m, Fourier heat rate
    w0, w1 = run["steady_window_s"]
    win = (t >= w0) & (t <= w1)
    if win.sum() < 3:
        raise ValueError(f"{run['name']}: fewer than 3 readings in the steady window")
    T_ss, T_ss_sd = Ts[win].mean(axis=0), Ts[win].std(axis=0, ddof=1)
    Tb_ss = Tb_series[win].mean()
    m_guess = fin.fin_parameter(15.0, spec.P, k, spec.Ac)
    fit_m, h_over_k = inverse.fit_steady_profile(x_s, T_ss, Tb_ss, T_inf, spec.Ac, spec.P, L, m_guess)
    m = fit_m["m"]
    Lc = fin.corrected_length(L, spec.Ac, spec.P)
    q_fin = k * spec.Ac * m * (Tb_ss - T_inf) * np.tanh(m * Lc)
    q_2pt = -k * spec.Ac * (T_ss[1] - T_ss[0]) / (x_s[1] - x_s[0])

    # ---- a priori model: correlation h(T) + radiation, handbook k (no fitting)
    def loss_apriori(kk, eps, Tinf):
        return SurfaceLoss(lambda T: h_free_horizontal_cylinder(spec.diameter, T, Tinf, P), Tinf,
                           eps, Tinf + (T_sur - T_inf))

    def rod_with(kk):
        return Rod(L, an["n_cells"], a, p, kk, spec.material.rho * spec.material.cp, "convective")

    rod0 = rod_with(k)
    T_pred_ss = np.interp(x_s, rod0.x, rod0.steady(Tb_ss, loss_apriori(k, spec.eps, T_inf)))
    mc = []
    for _ in range(an["monte_carlo_n"]):
        kk = k * (1 + spec.k_rel_unc * rng.standard_normal())
        eps = rng.uniform(*spec.eps_range)
        Ti = T_inf + 0.5 * rng.standard_normal()
        r = rod_with(kk)
        mc.append(np.interp(x_s, r.x, r.steady(Tb_ss, loss_apriori(kk, eps, Ti))))
    mc = np.array(mc)
    band = np.percentile(mc, [5, 95], axis=0)

    # ---- transient fit: h and k separately (alpha = k / rho c carries the time scale)
    # Base boundary condition: the mean of the measured base (boiling-water)
    # temperature over the run; its scatter is reported. Interpolating the raw,
    # noisy base series would force the solver with measurement noise.
    Tb_const = float(Tb_series.mean())
    Tb_fn = lambda tt: Tb_const  # noqa: E731
    T_init = np.full(an["n_cells"] + 1, T_inf)
    T_init[0] = Tb_const
    fit_hk, T_fit = inverse.fit_transient_rod(t, x_s, Ts, rod0, Tb_fn, T_init, T_inf,
                                              {"kind": "constant"}, p0=[h_over_k * k, k],
                                              n_cells=an["n_cells"])
    h_corr = lambda T: h_free_horizontal_cylinder(spec.diameter, T, T_inf, P)  # noqa: E731
    fit_ck, _ = inverse.fit_transient_rod(
        t, x_s, Ts, rod0, Tb_fn, T_init, T_inf,
        {"kind": "correlation", "h_corr": h_corr, "eps": spec.eps, "T_sur": T_sur},
        p0=[1.0, k], n_cells=an["n_cells"])
    T_apriori = rod0.transient(t, Tb_fn, T_init, loss_apriori(k, spec.eps, T_inf))
    T_apriori_s = np.array([np.interp(x_s, rod0.x, row) for row in T_apriori])
    tau1 = fin.slowest_time_constant(L, np.sqrt(fit_hk["h_eff"] * spec.P / (fit_hk["k"] * spec.Ac)),
                                     fit_hk["k"] / (spec.material.rho * spec.material.cp))

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    th_b = Tb_ss - T_inf
    xx = np.linspace(0, L, 200)
    ax[0].errorbar(x_s * 100, (T_ss - T_inf) / th_b, yerr=T_ss_sd / th_b, fmt="o",
                   color=COLORS["data"], label="data (steady window, mean ± sd)")
    ax[0].plot(xx * 100, fin.theta_adiabatic_tip(xx, Lc, m), color=COLORS["model"],
               label=f"fin fit: m = {m:.1f} ± {fit_m.ci('m'):.1f} 1/m")
    ax[0].plot(x_s * 100, (T_pred_ss - T_inf) / th_b, "s--", color=COLORS["alt"], mfc="none",
               label="a priori: correlation h(T) + radiation")
    ax[0].fill_between(x_s * 100, (band[0] - T_inf) / th_b, (band[1] - T_inf) / th_b,
                       color=COLORS["alt"], alpha=0.15, label="a priori 5-95 % (k, ε, T∞)")
    ax[0].set_yscale("log")
    ax[0].set(xlabel="x above the water surface [cm]", ylabel=r"$\theta/\theta_b$",
              title=f"{run['name']}: steady profile")
    ax[0].legend(fontsize=8)
    step = max(1, t.size // 60)
    for i in range(n_s):
        c = plt.cm.viridis(i / max(1, n_s - 1))
        ax[1].plot(t[::step], Ts[::step, i] - 273.15, "o", ms=3, color=c,
                   label=f"x = {x_s[i] * 100:g} cm")
        ax[1].plot(t, T_fit[:, i] - 273.15, "-", color=c, lw=1.2)
        ax[1].plot(t, T_apriori_s[:, i] - 273.15, ":", color=c, lw=1)
    ax[1].set(xlabel="t [s]", ylabel="T [°C]", title="transient: constant-h fit (lines), a priori (dotted)")
    ax[1].text(0.98, 0.205, f"constant-h fit: h = {fit_hk['h_eff']:.1f} W/m²K, k = {fit_hk['k']:.1f} W/m K\n"
               f"correlation-h fit: C_h = {fit_ck['C_h']:.2f}, k = {fit_ck['k']:.1f} W/m K",
               transform=ax[1].transAxes, ha="right", va="bottom", fontsize=8,
               bbox=dict(fc="white", ec="0.8"))
    ax[1].legend(fontsize=7, ncol=2)
    save(fig, out / f"{run['name']}.png", syn)

    res = {
        "rod": spec.name, "T_base_C": Tb_ss - 273.15,
        "steady_fit": fit_m.to_dict(), "h_over_k_per_m": h_over_k,
        "h_from_steady_with_handbook_k": h_over_k * k,
        "q_base_fin_W": q_fin, "q_base_two_point_fourier_W": q_2pt,
        "T_base_sd_K": float(Tb_series.std(ddof=1)),
        "transient_fit": fit_hk.to_dict(), "transient_fit_correlation_h": fit_ck.to_dict(),
        "k_model_form_spread": abs(fit_hk["k"] - fit_ck["k"]), "tau1_s": tau1,
        "apriori_steady_C": (T_pred_ss - 273.15).tolist(),
        "apriori_band_C": (band - 273.15).tolist(),
        "measured_steady_C": (T_ss - 273.15).tolist(),
    }
    if truth is not None:
        res["truth_k"] = spec.material.k * truth["k_factor"][spec.name]
    return res


def cooling_run(run, spec, env, an, syn, out, truth):
    T_inf, T_sur, P = env
    d = read_table(run["csv"], ["t_s", "T_C"])
    check_time_increasing(d["t_s"], run["csv"])
    sel = d["t_s"] >= run["fit_start_s"]
    t = d["t_s"][sel] - d["t_s"][sel][0]
    T = c_to_k(d["T_C"][sel])
    C, A = spec.C, spec.A_surface

    exp = inverse.fit_cooling_exponential(t, T, T_inf, tau0=200.0)
    h_avg = C / (A * exp["tau"])
    rel_unc_geom = np.hypot(CP_REL_UNC, D_ABS_UNC_MM * build.MM / spec.diameter)
    if run["mode"] == "natural":
        h_corr = lambda Ts: h_free_horizontal_cylinder(spec.diameter, Ts, T_inf, P)  # noqa: E731
        label = "C_h x Churchill-Chu"
    elif run["mode"] == "forced":
        H_REF = 50.0  # W/(m^2 K); fitted C_h * H_REF is the constant convective h
        h_corr = lambda Ts: np.full_like(np.asarray(Ts, dtype=float), H_REF)  # noqa: E731
        label = "constant h_conv"
    else:
        raise ValueError(f"mode must be 'natural' or 'forced', got {run['mode']!r}")
    phys, T_phys = inverse.fit_cooling_physical(t, T, C, A, h_corr, T_inf, T_sur,
                                                fit=("T_i", "C_h"), fixed={"eps": spec.eps})
    T_mid = 0.5 * (phys["T_i"] + T_inf)
    res = {"rod": spec.name, "mode": run["mode"], "tau_s": exp["tau"], "tau_ci95_s": exp.ci("tau"),
           "h_avg_total_W_m2K": h_avg,
           "h_avg_total_rel_unc": float(np.hypot(exp.ci("tau") / exp["tau"], rel_unc_geom)),
           "biot": biot(h_avg, spec.Ac * spec.total_length, A, spec.material.k),
           "exp_fit_residual_K": exp.s, "physical_fit": phys.to_dict(),
           "physical_fit_residual_K": phys.s,
           "heat_released_J": C * (phys["T_i"] - T_inf)}
    below = np.nonzero(T - T_inf < 2.0)[0]
    res["time_to_within_2K_s_measured"] = float(t[below[0]]) if below.size else None
    res["time_to_within_2K_s_exponential"] = float(exp["tau"] * np.log((exp["T_i"] - T_inf) / 2.0))

    if run["mode"] == "forced":
        h_conv = 50.0 * phys["C_h"]
        h_conv_ci = 50.0 * phys.ci("C_h")
        corr = an["forced_correlation"]
        U = inverse.velocity_from_h(h_conv, spec.diameter, T_mid, T_inf, P, corr, (0.02, 30.0))
        U_lo = inverse.velocity_from_h(h_conv - h_conv_ci, spec.diameter, T_mid, T_inf, P, corr, (0.01, 30.0))
        U_hi = inverse.velocity_from_h(h_conv + h_conv_ci, spec.diameter, T_mid, T_inf, P, corr, (0.02, 40.0))
        res.update({"fan_label": run["fan_label"], "h_conv_W_m2K": h_conv, "h_conv_ci95": h_conv_ci,
                    "U_eff_m_s": U, "U_eff_range_m_s": [U_lo, U_hi], "U_eff_correlation": corr})
        if "U_measured_m_s" in run:
            h_pred = h_forced_cylinder(spec.diameter, run["U_measured_m_s"], T_mid, T_inf, P, corr)
            res.update({"U_measured_m_s": run["U_measured_m_s"], "h_conv_predicted_W_m2K": h_pred,
                        "h_conv_measured_over_predicted": h_conv / h_pred})
        if truth is not None:
            res["truth_U_m_s"] = truth["fan_U_m_s"][run["fan_label"]]
            res["truth_C_h_forced"] = truth["C_h_forced"]
    else:
        res["C_h_natural"] = phys["C_h"]
        if truth is not None:
            res["truth_C_h"] = truth["C_h_natural"]

    # ---- instantaneous h(ΔT) and figures
    win = an["savgol_window"]
    h_inst = instantaneous_h(t, T, T_inf, C, A, win, an["savgol_polyorder"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(t, T - T_inf, ".", ms=3, color=COLORS["data"], label="data")
    ax[0].semilogy(t, T_phys - T_inf, color=COLORS["model"], label=f"fit: {label} + radiation")
    ax[0].semilogy(t, (exp["T_i"] - T_inf) * np.exp(-t / exp["tau"]), "--", color=COLORS["alt"],
                   label=f"single exponential, τ = {exp['tau']:.0f} s")
    ax[0].set(xlabel="t [s]", ylabel=r"$T - T_\infty$ [K]", title=f"{run['name']}: cooling curve")
    ax[0].set_ylim(bottom=max(0.3, float(np.min(T - T_inf)) * 0.8))
    ax[0].legend(fontsize=8)
    good = (T - T_inf) > 3.0
    ax[1].plot((T - T_inf)[good], h_inst[good], ".", ms=3, color=COLORS["data"],
               label=f"data: -C dT/dt / (A ΔT), SG window {win}")
    dT = np.linspace(max(1.0, float((T - T_inf)[good].min())), float((T - T_inf).max()), 100)
    Ts_ = T_inf + dT
    ax[1].plot(dT, phys["C_h"] * h_corr(Ts_) + h_radiation(spec.eps, Ts_, T_sur), color=COLORS["model"],
               label="fitted model (convection + radiation)")
    if run["mode"] == "natural":
        ax[1].plot(dT, h_corr(Ts_) + h_radiation(spec.eps, Ts_, T_sur), "--", color=COLORS["alt"],
                   label="Churchill-Chu + radiation (C_h = 1)")
    ax[1].plot(dT, h_radiation(spec.eps, Ts_, T_sur), ":", color=COLORS["aux"], label="radiation alone")
    ax[1].set(xlabel=r"$T - T_\infty$ [K]", ylabel=r"total $h$ [W m$^{-2}$ K$^{-1}$]",
              title="effective heat-transfer coefficient")
    ax[1].legend(fontsize=8)
    save(fig, out / f"{run['name']}.png", syn)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    syn = bool(cfg["meta"]["synthetic"])
    out = Path(cfg["meta"]["output_dir"])
    env = build.environment(cfg["environment"])
    an = cfg["analysis"]
    rng = np.random.default_rng(an["seed"])
    truth = cfg.get("truth") if syn else None
    rods = {r["name"]: build.rod_spec(r) for r in cfg["rods"]}
    results = {"config": cfg["_path"], "synthetic": syn, "rods": {}, "heating": {}, "cooling": {}}
    for name, s in rods.items():
        results["rods"][name] = {"material": s.material.name, "k_handbook": s.material.k,
                                 "k_source": s.material.source, "C_J_per_K": s.C,
                                 "A_surface_m2": s.A_surface, "density_check": s.density_check}
    for run in cfg.get("heating", []):
        print(f"heating: {run['name']}")
        results["heating"][run["name"]] = heating_run(run, rods[run["rod"]], env, an, rng, syn, out, truth)
    for run in cfg.get("cooling", []):
        print(f"cooling: {run['name']}")
        results["cooling"][run["name"]] = cooling_run(run, rods[run["rod"]], env, an, syn, out, truth)
    out.mkdir(parents=True, exist_ok=True)
    (out / "week1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, out / "summary.md")
    print(f"results in {out}/")


def write_summary(r, path):
    L = ["# Week 1 results" + (" (SYNTHETIC DATA - pipeline test)" if r["synthetic"] else ""), "",
         f"Config: `{r['config']}`", "", "## Rods", "",
         "| rod | material | k handbook [W/m K] | measured/handbook density | C [J/K] |",
         "|---|---|---|---|---|"]
    for n, s in r["rods"].items():
        L.append(f"| {n} | {s['material']} | {s['k_handbook']:.1f} | {s['density_check']:.3f} | {s['C_J_per_K']:.2f} |")
    L += ["", "## Heating (fin)", "",
          "| run | m [1/m] | h/k [1/m] | q_base fin [W] | q two-point Fourier [W] | h_eff (const-h fit) | k (const-h fit) | C_h, k (correlation-h fit) | corr(h,k) | tau1 [s] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, h in r["heating"].items():
        tf, cf = h["transient_fit"], h["transient_fit_correlation_h"]
        L.append(f"| {n} | {h['steady_fit']['values']['m']:.2f} ± {h['steady_fit']['ci95']['m']:.2f} "
                 f"| {h['h_over_k_per_m']:.3f} | {h['q_base_fin_W']:.3f} | {h['q_base_two_point_fourier_W']:.3f} "
                 f"| {tf['values']['h_eff']:.2f} ± {tf['ci95']['h_eff']:.2f} "
                 f"| {tf['values']['k']:.1f} ± {tf['ci95']['k']:.1f}"
                 + (f" (truth {h['truth_k']:.1f})" if "truth_k" in h else "")
                 + f" | {cf['values']['C_h']:.2f}, {cf['values']['k']:.1f} ± {cf['ci95']['k']:.1f}"
                 + f" | {tf['corr'][0][1]:+.2f} | {h['tau1_s']:.0f} |")
    L += ["", "## Cooling (lumped)", "",
          "| run | tau [s] | h total avg [W/m²K] | Bi | C_h or h_conv | U_eff [m/s] | to within 2 K [s] |",
          "|---|---|---|---|---|---|---|"]
    for n, c in r["cooling"].items():
        if c["mode"] == "natural":
            extra = f"C_h = {c['C_h_natural']:.3f} ± {c['physical_fit']['ci95']['C_h']:.3f}" + (
                f" (truth {c['truth_C_h']})" if "truth_C_h" in c else "")
            U = "-"
        else:
            extra = f"h_conv = {c['h_conv_W_m2K']:.1f} ± {c['h_conv_ci95']:.1f}"
            U = f"{c['U_eff_m_s']:.2f} [{c['U_eff_range_m_s'][0]:.2f}, {c['U_eff_range_m_s'][1]:.2f}]" + (
                f" (truth U {c['truth_U_m_s']}, true h = {c['truth_C_h_forced']} x correlation)"
                if "truth_U_m_s" in c else "")
        t2 = c["time_to_within_2K_s_measured"]
        L.append(f"| {n} | {c['tau_s']:.1f} ± {c['tau_ci95_s']:.1f} | {c['h_avg_total_W_m2K']:.1f} "
                 f"| {c['biot']:.1e} | {extra} | {U} | {t2 if t2 is not None else 'not reached'} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

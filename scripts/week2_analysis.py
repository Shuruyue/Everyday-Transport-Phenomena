"""Week 2 analysis: Stefan tubes (D_AB) and open-dish evaporation (k_c, cooling, time).

    python scripts/week2_analysis.py config/week2.toml
    python scripts/week2_analysis.py config/synthetic/week2.toml   # pipeline test

Writes figures, week2_results.json and summary.md to [meta].output_dir.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from etp import build
from etp.constants import R, c_to_k
from etp.diffusivity import d_chapman_enskog, d_fuller, scale_diffusivity
from etp.io import check_time_increasing, load_config, read_table
from etp.mass import evaporation as ev
from etp.mass import stefan
from etp.mass.activity import ETHANOL_WATER_NRTL, Ideal
from etp.mass.inverse import fit_evaporation, mass_rate
from etp.plotting import COLORS, save
from etp.properties import ETHANOL, M_AIR, WATER, dew_point

T_UNC_K = 0.5  # room-temperature uncertainty used in the Monte Carlo


def stefan_run(st, amb, an, rng, syn, out, truth):
    col = "mass_g" if st["reading"] == "mass" else "z_mm"
    if st["reading"] not in ("mass", "level"):
        raise ValueError(f"reading must be 'mass' or 'level', got {st['reading']!r}")
    d = read_table(st["csv"], ["t_h", col])
    check_time_increasing(d["t_h"], st["csv"])
    t = d["t_h"] * 3600.0
    A = np.pi * (st["inner_diameter_mm"] * build.MM) ** 2 / 4
    z0 = st["z0_mm"] * build.MM
    T = amb.T_inf
    rho_L = ETHANOL.rho(T)
    if st["reading"] == "mass":
        z = stefan.level_from_mass(d["mass_g"] * build.G, d["mass_g"][0] * build.G, z0, rho_L, A)
    else:
        z = d["z_mm"] * build.MM
    c, c_L, y_s = amb.P / (R * T), rho_L / ETHANOL.M, ETHANOL.psat(T) / amb.P
    fit = stefan.fit_stefan_tube(t, z, z0, c, c_L, y_s, 0.0)
    # Monte Carlo: slope uncertainty and room temperature (P_sat changes ~6 %/K)
    se_K = fit.K_ci95 / 1.96
    D_mc = []
    for _ in range(an["monte_carlo_n"] * 10):
        Ti = T + T_UNC_K * rng.standard_normal()
        K = fit.K + se_K * rng.standard_normal()
        ys = ETHANOL.psat(Ti) / amb.P
        D_mc.append(K / (2 * (amb.P / (R * Ti)) / (ETHANOL.rho(Ti) / ETHANOL.M)
                         * stefan.log_factor(ys, 0.0)))
    D_lo, D_hi = np.percentile(D_mc, [2.5, 97.5])
    D25 = scale_diffusivity(fit.D, T, 298.15, 1.75)
    res = {"D_m2_s": fit.D, "D_ci95_slope_only": fit.D_ci95, "D_95pct_range_with_T": [D_lo, D_hi],
           "D_at_25C_m2_s": D25, "D_fuller_at_T": float(d_fuller("ethanol", "air", T, amb.P)),
           "D_chapman_enskog_at_T": float(d_chapman_enskog("ethanol", "air", T, amb.P)),
           "intercept_m2": fit.intercept, "intercept_ci95": fit.intercept_ci95, "r2": fit.r2,
           "pseudo_steady_ratio_c_over_cL": c / c_L,
           "diffusion_time_z0sq_over_D_s": z0**2 / fit.D}
    if truth is not None:
        res["truth_D_at_25C"] = truth["D_ethanol_25C_m2_s"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    y = (z**2 - z0**2) * 1e6
    ax[0].plot(t / 3600, y, "o", color=COLORS["data"], label=f"data ({st['reading']} readings)")
    ax[0].plot(t / 3600, (fit.K * t + fit.intercept) * 1e6, color=COLORS["model"],
               label=f"fit: D = {fit.D * 1e5:.3f} ± {fit.D_ci95 * 1e5:.3f} ×10⁻⁵ m²/s")
    ax[0].plot(t / 3600, 2 * c / c_L * res["D_fuller_at_T"] * stefan.log_factor(y_s, 0) * t * 1e6,
               "--", color=COLORS["alt"], label="Fuller prediction")
    ax[0].set(xlabel="t [h]", ylabel=r"$z^2 - z_0^2$ [mm²]", title=f"{st['name']}: Stefan tube")
    ax[0].legend(fontsize=8)
    ax[1].plot(t / 3600, fit.residuals * 1e6, "o-", color=COLORS["data"])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="t [h]", ylabel="residual [mm²]", title="residuals (no trend = law holds)")
    save(fig, out / f"{st['name']}.png", syn)
    return res


def dish_setup(dc, V0, UA):
    """Dish geometry and thermal data; recess0 from the rim height and liquid volume V0 [m^3]."""
    R_d = dc["inner_diameter_mm"] * build.MM / 2
    recess0 = dc["rim_height_mm"] * build.MM - V0 / (np.pi * R_d**2)
    if recess0 < 0:
        raise ValueError("liquid volume exceeds the dish: check rim height and diameter")
    return ev.Dish(R_d, recess0, dc["dish_mass_g"] * build.G * dc["dish_cp_J_per_kgK"], UA,
                   dc["eps_liquid"])


def mt_model(run, dish, multiplier, below_range="raise"):
    if run["mode"] == "natural":
        return ev.FreeConvectionSurface(dish.L_char, multiplier, below_range)
    if run["mode"] == "forced":
        return ev.ForcedFlatPlate(run["U_m_s"], 2 * dish.radius, multiplier)
    raise ValueError(f"mode must be 'natural' or 'forced', got {run['mode']!r}")


def dish_run(run, cfg, amb, diff, an, rng, syn, out, fitted, truth):
    dc = cfg["dish"]
    d = read_table(run["csv"], ["t_min", "mass_g", "T_liquid_C"])
    check_time_increasing(d["t_min"], run["csv"])
    t = d["t_min"] * 60.0
    m = (d["mass_g"] - dc["tare_g"]) * build.G
    Ts = c_to_k(d["T_liquid_C"])
    m0 = float(m[0])
    T0 = amb.T_inf
    if run["liquid"] == "ethanol":
        n_E, n_W, act, wx = m0 / ETHANOL.M, 0.0, Ideal(), False
    elif run["liquid"] == "aqueous":
        w = run["ethanol_mass_fraction"]
        n_E, n_W, act, wx = w * m0 / ETHANOL.M, (1 - w) * m0 / WATER.M, ETHANOL_WATER_NRTL, True
    else:
        raise ValueError(f"liquid must be 'ethanol' or 'aqueous', got {run['liquid']!r}")
    V0 = n_E * ETHANOL.M / ETHANOL.rho(T0) + n_W * WATER.M / WATER.rho(T0)
    dish = dish_setup(dc, V0, dc["UA_bottom_W_per_K"])
    A = dish.area
    w0, w1 = (x * 60.0 for x in run["rate_window_min"])
    rate, rate_ci, _ = mass_rate(t, m, w0, w1)
    win = (t >= w0) & (t <= w1)
    Ts_win = float(Ts[win].mean())
    T_f = 0.5 * (Ts_win + amb.T_inf)
    D_E = diff.ethanol(T_f)
    res = {"liquid": run["liquid"], "mode": run["mode"], "m0_g": m0 * 1e3,
           "recess0_mm": dish.recess0 * 1e3, "rate_g_per_h": rate * 3.6e6,
           "rate_ci95_g_per_h": rate_ci * 3.6e6, "T_liquid_window_mean_C": Ts_win - 273.15,
           "dew_point_C": dew_point(amb.RH, amb.T_inf, amb.P) - 273.15}
    sim = lambda **kw: ev.simulate_evaporation(  # noqa: E731
        kw.get("t", t), n_E, n_W, T0, kw["dish"], amb, kw["model"], diff, kw.get("act", act),
        kw.get("wx", wx), kw.get("iso", False), an["recess_model"], kw.get("stop", 1e-3))

    if run["liquid"] == "ethanol":
        kc_eff = ev.kc_from_mass_rate(rate, A, Ts_win, amb)
        y_s = ETHANOL.psat(Ts_win) / amb.P
        rho_s = amb.P * (y_s * ETHANOL.M + (1 - y_s) * M_AIR) / (R * Ts_win)
        kc_corr, _ = mt_model(run, dish, 1.0).kc(Ts_win, rho_s, amb, D_E)
        kc_disk, _ = ev.DiffusionLimitDisk(dish.radius).kc(Ts_win, rho_s, amb, D_E)
        res.update({"kc_eff_measured_m_s": kc_eff, "film_thickness_mm": D_E / kc_eff * 1e3,
                    "kc_correlation_flush_m_s": kc_corr, "kc_diffusion_limit_m_s": kc_disk,
                    "kc_correlation_with_recess_m_s": 1 / (1 / kc_corr + dish.recess0 / D_E),
                    "surface_layer_stable": bool(rho_s > amb.rho)})
        fit, (m_fit, T_fit) = fit_evaporation(
            t, m, Ts, an["sigma_mass_g"] * build.G, an["sigma_T_K"], n_E, n_W, dish, amb,
            mt_model(run, dish, 1.0), diff, act, wx, an["recess_model"],
            fit=("multiplier", "UA_bottom"), p0=[1.0, dc["UA_bottom_W_per_K"]])
        fitted[run["name"]] = fit
        res["fit"] = fit.to_dict()
        if truth is not None:
            res["truth_multiplier"] = truth["multiplier"][run["mode"]]
            res["truth_UA_bottom"] = truth["UA_bottom_W_per_K"]
        mult, UA = fit["multiplier"], fit["UA_bottom"]
        apriori_iso = sim(model=mt_model(run, dish, 1.0), dish=dish, iso=True)
        apriori_cpl = sim(model=mt_model(run, dish, 1.0), dish=dish)
        fdish = replace(dish, UA_bottom=UA)
        fmodel = mt_model(run, dish, mult)
        long_t = np.linspace(0, 72 * 3600, 2001)
        full = sim(model=fmodel, dish=fdish, t=long_t, stop=0.02)
        res["t_98pct_model_h"] = None if full.stopped_at is None else full.stopped_at / 3600
        below = np.nonzero(m <= 0.02 * m0)[0]
        res["t_98pct_measured_h"] = float(t[below[0]] / 3600) if below.size else None
        res["wet_bulb_fitted_C"] = ev.wet_bulb_temperature(
            fdish, amb, fmodel, diff, Ideal(), an["recess_model"]) - 273.15
        # water uptake from humid air (the pure-ethanol fit ignores it). As water
        # accumulates the vapour layer can pass through neutral buoyancy, so the
        # free-convection correlation is extrapolated below Ra = 1e4 here.
        wet = ev.simulate_evaporation(long_t, n_E, 0.0, T0, fdish, amb,
                                      mt_model(run, dish, mult, "extrapolate"), diff,
                                      ETHANOL_WATER_NRTL, True, False, an["recess_model"], 0.02)
        res["water_uptake_max_g"] = float(wet.n_W.max() * WATER.M * 1e3)
        res["t_98pct_with_water_uptake_h"] = None if wet.stopped_at is None else wet.stopped_at / 3600
        # Monte Carlo on the evaporation time
        t98 = []
        for _ in range(an["monte_carlo_n"]):
            pm = max(0.05, mult + fit.stderr[0] * rng.standard_normal())
            pu = max(0.0, UA + fit.stderr[1] * rng.standard_normal())
            ambi = replace(amb, T_inf=amb.T_inf + T_UNC_K * rng.standard_normal())
            r = ev.simulate_evaporation(long_t, n_E, 0.0, ambi.T_inf, replace(dish, UA_bottom=pu), ambi,
                                        mt_model(run, dish, pm), diff, Ideal(), False, False,
                                        an["recess_model"], 0.02)
            t98.append(np.nan if r.stopped_at is None else r.stopped_at / 3600)
        t98 = np.array(t98)
        if np.isnan(t98).any():
            raise RuntimeError("some Monte Carlo runs did not dry within 72 h; extend long_t")
        res["t_98pct_95pct_range_h"] = np.percentile(t98, [2.5, 97.5]).tolist()
        curves = [(apriori_iso, ":", COLORS["alt"], "a priori, liquid at room T (textbook)"),
                  (apriori_cpl, "--", COLORS["aux2"], "a priori, with evaporative cooling"),
                  (full, "-", COLORS["model"], f"fit: multiplier {mult:.2f}, UA {UA * 1e3:.0f} mW/K")]
    else:
        src = fitted[run["multiplier_from"]]
        mult, UA = src["multiplier"], src["UA_bottom"]
        fdish = replace(dish, UA_bottom=UA)
        fmodel = mt_model(run, dish, mult)
        long_t = np.linspace(0, max(float(t[-1]), 8 * 3600), 1201)
        pred = sim(model=fmodel, dish=fdish, t=long_t, stop=0.02)
        raoult = sim(model=fmodel, dish=fdish, t=long_t, act=Ideal(), stop=0.02)
        dry = ev.simulate_evaporation(long_t, n_E, n_W, T0, fdish, replace(amb, RH=0.0), fmodel, diff,
                                      act, True, False, an["recess_model"], 0.02)
        m_pred = np.interp(t, pred.t, pred.mass)
        res.update({"multiplier_from": run["multiplier_from"], "multiplier": mult,
                    "rms_mass_error_g": float(np.sqrt(np.mean((m_pred - m) ** 2)) * 1e3),
                    "residual_liquid_after_8h_g": float(pred.mass[-1] * 1e3),
                    "x_E_after_8h": float(pred.x_E[-1])})
        curves = [(raoult, ":", COLORS["alt"], "Raoult's law (γ = 1)"),
                  (dry, "--", COLORS["aux2"], "NRTL, dry air (no water uptake)"),
                  (pred, "-", COLORS["model"], f"NRTL + humid air, k_c from {run['multiplier_from']}")]

    fig, ax = plt.subplots(1, 3 if run["liquid"] == "aqueous" else 2,
                           figsize=(15 if run["liquid"] == "aqueous" else 11, 4.2))
    ax[0].plot(t / 3600, m * 1e3, "o", ms=3, color=COLORS["data"], label="data")
    for r, ls, col, lab in curves:
        ax[0].plot(r.t / 3600, r.mass * 1e3, ls, color=col, label=lab)
    ax[0].set_xlim(0, max(t[-1] / 3600 * 1.1, 0.5))
    ax[0].set(xlabel="t [h]", ylabel="liquid mass [g]", title=f"{run['name']}: mass")
    ax[0].legend(fontsize=7)
    ax[1].plot(t / 3600, Ts - 273.15, "o", ms=3, color=COLORS["data"], label="data")
    for r, ls, col, lab in curves:
        ax[1].plot(r.t / 3600, r.T_s - 273.15, ls, color=col)
    ax[1].axhline(res["dew_point_C"], color="0.5", lw=0.8, ls="-.",
                  label=f"dew point of room air {res['dew_point_C']:.1f} °C")
    ax[1].set_xlim(0, max(t[-1] / 3600 * 1.1, 0.5))
    ax[1].set(xlabel="t [h]", ylabel="liquid temperature [°C]", title="evaporative cooling")
    ax[1].legend(fontsize=7)
    if run["liquid"] == "aqueous":
        for r, ls, col, lab in curves:
            ax[2].plot(r.t / 3600, r.x_E, ls, color=col, label=lab)
        ax[2].set_xlim(0, max(t[-1] / 3600 * 1.1, 0.5))
        ax[2].set(xlabel="t [h]", ylabel="ethanol mole fraction in the liquid",
                  title="ethanol leaves first (model)")
        ax[2].legend(fontsize=7)
    save(fig, out / f"{run['name']}.png", syn)
    return res


def analogy_check(cfg, dish_res, amb, diff, an):
    """Week 1 fan speed -> flat-plate k_c -> compare with the Week 2 fitted k_c."""
    p = Path(an["week1_results"])
    if not p.exists():
        return {"status": f"{p} not found; run the Week 1 analysis first"}
    w1 = json.loads(p.read_text(encoding="utf-8"))
    by_fan = {c["fan_label"]: c for c in w1["cooling"].values() if c["mode"] == "forced"}
    out = {}
    for run in cfg["dish_runs"]:
        if run["liquid"] != "ethanol" or run["mode"] != "forced" or run.get("fan_label") not in by_fan:
            continue
        c1 = by_fan[run["fan_label"]]
        r = dish_res[run["name"]]
        Ts = c_to_k(r["T_liquid_window_mean_C"])
        y_s = ETHANOL.psat(Ts) / amb.P
        rho_s = amb.P * (y_s * ETHANOL.M + (1 - y_s) * M_AIR) / (R * Ts)
        D_E = diff.ethanol(0.5 * (Ts + amb.T_inf))
        L = cfg["dish"]["inner_diameter_mm"] * build.MM
        pred = [ev.ForcedFlatPlate(U, L, 1.0).kc(Ts, rho_s, amb, D_E)[0]
                for U in [c1["U_eff_m_s"], *c1["U_eff_range_m_s"]]]
        meas = r["fit"]["values"]["multiplier"] * ev.ForcedFlatPlate(run["U_m_s"], L, 1.0).kc(
            Ts, rho_s, amb, D_E)[0]
        out[run["name"]] = {"fan_label": run["fan_label"], "U_eff_week1_m_s": c1["U_eff_m_s"],
                            "kc_predicted_from_week1_m_s": pred[0],
                            "kc_predicted_range_m_s": [pred[1], pred[2]],
                            "kc_week2_fitted_m_s": meas, "measured_over_predicted": meas / pred[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    syn = bool(cfg["meta"]["synthetic"])
    out = Path(cfg["meta"]["output_dir"])
    an = cfg["analysis"]
    rng = np.random.default_rng(an["seed"])
    amb = build.ambient(cfg["environment"])
    diff = build.diffusivities(cfg["diffusivity"], amb.P)
    truth = cfg.get("truth") if syn else None
    results = {"config": cfg["_path"], "synthetic": syn, "stefan": {}, "dish": {}}
    for st in cfg.get("stefan", []):
        print(f"stefan: {st['name']}")
        results["stefan"][st["name"]] = stefan_run(st, amb, an, rng, syn, out, truth)
    fitted = {}
    runs = sorted(cfg.get("dish_runs", []), key=lambda r: r["liquid"] != "ethanol")  # pure first
    for run in runs:
        print(f"dish: {run['name']}")
        results["dish"][run["name"]] = dish_run(run, cfg, amb, diff, an, rng, syn, out, fitted, truth)
    results["analogy_week1_to_week2"] = analogy_check(cfg, results["dish"], amb, diff, an)
    out.mkdir(parents=True, exist_ok=True)
    (out / "week2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, out / "summary.md")
    print(f"results in {out}/")


def write_summary(r, path):
    L = ["# Week 2 results" + (" (SYNTHETIC DATA - pipeline test)" if r["synthetic"] else ""), "",
         f"Config: `{r['config']}`", "", "## Stefan tubes", "",
         "| tube | D at room T [1e-5 m²/s] | 95 % range incl. T | D at 25 °C | Fuller | Chapman-Enskog |",
         "|---|---|---|---|---|---|"]
    for n, s in r["stefan"].items():
        tr = f" (truth {s['truth_D_at_25C'] * 1e5:.3f})" if "truth_D_at_25C" in s else ""
        L.append(f"| {n} | {s['D_m2_s'] * 1e5:.3f} ± {s['D_ci95_slope_only'] * 1e5:.3f} "
                 f"| {s['D_95pct_range_with_T'][0] * 1e5:.3f}-{s['D_95pct_range_with_T'][1] * 1e5:.3f} "
                 f"| {s['D_at_25C_m2_s'] * 1e5:.3f}{tr} | {s['D_fuller_at_T'] * 1e5:.3f} "
                 f"| {s['D_chapman_enskog_at_T'] * 1e5:.3f} |")
    L += ["", "## Open dish", "",
          "| run | rate [g/h] | T liquid [°C] | k_c measured [mm/s] | k_c correlation flush / with recess [mm/s] "
          "| multiplier | UA [mW/K] | t 98 % model [h] (95 % range) | water uptake [g] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, d in r["dish"].items():
        if d["liquid"] != "ethanol":
            continue
        f = d["fit"]
        tm = f" (truth {d['truth_multiplier']})" if "truth_multiplier" in d else ""
        tu = f" (truth {d['truth_UA_bottom'] * 1e3:.0f})" if "truth_UA_bottom" in d else ""
        rng_ = d["t_98pct_95pct_range_h"]
        L.append(f"| {n} | {d['rate_g_per_h']:.2f} ± {d['rate_ci95_g_per_h']:.2f} "
                 f"| {d['T_liquid_window_mean_C']:.1f} | {d['kc_eff_measured_m_s'] * 1e3:.2f} "
                 f"| {d['kc_correlation_flush_m_s'] * 1e3:.2f} / {d['kc_correlation_with_recess_m_s'] * 1e3:.2f} "
                 f"| {f['values']['multiplier']:.3f} ± {f['ci95']['multiplier']:.3f}{tm} "
                 f"| {f['values']['UA_bottom'] * 1e3:.1f} ± {f['ci95']['UA_bottom'] * 1e3:.1f}{tu} "
                 f"| {d['t_98pct_model_h']:.2f} ({rng_[0]:.2f}-{rng_[1]:.2f}) | {d['water_uptake_max_g']:.2f} |")
    aq = {n: d for n, d in r["dish"].items() if d["liquid"] == "aqueous"}
    if aq:
        L += ["", "## Aqueous ethanol (prediction, no fitting)", "",
              "| run | k_c from | rms mass error [g] | liquid left after 8 h [g] | x_E after 8 h |",
              "|---|---|---|---|---|"]
        for n, d in aq.items():
            L.append(f"| {n} | {d['multiplier_from']} | {d['rms_mass_error_g']:.2f} "
                     f"| {d['residual_liquid_after_8h_g']:.2f} | {d['x_E_after_8h']:.3f} |")
    L += ["", "## Heat-mass analogy across the two assignments", ""]
    a = r["analogy_week1_to_week2"]
    if "status" in a:
        L.append(a["status"])
    for n, v in a.items():
        if n == "status":
            continue
        L.append(f"- {n}: Week 1 U_eff = {v['U_eff_week1_m_s']:.2f} m/s predicts k_c = "
                 f"{v['kc_predicted_from_week1_m_s'] * 1e3:.2f} mm/s "
                 f"({v['kc_predicted_range_m_s'][0] * 1e3:.2f}-{v['kc_predicted_range_m_s'][1] * 1e3:.2f}); "
                 f"Week 2 fit gives {v['kc_week2_fitted_m_s'] * 1e3:.2f} mm/s "
                 f"(ratio {v['measured_over_predicted']:.2f}).")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

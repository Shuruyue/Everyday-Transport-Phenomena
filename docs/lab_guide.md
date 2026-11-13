# Lab guide: from measurements to the report

This repository covers the two self-study assignments of Transport Phenomena II: Week 1, heat
transfer (a metal rod heated in boiling water and then cooled), and Week 2, mass transfer
(evaporation of ethanol). The code, the verification and the whole analysis pipeline are done. The
figures in `results/synthetic/` come from a pipeline test on synthetic data (they carry a watermark)
and **must not be handed in as experimental results**.

## 1. Before the experiment

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
pytest -q                            # all 47 tests pass: the code works
python scripts/design_estimates.py   # design tables → docs/design_estimates.md
```

The design estimates fix a few choices. Heat travels only about 3 cm up a stainless-steel chopstick,
so the sensors sit close to the water: 1, 2, 4, 7 and 12 cm above the surface. With a fan the cooling
time constant is only 40–70 s, so temperatures are read frame by frame from a video (one reading every
≤ 5 s). The ethanol surface cools to 6–11 °C, so the liquid temperature must be measured.

## 2. Recording the data

Save the data as CSV files in the column format of `data/templates/` and put them in `data/raw/`:

| Experiment | Columns | Note |
|---|---|---|
| Week 1 heating | `t_s, T_base_C, T1_C … T5_C` | T1…T5 in the same order as `sensor_positions_cm` in the config |
| Week 1 cooling | `t_s, T_C` | constant sampling interval (video read-out), otherwise the instantaneous h cannot be computed |
| Week 2 Stefan tube | `t_h, mass_g` or `t_h, z_mm` | weigh on a 0.01 g balance; use `z_mm` when reading the liquid level |
| Week 2 open dish | `t_min, mass_g, T_liquid_C` | `mass_g` is dish plus liquid; the empty dish goes in `tare_g` of the config |

## 3. Filling in the configs

Open `config/week1.toml` and `config/week2.toml` and replace every `"MEASURE"` with your reading
(units are in the key names). While any `"MEASURE"` is left, the analysis stops and lists what is
missing; it never runs on a placeholder.

Entries that are easy to get wrong:

- `material`: do the magnet test first. 304 is barely magnetic, 430 is clearly magnetic, and their
  k differ by about 75 %.
- `emissivity`: about 0.95 under black tape, 0.1–0.4 for bare stainless steel; give the low and high
  values in `emissivity_range` for the uncertainty analysis.
- `steady_window_s`: the part of the heating curve that is already flat.
- `fit_start_s`: skip the first seconds of cooling, while the rod is just out of the water and the
  water film is being wiped off.
- `fan_label`: when Week 1 and Week 2 use the same fan at the same setting and distance, give them the
  same label; Week 2 then checks the heat-mass analogy across the two assignments automatically.
- `rate_window_min`: the early, nearly straight part of the open-dish mass curve.

## 4. Running the analysis

```bash
python scripts/week1_analysis.py config/week1.toml
python scripts/week2_analysis.py config/week2.toml   # needs the Week 1 results for the analogy check
```

The results go to `results/week1/` and `results/week2/`: one figure per run, `summary.md` (tables
ready for the slides) and `*_results.json` (every number with its confidence interval).

## 5. Where each slide comes from

| Slide content | Source |
|---|---|
| Fourier heat rate, fin parameter m | `summary.md`, Heating table: `q_base`, `m` |
| h and k separated, model-form uncertainty | difference between the k fitted with the two h models |
| Cooling time constant, time to room temperature, h vs correlation | Cooling table and the right panel of each cooling-run figure |
| Air speed U_eff from the cooling curve | `U_eff_m_s` of the forced cooling runs |
| D_AB vs Fuller and Chapman–Enskog | Week 2 Stefan table |
| k_c, film thickness, evaporative cooling, time to dry out (95 % range) | Week 2 open-dish table and figures |
| 75 % ethanol, water uptake | Aqueous table, `water_uptake_max_g` |
| Heat-mass analogy across the assignments | last section of `summary.md` |
| Correctness of the code | `figures/verification/` (appendix) |

The derivations and the reason for every assumption are in [`derivations.md`](derivations.md).

# Week 2 results (SYNTHETIC DATA - pipeline test)

Config: `config/synthetic/week2.toml`

## Stefan tubes

| tube | D at room T [1e-5 m²/s] | 95 % range incl. T | D at 25 °C | Fuller | Chapman-Enskog |
|---|---|---|---|---|---|
| tube-A | 1.183 ± 0.020 | 1.114-1.253 | 1.183 (truth 1.180) | 1.234 | 1.188 |
| tube-B | 1.205 ± 0.025 | 1.136-1.275 | 1.205 (truth 1.180) | 1.234 | 1.188 |

## Open dish

| run | rate [g/h] | T liquid [°C] | k_c measured [mm/s] | k_c correlation flush / with recess [mm/s] | multiplier | UA [mW/K] | t 98 % model [h] (95 % range) | water uptake [g] |
|---|---|---|---|---|---|---|---|---|
| ethanol-still-1 | 3.70 ± 0.17 | 16.3 | 1.81 | 2.30 / 1.70 | 1.269 ± 0.007 (truth 1.35) | 29.8 ± 0.5 (truth 30) | 4.89 (4.76-5.04) | 1.82 |
| ethanol-fan-low-1 | 6.27 ± 0.85 | 14.9 | 3.32 | 9.33 / 3.84 | 1.095 ± 0.015 (truth 1.25) | 29.5 ± 0.7 (truth 30) | 3.42 (3.33-3.53) | 2.13 |

## Aqueous ethanol (prediction, no fitting)

| run | k_c from | rms mass error [g] | liquid left after 8 h [g] | x_E after 8 h |
|---|---|---|---|---|
| aqueous-75-fan-low-1 | ethanol-fan-low-1 | 0.01 | 4.01 | 0.000 |

## Heat-mass analogy across the two assignments

- ethanol-fan-low-1: Week 1 U_eff = 1.98 m/s predicts k_c = 10.37 mm/s (10.31-10.42); Week 2 fit gives 10.22 mm/s (ratio 0.99).

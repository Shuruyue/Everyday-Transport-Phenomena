# Week 1 results (SYNTHETIC DATA - pipeline test)

Config: `config/synthetic/week1.toml`

## Rods

| rod | material | k handbook [W/m K] | measured/handbook density | C [J/K] |
|---|---|---|---|---|
| stainless | AISI 304 stainless steel | 14.9 | 0.998 | 10.40 |
| aluminum | aluminum alloy 6061-T6 | 167.0 | 0.998 | 6.68 |

## Heating (fin)

| run | m [1/m] | h/k [1/m] | q_base fin [W] | q two-point Fourier [W] | h_eff (const-h fit) | k (const-h fit) | C_h, k (correlation-h fit) | corr(h,k) | tau1 [s] |
|---|---|---|---|---|---|---|---|---|---|
| stainless-heating-1 | 37.49 ± 1.46 | 1.406 | 0.522 | 0.294 | 21.31 ± 0.25 | 15.1 ± 0.2 (truth 14.9) | 1.20, 14.9 ± 0.0 | +0.97 | 164 |
| aluminum-heating-1 | 11.97 ± 0.17 | 0.143 | 1.769 | 1.476 | 22.50 ± 0.14 | 157.2 ± 1.0 (truth 153.6) | 1.20, 153.5 ± 0.4 | +0.98 | 61 |

## Cooling (lumped)

| run | tau [s] | h total avg [W/m²K] | Bi | C_h or h_conv | U_eff [m/s] | to within 2 K [s] |
|---|---|---|---|---|---|---|
| stainless-natural-1 | 180.6 ± 1.3 | 20.6 | 1.4e-03 | C_h = 1.197 ± 0.004 (truth 1.2) | - | 675.0 |
| stainless-fan-low-1 | 46.0 ± 0.2 | 81.0 | 5.4e-03 | h_conv = 74.5 ± 0.4 | 1.98 [1.96, 2.00] (truth U 1.5, true h = 1.15 x correlation) | 155.0 |
| stainless-fan-high-1 | 33.1 ± 0.2 | 112.5 | 7.5e-03 | h_conv = 106.0 ± 0.6 | 4.01 [3.96, 4.05] (truth U 3.0, true h = 1.15 x correlation) | 110.0 |
| aluminum-natural-1 | 116.2 ± 0.9 | 20.6 | 1.2e-04 | C_h = 1.202 ± 0.005 (truth 1.2) | - | 435.0 |

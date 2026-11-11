# Derivations and modelling notes

Everything the code assumes, in the order the report needs it. Symbols are SI.

## 1. Rod in boiling water (fin)

Energy balance on a slice $dx$ of a rod with cross-section $A_c$, perimeter $P$:

$$
\rho c A_c \frac{\partial T}{\partial t} = \frac{\partial}{\partial x}\left(k A_c \frac{\partial T}{\partial x}\right) - P\,q''(T),
\qquad q''(T) = h_c(T)(T-T_\infty) + \varepsilon\sigma(T^4 - T_{sur}^4).
$$

With constant $h$ and $\theta = T - T_\infty$ the steady problem is $\theta'' = m^2\theta$, $m^2 = hP/(kA_c)$.
Base $\theta(0) = \theta_b$; an adiabatic tip at the corrected length $L_c = L + A_c/P$ gives

$$
\frac{\theta}{\theta_b} = \frac{\cosh m(L_c - x)}{\cosh mL_c},\qquad
q_b = \sqrt{hPkA_c}\,\theta_b \tanh mL_c .
$$

**What a steady profile can and cannot tell you.** The steady solution depends on $h$ and $k$ only
through $m$. A steady profile therefore fixes $h/k = m^2 A_c/P$, and $h$ follows only after a
handbook $k$ is assumed. The transient adds the diffusivity $\alpha = k/\rho c$:

**Transient series solution** (rod initially at $T_\infty$, base stepped to $T_b$ at $t=0$, adiabatic tip).
Write $\theta = \theta_{ss}(x) - \phi(x,t)$. Then $\phi_t = \alpha(\phi_{xx} - m^2\phi)$ with
$\phi(0,t)=0$, $\phi_x(L,t)=0$, $\phi(x,0)=\theta_{ss}$. Separation gives eigenfunctions
$\sin\lambda_n x$ with $\cos\lambda_n L = 0$, i.e. $\lambda_n = (2n-1)\pi/2L$, and decay rates
$\alpha(\lambda_n^2 + m^2)$. The coefficients are $B_n = \frac{2}{L}\int_0^L \theta_{ss}\sin\lambda_n x\,dx$.
Integrating by parts twice, using $\theta_{ss}'' = m^2\theta_{ss}$, $\cos\lambda_n L = 0$ and $\theta_{ss}'(L)=0$:

$$
\int_0^L \cosh m(L-x)\sin\lambda_n x\,dx = \frac{\lambda_n\cosh mL}{\lambda_n^2+m^2}
\quad\Rightarrow\quad
\frac{\theta}{\theta_b} = \frac{\cosh m(L-x)}{\cosh mL} - \sum_{n\ge1}\frac{2\lambda_n}{L(\lambda_n^2+m^2)}\sin(\lambda_n x)\,e^{-\alpha(\lambda_n^2+m^2)t}.
$$

The slowest mode sets the time to steady state, $\tau_1 = 1/[\alpha(m^2 + (\pi/2L)^2)]$. Because the
time scale carries $\alpha$ and the shape carries $m$, fitting the whole transient separates $h$ and $k$
(`fit_transient_rod`). The finite-volume solver (`fin_fv.py`) reproduces this series and the steady
solutions with observed order 2.0 (`figures/verification/fin_solver.png`).

## 2. Cooling in air (lumped capacitance)

For $Bi = h(V/A_s)/k < 0.1$ (here $10^{-3}$ to $10^{-2}$): $C\,dT/dt = -A_s q''(T)$, $C = mc$.
With constant $h$: $\theta/\theta_i = e^{-t/\tau}$, $\tau = C/(hA_s)$. Free convection
($h\propto\Delta T^{1/4}$) and radiation ($\propto T^3$) bend $\ln\theta$ vs $t$, so the code also fits the
nonlinear ODE and reports the instantaneous coefficient

$$
h(t) = -\frac{C}{A_s}\,\frac{dT/dt}{T - T_\infty},
$$

with $dT/dt$ from a Savitzky-Golay filter (window and order are reported with the result).

**Fan speed from the cooling curve.** With radiation subtracted, the convective $h$ is inverted through
the Churchill-Bernstein correlation to an effective speed $U_{eff}$. $U_{eff}$ absorbs any error of the
correlation (a real $h$ 15 % above the correlation reads as a 35 % higher speed, since $h\sim U^{0.47}$).
It is an effective, not a measured, speed. The Week 2 analogy check uses it on purpose: the same
correlation-plus-analogy chain is tested end to end.

## 3. Stefan tube

Vapour A diffuses through stagnant air B ($N_B = 0$): $N_A = y_A N_A - cD\,dy_A/dz$, so
$N_A = \frac{cD}{z}\ln\frac{1-y_\infty}{1-y_s}$ (Stefan's law). The liquid recedes as $c_L\,dz/dt = N_A$:

$$
z^2 - z_0^2 = K t,\qquad K = 2\frac{c}{c_L}D_{AB}\ln\frac{1-y_\infty}{1-y_s}.
$$

**Is pseudo-steady justified?** `stefan.moving_boundary` solves the full problem. In the interface frame
$\zeta = (\text{height above liquid})/Z(t)$. For an isothermal, isobaric ideal gas $c$ is constant, so total
continuity makes the molar flux $N = N_A + N_B$ uniform in height. At the interface moving down at
$u = \dot Z$, liquid lost $c_L u$ must supply both the outgoing flux and the vapour that fills the new gas
volume, while air flows down into that volume:

$$
c_L u = N_{A,s} + c\,y_s u,\qquad 0 = N_{B,s} + c(1-y_s)u
\;\Rightarrow\; N = (c_L - c)u,\quad u = -\frac{cD\,y_\zeta(0)}{Z\,c_L(1-y_s)}.
$$

The species equation in $\zeta$ becomes $y_t = \frac{D}{Z^2}y_{\zeta\zeta} - \left[\frac{N}{cZ} + (1-\zeta)\frac{u}{Z}\right]y_\zeta$.
With an initially vapour-free column, the full solution differs from the pseudo-steady law only by a
nearly constant offset of about $(c/c_L)z_0 y_s/3 \approx 2\ \mu$m: the column stores vapour worth
$y_s/2$ on average, minus the permeation time-lag deficit worth $y_s/6$. That is far below any level
reading, and the slope $K$ agrees to 0.2 % (`figures/verification/stefan_tube.png`).

## 4. Open dish: film model with Stefan correction

The code defines $k_c$ through

$$
N_A = k_c\,c_f\,(y_s - y_\infty)\,\phi,\qquad \phi = \frac{\ln[(1-y_\infty)/(1-y_s)]}{y_s - y_\infty},
$$

so $N_A = k_c c_f\ln[(1-y_\infty)/(1-y_s)]$, which is Stefan's law with $k_c = D/\delta$ (film thickness
$\delta$). The measured $k_c$ uses the **measured liquid temperature** in $y_s = P_{sat}(T_s)/P$.

**Correlations.** Forced flow: $Sh_L = 0.664Re_L^{1/2}Sc^{1/3}$ (course flat-plate form, $L$ = dish
diameter). Still air: free convection driven by the gas density difference

$$
Ra_L = \frac{g\,|\rho_s - \rho_\infty|}{\bar\rho}\frac{L^3}{\nu D_{AB}},\qquad L = A/P = R/2 .
$$

Ethanol vapour ($M = 46$) is heavier than air and the surface is cold, so $\rho_s > \rho_\infty$: a
**stable** layer, $Sh = 0.52Ra^{1/5}$ (Incropera Eq. 9.32), not the familiar $0.54Ra^{1/4}$ for an
evaporating water surface. The pure-diffusion disk, $\dot m = 4RD\Delta c$ (Weber), is the lower bound.

**Water in the same film.** The correlation is evaluated for ethanol ($Sc = 1.26$). For water
($Sc = 0.61$, below the $Pr\ge0.7$ range of the free-convection correlations) the code scales
$k_{c,W} = k_{c,E}(D_W/D_E)^n$ with the exponent the same correlation implies: $n = 2/3$ for
$Sh\propto Sc^{1/3}$, $n = 4/5$ for $Sh\propto Ra^{1/5}$ (since $Ra\propto1/D$), $n = 1$ for pure diffusion.

## 5. Evaporative cooling and the heat-mass analogy

Surface energy balance (liquid plus dish, well mixed):

$$
\Big(\sum_i n_iM_ic_{p,i} + C_{dish}\Big)\frac{dT_s}{dt} = (hA + UA_b)(T_\infty - T_s)
+ \varepsilon\sigma A(T_{sur}^4 - T_s^4) - A\sum_i N_i\,\Delta H_{vap,i}(T_s).
$$

$h$ comes from Chilton-Colburn, $j_H = j_D \Rightarrow h = k_c\rho c_p Le^{2/3}$, $Le = \alpha/D_{AB} = 1.8$.
Setting $dT_s/dt = 0$ gives the psychrometric wet-bulb temperature of ethanol. Without radiation and
bottom conduction both sides scale with $k_c$, so the wet-bulb temperature does not depend on the air
speed (3.9 °C in dry 25 °C air); radiation and the table raise it, more so in still air.

**Why it matters.** At 10 °C ethanol's vapour pressure is 40 % of its 25 °C value. Assuming the liquid
stays at room temperature over-predicts the rate 2-3 times (`docs/design_estimates.md`).

## 6. Liquid below the rim

`recess_model = "stagnant_layer"` adds $h_r/D$ to $1/k_c$ (and $h_r/k_{air}$ to $1/h$). That treats the
gas in the recess as stagnant, so it is an upper bound on the extra resistance. The 2D axisymmetric
diffusion solution in still air (`diffusion2d.py`) gives a smaller effect than even the 1D stagnant-layer
estimate (`figures/verification/dish_diffusion.png`). Flowing air ventilates a shallow recess further.
Practical consequence: use a shallow dish filled near the rim.

## 7. Aqueous ethanol

Surface vapour from modified Raoult's law, $y_{i,s} = x_i\gamma_iP_{sat,i}(T_s)/P$, with NRTL
($\tau_{ij} = a_{ij} + b_{ij}/T$, $\alpha = 0.3$; parameters and formula as in
[Ethanol-Water-VLE](https://github.com/Shuruyue/Ethanol-Water-VLE), which reproduces the 1 atm azeotrope;
`tests/test_evaporation.py` checks the same azeotrope here). The parameters were regressed near
350-370 K; at 280-300 K their temperature dependence is extrapolated. Humid air supplies water vapour:
when the surface is colder than the dew point, water condenses into the liquid ($N_W < 0$). This also
happens to "pure" ethanol, which the pure-ethanol fit neglects; the analysis reports how much.

## 8. Uncertainty

- Fitted parameters: linearized 95 % intervals from $\mathrm{Cov} = s^2(J^TJ)^{-1}$. They assume
  independent, unbiased residuals. When the model form is wrong (systematic residuals) they are too
  narrow, which is why the Week 1 transient is fitted with two loss models (constant $h$ vs
  correlation-shaped $h$) and the spread in $k$ is reported as model-form uncertainty.
- Property and set-up uncertainty ($k$, $\varepsilon$, $T_\infty$, $D_{AB}$, fitted coefficients):
  Monte Carlo with a recorded seed.
- Synthetic-data recovery (`tests/test_inverse.py`, `config/synthetic/`): each fit recovers parameters
  it did not see. In the synthetic Week 2 run the $k_c$ multiplier comes out lower than its true value
  because the analysis uses Fuller's $D$ and the anemometer reading, both slightly off on purpose. With
  the true $D$ and $U$ the same pipeline returns the true multiplier and bottom conductance. On real data
  the multiplier therefore measures "correlation + property + speed" error together.

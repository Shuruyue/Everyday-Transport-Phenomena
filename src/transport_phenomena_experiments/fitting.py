"""Parameter estimation with honest uncertainties.

Nonlinear fits use scipy.optimize.least_squares. The covariance is the usual
linearized estimate Cov = s^2 (J^T J)^-1 at the optimum, s^2 = SSR / (n - p),
computed through the SVD of J so that a rank-deficient (non-identifiable)
problem raises instead of returning meaningless error bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal, stats


@dataclass(frozen=True)
class FitResult:
    names: tuple[str, ...]
    values: np.ndarray
    stderr: np.ndarray
    ci95: np.ndarray  # half-width of the 95 % confidence interval
    cov: np.ndarray
    corr: np.ndarray
    residuals: np.ndarray
    dof: int
    s: float  # residual standard deviation

    def __getitem__(self, name):
        return float(self.values[self.names.index(name)])

    def ci(self, name):
        return float(self.ci95[self.names.index(name)])

    def summary(self):
        lines = [f"{'parameter':>16s} {'value':>14s} {'95% CI (+/-)':>14s} {'rel.':>8s}"]
        for n, v, c in zip(self.names, self.values, self.ci95):
            lines.append(f"{n:>16s} {v:14.6g} {c:14.4g} {abs(c / v) if v != 0 else np.inf:8.2%}")
        lines.append(f"residual s = {self.s:.4g}, dof = {self.dof}")
        if len(self.names) > 1:
            iu = np.triu_indices(len(self.names), 1)
            for i, j in zip(*iu):
                lines.append(f"corr({self.names[i]}, {self.names[j]}) = {self.corr[i, j]:+.3f}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "values": dict(zip(self.names, map(float, self.values))),
            "ci95": dict(zip(self.names, map(float, self.ci95))),
            "stderr": dict(zip(self.names, map(float, self.stderr))),
            "corr": self.corr.tolist(),
            "residual_s": self.s,
            "dof": self.dof,
        }


def _covariance_from_jacobian(J, s2, rcond):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    if S[-1] <= rcond * S[0]:
        raise np.linalg.LinAlgError(
            f"Jacobian is rank deficient (singular values {S}); the parameters are not "
            "identifiable from these data. Fix one of them or add data."
        )
    return s2 * (Vt.T / S**2) @ Vt


def fit_nonlinear(residual_fn, p0, names, bounds, x_scale, diff_step=None, rcond=1e-10):
    """Least-squares fit of ``residual_fn(p) -> residual vector``.

    bounds and x_scale are passed to least_squares explicitly because a wrong
    scale silently degrades convergence for parameters spanning many decades.
    diff_step: relative finite-difference step for the Jacobian. Residuals that
    come from an adaptive ODE solver carry solver noise of order rtol, so they
    need a step well above it (1e-4 is used for those fits); None keeps scipy's
    default (~1.5e-8) for closed-form models.

    The confidence intervals assume independent, unbiased residuals. When the
    model form is wrong (systematic residuals) they understate the real
    uncertainty; compare model variants to see the model-form part.
    """
    res = optimize.least_squares(residual_fn, np.asarray(p0, dtype=float), bounds=bounds,
                                 x_scale=x_scale, method="trf", xtol=1e-12, ftol=1e-12,
                                 gtol=1e-12, max_nfev=2000, diff_step=diff_step)
    if not res.success:
        raise RuntimeError(f"least_squares did not converge: {res.message}")
    n, p = res.fun.size, res.x.size
    dof = n - p
    if dof <= 0:
        raise ValueError(f"need more data points ({n}) than parameters ({p})")
    s2 = float(res.fun @ res.fun) / dof
    cov = _covariance_from_jacobian(res.jac, s2, rcond)
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    t = stats.t.ppf(0.975, dof)
    return FitResult(tuple(names), res.x, se, t * se, cov, corr, res.fun, dof, float(np.sqrt(s2)))


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    slope_ci95: float
    intercept_ci95: float
    r2: float
    dof: int
    residuals: np.ndarray


def fit_linear(x, y, through_origin):
    """Ordinary least squares y = a x (+ b), with standard errors and 95 % CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.ndim == 1
    X = x[:, None] if through_origin else np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    dof = x.size - X.shape[1]
    if dof <= 0:
        raise ValueError("not enough points for a linear fit")
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = stats.t.ppf(0.975, dof)
    ss_tot = float(((y - y.mean()) ** 2).sum()) if not through_origin else float((y**2).sum())
    r2 = 1.0 - float(r @ r) / ss_tot
    if through_origin:
        return LinearFit(float(coef[0]), 0.0, float(se[0]), 0.0, float(t * se[0]), 0.0, r2, dof, r)
    return LinearFit(float(coef[0]), float(coef[1]), float(se[0]), float(se[1]),
                     float(t * se[0]), float(t * se[1]), r2, dof, r)


def savgol_derivative(t, y, window, polyorder):
    """dy/dt by a Savitzky-Golay filter. Requires uniform sampling."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=0.0):
        raise ValueError("Savitzky-Golay derivative needs uniformly spaced samples")
    return signal.savgol_filter(np.asarray(y, dtype=float), window, polyorder, deriv=1,
                                delta=float(dt[0]), mode="interp")


def monte_carlo(model, sampler, n, rng):
    """Propagate input uncertainty: returns [model(sampler(rng)) for n draws].

    ``rng`` must be a numpy Generator created from a recorded seed.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("pass a numpy.random.Generator with a recorded seed")
    return np.array([model(sampler(rng)) for _ in range(n)])

"""1D finite-volume model of a rod heated at one end (vertex-centred grid).

    rho c A_c dT/dt = d/dx (k A_c dT/dx) - P q''(T)
    q''(T) = h_c(T) (T - T_inf) + eps sigma (T^4 - T_sur^4)

Nodes x_i = i dx, i = 0..N. Node 0 is the base (Dirichlet, T_b(t) measured).
Interior nodes own a control volume of length dx; the tip node owns dx/2, which
makes the adiabatic tip exactly the ghost-node condition and keeps the scheme
second-order accurate (verified in tests/test_fin.py). A_c(x) and P(x) may vary
along the rod (tapered chopstick, spoon handle).

The steady problem with temperature-dependent h is solved by Picard iteration
on the linearized loss coefficients; the transient problem by the method of
lines with a stiff (BDF) integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, linalg, sparse

from ..constants import SIGMA


@dataclass(frozen=True)
class SurfaceLoss:
    """Lateral heat loss q''(T) = h_conv(T) (T - T_inf) + eps sigma (T^4 - T_sur^4)."""

    h_conv: Callable[[np.ndarray], np.ndarray]  # W/(m^2 K) as a function of T [K]
    T_inf: float  # K
    eps: float  # -
    T_sur: float  # K

    def flux(self, T):
        T = np.asarray(T, dtype=float)
        return self.h_conv(T) * (T - self.T_inf) + self.eps * SIGMA * (T**4 - self.T_sur**4)

    def coefficients(self, T):
        """(H_c, H_r) with q'' = H_c (T - T_inf) + H_r (T - T_sur), exact at T."""
        T = np.asarray(T, dtype=float)
        H_r = self.eps * SIGMA * (T + self.T_sur) * (T**2 + self.T_sur**2)
        return self.h_conv(T), H_r


@dataclass(frozen=True)
class Rod:
    L: float  # exposed length [m], base at x = 0
    n_cells: int
    Ac: Callable[[np.ndarray], np.ndarray]  # cross-section area [m^2] vs x [m]
    P: Callable[[np.ndarray], np.ndarray]  # perimeter [m] vs x [m]
    k: float  # W/(m K)
    rho_c: float  # volumetric heat capacity [J/(m^3 K)]
    tip: str  # "adiabatic" or "convective"

    def __post_init__(self):
        if self.tip not in ("adiabatic", "convective"):
            raise ValueError(f"tip must be 'adiabatic' or 'convective', got {self.tip!r}")
        if self.n_cells < 2:
            raise ValueError("need at least 2 cells")

    @property
    def x(self):
        return np.linspace(0.0, self.L, self.n_cells + 1)

    def _geometry(self):
        x = self.x
        dx = x[1] - x[0]
        xf = 0.5 * (x[:-1] + x[1:])  # faces i+1/2, i = 0..N-1
        G = self.k * self.Ac(xf) / dx  # face conductances [W/K]
        Ac_n, P_n = self.Ac(x), self.P(x)
        w = np.ones_like(x)
        w[-1] = 0.5  # tip node owns half a cell
        V = Ac_n * dx * w  # node volumes (node 0 unused)
        S = P_n * dx * w  # lateral areas
        if self.tip == "convective":
            S = S.copy()
            S[-1] += Ac_n[-1]
        return x, dx, G, V, S

    # ------------------------------------------------------------------ steady
    def steady(self, T_b, loss, tol=1e-10, max_iter=200):
        """Steady temperatures at the nodes [K] for base temperature T_b [K]."""
        x, dx, G, V, S = self._geometry()
        N = self.n_cells
        T = np.full(N + 1, loss.T_inf, dtype=float)
        T[0] = T_b
        for it in range(max_iter):
            Hc, Hr = loss.coefficients(T[1:])
            Gw = G  # G[i-1] is the face between node i-1 and i, for unknown i = 1..N
            Ge = np.append(G[1:], 0.0)  # face to the east; zero at the tip
            diag = Gw + Ge + S[1:] * (Hc + Hr)
            rhs = S[1:] * (Hc * loss.T_inf + Hr * loss.T_sur)
            rhs[0] += G[0] * T_b
            ab = np.zeros((3, N))
            ab[0, 1:] = -Ge[:-1]  # super-diagonal
            ab[1, :] = diag
            ab[2, :-1] = -Gw[1:]  # sub-diagonal
            T_new = linalg.solve_banded((1, 1), ab, rhs)
            change = np.max(np.abs(T_new - T[1:]))
            T[1:] = T_new
            if change < tol:
                return T
        raise RuntimeError(f"Picard iteration did not converge in {max_iter} iterations")

    def base_heat_rate(self, T, loss):
        """Heat entering the rod at x = 0 [W] for a STEADY nodal solution T."""
        x, dx, G, V, S = self._geometry()
        half_cell_loss = self.P(np.array([0.0]))[0] * dx / 2 * loss.flux(T[0])
        return G[0] * (T[0] - T[1]) + half_cell_loss

    # --------------------------------------------------------------- transient
    def transient(self, t_eval, T_b_fn, T_init, loss, rtol=1e-8, atol=1e-8):
        """Nodal temperatures [K] at t_eval [s], shape (len(t_eval), N + 1).

        T_b_fn(t) gives the base temperature; T_init are the initial nodal values.
        """
        x, dx, G, V, S = self._geometry()
        N = self.n_cells
        T_init = np.asarray(T_init, dtype=float)
        assert T_init.shape == (N + 1,)
        Ge = np.append(G[1:], 0.0)
        C = self.rho_c * V[1:]

        def rhs(t, Tu):
            Tw = np.concatenate(([T_b_fn(t)], Tu[:-1]))
            Te = np.append(Tu[1:], 0.0)
            q = G * (Tw - Tu) + Ge * (Te - Tu) - S[1:] * loss.flux(Tu)
            return q / C

        jac_pattern = sparse.diags([np.ones(N - 1), np.ones(N), np.ones(N - 1)], [-1, 0, 1])
        sol = integrate.solve_ivp(rhs, (t_eval[0], t_eval[-1]), T_init[1:], method="BDF",
                                  t_eval=t_eval, rtol=rtol, atol=atol, jac_sparsity=jac_pattern)
        if not sol.success:
            raise RuntimeError(f"transient fin solve failed: {sol.message}")
        Tb = np.array([T_b_fn(t) for t in sol.t])
        return np.column_stack([Tb, sol.y.T])

    def energy_residual(self, t, T_nodes, loss):
        """Discrete energy balance check over a transient: returns
        (stored energy change, integrated base heat in - lateral loss) in J.
        The two agree to integration tolerance if the solver conserves energy."""
        x, dx, G, V, S = self._geometry()
        E = (self.rho_c * V[1:] * T_nodes[:, 1:]).sum(axis=1)
        q_in = G[0] * (T_nodes[:, 0] - T_nodes[:, 1])
        q_out = (S[1:] * loss.flux(T_nodes[:, 1:])).sum(axis=1)
        net = integrate.trapezoid(q_in - q_out, t)
        return E[-1] - E[0], net


def uniform_section(Ac, P):
    """Helpers turning constants into the callables Rod expects."""
    return (lambda x: np.full_like(np.asarray(x, dtype=float), Ac),
            lambda x: np.full_like(np.asarray(x, dtype=float), P))

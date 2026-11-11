"""Steady vapour diffusion from a dish into still air, axisymmetric (r, z).

    (1/r) d/dr (r dc/dr) + d^2c/dz^2 = 0

Normalized concentration c = 1 on the liquid surface (radius R, depth h_r below
the rim), dc/dn = 0 on the dish wall and on the plane z = 0 outside the dish,
and the monopole far-field condition c ~ 1/rho on the outer boundary, which
removes most of the domain-truncation error.

For h_r = 0 the exact evaporation rate is Q = 4 D R c_s (disk on an insulating
plane, Weber's result), i.e. Q / (D R c_s) = 4. That is the verification case.
With h_r > 0 the solver gives how much the dry wall above the liquid slows
evaporation, compared in docs with the 1D series-resistance estimate
1/Q = 1/(4 D R) + h_r / (D pi R^2).

This is pure diffusion (no convection): the lower bound for "still air".

Finite volumes on a tensor grid, graded geometrically towards the singular
lines (disk edge, rim corner). Solved with a sparse direct solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def _graded(a, b, h0, growth, h_max):
    """Faces from a to b (either direction): first cell h0 at a, growing by `growth`."""
    L = abs(b - a)
    sign = 1.0 if b > a else -1.0
    sizes = []
    h = h0
    total = 0.0
    while total + h < L:
        sizes.append(h)
        total += h
        h = min(h * growth, h_max)
    rem = L - total
    if sizes and rem < 0.5 * sizes[-1]:
        sizes[-1] += rem
    else:
        sizes.append(rem)
    return a + sign * np.concatenate(([0.0], np.cumsum(sizes)))


@dataclass(frozen=True)
class DishDiffusionResult:
    Q_norm: float  # Q / (D R c_s)
    n_cells: int
    r_faces: np.ndarray
    z_faces: np.ndarray
    c: np.ndarray  # (n_r, n_z) normalized concentration, NaN in the solid
    mask: np.ndarray  # active cells


def solve_dish(R, h_r, h0, growth, far_factor):
    """Q / (D R c_s) for a dish of radius R with the liquid h_r below the rim.

    h0: smallest cell (at the disk edge / rim), growth: geometric ratio,
    far_factor: outer boundary at far_factor * R.
    """
    h_max_near = R / 8
    r_in = _graded(R, 0.0, h0, growth, h_max_near)[::-1]  # 0 .. R, fine at R
    r_out = _graded(R, far_factor * R, h0, growth, far_factor * R)  # R .. far, fine at R
    rf = np.concatenate((r_in, r_out[1:]))
    zf_up = _graded(0.0, far_factor * R, h0, growth, far_factor * R)
    if h_r > 0:
        n_pocket = max(4, int(np.ceil(h_r / min(R / 8, 4 * h0))))
        # pocket: fine near the rim (z = 0) and near the liquid (z = -h_r)
        half = _graded(0.0, -h_r / 2, h0, growth, h_r / n_pocket * 2)
        other = _graded(-h_r, -h_r / 2, h0, growth, h_r / n_pocket * 2)
        zf_low = np.concatenate((other, half[::-1][1:]))  # -h_r .. 0
        zf = np.concatenate((zf_low, zf_up[1:]))
    else:
        zf = zf_up
    rc, zc = 0.5 * (rf[:-1] + rf[1:]), 0.5 * (zf[:-1] + zf[1:])
    dz = np.diff(zf)
    nr, nz = rc.size, zc.size
    active = (zc[None, :] > 0) | (rc[:, None] < R)
    idx = -np.ones((nr, nz), dtype=int)
    idx[active] = np.arange(active.sum())
    n = int(active.sum())

    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    rhs = np.zeros(n)
    j_liq = 0  # the liquid is the bottom face of the first z row (z = -h_r or 0)
    G_liq = np.zeros(nr)

    def couple(p, q, G):
        diag[p] += G
        diag[q] += G
        rows.extend((p, q))
        cols.extend((q, p))
        vals.extend((-G, -G))

    # r-direction faces (between i and i+1)
    for i in range(nr - 1):
        for j in range(nz):
            p, q = idx[i, j], idx[i + 1, j]
            if p >= 0 and q >= 0:
                couple(p, q, 2 * np.pi * rf[i + 1] * dz[j] / (rc[i + 1] - rc[i]))
    # z-direction faces
    for i in range(nr):
        A = np.pi * (rf[i + 1] ** 2 - rf[i] ** 2)
        for j in range(nz - 1):
            p, q = idx[i, j], idx[i, j + 1]
            if p >= 0 and q >= 0:
                couple(p, q, A / (zc[j + 1] - zc[j]))
        # liquid surface (Dirichlet c = 1) under the disk only
        if rc[i] < R:
            p = idx[i, j_liq]
            G = A / (zc[j_liq] - zf[0])
            diag[p] += G
            rhs[p] += G
            G_liq[i] = G
    # far field, monopole c ~ 1/rho centred at the origin
    for j in range(nz):  # outer r boundary
        p = idx[nr - 1, j]
        if p >= 0:
            rho_c, rho_f = np.hypot(rc[-1], zc[j]), np.hypot(rf[-1], zc[j])
            diag[p] += 2 * np.pi * rf[-1] * dz[j] * (rho_c / rho_f) * (rf[-1] / rho_f) / rho_f
    for i in range(nr):  # top boundary
        p = idx[i, nz - 1]
        rho_c, rho_f = np.hypot(rc[i], zc[-1]), np.hypot(rc[i], zf[-1])
        A = np.pi * (rf[i + 1] ** 2 - rf[i] ** 2)
        diag[p] += A * (rho_c / rho_f) * (zf[-1] / rho_f) / rho_f

    M = sparse.csr_matrix((np.concatenate((vals, diag)),
                           (np.concatenate((rows, np.arange(n))),
                            np.concatenate((cols, np.arange(n))))), shape=(n, n))
    c_vec = spla.spsolve(M.tocsc(), rhs)
    if not np.all(np.isfinite(c_vec)):
        raise RuntimeError("diffusion solve produced non-finite values")
    if c_vec.min() < -1e-9 or c_vec.max() > 1 + 1e-9:
        raise RuntimeError("maximum principle violated: concentration outside [0, 1]")
    c = np.full((nr, nz), np.nan)
    c[active] = c_vec
    Q = sum(G_liq[i] * (1.0 - c[i, j_liq]) for i in range(nr) if rc[i] < R)
    return DishDiffusionResult(Q_norm=Q / R, n_cells=n, r_faces=rf, z_faces=zf, c=c, mask=active)


def series_resistance_estimate(R, h_r):
    """1D estimate of Q/(D R c_s): disk-diffusion resistance plus a stagnant column."""
    return 1.0 / (1.0 / 4.0 + h_r / (np.pi * R))

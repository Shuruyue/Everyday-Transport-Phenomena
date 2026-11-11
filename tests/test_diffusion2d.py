import numpy as np

from etp.mass.diffusion2d import series_resistance_estimate, solve_dish


def test_flush_disk_converges_to_weber_solution():
    errs = [abs(solve_dish(1.0, 0.0, h0, 1.08, 60.0).Q_norm / 4.0 - 1)
            for h0 in (0.02, 0.01, 0.005)]
    assert errs[-1] < 0.004
    assert errs[0] > errs[1] > errs[2]


def test_far_boundary_is_not_limiting():
    a = solve_dish(1.0, 0.0, 0.01, 1.15, 40.0).Q_norm
    b = solve_dish(1.0, 0.0, 0.01, 1.15, 160.0).Q_norm
    assert abs(a / b - 1) < 1e-3


def test_recess_reduces_rate_monotonically():
    q = [solve_dish(1.0, hr, 0.01, 1.15, 60.0).Q_norm for hr in (0.0, 0.1, 0.3)]
    assert q[0] > q[1] > q[2]
    # the 1D stagnant-layer estimate overpredicts the rate here (it ignores how the
    # wall blocks the edge-concentrated flux) but stays within 10 %
    for hr, qq in zip((0.1, 0.3), q[1:]):
        assert abs(series_resistance_estimate(1.0, hr) / qq - 1) < 0.10


def test_concentration_bounded():
    r = solve_dish(1.0, 0.2, 0.02, 1.15, 40.0)
    c = r.c[r.mask]
    assert np.all((c >= 0) & (c <= 1))

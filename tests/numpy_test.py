"""Tests for the numpy version."""

import numpy as np

import sw_solver


def test_numpy():
    """Test against serialized data."""
    IC = sw_solver.ICType.RossbyHaurwitzWave

    T = 2

    M = 10
    N = 10

    CFL = 0.5

    diffusion = True

    save_data = {"interval": 392}
    sw_solver.numpy.solve(M, N, IC, T, CFL, diffusion, save_data=save_data)
    h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
    phi, theta = save_data["phi"], save_data["theta"]

    for array, var in zip(
        (h, u, v, t, phi, theta), ("h", "u", "v", "t", "phi", "theta")
    ):
        assert np.allclose(array, np.load(f"data/swes-numpy-ref-10-{var}.npy"))

"""Tests for the numpy version."""

import numpy as np
import pytest

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


@pytest.mark.parametrize("use_diffusion", (True, False))
def test_immutability(use_diffusion: bool):
    """Test to ensure that input arrays are not modified."""
    num_lon_pts = 4
    num_lat_pts = 4

    latlon_grid = sw_solver.grid.LatLonGrid(num_lon_pts, num_lat_pts)
    cart_grid = sw_solver.grid.CartesianGrid(latlon_grid)

    # Note: currently just a flat surface
    hs_0 = np.zeros(latlon_grid.shape, float)

    h_0, u_0, v_0, f_0 = sw_solver.ic.get_initial_conditions(
        sw_solver.ICType.RossbyHaurwitzWave, latlon_grid
    )

    h, u, v, f, hs = (var.copy() for var in (h_0, u_0, v_0, f_0, hs_0))

    dt = 1.25

    hnew, unew, vnew = sw_solver.numpy.lax_wendroff_update(
        latlon_grid, cart_grid, dt, f, hs, h, u, v
    )

    diff_coeff = (
        sw_solver.numpy.DiffusionCoefficients(cart_grid) if use_diffusion else None
    )

    # Diffusion
    if diff_coeff:
        hnew += (
            dt
            * sw_solver.utils.EARTH_CONSTANTS.nu
            * sw_solver.numpy.compute_diffusion(h, diff_coeff)
        )
        unew += (
            dt
            * sw_solver.utils.EARTH_CONSTANTS.nu
            * sw_solver.numpy.compute_diffusion(u, diff_coeff)
        )
        vnew += (
            dt
            * sw_solver.utils.EARTH_CONSTANTS.nu
            * sw_solver.numpy.compute_diffusion(v, diff_coeff)
        )

    for var_before, var_after in ((h_0, h), (u_0, u), (v_0, v), (f_0, f), (hs_0, hs)):
        assert np.allclose(var_before, var_after)

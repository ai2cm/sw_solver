"""Tests for the gt4py version."""

import gt4py
import numpy as np
import pytest

import sw_solver
from sw_solver.utils import EARTH_CONSTANTS, FloatT


@pytest.mark.parametrize(
    "ic_type",
    (sw_solver.ICType.RossbyHaurwitzWave, sw_solver.ICType.ZonalGeostrophicFlow),
)
def test_gt4py_lax_wendroff_numpy(ic_type: sw_solver.ICType):
    """Compare the GT4Py Lax-Wendroff update to numpy."""
    latlon_grid = sw_solver.LatLonGrid(10, 20)
    cart_grid = sw_solver.CartesianGrid(latlon_grid)

    # Note: currently just a flat surface
    hs = np.zeros(latlon_grid.shape, FloatT)

    # diff_coeff = DiffusionCoefficients(cart_grid) if use_diffusion else None
    h, u, v, f = sw_solver.ic.get_initial_conditions(ic_type, latlon_grid)

    # --- Compute timestep through CFL condition --- #
    # Compute flux Jacobian eigenvalues
    eigenx = (
        np.maximum(
            np.absolute(u - np.sqrt(EARTH_CONSTANTS.g * np.absolute(h))),
            np.maximum(
                np.absolute(u),
                np.absolute(u + np.sqrt(EARTH_CONSTANTS.g * np.absolute(h))),
            ),
        )
    ).max()

    eigeny = (
        np.maximum(
            np.absolute(v - np.sqrt(EARTH_CONSTANTS.g * np.absolute(h))),
            np.maximum(
                np.absolute(v),
                np.absolute(v + np.sqrt(EARTH_CONSTANTS.g * np.absolute(h))),
            ),
        )
    ).max()

    # Compute timestep
    dtmax = np.minimum(cart_grid.dxmin / eigenx, cart_grid.dymin / eigeny)
    dt = 0.5 * dtmax

    # --- Update solution --- #
    h_new_np, u_new_np, v_new_np = sw_solver.numpy.lax_wendroff_update(
        latlon_grid, cart_grid, None, dt, f, hs, h, u, v
    )

    # --- gt4py ---

    gt4py_backend = "gtc:numpy"

    h_gt, u_gt, v_gt = (
        sw_solver.gt4py.StorageAllocator().from_array(
            arr[:, :, np.newaxis],
            backend=gt4py_backend,
            default_origin=(0, 0, 0),
            shape=list(arr.shape) + [1],
        )
        for arr in (h, u, v)
    )

    h_new_gt, u_new_gt, v_new_gt = (
        sw_solver.gt4py.StorageAllocator().empty_like(var) for var in (h_gt, u_gt, v_gt)
    )

    f = sw_solver.gt4py.StorageAllocator().from_array(
        f, backend=gt4py_backend, default_origin=(0, 0), mask=(True, True, False)
    )

    phi = sw_solver.gt4py.StorageAllocator().from_array(
        latlon_grid.phi, backend=gt4py_backend, default_origin=(1, 1, 0)
    )
    theta = sw_solver.gt4py.StorageAllocator().from_array(
        latlon_grid.theta, backend=gt4py_backend, default_origin=(1, 1, 0)
    )

    lax_wendroff_stencil = gt4py.gtscript.stencil(
        definition=sw_solver.gt4py.lax_wendroff_definition, backend=gt4py_backend
    )

    lax_wendroff_stencil(
        phi,
        theta,
        f,
        hs,
        dt,
        h_gt,
        u_gt,
        v_gt,
        EARTH_CONSTANTS.a,
        EARTH_CONSTANTS.g,
        h_new_gt,
        u_new_gt,
        v_new_gt,
    )
    print(np.abs(h_new_np - h_new_gt[1:-1, 1:-1, 0]) / h_new_np)

    assert np.allclose(h_new_np, h_new_gt[1:-1, 1:-1, 0])
    assert np.allclose(u_new_np, u_new_gt[1:-1, 1:-1, 0])
    assert np.allclose(v_new_np, v_new_gt[1:-1, 1:-1, 0])

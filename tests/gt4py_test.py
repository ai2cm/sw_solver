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
    h_0, u_0, v_0, f = sw_solver.ic.get_initial_conditions(ic_type, latlon_grid)

    # Take copies to ensure nothing is mutated in the numpy version call.
    h = h_0.copy()
    u = u_0.copy()
    v = v_0.copy()

    # Probably does not satisfy physicality constraints,
    # but this is only a test of the update.
    dt = 1.25

    # --- Update solution --- #
    (
        h_new_np,
        u_new_np,
        v_new_np,
    ) = sw_solver.numpy.lax_wendroff_update(latlon_grid, cart_grid, dt, f, hs, h, u, v)

    # --- gt4py ---
    gt4py_backend = "gtc:numpy"
    nk_levels = 1

    h_gt, u_gt, v_gt = (
        sw_solver.gt4py.StorageAllocator().from_array(
            np.repeat(arr[:, :, np.newaxis], nk_levels, axis=2),
            backend=gt4py_backend,
            default_origin=(1, 1, 0),
        )
        for arr in (h_0, u_0, v_0)
    )

    h_new_gt, u_new_gt, v_new_gt = (
        sw_solver.gt4py.StorageAllocator().zeros(
            shape=list(var.shape) + [nk_levels],
            backend=gt4py_backend,
            default_origin=(0, 0, 0),
            dtype=var.dtype,
        )
        for var in (h_new_np, u_new_np, v_new_np)
    )

    f_gt = sw_solver.gt4py.StorageAllocator().from_array(
        f, backend=gt4py_backend, default_origin=(1, 1), mask=(True, True, False)
    )

    phi, theta = (
        sw_solver.gt4py.StorageAllocator().from_array(
            x, backend=gt4py_backend, default_origin=(1, 1, 0)
        )
        for x in (latlon_grid.phi, latlon_grid.theta)
    )

    lax_wendroff_stencil = gt4py.gtscript.stencil(
        definition=sw_solver.gt4py.lax_wendroff_definition,
        backend=gt4py_backend,
        oir_pipeline=sw_solver.gt4py.oir_pipeline,
    )

    hs_gt = sw_solver.gt4py.StorageAllocator().from_array(
        hs,
        backend=gt4py_backend,
        default_origin=(1, 1),
        shape=hs.shape,
        mask=(True, True, False),
    )

    lax_wendroff_stencil(
        phi,
        theta,
        f_gt,
        hs_gt,
        h_gt,
        u_gt,
        v_gt,
        h_new_gt,
        u_new_gt,
        v_new_gt,
        EARTH_CONSTANTS.a,
        EARTH_CONSTANTS.g,
        dt,
        domain=h_new_gt.shape,
    )

    assert np.allclose(h_new_np, h_new_gt[:, :, 0])
    assert np.allclose(u_new_np, u_new_gt[:, :, 0])
    assert np.allclose(v_new_np, v_new_gt[:, :, 0])

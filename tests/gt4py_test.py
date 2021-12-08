"""Tests for the gt4py version."""

import gt4py.definitions as gt_defs
import numpy as np
import pytest

import sw_solver
from sw_solver.utils import EARTH_CONSTANTS, FloatT


@pytest.mark.parametrize("use_diffusion", (True, False))
def test_gt4py_immutability(use_diffusion: bool):
    """Check that input-only fields are detected as such in the gt4py compiler."""
    if use_diffusion:
        pytest.skip("Diffusion not yet implemented")

    lax_wendroff_stencil = sw_solver.gt4py.make_stencil(
        definition=sw_solver.gt4py.lax_wendroff_definition
    )

    read_only_field_infos = [
        lax_wendroff_stencil.field_info[name].access
        for name in ("h", "u", "v", "f", "hs")
    ]
    assert all(x == gt_defs.AccessKind.READ for x in read_only_field_infos)


@pytest.mark.parametrize("gt4py_backend", ("gtc:numpy",))
@pytest.mark.parametrize(
    "ic_type",
    (sw_solver.ICType.RossbyHaurwitzWave, sw_solver.ICType.ZonalGeostrophicFlow),
)
def test_gt4py_lax_wendroff_numpy(ic_type: sw_solver.ICType, gt4py_backend: str):
    """Compare the GT4Py Lax-Wendroff update to numpy."""
    latlon_grid = sw_solver.LatLonGrid(10, 20)
    cart_grid = sw_solver.CartesianGrid(latlon_grid)

    # Note: currently just a flat surface
    hs = np.zeros(latlon_grid.shape, FloatT)

    # diff_coeff = DiffusionCoefficients(cart_grid) if use_diffusion else None
    h_0, u_0, v_0, f = sw_solver.ic.get_initial_conditions(ic_type, latlon_grid)

    # Probably does not satisfy physicality constraints,
    # but this is only a test of the update.
    dt = 1.25

    # --- numpy ---
    (h_new_np, u_new_np, v_new_np,) = sw_solver.numpy.lax_wendroff_update(
        latlon_grid, cart_grid, dt, f, hs, h_0, u_0, v_0
    )

    # --- gt4py ---
    nk_levels = 1

    def make_storage_from_array(*args, **kwargs):
        return sw_solver.gt4py.make_storage_from_array(
            *args, **kwargs, backend=gt4py_backend
        )

    def make_storage_from_shape(*args, **kwargs):
        return sw_solver.gt4py.make_storage_from_shape(
            *args, **kwargs, backend=gt4py_backend
        )

    h_gt, u_gt, v_gt = (
        make_storage_from_array(
            np.repeat(arr[:, :, np.newaxis], nk_levels, axis=2),
            default_origin=(1, 1, 0),
        )
        for arr in (h_0, u_0, v_0)
    )

    # NOTE: The output storages are two smaller in I and J, since input points are consumed.
    h_new_gt, u_new_gt, v_new_gt = (
        make_storage_from_shape(list(var.shape) + [nk_levels])
        for var in (h_new_np, u_new_np, v_new_np)
    )

    f_gt = make_storage_from_array(f, default_origin=(1, 1))

    phi, theta = (
        make_storage_from_array(x, default_origin=(1, 1))
        for x in (latlon_grid.phi, latlon_grid.theta)
    )

    lax_wendroff_stencil = sw_solver.gt4py.make_stencil(
        definition=sw_solver.gt4py.lax_wendroff_definition
    )

    hs_gt = make_storage_from_array(hs, default_origin=(1, 1))

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

    for np_var, gt_var in (
        (h_new_np, h_new_gt),
        (u_new_np, u_new_gt),
        (v_new_np, v_new_gt),
    ):
        assert np.allclose(np_var, gt_var.data[:, :, 0])

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
        pytest.skip("Diffusion not yet functional")

    lax_wendroff_stencil = sw_solver.gt4py.make_stencil(
        definition=sw_solver.gt4py.lax_wendroff_definition
    )

    read_only_field_infos = [
        lax_wendroff_stencil.field_info[name].access
        for name in ("h", "u", "v", "f", "hs")
    ]
    assert all(x == gt_defs.AccessKind.READ for x in read_only_field_infos)


@pytest.mark.parametrize(
    "ic_type",
    (sw_solver.ICType.RossbyHaurwitzWave, sw_solver.ICType.ZonalGeostrophicFlow),
)
@pytest.mark.parametrize("use_diffusion", (False, True))
@pytest.mark.parametrize("gt4py_backend", ("gtc:numpy",))
def test_gt4py_lax_wendroff_numpy(
    ic_type: sw_solver.ICType, use_diffusion: bool, gt4py_backend: str
):
    """Compare the GT4Py Lax-Wendroff update to numpy."""
    latlon_grid = sw_solver.LatLonGrid(5, 6)
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

    diff_coeff = (
        sw_solver.numpy.DiffusionCoefficients(cart_grid) if use_diffusion else None
    )

    def add_diffusion(var):
        return (
            dt * EARTH_CONSTANTS.nu * sw_solver.numpy.compute_diffusion(var, diff_coeff)
        )

    if diff_coeff:
        h_new_np += add_diffusion(h_0)
        u_new_np += add_diffusion(u_0)
        v_new_np += add_diffusion(v_0)

    # --- gt4py ---
    nk_levels = 1

    def storage_from_array(*args, **kwargs):
        return sw_solver.gt4py.make_storage_from_array(
            *args, **kwargs, backend=gt4py_backend
        )

    def storage_from_shape(*args, **kwargs):
        return sw_solver.gt4py.make_storage_from_shape(
            *args, **kwargs, backend=gt4py_backend
        )

    h_gt, u_gt, v_gt = (
        storage_from_array(
            np.repeat(arr[:, :, np.newaxis], nk_levels, axis=2),
            default_origin=(1, 1, 0),
        )
        for arr in (h_0, u_0, v_0)
    )

    # NOTE: The output storages are two smaller in I and J, since input points are consumed.
    h_new_gt, u_new_gt, v_new_gt = (
        storage_from_shape(list(var.shape) + [nk_levels])
        for var in (h_new_np, u_new_np, v_new_np)
    )

    f_gt = storage_from_array(f, default_origin=(1, 1))

    extended = storage_from_shape(
        [s + 2 for s in h_gt.shape[:-1]] + [h_gt.shape[-1]], default_origin=(2, 2, 0)
    )

    lap_var = storage_from_shape(h_new_gt.shape)

    phi, theta = (
        storage_from_array(x, default_origin=(1, 1))
        for x in (latlon_grid.phi, latlon_grid.theta)
    )

    def stencilize(definition):
        return sw_solver.gt4py.make_stencil(definition)

    lax_wendroff_update = stencilize(sw_solver.gt4py.lax_wendroff_definition)
    compute_laplacian = stencilize(sw_solver.gt4py.laplacian_definition)

    hs_gt = storage_from_array(hs, default_origin=(1, 1))

    lax_wendroff_update(
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

    if use_diffusion:
        pytest.skip("Diffusion not yet functional")
        extended = sw_solver.gt4py._extend_storage(h_gt, extended)
        compute_laplacian(
            phi,
            theta,
            extended,
            lap_var,
            EARTH_CONSTANTS.a,
            origin={"phi": (2, 2), "theta": (2, 2)},
        )
        h_new_gt += dt * EARTH_CONSTANTS.nu * lap_var

        extended = sw_solver.gt4py._extend_storage(u_gt, extended)
        compute_laplacian(
            phi,
            theta,
            extended,
            lap_var,
            EARTH_CONSTANTS.a,
            origin={"phi": (2, 2), "theta": (2, 2)},
        )
        u_new_gt += dt * EARTH_CONSTANTS.nu * lap_var

        extended = sw_solver.gt4py._extend_storage(v_gt, extended)
        compute_laplacian(
            phi,
            theta,
            extended,
            lap_var,
            EARTH_CONSTANTS.a,
            origin={"phi": (2, 2), "theta": (2, 2)},
        )
        v_new_gt += dt * EARTH_CONSTANTS.nu * lap_var

    for np_var, gt_var in (
        (h_new_np, h_new_gt),
        (u_new_np, u_new_gt),
        (v_new_np, v_new_gt),
    ):
        assert np.allclose(np_var, gt_var.data[:, :, 0])

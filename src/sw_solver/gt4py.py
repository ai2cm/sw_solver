"""GT4Py version of the SWES solver."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import gt4py
import gt4py.storage as gt_storage
import numpy as np
from gt4py import gtscript
from gt4py.gtscript import (
    IJ,
    IJK,
    PARALLEL,
    Field,
    computation,
    cos,
    interval,
    sin,
    tan,
)
from gtc.passes.oir_pipeline import DefaultPipeline

from .grid import CartesianGrid, LatLonGrid
from .ic import ICType, get_initial_conditions
from .utils import EARTH_CONSTANTS, FloatT

DEFAULT_BACKEND = "gtc:numpy"


def _default_origin_and_mask(
    axes: Union[gtscript.Axis, Tuple[gtscript.Axis, ...]],
    default_origin: Optional[Tuple[int, ...]] = None,
) -> Tuple[Tuple[int, ...], Tuple[bool, bool, bool]]:
    if axes == gtscript.IJK:
        default_origin = default_origin or (0, 0, 0)
        mask = (True, True, True)
    elif axes == gtscript.IJ:
        default_origin = default_origin or (0, 0)
        mask = (True, True, False)
    else:
        raise NotImplementedError(f"Unrecognized axes: {axes}")

    return default_origin, mask


def _num_axes_to_gtscript(
    num_axes: int,
) -> Union[gtscript.Axis, Tuple[gtscript.Axis, ...]]:
    if num_axes == 3:
        return gtscript.IJK
    elif num_axes == 2:
        return gtscript.IJ
    else:
        raise NotImplementedError(f"Unrecognized number of axes: {num_axes}")


def make_storage_from_shape(
    shape: Tuple[int, ...],
    dtype: Any = FloatT,
    backend: str = DEFAULT_BACKEND,
    default_origin: Optional[Tuple[int, ...]] = None,
    axes: Optional[Union[gtscript.Axis, Tuple[gtscript.Axis, ...]]] = None,
) -> gt_storage.Storage:
    """Create gt4py storage from a given shape.

    Parameters
    ----------
    shape : tuple
        Tuple of integers indicating the size of each axis.
    dtype : np.dtype
        Data type of each element.
    backend : str, optional
        String describing the gt4py backend.
    default_origin : tuple, optional
        Data index mapping to the origin of the stencil compute domain.
    axes
        The gtscript Axis definition.

    Returns
    -------
    gt_storage.Storage
        The resulting gt4py storage.

    """
    axes = axes or _num_axes_to_gtscript(len(shape))
    default_origin, mask = _default_origin_and_mask(axes, default_origin)

    return gt_storage.empty(backend, default_origin, shape, dtype, mask=mask)


def make_storage_from_array(
    data: Any,
    backend: str = DEFAULT_BACKEND,
    default_origin: Optional[Tuple[int, ...]] = None,
    axes: Optional[Union[gtscript.Axis, Tuple[gtscript.Axis, ...]]] = None,
) -> gt_storage.Storage:
    """Create gt4py storage from an existing data array.

    Wrapper around gt_storage.make_storage_from_array. Will copy the storage if using
    a GPU backend and a numpy data array is given.

    Parameters
    ----------
    data :
        Input data array, follows buffer protocol.
    backend : str, optional
        String describing the gt4py backend.
    default_origin : tuple, optional
        Data index mapping to the origin of the stencil compute domain.
    axes
        The gtscript Axis definition.

    Returns
    -------
    gt_storage.Storage
        The resulting gt4py storage.

    """
    axes = axes or _num_axes_to_gtscript(data.ndim)
    default_origin, mask = _default_origin_and_mask(axes, default_origin)

    return gt_storage.from_array(
        data, backend, default_origin, shape=data.shape, dtype=data.dtype, mask=mask
    )


def make_stencil(definition: Callable[..., None], **kwargs: Any) -> gt4py.StencilObject:
    """Wrap around gtscript.Stencil.

    Parameters
    ----------
    definition : func
        The gtscript stencil definition
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    gt4py.StencilObject
        The resulting stencil.

    """
    kwargs.setdefault("oir_pipeline", DefaultPipeline(skip=DefaultPipeline.all_steps()))
    kwargs.setdefault("backend", "gtc:numpy")
    kwargs.setdefault("externals", {"USE_DIFFUSION": False})

    return gt4py.gtscript.stencil(definition=definition, **kwargs)


def laplacian_definition(
    phi: Field[IJ, FloatT],
    theta: Field[IJ, FloatT],
    var: Field[IJK, FloatT],
    lap_var: Field[IJK, FloatT],
    const_a: FloatT,
):
    """Compute the diffusion term."""
    # a*, b*, c* are the centered, upwind, and downwind cofficients, respectively.

    with computation(PARALLEL), interval(...):
        x = const_a * cos(theta) * phi
        y = const_a * theta

        dx = x - x[-1, 0, 0]
        dy = y - y[0, -1, 0]

        ax = (dx[1, 0, 0] - dx) / (dx[1, 0, 0] * dx)
        bx = dx / dx[1, 0, 0] * (dx[1, 0, 0] + dx)
        cx = -dx[1, 0, 0] / dx * (dx[1, 0, 0] + dx)

        vxx_0 = ax * (
            ax[0, 0, 0] * var[0, 0, 0]
            + bx[0, 0, 0] * var[1, 0, 0]
            + cx[0, 0, 0] * var[-1, 0, 0]
        )
        vxx_u = bx * (
            ax[1, 0, 0] * var[1, 0, 0]
            + bx[1, 0, 0] * var[2, 0, 0]
            + cx[1, 0, 0] * var[0, 0, 0]
        )
        vxx_d = cx * (
            ax[-1, 0, 0] * var[-1, 0, 0]
            + bx[-1, 0, 0] * var[0, 0, 0]
            + cx[-1, 0, 0] * var[-2, 0, 0]
        )

        vxx = (  # noqa: F841  # local variable 'vxx' is assigned to but never used
            vxx_0 + vxx_u + vxx_d
        )

        ay = (dy[0, 1, 0] - dy) / (dy[0, 1, 0] * dy)
        by = dy / dy[0, 1, 0] * (dy[0, 1, 0] + dy)
        cy = -dy[0, 1, 0] / dy * (dy[0, 1, 0] + dy)

        vyy_0 = ay * (
            ay[0, 0, 0] * var[0, 0, 0]
            + by[0, 0, 0] * var[0, 1, 0]
            + cy[0, 0, 0] * var[0, -1, 0]
        )
        vyy_u = by * (
            ay[0, 1, 0] * var[0, 1, 0]
            + by[0, 1, 0] * var[0, 2, 0]
            + cy[0, 1, 0] * var[0, 0, 0]
        )
        vyy_d = cy * (
            ay[0, -1, 0] * var[0, -1, 0]
            + by[0, -1, 0] * var[0, 0, 0]
            + cy[0, -1, 0] * var[0, -2, 0]
        )

        vyy = (  # noqa: F841  # local variable 'vyy' is assigned to but never used
            vyy_0 + vyy_u + vyy_d
        )

        lap_var = vxx_d  # noqa: F841  # local variable 'lap_var' is assigned to but never used


def lax_wendroff_definition(
    phi: Field[IJ, FloatT],
    theta: Field[IJ, FloatT],
    f: Field[IJ, FloatT],
    hs: Field[IJ, FloatT],
    h: Field[IJK, FloatT],
    u: Field[IJK, FloatT],
    v: Field[IJK, FloatT],
    h_new: Field[IJK, FloatT],
    u_new: Field[IJK, FloatT],
    v_new: Field[IJK, FloatT],
    const_a: FloatT,
    const_g: FloatT,
    dt: FloatT,
):
    """Definition of the Lax-Wendroff finite difference coupled update."""
    with computation(PARALLEL), interval(...):
        # Compute grid variables

        # # Latitude-Longitude Grid
        grid_c = cos(theta)
        grid_tg = tan(theta)
        c_mid_y = cos(0.5 * (theta + theta[0, -1]))
        tg_mid_x = tan(0.5 * (theta[-1, 0] + theta))
        tg_mid_y = tan(0.5 * (theta[0, -1] + theta))

        # # Cartesian Grid
        x = const_a * cos(theta) * phi
        y = const_a * theta
        y1 = const_a * sin(theta)

        dx = x - x[-1, 0, 0]
        dy = y - y[0, -1, 0]
        dy1 = y1 - y1[0, -1, 0]

        dxc = 0.5 * (dx + dx[1, 0, 0])
        dyc = 0.5 * (dy + dy[0, 1, 0])
        dy1c = 0.5 * (dy1 + dy1[0, 1, 0])

        # --- Auxiliary variables --- #
        v1 = v * grid_c
        hu = h * u
        hv = h * v
        hv1 = h * v1

        # --- Compute mid-point values after half timestep --- #
        h_mid_x = 0.5 * (h + h[-1, 0, 0]) - 0.5 * (dt / dx) * (hu - hu[-1, 0, 0])
        h_mid_y = 0.5 * (h + h[0, -1, 0]) - 0.5 * (dt / dy1) * (hv1 - hv1[0, -1, 0])

        # Mid-point value for hu along x
        ux = hu * u + 0.5 * const_g * h * h
        hu_mid_x = (
            0.5 * (hu + hu[-1, 0, 0])
            - 0.5 * dt / dx * (ux - ux[-1, 0, 0])
            + 0.5
            * dt
            * (0.5 * (f + f[-1, 0]) + 0.5 * (u + u[-1, 0, 0]) * tg_mid_x / const_a)
            * (0.5 * (hv + hv[-1, 0, 0]))
        )

        # Mid-point value for hu along y
        uy = hu * v1
        hu_mid_y = (
            0.5 * (hu + hu[0, -1, 0])
            - 0.5 * dt / dy1 * (uy - uy[0, -1, 0])
            + 0.5
            * dt
            * (0.5 * (f + f[0, -1]) + 0.5 * (u + u[0, -1, 0]) * tg_mid_y / const_a)
            * (0.5 * (hv + hv[0, -1, 0]))
        )

        # Mid-point value for hv along x
        vx = hu * v
        hv_mid_x = (
            0.5 * (hv + hv[-1, 0, 0])
            - 0.5 * dt / dx * (vx - vx[-1, 0, 0])
            - 0.5
            * dt
            * (0.5 * (f + f[-1, 0]) + 0.5 * (u + u[-1, 0, 0]) * tg_mid_x / const_a)
            * (0.5 * (hu + hu[-1, 0, 0]))
        )

        # Mid-point value for hv along y
        vy1 = hv * v1
        vy2 = 0.5 * const_g * h * h
        hv_mid_y = (
            0.5 * (hv + hv[0, -1, 0])
            - 0.5 * dt / dy1 * (vy1 - vy1[0, -1, 0])
            - 0.5 * dt / dy * (vy2 - vy2[0, -1, 0])
            - 0.5
            * dt
            * (0.5 * (f + f[0, -1]) + 0.5 * (u + u[0, -1, 0]) * tg_mid_y / const_a)
            * (0.5 * (hu + hu[0, -1, 0]))
        )

        # --- Compute solution at next timestep --- #

        # # Update fluid height
        h_new = (
            h
            - dt / dxc * (hu_mid_x[1, 0, 0] - hu_mid_x)
            - dt / dy1c * (hv_mid_y[0, 1, 0] * c_mid_y[0, 1, 0] - hv_mid_y * c_mid_y)
        )

        f_update = (
            f
            + 0.25
            * (
                hu_mid_x / h_mid_x
                + hu_mid_x[1, 0, 0] / h_mid_x[1, 0, 0]
                + hu_mid_y / h_mid_y
                + hu_mid_y[0, 1, 0] / h_mid_y[0, 1, 0]
            )
            * grid_tg
            / const_a
        )

        # Update longitudinal moment
        ux_mid_base = 0.5 * const_g * h_mid_x * h_mid_x
        if h_mid_x > 0.0:
            ux_mid = ux_mid_base + hu_mid_x * hu_mid_x / h_mid_x
        else:
            ux_mid = ux_mid_base

        if h_mid_y > 0.0:
            uy_mid = hv_mid_y * c_mid_y * hu_mid_y / h_mid_y
        else:
            uy_mid = 0.0

        hu_new = (
            hu
            - dt / dxc * (ux_mid[1, 0, 0] - ux_mid)
            - dt / dy1c * (uy_mid[0, 1, 0] - uy_mid)
            + dt
            * f_update
            * 0.25
            * (hv_mid_x + hv_mid_x[1, 0, 0] + hv_mid_y + hv_mid_y[0, 1, 0])
            - dt
            * const_g
            * 0.25
            * (h_mid_x + h_mid_x[1, 0, 0] + h_mid_y + h_mid_y[0, 1, 0])
            * (hs[1, 0] - hs[-1, 0])
            / (dx + dx[1, 0, 0])
        )

        # Update latitudinal moment
        if h_mid_x > 0.0:
            vx_mid = hv_mid_x * hu_mid_x / h_mid_x
        else:
            vx_mid = 0.0

        if h_mid_y > 0.0:
            vy1_mid = hv_mid_y * hv_mid_y / h_mid_y * c_mid_y
        else:
            vy1_mid = 0.0
        vy2_mid = 0.5 * const_g * h_mid_y * h_mid_y

        hv_new = (
            hv
            - dt / dxc * (vx_mid[1, 0, 0] - vx_mid)
            - dt / dy1c * (vy1_mid[0, 1, 0] - vy1_mid)
            - dt / dyc * (vy2_mid[0, 1, 0] - vy2_mid)
            - dt
            * f_update
            * 0.25
            * (hu_mid_x + hu_mid_x[1, 0, 0] + hu_mid_y + hu_mid_y[0, 1, 0])
            - dt
            * const_g
            * 0.25
            * (h_mid_x + h_mid_x[1, 0, 0] + h_mid_y + h_mid_y[0, 1, 0])
            * (hs[0, 1] - hs[0, -1])
            / (dy1 + dy1[0, 1, 0])
        )

        # # Come back to original variables
        u_new = (  # noqa: F841 local variable 'u_new' is assigned to but never used
            hu_new / h_new
        )
        v_new = (  # noqa: F841 local variable 'u_new' is assigned to but never used
            hv_new / h_new
        )


def _apply_bcs(q, qnew):
    """
    Apply boundary conditions to a quantity.

    Parameters
    ----------
    q : FloatArray2D
        Input quantity (this is modified).
    qnew : FloatArray2D
        Updated quantity.

    Returns
    -------
    FloatArray2D
        Quantity with boundary conditions applied.

    """
    q[:, 1:-1, :] = np.concatenate((qnew[-2:-1, :, :], qnew, qnew[1:2, :, :]), axis=0)
    q[:, 0, :] = q[:, 1, :]
    q[:, -1, :] = q[:, -2, :]

    return q


def _extend_storage(
    quantity: gt_storage.Storage, extended: gt_storage.Storage
) -> gt_storage.Storage:
    for k in range(quantity.shape[2]):
        extended_data = np.concatenate(
            (
                quantity.data[-4:-3, :, k],
                quantity.data[:, :, k],
                quantity.data[3:4, :, k],
            ),
            axis=0,
        )
        extended_data = np.concatenate(
            (extended_data[:, 0:1], extended_data, extended_data[:, -1:]), axis=1
        )
        extended[:, :, k] = extended_data[:, :]

    return extended


def solve(
    num_lon_pts: int,
    num_lat_pts: int,
    ic_type: ICType,
    sim_days: float,
    courant: float,
    use_diffusion: bool = True,
    save_data: Optional[Dict[str, Any]] = None,
    print_interval: int = -1,
    gt4py_backend: str = "gtc:numpy",
    nk_levels: int = 1,
):
    """
    Numpy implementation of the SWES solver.

    Parameters
    ----------
    num_lon_pts : int
        Number of longitudinal gridpoints.
    num_lat_pts : int
        Number of latitudinal gridpoints.
    ic_type : ICType
        Initial condition type.
    sim_days : float
        Simulation days.
    courant : float
        CFL number.
    use_diffusion : bool, optional
        If True, adds a diffusion term (default: True).
    save_data : dict, optional
        If passed with the key "interval" and value > 0, will output
        the following arrays in this dict:
            * "h" : height.
            * "u" : x-velocity.
            * "v" : y-velocity.
            * "t" : times for each savepoint (in secs).
            * "phi" : grid data.
            * "theta" : grid data.
    print_interval : int
        If positive, print to screen information about the solution
        every 'verbose' timesteps.
    gt4py_backend : str
        Stencil backend.
    nk_levels : int
        Number of vertical levels.

    Notes
    -----
    The following notation is used:
    - h : fluid height
    - hs : terrain height (topography)
    - ht : total height (ht = hs + h)
    - phi : longitude
    - R : planet radius
    - theta : latitude
    - u : longitudinal fluid velocity
    - v : latitudinal fluid velocity

    """

    def stencilize(definition):
        return make_stencil(definition=definition, backend=gt4py_backend)

    lax_wendroff_update = stencilize(lax_wendroff_definition)
    compute_laplacian = stencilize(laplacian_definition)

    save_data = save_data or {}

    latlon_grid = LatLonGrid(num_lon_pts, num_lat_pts)
    cart_grid = CartesianGrid(latlon_grid)

    if sim_days < 0.0:
        raise ValueError(f"Final time {sim_days} must be non-negative.")

    hours_per_day: int = 24
    seconds_per_hour: int = 3600
    final_time = hours_per_day * seconds_per_hour * sim_days

    def storage_from_array(*args, **kwargs):
        return make_storage_from_array(*args, **kwargs, backend=gt4py_backend)

    def storage_from_shape(*args, **kwargs):
        return make_storage_from_shape(*args, **kwargs, backend=gt4py_backend)

    # Note: currently just a flat surface
    hs = storage_from_shape(latlon_grid.shape, default_origin=(1, 1))

    if not isinstance(ic_type, ICType):
        raise TypeError(
            f"Invalid problem IC: {ic_type}. See code documentation for implemented initial conditions."
        )

    h_arr, u_arr, v_arr, f_arr = get_initial_conditions(ic_type, latlon_grid)

    save_interval: int = save_data.get("interval", 0)

    if save_interval > 0:
        tsave = [0.0]
        hsave = h_arr[1:-1, :, np.newaxis].copy()
        usave = u_arr[1:-1, :, np.newaxis].copy()
        vsave = v_arr[1:-1, :, np.newaxis].copy()

    h, u, v = (
        storage_from_array(np.tile(x, (1, 1, nk_levels)), default_origin=(1, 1, 0))
        for x in (h_arr, u_arr, v_arr)
    )

    f = storage_from_array(f_arr, default_origin=(1, 1))

    h_new, u_new, v_new = (
        storage_from_shape([s - 2 for s in var.shape[:-1]] + [var.shape[-1]])
        for var in (h, u, v)
    )

    extended = storage_from_shape(
        [s + 2 for s in h.shape[:-1]] + [h.shape[-1]], default_origin=(2, 2, 0)
    )

    lap_var = storage_from_shape(h_new.shape)

    phi, theta = (
        storage_from_array(var, default_origin=(1, 1))
        for var in (latlon_grid.phi, latlon_grid.theta)
    )

    num_steps = 0
    time = 0.0

    while time < final_time:
        num_steps += 1

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
        dt = courant * dtmax

        lax_wendroff_update(
            phi,
            theta,
            f,
            hs,
            h,
            u,
            v,
            h_new,
            u_new,
            v_new,
            EARTH_CONSTANTS.a,
            EARTH_CONSTANTS.g,
            dt,
        )

        if use_diffusion:
            extended = _extend_storage(h, extended)
            compute_laplacian(phi, theta, extended, lap_var, EARTH_CONSTANTS.a)
            h_new += dt * EARTH_CONSTANTS.nu * lap_var

            extended = _extend_storage(u, extended)
            compute_laplacian(phi, theta, extended, lap_var, EARTH_CONSTANTS.a)
            u_new += dt * EARTH_CONSTANTS.nu * lap_var

            extended = _extend_storage(v, extended)
            compute_laplacian(phi, theta, extended, lap_var, EARTH_CONSTANTS.a)
            v_new += dt * EARTH_CONSTANTS.nu * lap_var

        # --- Update solution applying BCs --- #
        h = _apply_bcs(h, h_new)
        u = _apply_bcs(u, u_new)
        v = _apply_bcs(v, v_new)

        # If needed, adjust timestep not to exceed final time
        if time + dt > final_time:
            dt = final_time - time
            time = final_time
        else:
            time += dt

        # --- Print and save --- #
        if print_interval > 0 and (num_steps % print_interval == 0):
            norm = np.sqrt(u * u + v * v)
            umax = norm.max()
            print(
                f"Time = {time / 3600.0:6.2f} hours (max {int(final_time / 3600.0)}); max(|u|) = {umax:16.16f}"
            )

        if save_interval > 0 and (num_steps % save_interval == 0):
            tsave.append(time)
            hsave = np.concatenate((hsave, h[1:-1, :, np.newaxis]), axis=2)
            usave = np.concatenate((usave, u[1:-1, :, np.newaxis]), axis=2)
            vsave = np.concatenate((vsave, v[1:-1, :, np.newaxis]), axis=2)

    if save_interval > 0:
        tsave = np.asarray(tsave)
        save_data["h"] = hsave
        save_data["u"] = usave
        save_data["v"] = vsave
        save_data["t"] = tsave
        save_data["phi"] = latlon_grid.phi
        save_data["theta"] = latlon_grid.theta

    # --- Return --- #
    return h, u, v

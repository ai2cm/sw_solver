"""GT4Py version of the SWES solver."""

from typing import Any, Dict, Optional, Type

import gt4py
import numpy as np
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

from .grid import CartesianGrid, LatLonGrid
from .ic import ICType, get_initial_conditions
from .utils import EARTH_CONSTANTS, FloatT


class SingletonMeta(type):
    """Singleton Metaclass."""

    _instances: Dict[Type["SingletonMeta"], Any] = {}

    def __call__(  # noqa: D102  # Missing docstring in public method
        cls, *args, **kwargs
    ):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class StorageAllocator(metaclass=SingletonMeta):
    """Memory tracker."""

    def __init__(self):
        self.num_storages = {num_dim: 0 for num_dim in range(1, 4)}
        self.total_bytes: int = 0

    def _add_storage(self, storage: gt4py.storage.Storage):
        num_axes = sum(storage.mask)
        self.num_storages[num_axes] += 1
        self.total_bytes += np.prod(storage.shape) * np.dtype(storage.dtype).itemsize
        return storage

    def ones(self, backend, default_origin, shape, dtype, mask=None):
        """Wrap gt4py.storage.ones."""
        return self._add_storage(
            gt4py.storage.ones(backend, default_origin, shape, dtype, mask)
        )

    def zeros(self, backend, default_origin, shape, dtype, mask=None):
        """Wrap gt4py.storage.zeros."""
        return self._add_storage(
            gt4py.storage.zeros(backend, default_origin, shape, dtype, mask)
        )

    def from_array(self, data, backend, default_origin, shape=None, mask=None):
        """Wrap gt4py.storage.from_array."""
        shape = shape or data.shape
        return self._add_storage(
            gt4py.storage.from_array(
                data, backend, default_origin, shape, data.dtype, mask
            )
        )

    def empty_like(self, storage):
        """Wrap gt4py.storage.empty."""
        return self._add_storage(
            gt4py.storage.empty(
                storage.backend,
                storage.default_origin,
                storage.shape,
                storage.dtype,
                storage.mask,
            )
        )


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
    lax_wendroff_update = gt4py.gtscript.stencil(
        definition=lax_wendroff_definition, backend=gt4py_backend
    )
    if save_data is None:
        save_data = {}

    latlon_grid = LatLonGrid(num_lon_pts, num_lat_pts)
    cart_grid = CartesianGrid(latlon_grid)

    if sim_days < 0.0:
        raise ValueError(f"Final time {sim_days} must be non-negative.")

    hours_per_day: int = 24
    seconds_per_hour: int = 3600
    final_time = hours_per_day * seconds_per_hour * sim_days

    # Note: currently just a flat surface
    hs = StorageAllocator().zeros(
        backend=gt4py_backend,
        default_origin=(0, 0, 0),
        shape=latlon_grid.shape,
        dtype=FloatT,
    )

    if not isinstance(ic_type, ICType):
        raise TypeError(
            f"Invalid problem IC: {ic_type}. See code documentation for implemented initial conditions."
        )

    h_arr, u_arr, v_arr, f_arr = get_initial_conditions(ic_type, latlon_grid)

    h, u, v = (
        StorageAllocator().from_array(
            x[:, :, np.newaxis],
            backend=gt4py_backend,
            default_origin=(1, 0, 0),
            shape=list(x.shape) + [1],
        )
        for x in (h_arr, u_arr, v_arr)
    )

    f = StorageAllocator().from_array(
        f_arr, backend=gt4py_backend, default_origin=(1, 0), mask=(True, True, False)
    )

    print(StorageAllocator().num_storages, StorageAllocator().total_bytes)

    # TODO: loop
    time = 0.0

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

    h_new, u_new, v_new = (StorageAllocator().empty_like(var) for var in (h, u, v))

    phi = StorageAllocator().from_array(
        latlon_grid.phi, backend=gt4py_backend, default_origin=(1, 0, 0)
    )
    theta = StorageAllocator().from_array(
        latlon_grid.theta, backend=gt4py_backend, default_origin=(1, 0, 0)
    )

    lax_wendroff_update(
        phi,
        theta,
        f,
        hs,
        dt,
        h,
        u,
        v,
        EARTH_CONSTANTS.a,
        EARTH_CONSTANTS.g,
        h_new,
        u_new,
        v_new,
    )

    # If needed, adjust timestep not to exceed final time
    if time + dt > final_time:
        dt = final_time - time
        time = final_time
    else:
        time += dt

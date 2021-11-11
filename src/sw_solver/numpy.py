"""Numpy version of the SWES solver."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .grid import CartesianGrid, LatLonGrid
from .ic import ICType, get_initial_conditions
from .utils import EARTH_CONSTANTS, FloatArray2D, FloatT


class DiffusionCoefficients:
    """Diffusion coefficient array container.

    TODO: Document attributes.

    """

    Ax: FloatArray2D
    Bx: FloatArray2D
    Cx: FloatArray2D

    Ay: FloatArray2D
    By: FloatArray2D
    Cy: FloatArray2D

    def __init__(self, cart_grid: CartesianGrid):
        """Compute the coefficients."""
        # Centered finite difference along longitude
        # Ax, Bx and Cx denote the coefficients associated
        # to the centred, upwind and downwind point, respectively
        Ax = (cart_grid.dx[1:, 1:-1] - cart_grid.dx[:-1, 1:-1]) / (
            cart_grid.dx[1:, 1:-1] * cart_grid.dx[:-1, 1:-1]
        )
        self.Ax = np.concatenate((Ax[-2:-1, :], Ax, Ax[1:2, :]), axis=0)

        Bx = cart_grid.dx[:-1, 1:-1] / (
            cart_grid.dx[1:, 1:-1] * (cart_grid.dx[1:, 1:-1] + cart_grid.dx[:-1, 1:-1])
        )
        self.Bx = np.concatenate((Bx[-2:-1, :], Bx, Bx[1:2, :]), axis=0)

        Cx = -cart_grid.dx[1:, 1:-1] / (
            cart_grid.dx[:-1, 1:-1] * (cart_grid.dx[1:, 1:-1] + cart_grid.dx[:-1, 1:-1])
        )
        self.Cx = np.concatenate((Cx[-2:-1, :], Cx, Cx[1:2, :]), axis=0)

        # Centered finite difference along latitude
        # Ay, By and Cy denote the coefficients associated
        # to the centred, upwind and downwind point, respectively
        Ay = (cart_grid.dy[1:-1, 1:] - cart_grid.dy[1:-1, :-1]) / (
            cart_grid.dy[1:-1, 1:] * cart_grid.dy[1:-1, :-1]
        )
        self.Ay = np.concatenate((Ay[:, 0:1], Ay, Ay[:, -1:]), axis=1)

        By = cart_grid.dy[1:-1, :-1] / (
            cart_grid.dy[1:-1, 1:] * (cart_grid.dy[1:-1, 1:] + cart_grid.dy[1:-1, :-1])
        )
        self.By = np.concatenate((By[:, 0:1], By, By[:, -1:]), axis=1)

        Cy = -cart_grid.dy[1:-1, 1:] / (
            cart_grid.dy[1:-1, :-1] * (cart_grid.dy[1:-1, 1:] + cart_grid.dy[1:-1, :-1])
        )
        self.Cy = np.concatenate((Cy[:, 0:1], Cy, Cy[:, -1:]), axis=1)


def compute_laplacian(coeff: DiffusionCoefficients, q: FloatArray2D) -> FloatArray2D:
    """
    Evaluate the Laplacian of a given quantity.

    The approximations is given by applying twice a centre finite difference
    formula along both axis.

    Parameters
    ----------
    coeff : DiffusionCoefficients
        The computed diffusion coefficients.
    q : FloatArray2D
        The field to take differences from.

    Returns
    -------
    FloatArray2D
        The Laplacian of q.

    """
    # Compute second order derivative along longitude
    qxx = (
        coeff.Ax[1:-1, :]
        * (
            coeff.Ax[1:-1, :] * q[2:-2, 2:-2]
            + coeff.Bx[1:-1, :] * q[3:-1, 2:-2]
            + coeff.Cx[1:-1, :] * q[1:-3, 2:-2]
        )
        + coeff.Bx[1:-1, :]
        * (
            coeff.Ax[2:, :] * q[3:-1, 2:-2]
            + coeff.Bx[2:, :] * q[4:, 2:-2]
            + coeff.Cx[2:, :] * q[2:-2, 2:-2]
        )
        + coeff.Cx[1:-1, :]
        * (
            coeff.Ax[:-2, :] * q[1:-3, 2:-2]
            + coeff.Bx[:-2, :] * q[2:-2, 2:-2]
            + coeff.Cx[:-2, :] * q[:-4, 2:-2]
        )
    )

    # Compute second order derivative along latitude
    qyy = (
        coeff.Ay[:, 1:-1]
        * (
            coeff.Ay[:, 1:-1] * q[2:-2, 2:-2]
            + coeff.By[:, 1:-1] * q[2:-2, 3:-1]
            + coeff.Cy[:, 1:-1] * q[2:-2, 1:-3]
        )
        + coeff.By[:, 1:-1]
        * (
            coeff.Ay[:, 2:] * q[2:-2, 3:-1]
            + coeff.By[:, 2:] * q[2:-2, 4:]
            + coeff.Cy[:, 2:] * q[2:-2, 2:-2]
        )
        + coeff.Cy[:, 1:-1]
        * (
            coeff.Ay[:, :-2] * q[2:-2, 1:-3]
            + coeff.By[:, :-2] * q[2:-2, 2:-2]
            + coeff.Cy[:, :-2] * q[2:-2, :-4]
        )
    )

    # Compute Laplacian
    qlap = qxx + qyy

    return qlap


def lax_wendroff_update(
    latlon_grid: LatLonGrid,
    cart_grid: CartesianGrid,
    diffusion: Optional[DiffusionCoefficients],
    dt: FloatT,
    f: FloatArray2D,
    hs: FloatArray2D,
    h: FloatArray2D,
    u: FloatArray2D,
    v: FloatArray2D,
) -> Tuple[FloatArray2D, FloatArray2D, FloatArray2D]:
    """
    Update solution through finite difference Lax-Wendroff scheme.

    The Coriolis effect is taken into account in Lax-Wendroff step,
    while diffusion is separately added afterwards.

    Parameters
    ----------
    latlon_grid : LatLonGrid
        Grid on the latitude-longitude coordinates.
    cart_grid : CartesianGrid
        Cartesian grid.
    diffusion : DiffusionCoefficient, optional
        Computed diffusion coefficients, if using.
    dt : FloatT
        Timestep.
    f : FloatArray2D
        Coriolis force.
    hs : FloatArray2D
        Terrain height.
    h : FloatArray2D
        Fluid height at current timestep.
    u : FloatArray2D
        Longitudinal velocity at current timestep.
    v : FloatArray2D
        Latitudinal velocity at current timestep.

    Returns
    -------
    hnew : FloatArray2D
        Updated fluid height.
    unew : FloatArray2D
        Updated longitudinal velocity.
    vnew : FloatArray2D
        Updated latitudinal velocity.

    """
    # --- Auxiliary variables --- #

    v1 = v * latlon_grid.c
    hu = h * u
    hv = h * v
    hv1 = h * v1

    # --- Compute mid-point values after half timestep --- #

    # Mid-point value for h along x
    hMidx = 0.5 * (h[1:, 1:-1] + h[:-1, 1:-1]) - 0.5 * dt / cart_grid.dx[:, 1:-1] * (
        hu[1:, 1:-1] - hu[:-1, 1:-1]
    )

    # Mid-point value for h along y
    hMidy = 0.5 * (h[1:-1, 1:] + h[1:-1, :-1]) - 0.5 * dt / cart_grid.dy1[1:-1, :] * (
        hv1[1:-1, 1:] - hv1[1:-1, :-1]
    )

    # Mid-point value for hu along x
    Ux = hu * u + 0.5 * EARTH_CONSTANTS.g * h * h
    huMidx = (
        0.5 * (hu[1:, 1:-1] + hu[:-1, 1:-1])
        - 0.5 * dt / cart_grid.dx[:, 1:-1] * (Ux[1:, 1:-1] - Ux[:-1, 1:-1])
        + 0.5
        * dt
        * (
            0.5 * (f[1:, 1:-1] + f[:-1, 1:-1])
            + 0.5
            * (u[1:, 1:-1] + u[:-1, 1:-1])
            * latlon_grid.tgMidx
            / EARTH_CONSTANTS.a
        )
        * (0.5 * (hv[1:, 1:-1] + hv[:-1, 1:-1]))
    )

    # Mid-point value for hu along y
    Uy = hu * v1
    huMidy = (
        0.5 * (hu[1:-1, 1:] + hu[1:-1, :-1])
        - 0.5 * dt / cart_grid.dy1[1:-1, :] * (Uy[1:-1, 1:] - Uy[1:-1, :-1])
        + 0.5
        * dt
        * (
            0.5 * (f[1:-1, 1:] + f[1:-1, :-1])
            + 0.5
            * (u[1:-1, 1:] + u[1:-1, :-1])
            * latlon_grid.tgMidy
            / EARTH_CONSTANTS.a
        )
        * (0.5 * (hv[1:-1, 1:] + hv[1:-1, :-1]))
    )

    # Mid-point value for hv along x
    Vx = hu * v
    hvMidx = (
        0.5 * (hv[1:, 1:-1] + hv[:-1, 1:-1])
        - 0.5 * dt / cart_grid.dx[:, 1:-1] * (Vx[1:, 1:-1] - Vx[:-1, 1:-1])
        - 0.5
        * dt
        * (
            0.5 * (f[1:, 1:-1] + f[:-1, 1:-1])
            + 0.5
            * (u[1:, 1:-1] + u[:-1, 1:-1])
            * latlon_grid.tgMidx
            / EARTH_CONSTANTS.a
        )
        * (0.5 * (hu[1:, 1:-1] + hu[:-1, 1:-1]))
    )

    # Mid-point value for hv along y
    Vy1 = hv * v1
    Vy2 = 0.5 * EARTH_CONSTANTS.g * h * h
    hvMidy = (
        0.5 * (hv[1:-1, 1:] + hv[1:-1, :-1])
        - 0.5 * dt / cart_grid.dy1[1:-1, :] * (Vy1[1:-1, 1:] - Vy1[1:-1, :-1])
        - 0.5 * dt / cart_grid.dy[1:-1, :] * (Vy2[1:-1, 1:] - Vy2[1:-1, :-1])
        - 0.5
        * dt
        * (
            0.5 * (f[1:-1, 1:] + f[1:-1, :-1])
            + 0.5
            * (u[1:-1, 1:] + u[1:-1, :-1])
            * latlon_grid.tgMidy
            / EARTH_CONSTANTS.a
        )
        * (0.5 * (hu[1:-1, 1:] + hu[1:-1, :-1]))
    )

    # --- Compute solution at next timestep --- #

    # Update fluid height
    hnew = (
        h[1:-1, 1:-1]
        - dt / cart_grid.dxc * (huMidx[1:, :] - huMidx[:-1, :])
        - dt
        / cart_grid.dy1c
        * (
            hvMidy[:, 1:] * latlon_grid.cMidy[:, 1:]
            - hvMidy[:, :-1] * latlon_grid.cMidy[:, :-1]
        )
    )

    # Update longitudinal moment
    UxMid = np.where(
        hMidx > 0.0,
        huMidx * huMidx / hMidx + 0.5 * EARTH_CONSTANTS.g * hMidx * hMidx,
        0.5 * EARTH_CONSTANTS.g * hMidx * hMidx,
    )
    UyMid = np.where(hMidy > 0.0, hvMidy * latlon_grid.cMidy * huMidy / hMidy, 0.0)
    hunew = (
        hu[1:-1, 1:-1]
        - dt / cart_grid.dxc * (UxMid[1:, :] - UxMid[:-1, :])
        - dt / cart_grid.dy1c * (UyMid[:, 1:] - UyMid[:, :-1])
        + dt
        * (
            f[1:-1, 1:-1]
            + 0.25
            * (
                huMidx[:-1, :] / hMidx[:-1, :]
                + huMidx[1:, :] / hMidx[1:, :]
                + huMidy[:, :-1] / hMidy[:, :-1]
                + huMidy[:, 1:] / hMidy[:, 1:]
            )
            * latlon_grid.tg
            / EARTH_CONSTANTS.a
        )
        * 0.25
        * (hvMidx[:-1, :] + hvMidx[1:, :] + hvMidy[:, :-1] + hvMidy[:, 1:])
        - dt
        * EARTH_CONSTANTS.g
        * 0.25
        * (hMidx[:-1, :] + hMidx[1:, :] + hMidy[:, :-1] + hMidy[:, 1:])
        * (hs[2:, 1:-1] - hs[:-2, 1:-1])
        / (cart_grid.dx[:-1, 1:-1] + cart_grid.dx[1:, 1:-1])
    )

    # Update latitudinal moment
    VxMid = np.where(hMidx > 0.0, hvMidx * huMidx / hMidx, 0.0)
    Vy1Mid = np.where(hMidy > 0.0, hvMidy * hvMidy / hMidy * latlon_grid.cMidy, 0.0)
    Vy2Mid = 0.5 * EARTH_CONSTANTS.g * hMidy * hMidy
    hvnew = (
        hv[1:-1, 1:-1]
        - dt / cart_grid.dxc * (VxMid[1:, :] - VxMid[:-1, :])
        - dt / cart_grid.dy1c * (Vy1Mid[:, 1:] - Vy1Mid[:, :-1])
        - dt / cart_grid.dyc * (Vy2Mid[:, 1:] - Vy2Mid[:, :-1])
        - dt
        * (
            f[1:-1, 1:-1]
            + 0.25
            * (
                huMidx[:-1, :] / hMidx[:-1, :]
                + huMidx[1:, :] / hMidx[1:, :]
                + huMidy[:, :-1] / hMidy[:, :-1]
                + huMidy[:, 1:] / hMidy[:, 1:]
            )
            * latlon_grid.tg
            / EARTH_CONSTANTS.a
        )
        * 0.25
        * (huMidx[:-1, :] + huMidx[1:, :] + huMidy[:, :-1] + huMidy[:, 1:])
        - dt
        * EARTH_CONSTANTS.g
        * 0.25
        * (hMidx[:-1, :] + hMidx[1:, :] + hMidy[:, :-1] + hMidy[:, 1:])
        * (hs[1:-1, 2:] - hs[1:-1, :-2])
        / (cart_grid.dy1[1:-1, :-1] + cart_grid.dy1[1:-1, 1:])
    )

    # Come back to original variables
    unew = hunew / hnew
    vnew = hvnew / hnew

    # --- Add diffusion --- #

    if diffusion:
        # Extend fluid height
        hext = np.concatenate((h[-4:-3, :], h, h[3:4, :]), axis=0)
        hext = np.concatenate((hext[:, 0:1], hext, hext[:, -1:]), axis=1)

        # Add the Laplacian
        hnew += dt * EARTH_CONSTANTS.nu * compute_laplacian(diffusion, hext)

        # Extend longitudinal velocity
        uext = np.concatenate((u[-4:-3, :], u, u[3:4, :]), axis=0)
        uext = np.concatenate((uext[:, 0:1], uext, uext[:, -1:]), axis=1)

        # Add the Laplacian
        unew += dt * EARTH_CONSTANTS.nu * compute_laplacian(diffusion, uext)

        # Extend fluid height
        vext = np.concatenate((v[-4:-3, :], v, v[3:4, :]), axis=0)
        vext = np.concatenate((vext[:, 0:1], vext, vext[:, -1:]), axis=1)

        # Add the Laplacian
        vnew += dt * EARTH_CONSTANTS.nu * compute_laplacian(diffusion, vext)

    return hnew, unew, vnew


def _apply_bcs(q: FloatArray2D, qnew: FloatArray2D):
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
    q[:, 1:-1] = np.concatenate((qnew[-2:-1, :], qnew, qnew[1:2, :]), axis=0)
    q[:, 0] = q[:, 1]
    q[:, -1] = q[:, -2]

    return q


def solve(
    num_lon_pts: int,
    num_lat_pts: int,
    ic_type: ICType,
    sim_days: float,
    courant: float,
    use_diffusion: bool = True,
    save_data: Optional[Dict[str, Any]] = None,
    print_interval: int = -1,
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
    hs = np.zeros(latlon_grid.shape, FloatT)

    if use_diffusion:
        diff_coeff = DiffusionCoefficients(cart_grid)

    if not isinstance(ic_type, ICType):
        raise TypeError(
            f"Invalid problem IC: {ic_type}. See code documentation for implemented initial conditions."
        )

    h, u, v, f = get_initial_conditions(ic_type, latlon_grid)

    save_interval: int = save_data.get("interval", 0)

    if save_interval > 0:
        tsave = [0.0]
        hsave = h[1:-1, :, np.newaxis].copy()
        usave = u[1:-1, :, np.newaxis].copy()
        vsave = v[1:-1, :, np.newaxis].copy()

        num_steps = 0
        time = 0.0
        while time < final_time:

            # Update number of iterations
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

            # If needed, adjust timestep not to exceed final time
            if time + dt > final_time:
                dt = final_time - time
                time = final_time
            else:
                time += dt

            # --- Update solution --- #
            hnew, unew, vnew = lax_wendroff_update(
                latlon_grid, cart_grid, diff_coeff, dt, f, hs, h, u, v
            )

            # --- Update solution applying BCs --- #
            h = _apply_bcs(h, hnew)
            u = _apply_bcs(u, unew)
            v = _apply_bcs(v, vnew)

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

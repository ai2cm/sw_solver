"""Initial condition-related objects and functions."""

import enum
import math

import numpy as np

from .grid import LatLonGrid
from .utils import EARTH_CONSTANTS


class ICType(enum.IntEnum):
    """Initial condition Types."""

    RossbyHaurwitzWave = 0
    """Test case 6 by Williamson."""

    ZonalGeostrophicFlow = 1
    """Test case 2 by Williamson."""


def get_initial_conditions(ic_type: ICType, latlon_grid: LatLonGrid):
    """
    Compute the initial condition state based on the grid and type.

    Parameters
    ----------
    ic_type : ICType
        The initial condition type (see enum).
    latlon_grid : LatLonGrid
        Input grid on lat-lon points.

    Returns
    -------
    h : FloatArray2D
        Initial fluid height.
    u : FloatArray2D
        Initial longitudinal velocity.
    v : FloatArray2D
        Initial latitudinal velocity.
    f : FloatArray2D
        Coriolis parameter [Hz].

    """
    # Coriolis force
    f = 2.0 * EARTH_CONSTANTS.omega * np.sin(latlon_grid.theta)

    if ic_type == ICType.RossbyHaurwitzWave:
        # --- IC 0: sixth test case taken from Williamson's suite --- #
        # ---       Rossby-Haurwitz Wave                          --- #

        # Set constants
        w = 7.848e-6
        K = 7.848e-6
        h0 = 8e3
        R = 4.0

        # Compute initial fluid height
        A = 0.5 * w * (2.0 * EARTH_CONSTANTS.omega + w) * (
            np.cos(latlon_grid.theta) ** 2.0
        ) + 0.25 * (K ** 2.0) * (np.cos(latlon_grid.theta) ** (2.0 * R)) * (
            (R + 1.0) * (np.cos(latlon_grid.theta) ** 2.0)
            + (2.0 * (R ** 2.0) - R - 2.0)
            - 2.0 * (R ** 2.0) * (np.cos(latlon_grid.theta) ** (-2.0))
        )
        B = (
            (2.0 * (EARTH_CONSTANTS.omega + w) * K)
            / ((R + 1.0) * (R + 2.0))
            * (np.cos(latlon_grid.theta) ** R)
            * (
                ((R ** 2.0) + 2.0 * R + 2.0)
                - ((R + 1.0) ** 2.0) * (np.cos(latlon_grid.theta) ** 2.0)
            )
        )
        C = (
            0.25
            * (K ** 2.0)
            * (np.cos(latlon_grid.theta) ** (2.0 * R))
            * ((R + 1.0) * (np.cos(latlon_grid.theta) ** 2.0) - (R + 2.0))
        )

        h = (
            h0
            + (
                (EARTH_CONSTANTS.a ** 2.0) * A
                + (EARTH_CONSTANTS.a ** 2.0) * B * np.cos(R * latlon_grid.phi)
                + (EARTH_CONSTANTS.a ** 2.0) * C * np.cos(2.0 * R * latlon_grid.phi)
            )
            / EARTH_CONSTANTS.g
        )

        # Compute initial wind
        u = EARTH_CONSTANTS.a * w * np.cos(
            latlon_grid.theta
        ) + EARTH_CONSTANTS.a * K * (np.cos(latlon_grid.theta) ** (R - 1.0)) * (
            R * (np.sin(latlon_grid.theta) ** 2.0) - (np.cos(latlon_grid.theta) ** 2.0)
        ) * np.cos(
            R * latlon_grid.phi
        )
        v = (
            -EARTH_CONSTANTS.a
            * K
            * R
            * (np.cos(latlon_grid.theta) ** (R - 1.0))
            * np.sin(latlon_grid.theta)
            * np.sin(R * latlon_grid.phi)
        )

    elif ic_type == ICType.ZonalGeostrophicFlow:
        # --- IC 1: second test case taken from Williamson's suite --- #
        # ----      Steady State Nonlinear Zonal Geostrophic Flow  --- #

        # Suggested values for $\alpha$ for second
        # test cases of Williamson's suite:
        # 	- 0
        # 	- 0.05
        # 	- pi/2 - 0.05
        # 	- pi/2
        alpha = math.pi / 2

        # Set constants
        u0 = 2.0 * math.pi * EARTH_CONSTANTS.a / (12.0 * 24.0 * 3600.0)
        h0 = 2.94e4 / EARTH_CONSTANTS.g

        # Make Coriolis parameter dependent on longitude and latitude
        f = (
            2.0
            * EARTH_CONSTANTS.omega
            * (
                -np.cos(latlon_grid.phi) * np.cos(latlon_grid.theta) * np.sin(alpha)
                + np.sin(latlon_grid.theta) * np.cos(alpha)
            )
        )

        # Compute initial height
        h = (
            h0
            - (EARTH_CONSTANTS.a * EARTH_CONSTANTS.omega * u0 + 0.5 * (u0 ** 2.0))
            * (
                (
                    -np.cos(latlon_grid.phi) * np.cos(latlon_grid.theta) * np.sin(alpha)
                    + np.sin(latlon_grid.theta) * np.cos(alpha)
                )
                ** 2.0
            )
            / EARTH_CONSTANTS.g
        )

        # Compute initial wind
        u = u0 * (
            np.cos(latlon_grid.theta) * np.cos(alpha)
            + np.cos(latlon_grid.phi) * np.sin(latlon_grid.theta) * np.sin(alpha)
        )
        v = -u0 * np.sin(latlon_grid.phi) * np.sin(alpha)

    else:
        raise ValueError(f"Unknown ICType: {ic_type}")

    return h, u, v, f

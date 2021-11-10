"""Grid types."""

import math
from typing import Tuple

import numpy as np

from .utils import EARTH_CONSTANTS, FloatArray2D, FloatT


class CartesianGrid:
    """Cartesian Grid.

    TODO: Document attributes.

    """

    x: FloatArray2D
    y: FloatArray2D

    dx: FloatArray2D
    dy: FloatArray2D
    dy1: FloatArray2D

    dxmin: FloatT
    dymin: FloatT

    dxc: FloatArray2D
    dyc: FloatArray2D
    dy1c: FloatArray2D

    def __init__(self, latlon_grid: "LatLonGrid"):
        """Set up the grid."""
        # Coordinates
        self.x = EARTH_CONSTANTS.a * np.cos(latlon_grid.theta) * latlon_grid.phi
        self.y = EARTH_CONSTANTS.a * latlon_grid.theta
        y1 = EARTH_CONSTANTS.a * np.sin(latlon_grid.theta)

        # Increments
        self.dx = self.x[1:, :] - self.x[:-1, :]
        self.dy = self.y[:, 1:] - self.y[:, :-1]
        self.dy1 = y1[:, 1:] - y1[:, :-1]

        # Compute mimimum distance between grid points on the sphere.
        # This will be useful for CFL condition
        self.dxmin = self.dx.min()
        self.dymin = self.dy.min()

        # "Centered" increments. Useful for updating solution
        # with Lax-Wendroff scheme
        self.dxc = 0.5 * (self.dx[:-1, 1:-1] + self.dx[1:, 1:-1])
        self.dyc = 0.5 * (self.dy[1:-1, :-1] + self.dy[1:-1, 1:])
        self.dy1c = 0.5 * (self.dy1[1:-1, :-1] + self.dy1[1:-1, 1:])


class LatLonGrid:
    """Latitude-Longitude Grid.

    TODO: Document attributes.

    """

    phi: FloatArray2D
    theta: FloatArray2D

    c: FloatArray2D

    cMidy: FloatArray2D

    tg: FloatArray2D
    tgMidx: FloatArray2D
    tgMidy: FloatArray2D

    def __init__(self, M: int, N: int):
        """Set up the grid."""
        has_points = (M > 1) and (N > 1)
        if not has_points:
            raise ValueError(
                f"Number of grid points along each direction must be greater than one. Found {(M, N)}"
            )

        # Enforce an even number of points along latitude
        N = N if N % 2 == 0 else N + 1

        # Discretize longitude
        dphi = 2.0 * math.pi / M
        phi1D = np.linspace(-dphi, 2.0 * math.pi + dphi, M + 3)

        # Discretize latitude
        # Note: we exclude the poles and only consider a channel from -85 S to 85 N to avoid pole problem
        # Note: the number of grid points must be even to prevent f to vanish
        #       (important for computing initial height and velocity in geostrophic balance)
        theta_range = 85.0
        theta1D = np.linspace(
            -theta_range / 180.0 * math.pi, theta_range / 180.0 * math.pi, N
        )

        # Build grid
        self.phi, self.theta = np.meshgrid(phi1D, theta1D, indexing="ij")

        # Cosine of mid-point values for theta along y
        self.c = np.cos(self.theta)
        self.cMidy = np.cos(0.5 * (self.theta[1:-1, 1:] + self.theta[1:-1, :-1]))

        # Compute $\tan(\theta)$
        self.tg = np.tan(self.theta[1:-1, 1:-1])
        self.tgMidx = np.tan(0.5 * (self.theta[:-1, 1:-1] + self.theta[1:, 1:-1]))
        self.tgMidy = np.tan(0.5 * (self.theta[1:-1, :-1] + self.theta[1:-1, 1:]))

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the grid size."""
        return self.phi.shape

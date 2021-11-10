"""Miscellaneous utility methods and types."""

from dataclasses import dataclass
from typing import Type

import numpy as np

FloatT = Type[np.float64]

FloatArray1D = Type[np.ndarray]
FloatArray2D = Type[np.ndarray]
FloatArray3D = Type[np.ndarray]


@dataclass(frozen=True)
class EarthConstants:
    """
    Constants for Earth.

    g : gravity [m/s2]
    rho: average atmosphere density [kg/m3]
    a : average radius [m]
    omega : rotation rate [Hz]
    scaleHeight : atmosphere scale height [m]
    nu : viscosity [m2/s]

    """

    g = 9.80616
    rho = 1.2
    a = 6.37122e6
    omega = 7.292e-5
    scaleHeight = 8.0e3
    nu = 5.0e5


EARTH_CONSTANTS = EarthConstants()

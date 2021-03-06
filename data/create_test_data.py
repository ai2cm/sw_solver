"""
Re-gerates the test data for the base numpy backend.

Note that this must match the setup in numpy_test.py:test_numpy.
"""

import numpy as np

import sw_solver

IC = sw_solver.ICType.RossbyHaurwitzWave

M = 10
N = 10
T = 2
CFL = 0.5
save = 392

diffusion = True

verbose = 50
save_data = {"interval": save}
sw_solver.numpy.solve(
    M, N, IC, T, CFL, diffusion, save_data=save_data, print_interval=verbose
)
h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
phi, theta = save_data["phi"], save_data["theta"]

# Numpy serialized data:
for array, var in zip((h, u, v, t, phi, theta), ("h", "u", "v", "t", "phi", "theta")):
    np.save(f"swes-numpy-ref-10-{var}.npy", array)

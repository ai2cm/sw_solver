#!/usr/bin/env python3
"""Driver for running the solver for the Shallow Water Equations on a Sphere (SWES)."""

import sw_solver

IC = sw_solver.ICType.RossbyHaurwitzWave

T = 5

M = 50
N = 50

CFL = 0.5

diffusion = True

verbose = 500
save = 500


save_data = {"interval": 500}
sw_solver.numpy.solve(
    M, N, IC, T, CFL, diffusion, save_data=save_data, print_interval=1
)
h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
phi, theta = save_data["phi"], save_data["theta"]

# Numpy serialized data:
# M = 10, N = 10, T = 2, CFL = 0.5, save = 392
# for array, var in zip((h, u, v, t, phi, theta), ("h", "u", "v", "t", "phi", "theta")):
#     np.save(f"data/swes-numpy-ref-10-{var}.npy", array)

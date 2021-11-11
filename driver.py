#!/usr/bin/env python3
"""Driver for running the solver for the Shallow Water Equations on a Sphere (SWES)."""

import pickle

import sw_solver

# --- SETTINGS --- #

# Initial condition:
# 	* 0: sixth test case of Williamson's suite
# 	* 1: second test case of Williamson's suite
IC = sw_solver.ICType.RossbyHaurwitzWave

# Simulation length (in days); better to use integer values.
# Suggested simulation length for Williamson's test cases:
T = 5

# Grid dimensions
M = 180
N = 90

# CFL number
CFL = 0.5

# Various solver settings:
# 	* diffusion: take diffusion into account
diffusion = True

# Output settings:
# 	* verbose: 	specify number of iterations between two consecutive output
# 	* save:		specify number of iterations between two consecutive stored timesteps
verbose = 500
save = 500

# --- RUN THE SOLVER --- #

save_data = {"interval": save}
sw_solver.numpy.solve(
    M, N, IC, T, CFL, diffusion, save_data=save_data, print_interval=verbose
)
h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
phi, theta = save_data["phi"], save_data["theta"]

# --- STORE THE SOLUTION --- #

if save > 0:
    baseName = "./data/swes-%s-%s-M%i-N%i-T%i-%i-" % (
        "numpy",
        str(IC),
        M,
        N,
        T,
        diffusion,
    )

    # Save h
    with open(baseName + "h", "wb") as f:
        pickle.dump([M, N, t, phi, theta, h], f, protocol=2)

    # Save u
    with open(baseName + "u", "wb") as f:
        pickle.dump([M, N, t, phi, theta, u], f, protocol=2)

    # Save v
    with open(baseName + "v", "wb") as f:
        pickle.dump([M, N, t, phi, theta, v], f, protocol=2)

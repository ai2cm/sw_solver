"""
Example 1: Base Numpy Solver.

This demonstrates how to run the base numpy implementation using the driver function,
and extract save data.

"""

import sw_solver

# Type of initial condition
IC = sw_solver.ICType.RossbyHaurwitzWave

# This many days
T = 4

# Grid size
M = 30
N = 30

# Courant number
CFL = 0.5

# If True, adds a diffusion term
diffusion = False

# If save > 0, save_data will be updated with 'h', 'u', 'v', 'phi', and 'theta' keys after the call.
# These will contain the save data at 'save' intervals.
save_data = {"interval": 50}

sw_solver.numpy.solve(
    M, N, IC, T, CFL, diffusion, save_data=save_data, print_interval=50
)
h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
phi, theta = save_data["phi"], save_data["theta"]

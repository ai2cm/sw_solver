#!/usr/bin/env python3
"""Driver for running the solver for the Shallow Water Equations on a Sphere (SWES)."""

import sw_solver

IC = sw_solver.ICType.RossbyHaurwitzWave

T = 8

M = 30
N = 30

CFL = 0.5

diffusion = False

verbose = 50
save = 50

gt4py_backend = "gtc:numpy"

save_data = {"interval": save}
sw_solver.gt4py.solve(
    M,
    N,
    IC,
    T,
    CFL,
    diffusion,
    save_data=save_data,
    print_interval=verbose,
    gt4py_backend=gt4py_backend,
)
h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
phi, theta = save_data["phi"], save_data["theta"]

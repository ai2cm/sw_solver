#!/usr/bin/env python3
"""Driver for running the solver for the Shallow Water Equations on a Sphere (SWES)."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import sw_solver

IC = sw_solver.ICType.RossbyHaurwitzWave

T = 8

M = 30
N = 30

CFL = 0.5

diffusion = True

verbose = 50
save = 50

save_data = {"interval": save}
sw_solver.numpy.solve(
    M, N, IC, T, CFL, diffusion, save_data=save_data, print_interval=verbose
)
h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
phi, theta = save_data["phi"], save_data["theta"]

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.contourf(phi[1:-1, :], theta[1:-1, :], u[:, :, -1])
plt.show()

# Numpy serialized data:
# M = 10, N = 10, T = 2, CFL = 0.5, save = 392
# for array, var in zip((h, u, v, t, phi, theta), ("h", "u", "v", "t", "phi", "theta")):
#     np.save(f"data/swes-numpy-ref-10-{var}.npy", array)

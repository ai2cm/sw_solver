"""
Example 2: GT4Py version.

In this example we demonstrate the gt4py version of the solver.

"""
import sw_solver

IC = sw_solver.ICType.RossbyHaurwitzWave

T = 8

M = 30
N = 30

CFL = 0.5

diffusion = False

save_data = {"interval": 50}
sw_solver.gt4py.solve(
    M,
    N,
    IC,
    T,
    CFL,
    diffusion,
    save_data=save_data,
    print_interval=50,
    gt4py_backend="gtc:numpy",
)
h, u, v, t = save_data["h"], save_data["u"], save_data["v"], save_data["t"]
phi, theta = save_data["phi"], save_data["theta"]

import pickle

import numpy as np

import sw_solver


def test_numpy():

    # --- SETTINGS --- #

    # Solver version:
    # 	* numpy (NumPy version)
    #   * gt4py (DSL version)
    version = "numpy"

    # Initial condition:
    # 	* 0: sixth test case of Williamson's suite
    # 	* 1: second test case of Williamson's suite
    IC = 0

    # Simulation length (in days); better to use integer values.
    # Suggested simulation length for Williamson's test cases:
    T = 2

    # Grid dimensions
    M = 10
    N = 10

    # CFL number
    CFL = 0.5

    # Various solver settings:
    # 	* diffusion: take diffusion into account
    diffusion = True

    # --- RUN THE SOLVER --- #

    pb = sw_solver.numpy.Solver(T, M, N, IC, CFL, diffusion)
    t, phi, theta, h, u, v = pb.solve(0, 500)

    # --- VALIDATE THE SOLUTION --- #

    refBaseName = "./data/swes-ref-%s-%s-M%i-N%i-T%i-%i-" % (
        version,
        str(IC),
        M,
        N,
        T,
        diffusion,
    )
    for var, name in zip((h, u, v), ("h", "u", "v")):
        Mf, Nf, tf, phif, thetaf, varf = pickle.load(
            open(refBaseName + name, mode="rb")
        )
        # Uses integer time
        assert np.allclose(t, tf)
        assert np.allclose(phi, phif)
        assert np.allclose(theta, thetaf)
        assert np.allclose(var, varf)

import numpy as np
from constants import *
from spsolver import SPSolver3D
from initial_configurations import rho0_soliton_3D


# Simulation grid
L = 0.25 # Mpc
N = 512 # ^3

solver = SPSolver3D(N, L)
solver.set_initial_density(rho0_soliton_3D, n = 15, k = 5, r_c_b = 0.0002, r_c_e = 0.0005)

psi, t = solver.run(10, 0.01, store = False, store_every = 10, draw = True, draw_every = 1, save_draw = True, draw_vmin = 9, draw_vmax = "auto", save_animation = True, draw_cmap = "jet")


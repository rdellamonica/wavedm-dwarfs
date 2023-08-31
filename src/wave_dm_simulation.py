import numpy as np
from constants import *
from wave_dm_simulation import SPSolver3D, rho0_soliton_3D


# Simulation grid
L = 0.25 # Mpc
N = 512 # ^3

solver = SPSolver3D(N, L)
solver.set_initial_density(rho0_soliton_3D)

psi, t = solver.run(10, 0.01, store = False, store_every = 10, draw = True, draw_every = 1, save_draw = True, draw_vmin = 9, draw_vmax = "auto", save_animation = True, draw_cmap = "jet")


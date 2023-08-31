import numpy as np
from constants import *

def rho0_soliton_3D(x, y, z, L, n = 10, k = 5, r_c_b = 0.0003, r_c_e = 0.0004):

    rho = 0
    
    # Concentrating initial solitons within a central window in the grid
    l0 = L/k

    # Bunch of soliton scattered in the simulation grid (on the equatorial plane) with random core radii
    r_c_arr = np.random.uniform(r_c_b, r_c_e, n)
    x0_arr = np.random.uniform(0, 1, n)
    y0_arr = np.random.uniform(0, 1, n)
    z0_arr = np.random.uniform(0, 1, n)
    z0 = L/2

    M = 0
    
    for r_c, x0, y0, z0 in zip(r_c_arr, x0_arr, y0_arr, z0_arr):
        # Central soliton density
        rho0 = 3.1e+15*(2.5e-22*eV/Msun/m_a)**2*(0.001/r_c)**4
        # Scaling relation 
        M_s = 2.2e+10/((m_a/(1e-23*eV/Msun))**2*(r_c/0.001))
        
        M += M_s
        x0 = x0*l0+(L/2-l0/2)
        y0 = y0*l0+(L/2-l0/2)
        z0 = L/2
        r = np.sqrt(((x-x0)**2+(y-y0)**2+(z-z0)**2))
        # Soliton empirical profile from Schive et al. (2014)
        rho += rho0*(1+0.091*(r/r_c)**2)**(-8)
    
    print(f"Central density order of magnitude: 10^{np.log10(rho0):.0f}")
    print(f"Total mass: {M:e} M_sun")
    
    return rho, np.log10(rho0)
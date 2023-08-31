import numpy as np
import matplotlib.pyplot as plt
import pickle 
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import quad
from star import Star
from multiprocessing import Pool
import os
from constants import *

# Opening a specific FDM simulation run

timestamp = "XXX"

with open(f"{timestamp}/{timestamp}.run", "rb") as f:
    run = pickle.load(f)

# Getting run parameters

m_a = run["m_a"]
L = run["L"]
N = run["N"]

# Density

print("Computing density profile...")

psi = run['psi'][-1]
rho = np.abs(psi)**2

j = np.unravel_index(np.argmax(rho), rho.shape)
x = np.linspace(-L/2, L/2, N)

xx, yy, zz = np.meshgrid(x, x, x)

x0 = xx[j]
y0 = yy[j]
z0 = zz[j]

if True:

    rr = np.logspace(np.log10(0.0001), np.log10(.05), 50)

    r = np.sqrt((xx-x0)**2+(yy-y0)**2+(zz-z0)**2)

    rho_rr = []

    r_m_rho = []

    for i in range(len(rr)-1):
        rho_r = np.mean(rho[(r <= rr[i+1]) & (r >= rr [i])])
        rho_rr.append(rho_r)
        r_m_rho.append((rr[i+1]+rr[i])/2)

    rho_rr = np.array(rho_rr)
    index = np.isnan(rho_rr)
    rho_rr = rho_rr[~index]
    rho_rr = np.insert(rho_rr, 0, rho_rr[0])

    r_m_rho = np.array(r_m_rho)
    r_m_rho = r_m_rho[~index]
    r_m_rho = np.insert(r_m_rho, 0, 0)

    rho_rr_int = interp1d(r_m_rho, rho_rr*r_m_rho**2)

# Mass

def M_r(r):
    return 4*np.pi*quad(rho_rr_int, 0.000001, r)[0]

M_arr = np.vectorize(M_r)

print("Ok!")

# Velocity dispersion

print("Computing velocity dispersion profile...")

v = np.gradient(np.angle(psi)*hbar, x, x, x)

vx = v[0]*Mpc/Gyr/1000/m_a
vy = v[1]*Mpc/Gyr/1000/m_a
vz = v[2]*Mpc/Gyr/1000/m_a

v2 = np.sqrt(vx**2+vy**2+vz**2)

j = np.unravel_index(np.argmax(rho), rho.shape)

x = np.linspace(0, L, N)

xx, yy, zz = np.meshgrid(x, x, x)

x0 = xx[j]
y0 = yy[j]
z0 = zz[j]

r = np.sqrt((xx-x0)**2+(yy-y0)**2+(zz-z0)**2)

sigma_rr = []

r_m_sigma = []

for i in range(len(rr)-1):
    v2_r = v2[(r <= rr[i+1]) & (r >= rr [i])]

    sigma_r = np.std(v2_r)
    sigma_rr.append(sigma_r)
    r_m_sigma.append((rr[i+1]+rr[i])/2)

sigma_rr = np.array(sigma_rr)
index = np.isnan(sigma_rr)
sigma_rr = sigma_rr[~index]
sigma_rr = np.insert(sigma_rr, 0, sigma_rr[0])

# Size of the wavelets

l_c  = 2*np.pi*hbar/(m_a*sigma_rr*np.sqrt(3/2))
l_c_int = interp1d(r_m_rho, l_c)

print("Ok!")

# Gravitational potential

print("Computing gravitational potential...")

x = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)

klin = 2.0 * np.pi/L * np.arange(-N/2,N/2)
kx, ky, kz = np.meshgrid(klin, klin, klin)

kx = np.fft.ifftshift(kx)
ky = np.fft.ifftshift(ky)
kz = np.fft.ifftshift(kz)

kSq = kx**2 + ky**2 + kz**2
rhobar = np.mean(rho)
Vhat = np.fft.fftn(rho-rhobar) / ( kSq  + (kSq==0))
V = -4.0*np.pi*G_N*np.real(np.fft.ifftn(Vhat))

Phi = RegularGridInterpolator((x, y, z), V, method = "linear")

# Centering

j = np.unravel_index(np.argmin(V), V.shape)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

x0_c = xx[j]
y0_c = yy[j]
z0_c = zz[j]

print("Ok!")

# Population of stars
# Generating the CFD to extract the stars

N_points = 10000                    # number of points at which evaluate the PDF to extraxt stars
r = np.linspace(0, 10, N_points)
dx = r[1]-r[0]

rd = .5                             # scale radius of exponential disk
Sigma0 = 1/rd                       # central density
h0 = 1/5                            # for the normalization

PDF = Sigma0*np.exp(-r/rd)

sigma = .380                        # central core FWHM

gaussian_x = np.arange(-5*sigma, 5*sigma, dx)
gaussian_kernel = np.exp(-(gaussian_x/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)

cored_PDF = np.convolve(PDF, gaussian_kernel, mode = "same")
cored_PDF_int = interp1d(r, cored_PDF)

cored_CDF = np.zeros_like(cored_PDF)

for i in range(len(cored_CDF)):
    cored_CDF[i] = quad(cored_PDF_int, 0, r[i])[0]

cored_CDF = cored_CDF/np.max(cored_CDF)

CDF_inv_int = interp1d(r, cored_CDF)
CDF_min = CDF_inv_int(0.35)

CDF_int = interp1d(cored_CDF, r)

# Generating the CFD to extract the stars

N_stars = 10000                                                         # number of stars

np.random.seed(42)

r_arr = CDF_int(np.random.uniform(CDF_min, 1, N_stars))/1000            # extracting randomly positions from the PDF
phi_arr = np.random.rand(N_stars)*2*np.pi
h_arr = (np.random.rand(N_stars)*h0-h0/2)/1000

R_arr = np.sqrt(r_arr**2+h_arr**2)
v0_arr = np.sqrt(G_N*M_arr(R_arr)/(R_arr))                              # assigning circular velocity from integrated mass

x0_arr = r_arr*np.cos(phi_arr)
y0_arr = r_arr*np.sin(phi_arr)
z0_arr = h_arr
vx_arr = -np.sin(phi_arr)*v0_arr
vy_arr = np.cos(phi_arr)*v0_arr

initial_conditions = np.c_[x0_arr, y0_arr, z0_arr, vx_arr, vy_arr]      # array of all the initial conditions

# Star integration
n_processes = 30

foldername = timestamp+"/Stars/"

if not os.path.exists(foldername):
    os.mkdir(foldername)

def integrate_star(pos):
    x0, y0, z0, vx0, vy0 = pos

    s = Star([x0_c, y0_c, z0_c], Phi)

    s.set_initial_position(x0, y0, z0)
    s.set_initial_velocity(vx0, vy0, 0)

    s.evolve(Phi, 10, 0.001)

    return s

if __name__ == '__main__':
    with Pool(n_processes) as pool:
        k = 0
        N = len(initial_conditions)
        for i, s in enumerate(pool.imap_unordered(integrate_star, initial_conditions)):
            k += 1
            print(f"{k/N*100:.3f} % completed", end = "\r")
            
            filename = foldername + f"/{i:05}.s"

            try:
                s.save(filename)
            except ValueError:
                print(f"{i}-th star not saved")



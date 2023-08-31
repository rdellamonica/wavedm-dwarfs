import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pickle
from tqdm import tqdm
import pyfftw
from constants import *

FFTW = pyfftw.interfaces.numpy_fft

# Number of cores

pyfftw.config.NUM_THREADS = 32

# 3D Schrödinger-Poisson Solver with kick-drift-kick method
class SPSolver3D:
    
    def __init__(self, N, L):
        
        print("Initializing the 3D Schrödinger-Poisson Solver")
        
        self.N = N
        self.L = L
        
        print(f"Box side: {L} Gpc")
        print(f"Simulation resolution: {N}^3")        
        
        xlin = np.linspace(0, L, num = N+1)
        xlin = xlin[0:N]
        
        print(f"Creating grid...")
        self.xx, self.yy, self.zz = np.meshgrid(xlin, xlin, xlin)
    
    def set_initial_density(self, rho_f, **rho_f_kwargs):
        print(f"Setting up initial conditions...")
        self.rho0, rhoc = rho_f(self.xx, self.yy, self.zz, self.L, **rho_f_kwargs)
        self.vmax_auto  = rhoc
        self.rhobar = np.mean(self.rho0)
    
    def run(self, t_f, dt, verbose = True, store = True, save_file = True, store_every = 1, draw = False, draw_every = np.inf, draw_vmin = 9, draw_vmax = 15, save_draw = False, save_animation = False, draw_cmap = "inferno"):
        
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        print(f"Simulation running with timestamp {run_timestamp}")

        if draw_vmax == "auto":
            draw_vmax = self.vmax_auto

        try: 
            os.mkdir(run_timestamp)
        except FileExistsError:
            run_timestamp = run_timestamp + "_1"

        rho = self.rho0
        rhobar = self.rhobar

        psi = np.sqrt(self.rho0)

        t = 0
        
        klin = 2.0 * np.pi / self.L * np.arange(-self.N/2, self.N/2)
        kx, ky, kz = np.meshgrid(klin, klin, klin)
        
        kx = FFTW.ifftshift(kx)
        ky = FFTW.ifftshift(ky)
        kz = FFTW.ifftshift(kz)

        kSq = kx**2 + ky**2 + kz**2

        Vhat = FFTW.fftn(rho-rhobar) / ( kSq  + (kSq==0))
        V = -4.0*np.pi*G_N*np.real(FFTW.ifftn(Vhat))

        if dt == "auto":
            dt_kind = "auto"
            dt = np.max([m_a/(6*hbar)*(self.L/self.N)**2, hbar/(m_a*np.max(V))])/10
        else:
            dt_kind = "fixed"

        N_t = int(np.ceil(t_f/dt))
        
        if store:
            psi_arr = np.empty((int(np.ceil(N_t/store_every)+1), self.N, self.N, self.N), dtype = np.complex_)
        else:
            psi_arr = np.empty((2, self.N, self.N, self.N), dtype = np.complex_)

        if draw:
            plt.close('all')
            _, ax = plt.subplots(figsize = (5,5))
            ax.imshow(np.log10(np.abs(psi[:,:,int(np.ceil(self.N/2))])**2), cmap = draw_cmap, vmin = draw_vmin, vmax = draw_vmax, extent=[-self.L/2,self.L/2,-self.L/2,self.L/2])
            ax.text(0.95, 0.95, f"{t/1000:.3f} Gyr", transform = ax.transAxes, color = "white", ha = "right", va = "top")

            ax.set_xlabel("x (Mpc)")
            ax.set_ylabel("y (Mpc)")

            plt.show()
            draw_i = 0
        
        psi_arr[0] = psi
        t_arr = [t]

        draw_i = np.inf
        store_i = np.inf
        store_k = 1
        frame_i = 0

        for i in tqdm(range(N_t-1), position=0, leave=True):

            try:
                
                psi = np.exp(-1.j*dt/2.0*(m_a/hbar)*V) * psi               # Half kick

                psihat = FFTW.fftn(psi)                         
                psihat = np.exp(dt * (-1.j*kSq/2.*(hbar/m_a)))  * psihat   # Drift
                psi = FFTW.ifftn(psihat)

                rho = np.abs(psi)**2
                rhobar = np.mean(rho)

                Vhat = FFTW.fftn(rho-rhobar) / ( kSq  + (kSq==0))
                V = -4.0*np.pi*G_N*np.real(FFTW.ifftn(Vhat))

                if dt_kind == "auto":
                    dt = np.max([m_a/(6*hbar)*(self.L/self.N)**2, hbar/(m_a*np.max(V))])/10

                psi = np.exp(-1.j*dt/2.0*(m_a/hbar)*V) * psi               # Half kick
                t += dt

                draw_i += 1
                store_i += 1
                
                if draw:
                    if draw_i >= draw_every:
                        plt.close('all')
                        fig, ax = plt.subplots(figsize = (5,5))
                        ax.imshow(np.log10(np.abs(psi[:,:,int(np.ceil(self.N/2))])**2), cmap = draw_cmap, vmin = draw_vmin, vmax = draw_vmax, extent=[-self.L/2,self.L/2,-self.L/2,self.L/2])
                        ax.text(0.95, 0.95, f"{t:.3f} Gyr", transform = ax.transAxes, color = "white", ha = "right", va = "top")

                        ax.set_xlabel("x (Mpc)")
                        ax.set_ylabel("y (Mpc)")

                        fig.patch.set_facecolor('white')

                        if save_draw:
                            fig.savefig(f"{run_timestamp}/{str(frame_i).zfill(6)}.png")

                        draw_i = 0
                        frame_i += 1
                    
            except KeyboardInterrupt:
                break

            if store:
                if store_i >= store_every:
                    psi_arr[store_k] = psi
                    store_i = 0
                    store_k += 1
                    t_arr.append(t)
        
        if save_draw and save_animation:
            print("Producing animation...")
            os.system(f'ffmpeg -r 24 -i {run_timestamp}/%06d.png -c:v libx264 -pix_fmt yuv420p {run_timestamp}/Animation.mp4')

        if not store:
            psi_arr[1] = psi
            t_arr.append(t)

        if save_file:
            print("Saving the run...")
            with open(run_timestamp+"/"+ run_timestamp + ".run", "wb") as f:
                
                run = {
                    "timestamp": run_timestamp,
                    "L": self.L,
                    "N": self.N,
                    "psi": psi_arr,
                    "m_a": m_a
                }
                
                pickle.dump(run, f)
        
        print("Done!")
        return psi_arr, t_arr
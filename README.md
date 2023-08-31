# wavedm-dwarfs
Simulation of stellar disk in Wave Dark Matter halo. For details see the orignal [paper](https://arxiv.org/abs/2308.14664).



### The Wave DM halo simulation
<img width="1255" alt="Simulation" src="https://github.com/rdellamonica/wavedm-dwarfs/assets/53187090/f8e6829d-538a-4631-9cfe-44d0a4830a18">

The main file to execute is `wave_dm_simulation.py`. Here one can change the parameter `L` and `N` representing the simulation box side and linear resolution, respectively. This file calls the `SPSolver3D` class and initializes the resolution grid. The `rho0_soliton_3D` function initializes the simulation with `n` solitons with core radii between `r_c_b` and `r_c_e`, thus fixing the conserved total mass of the halo. 

The simulation is than carried out with the `SPSolver3D.run` function up to the time instant `t_f` starting with an initial timestep `dt`.

### The stellar orbits simulation

This script opens the Wave DM simulation carried out in the previous step by its unique timestamp, and the computes the density field, the velocity dispersion field (which are used to compute the characteristic wavelets size at each point on the grid) and then computes the gravitational potential (interpolated on a regular grid). It then initializes `N_stars` initial positions and velocities for point-like test particles randomly distributed in the halo according to a radial profile made up of an exponential thin disk and a cored distribution (details and motivations in the paper). The orbits of such stellar population are then integrate independently (in parallel) up to `t_f` on the Wave DM halo potential, taking into account interactions of the stars with the inherent granularity of the Wave DM halo.

### Animation

In the `animation` folder it is possible to find the animation corresponding the $N=512^3$ simulation used in the paper.

https://github.com/rdellamonica/wavedm-dwarfs/assets/53187090/a1ae7805-ed87-49c0-b779-fffcf944c90f


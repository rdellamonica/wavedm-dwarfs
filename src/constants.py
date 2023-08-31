from astropy import constants, units

# Constants

Mpc = units.Mpc.to('m')
Msun = units.Msun.to('kg')
Gyr = units.Gyr.to('s')
eV = 1.7826619216279e-36 #kg 

G_N = constants.G.to('Mpc**3/Msun*Gyr^2').value
hbar = constants.hbar.to('Msun*Mpc**2/Gyr').value
c = constants.c.to('Mpc/Gyr').value

# Wave DM boson mass used for the simulation
m_a = 1.6e-22*eV/Msun
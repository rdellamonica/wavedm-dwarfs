import numpy as np

# Class for the integration of stellar orbits

class Star:

    def __init__(self, pos_center, potential):
        self.x0_c, self.y0_c, self.z0_c = pos_center
        self.phi = potential

    def set_initial_position(self, x, y, z):
        self.x0 = x+self.x0_c
        self.y0 = y+self.y0_c
        self.z0 = z+self.z0_c
    
    def set_initial_velocity(self, vx, vy, vz):
        self.vx0 = vx
        self.vy0 = vy
        self.vz0 = vz
    
    def evolve_adaptive(self, phi, wavelet_sizes, tf, Prec = 2):

        r = np.sqrt(self.x0**2+self.y0**2+self.z0**2)
        v0 = np.sqrt(self.vx0**2 + self.vy0**2 + self.vz0**2)
        dt = wavelet_sizes(r)/(5*v0)

        x_arr = [np.array([self.x0, self.y0, self.z0])]
        v_arr = [np.array([self.vx0, self.vy0, self.vz0])]
        t_arr = [0]

        t = 0

        while t < tf:

            try:
                v = np.linalg.norm(v_arr[-1])
                r = np.linalg.norm(x_arr[-1]-np.array([self.x0_c, self.y0_c, self.z0_c]))
                dt = wavelet_sizes(r)/(Prec*v)
                t += dt
                dx = abs(max(v_arr[-1]))*dt/100
                
                x_arr.append(x_arr[-1] + v_arr[-1]*dt)

                ax = -(phi(x_arr[-1]+np.array([dx, 0, 0]))[0]-phi(x_arr[-1]-np.array([dx, 0, 0]))[0])/(2*dx)
                ay = -(phi(x_arr[-1]+np.array([0, dx, 0]))[0]-phi(x_arr[-1]-np.array([0, dx, 0]))[0])/(2*dx)
                az = -(phi(x_arr[-1]+np.array([0, 0, dx]))[0]-phi(x_arr[-1]-np.array([0, 0, dx]))[0])/(2*dx)
                
                a = np.array([ax, ay, az])

                v_arr.append(v_arr[-1]+ a*dt)

                t_arr.append(t)
                
            except ValueError:
                break
    
        self.t = t_arr
        self.x_arr = np.array(x_arr)
        self.v_arr = np.array(v_arr)
    
    def evolve(self, phi, tf, dt = 0.001):

        N_t = int(np.ceil(tf / dt))

        t = np.linspace(0, tf, N_t)

        x_arr = [np.array([self.x0, self.y0, self.z0])]
        v_arr = [np.array([self.vx0, self.vy0, self.vz0])]

        dx = max([abs(self.vx0), abs(self.vy0), abs(self.vz0)])*dt/1000

        for i in range(len(t)):
            
            try:
                x_arr.append(x_arr[-1] + v_arr[-1]*dt)

                ax = -(phi(x_arr[-1]+np.array([dx, 0, 0]))[0]-phi(x_arr[-1]-np.array([dx, 0, 0]))[0])/(2*dx)
                ay = -(phi(x_arr[-1]+np.array([0, dx, 0]))[0]-phi(x_arr[-1]-np.array([0, dx, 0]))[0])/(2*dx)
                az = -(phi(x_arr[-1]+np.array([0, 0, dx]))[0]-phi(x_arr[-1]-np.array([0, 0, dx]))[0])/(2*dx)
                
                a = np.array([ax, ay, az])

                v_arr.append(v_arr[-1]+ a*dt)
                
            except ValueError:
                break
            
        n = len(x_arr)-1
        self.t = t[:n]
        self.x_arr = np.array(x_arr[:n])
        self.v_arr = np.array(v_arr[:n])

    def save(self, filename):

        out = np.concatenate((np.reshape(self.t, (-1,1)), self.x_arr, self.v_arr), axis = 1)
        np.savetxt(filename, out)
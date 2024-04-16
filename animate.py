# %%
import sympy as sp
import numpy as np
import scipy as sc
from scipy.integrate import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import symbtools as st
import symbtools.modeltools as mt
import symbtools.noncommutativetools as nct
import symbtools.modeltools as mt

# %%
n_charges = 3
charges = np.ones(n_charges) #* -1
if n_charges == 1:
    magnet_positions = np.array([[0,0]])
else:
    magnet_positions = np.array([[np.cos(phi), np.sin(phi)] for phi in [i*2*np.pi/n_charges for i in np.arange(n_charges)]])
h = 0.3
b = 0.2
w0 = 0.5

# %%
def rhs_single(t, x):
    s = np.array([0,0], dtype=float)
    for pos, p in zip(magnet_positions, charges):
        s += p * (pos- x[0:2]) * ((pos[0]-x[0])**2 + (pos[1]-x[1])**2 + h**2)**(-3/2)

    dx1 = x[2]
    dx2 = x[3]
    dx3 = s[0] - b*x[2] - w0**2*x[0]
    dx4 = s[1] - b*x[3] - w0**2*x[1]

    return np.array([dx1, dx2, dx3, dx4])
def rhs(t, x):
    n, m = x.shape
    s = np.zeros((2, m), dtype=float)
    for _pos, p in zip(magnet_positions, charges):
        pos = np.zeros((m, 2), dtype=float)
        pos[:] = _pos
        pos = pos.T
        s += p * (pos- x[0:2]) * ((pos[0]-x[0])**2 + (pos[1]-x[1])**2 + h**2)**(-3/2)

    dx1 = x[2]
    dx2 = x[3]
    dx3 = s[0] - b*x[2] - w0**2*x[0]
    dx4 = s[1] - b*x[3] - w0**2*x[1]

    return np.array([dx1, dx2, dx3, dx4])



# %%
limits = [[-2, 2], [-2, 2]]
num_interpolation_points = 200

mesh_index = []
for i in range(len(limits)):
    mesh_index.append(np.linspace(*limits[i], num_interpolation_points))
meshgrid = np.meshgrid(*mesh_index) # shape [(NIP, NIP, ..), (NIP, NIP, ..)]
points = np.vstack([x.ravel() for x in meshgrid]).T




# %%
xx0 = np.zeros((4,2))
xx0[:2,0] = points[9012]
xx0[:2,1] = points[9013]

tend = 20
tt = np.linspace(0, tend, 1000)
fig = plt.figure()
ax = fig.add_subplot(111)
solutions = []
for x0 in xx0.T:
    sol = solve_ivp(rhs_single, (0, tend), x0, "RK45", t_eval=tt, rtol=1e-7, atol=1e-7)
    solutions.append(sol)
x1 = []
y1 = []
x2 = []
y2 = []
plt.xlim(-2,2)
plt.ylim(-2,2)
line, = ax.plot([], [], lw = 3)
def init():
    ax.scatter(solutions[0].y[0][0], solutions[0].y[1][0])
    ax.scatter(solutions[1].y[0][0], solutions[1].y[1][0])
    ax.grid()
    line.set_data([],[])
    ax.scatter(*magnet_positions.T, color= "red")
    return line,

def animate(i):
    assert i<len(sol.t)
    line.set_data(sol.y[0, :i], sol.y[1, :i])
    return line,

anim = FuncAnimation(fig, animate, init_func = init, frames = 1000, interval = 1, blit = True)

# for i in range(len(solutions[0].t)):
#     # for sol in solutions:
#     x1.append(solutions[0].y[0, i])
#     y1.append(solutions[0].y[1, i])
#     plt.plot(x1, y1, color="blue")
#     x2.append(solutions[1].y[0, i])
#     y2.append(solutions[1].y[1, i])
#     plt.plot(x2, y2, color="orange")
#     plt.pause(0.001)



# anim.save('test.mp4', writer = 'ffmpeg', fps = 30)
plt.show()

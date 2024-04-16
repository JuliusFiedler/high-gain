import sympy as sp
import numpy as np
import scipy as sc
from scipy.integrate import *
import matplotlib.pyplot as plt
from ipydex import IPS
import symbtools as st
import symbtools.modeltools as mt
import symbtools.noncommutativetools as nct
import symbtools.modeltools as mt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

G = 6.674e-3
m1 = 1
m2 = 1
m3 = 1
masses = [m1, m2, m3]
# x = [rr1, rr2, rr3, drr1, drr2, drr3]
def rhs(t, x):
    rr1 = x[0:3]
    rr2 = x[3:6]
    rr3 = x[6:9]

    drr1 = x[9:12]
    drr2 = x[12:15]
    drr3 = x[15:18]

    # denominators, as factors for adolc -> minus in exponent
    den12 = ((rr1[0] - rr2[0])**2 + (rr1[1] - rr2[1])**2 + (rr1[2] - rr2[2])**2) ** (-3/2)
    den13 = ((rr1[0] - rr3[0])**2 + (rr1[1] - rr3[1])**2 + (rr1[2] - rr3[2])**2) ** (-3/2)
    den23 = ((rr2[0] - rr3[0])**2 + (rr2[1] - rr3[1])**2 + (rr2[2] - rr3[2])**2) ** (-3/2)

    ddrr1 = [-G*m2 * (rr1[i]-rr2[i]) * den12 - G*m3 * (rr1[i]- rr3[i]) * den13 for i in range(3)]
    ddrr2 = [-G*m2 * (rr2[i]-rr3[i]) * den23 - G*m3 * (rr2[i]- rr1[i]) * den12 for i in range(3)]
    ddrr3 = [-G*m2 * (rr3[i]-rr1[i]) * den13 - G*m3 * (rr3[i]- rr2[i]) * den23 for i in range(3)]

    # ddrr1 = -G*m2 * (rr1-rr2) / ((rr1[0] - rr2[0])**2 + (rr1[1] - rr2[1])**2 + (rr1[2] - rr2[2])**2) ** (3/2) \
    #         -G*m3 * (rr1-rr3) / ((rr1[0] - rr3[0])**2 + (rr1[1] - rr3[1])**2 + (rr1[2] - rr3[2])**2) ** (3/2)
    # ddrr2 = -G*m3 * (rr2-rr3) / ((rr2[0] - rr3[0])**2 + (rr2[1] - rr3[1])**2 + (rr2[2] - rr3[2])**2) ** (3/2) \
    #         -G*m1 * (rr2-rr1) / ((rr2[0] - rr1[0])**2 + (rr2[1] - rr1[1])**2 + (rr2[2] - rr1[2])**2) ** (3/2)
    # ddrr3 = -G*m1 * (rr3-rr1) / ((rr3[0] - rr1[0])**2 + (rr3[1] - rr1[1])**2 + (rr3[2] - rr1[2])**2) ** (3/2) \
    #         -G*m2 * (rr3-rr2) / ((rr3[0] - rr2[0])**2 + (rr3[1] - rr2[1])**2 + (rr3[2] - rr2[2])**2) ** (3/2)
    return np.array([*drr1, *drr2, *drr3, *ddrr1, *ddrr2, *ddrr3])

tend = 10
tt = np.linspace(0, tend, 5000)
xx0 = [.1,0,0, 0,.2,0, 0,0,.15, 0,0,0, 0,0,0, 0,0,0]
# xx0 = [ 0.03812932,  0.09211813,  0.02371743,  0.05465057,  0.00811377,  0.06193882,  0.00722011,  0.0997681 ,  0.06434376, -0.24385575,  0.18341026,  0.22822593,  0.10084323, -0.27506154,  0.05503131,  0.14301252,  0.09165128, -0.28325724]
sol = solve_ivp(rhs, (0, tend), xx0, "RK45", t_eval=tt, rtol=1e-7, atol=1e-7)

center_of_mass = (sol.y[0:3]*m1 + sol.y[3:6]*m2 + sol.y[6:9]*m3) / (m1+m2+m3)
# IPS()
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(np.min(sol.y[0:9:3]*1.01), np.max(sol.y[0:9:3]*1.01))
ax.set_ylim(np.min(sol.y[1:10:3]*1.01), np.max(sol.y[1:10:3]*1.01))
ax.set_zlim(np.min(sol.y[2:11:3]*1.01), np.max(sol.y[2:11:3]*1.01))
colors = ["red", "blue", "green", "black"]
lines = [ax.plot([], [], [], lw = 1, color=colors[i], alpha=0.5)[0] for i in range(4)]
def init():
    for line in lines:
        line.set_data([],[])
        line.set_3d_properties([])
    return lines

def animate(i):
    assert i<len(sol.t)
    lines = [ax.plot(sol.y[0+3*k, :i], sol.y[1+3*k, :i], sol.y[2+3*k, :i], lw = 1, color=colors[k])[0] for k in range(3)]
    lines.append(ax.plot(center_of_mass[0, :i], center_of_mass[1, :i], center_of_mass[2, :i], lw=3, color="black")[0])
    ax.set_title(f"Time: {sol.t[i]}")
    return lines

# anim = FuncAnimation(fig, animate, init_func = init, frames = len(sol.t), interval = 1, blit = True)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for i in range(3):
    ax.plot(sol.y[0+3*i, :], sol.y[1+3*i, :], sol.y[2+3*i, :], color=colors[i], label=f"Body {i+1}, m={masses[i]}")
    ax.scatter(sol.y[0+3*i, 0], sol.y[1+3*i, 0], sol.y[2+3*i, 0], color=colors[i])
ax.plot(center_of_mass[0], center_of_mass[1], center_of_mass[2], color="black")
ax.set_xlim(np.min(sol.y[0:9:3]*1.01), np.max(sol.y[0:9:3]*1.01))
ax.set_ylim(np.min(sol.y[1:10:3]*1.01), np.max(sol.y[1:10:3]*1.01))
ax.set_zlim(np.min(sol.y[2:11:3]*1.01), np.max(sol.y[2:11:3]*1.01))
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.legend()
plt.tight_layout()
plt.show()



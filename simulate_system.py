import numpy as np
import sympy as sp
import os
from systems import *
from net import Net
from ipydex import IPS
import adolc as ac
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


system = InvPendulum2()

rhs = system.rhs


t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 50000)

x0 = [1,0,0]

sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval)


fig, ax = plt.subplots(system.n, 1, sharex=True)
for i in range(system.n):
    ax[i].plot(sol.t, sol.y[i, :], label=f"$x_{i}$ System")
    ax[i].set_ylabel(f'$x_{i}$')
    ax[i].legend()
    ax[i].grid()
ax[-1].set_xlabel('Time (s)')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', 'box')
ax.plot(sol.y[0], sol.y[1], color="tab:blue")
ax.scatter(*sol.y[:2,0], color="tab:blue")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
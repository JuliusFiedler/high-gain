import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from systems import *
import adolc
from ipydex import IPS


sys = Roessler()
rhs = sys.rhs

# Initial State values
xx0 = [2, 3, 4]
t_end = 300
# Note: The system is simulated for 300s to generate a nice plot, but due to numerical differences
# on different hardware, the evaluation is performed at half that time.
tt = np.linspace(0, t_end, 12000 - 1)
sim = solve_ivp(rhs, (0, t_end), xx0, t_eval=tt)
y = sim.y.tolist()

Tape_F = 0
Tape_H = 1

n = sys.n
adolc.trace_on(Tape_F)
af = [adolc.adouble() for _ in range(n)]
for i in range(n):
    af[i].declareIndependent()
vf = sys.rhs(0, af)
for a in vf:
    a.declareDependent()
adolc.trace_off()

adolc.trace_on(Tape_H)
ah = [adolc.adouble() for _ in range(n)]
for i in range(n):
    ah[i].declareIndependent()
vh = sys.get_output(ah)
vh.declareDependent()
adolc.trace_off()

d = sys.N
lie_derivs = []
for x in sim.y.T:
    lie_derivs.append(adolc.lie_scalarc(Tape_F, Tape_H, x, d))
lie_derivs = np.array(lie_derivs)

IPS()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(y[0], y[1], y[2], label="Phase portrait", lw=1, c="k")
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_zlabel("z", fontsize=15)
ax.legend()
ax.grid()

fig, ax = plt.subplots(4,1)

ax[0].plot(sim.t, y[0], label="x1")
ax[1].plot(sim.t, y[1], label="x2")
ax[2].plot(sim.t, y[2], label="x3")
ax[3].plot(sim.t, lie_derivs[:, -1], label="alpha")
plt.legend()
plt.grid()
plt.tight_layout()

# plt.figure()


plt.show()
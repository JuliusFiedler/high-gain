import os
import torch
import numpy as np
import joblib
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import adolc

from systems import *
from net import Net
import util as u
from ipydex import activate_ips_on_exception, IPS
activate_ips_on_exception()

add_path = None
######################################################
s = 8
if s == 1:
    system = UndampedHarmonicOscillator()
    # L = np.array([40, 600, 4000, 10000])
    L = u.get_coefs(np.ones(system.N) * -10)
    x0 = np.array([0.5, 0.5])
    z_hat0 = np.zeros(system.N)
if s == 2:
    system = DuffingOscillator()
    L = u.get_coefs(np.ones(system.N) * -400)
    x0 = np.array([0.8, 0])
    z_hat0 = np.zeros(system.N)
if s == 3:
    system = VanderPol()
    L = u.get_coefs(np.ones(system.N) * -200)
    x0 = np.array([-1, 0])
    z_hat0 = np.zeros(system.N)
if s == 4:
    system = Lorenz()
    L = u.get_coefs(np.ones(system.N) * -200)
    x0 = np.array([0, 1, 10])
    z_hat0 = np.zeros(system.N)
if s == 5:
    system = Roessler()
    L = u.get_coefs(np.ones(system.N) * -200)
    x0 = np.array([0, 1, 5])
    z_hat0 = np.zeros(system.N)
    # add_path = f"N{system.N}"
if s == 6:
    system = DoublePendulum()
    L = u.get_coefs(np.ones(system.N) * -200)
    x0 = np.array([-0.1, 0, 0, 0])
    z_hat0 = np.zeros(system.N)
    # add_path = f"N{system.N}"
if s == 7:
    system = InvPendulum2()
    L = u.get_coefs(np.ones(system.N) * -200)
    phi = np.pi/2+0.1
    x0 = np.array([np.cos(phi), np.sin(phi), 0])

    z_hat0 = np.zeros(system.N)
    add_path = f"measure_x2"

if s == 8:
    system = DoublePendulum2()
    L = u.get_coefs(np.ones(system.N) * -200)
    x0 = system.p_x(1,0,0,0).T[0]
    z_hat0 = system.p_x(0,0,0,0).T[0]
    add_zeros = np.zeros(system.N-len(z_hat0))
    z_hat0 = np.concatenate((z_hat0, add_zeros))
    # add_path = f"N{system.N}"

# IPS()
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 10000)
######################################################

folder_path = os.path.join("models", system.name)
if add_path:
    folder_path = os.path.join(folder_path, add_path)
model_path = os.path.join(folder_path, "model_state_dict.pth")
scaler_in = joblib.load(os.path.join(folder_path, 'scaler_in.pkl'))
scaler_lab = joblib.load(os.path.join(folder_path, 'scaler_lab.pkl'))

model = Net(n=system.n, N=system.N)
model.load_state_dict(torch.load(model_path))

# Observability Canonical Form
A = np.eye(system.N, k=1)
b = np.zeros(system.N)
b[-1] = 1


def system_with_observer_rhs(t, state):
    assert len(state) == system.n + system.N, "Dimension Error"
    x = state[:system.n]
    z_hat = state[system.n:]

    z_hat_normalized = scaler_in.transform([z_hat])[0]

    z_tensor = torch.tensor(z_hat_normalized, dtype=torch.float32).unsqueeze(0)

    # output nerual net
    with torch.no_grad():
        x_hat_normalized = model(z_tensor).numpy()[0]
    x_hat = scaler_lab.inverse_transform([x_hat_normalized])[0][:-1]
    alpha_hat = scaler_lab.inverse_transform([x_hat_normalized])[0][-1]

    # original system
    dxdt = system.rhs(t, x)
    y = system.get_output(x)

    # observer
    dz_hatdt = A @ z_hat + b * alpha_hat - L * (z_hat[0] - y)

    return np.concatenate((dxdt, dz_hatdt))

# simulation


w0 = np.concatenate((x0, z_hat0))

sol = solve_ivp(system_with_observer_rhs, t_span, w0, t_eval=t_eval)

t = sol.t
x_solution = sol.y[:system.n, :]
z_hat_solution = sol.y[system.n:, :]

# calc nominal z values
# z_nom = system.get_q_func()(*x_solution)
Tape_F = 0
Tape_H = 1

n = system.n
adolc.trace_on(Tape_F)
af = [adolc.adouble() for _ in range(n)]
for i in range(n):
    af[i].declareIndependent()
vf = system.rhs(0, af)
for a in vf:
    a.declareDependent()
adolc.trace_off()

adolc.trace_on(Tape_H)
ah = [adolc.adouble() for _ in range(n)]
for i in range(n):
    ah[i].declareIndependent()
vh = system.get_output(ah)
vh.declareDependent()
adolc.trace_off()

d = system.N-1
lie_derivs = []

for i, p0 in enumerate(x_solution.T):
    lie = adolc.lie_scalarc(Tape_F, Tape_H, p0, d)
    lie_derivs.append(lie)

z_nom = np.array(lie_derivs).T

# reconstruct x from observer state
z_hat_solution_normalized = scaler_in.transform(z_hat_solution.T)
with torch.no_grad():
    x_hat_normalized = model(torch.from_numpy(z_hat_solution_normalized).float()).numpy()
x_hat = scaler_lab.inverse_transform(x_hat_normalized)


folder_path = os.path.join(folder_path, f"aw_{x0}")
os.makedirs(folder_path, exist_ok=True)
# plot x
fig, ax = plt.subplots(system.n, 2, sharex=True, figsize=(12,2*system.n))
for i in range(system.n):
    ax[i, 0].plot(t, x_solution[i, :], label=f"$x_{i}$ System")
    ax[i, 0].plot(t, x_hat[:, i], label=f"$\hat x_{i}$ Observer", linestyle='dashed')
    ax[i, 0].set_ylabel(f'$x_{i}$')
    ax[i, 0].set_ylim((min(x_solution[i, :])-1), max(x_solution[i, :])+1)
    ax[i, 0].legend()
    ax[i, 0].grid()
ax[-1, 0].set_xlabel('Time (s)')
for i in range(system.n):
    ax[i, 1].plot(t, np.abs(x_solution[i, :]-x_hat[:, i]), label=f"$x_{i}$ Error")
    ax[i, 1].set_ylabel(f'$x_{i}$')
    ax[i, 1].legend()
    ax[i, 1].set_yscale("log")
    ax[i, 1].grid()
ax[-1, 1].set_xlabel('Time (s)')
plt.tight_layout()
fig.suptitle(f"Observer in original coordinates\nMeasuring {system.h_symb}")
plt.savefig(os.path.join(folder_path, "x.pdf"), format="pdf")

# plot z
fig, ax = plt.subplots(system.N, 2, sharex=True, figsize=(12,2*system.N))
for i in range(system.N):
    ax[i, 0].plot(t, z_nom[i, :], label=f"$z_{i}$ Nom")
    ax[i, 0].plot(t, z_hat_solution[i, :], label=f"$\hat z_{i}$ Observer", linestyle='dashed')
    ax[i, 0].set_ylabel(f'$z_{i}$')
    ax[i, 0].set_ylim((min(z_nom[i, :])-1), max(z_nom[i, :])+1)
    ax[i, 0].legend()
    ax[i, 0].grid()
ax[-1, 0].set_xlabel('Time (s)')
for i in range(system.N):
    ax[i, 1].plot(t, np.abs(z_nom[i, :]-z_hat_solution[i, :]), label=f"$z_{i}$ Error")
    ax[i, 1].set_ylabel(f'$z_{i}$')
    ax[i, 1].legend()
    ax[i, 1].set_yscale("log")
    ax[i, 1].grid()
ax[-1, 1].set_xlabel('Time (s)')
plt.tight_layout()
fig.suptitle(f"embedded Observer\nMeasuring {system.h_symb}")
plt.savefig(os.path.join(folder_path, "z.pdf"), format="pdf")


if system.n == 2:
    fig = plt.figure()
    plt.plot(x_solution[0, :], x_solution[1, :], label="System", color="tab:blue")
    plt.plot(x_hat[:, 0], x_hat[:, 1], label="Observer", linestyle="dashed", color="tab:orange")
    plt.scatter(x_solution[0, 0], x_solution[1, 0], color="tab:blue")
    plt.scatter(x_hat[0, 0], x_hat[0, 1], color="tab:orange")
    plt.xlim(min(x_solution[0, :])-0.5, max(x_solution[0, :]) +0.5)
    plt.ylim(min(x_solution[1, :])-0.5, max(x_solution[1, :]) +0.5)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.title("Phase Diagramm")
elif system.n == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(x_solution[0, :], x_solution[1, :], x_solution[2, :], label="System")
    ax.plot(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], label="Observer", linestyle="dashed")
    ax.scatter(x_solution[0, 0], x_solution[1, 0], x_solution[2, 0], color="tab:blue")
    ax.scatter(x_hat[0, 0], x_hat[0, 1], x_hat[0, 2], color="tab:orange")
    ax.set_xlim((min(x_solution[0, :])-1), max(x_solution[0, :])+1)
    ax.set_ylim((min(x_solution[1, :])-1), max(x_solution[1, :])+1)
    ax.set_zlim((min(x_solution[2, :])-1), max(x_solution[2, :])+1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.title("Phase Diagramm")

elif system.name == "DoublePendulum2":
    fig = plt.figure()
    # inner arm G1
    ax = fig.add_subplot(111)
    # System
    ax.plot(x_solution[0]*2, x_solution[1]*2, color="tab:blue", label="System", alpha=0.5)
    ax.scatter(x_solution[0,0]*2, x_solution[1,0]*2, color="tab:blue") # TODO *2 = *l/s1
    # Observer
    ax.plot(x_hat[:,0]*2, x_hat[:,1]*2, color="tab:orange", label="Observer")
    ax.scatter(x_hat[0,0]*2, x_hat[1,0]*2, color="tab:orange") # TODO *2 = *l/s1

    # outer arm G2
    # System
    G1_ = np.array([x_solution[0]*2, x_solution[1]*2])
    S2_ = np.array([x_solution[2], x_solution[3]])
    G2_ = G1_ + (S2_-G1_)*2
    ax.plot(*G2_, color="tab:blue", linestyle="dashed")
    ax.scatter(*G2_[:,0], color="tab:blue")
    # Obeserver
    G1_ = np.array([x_hat[:,0]*2, x_hat[:,1]*2])
    S2_ = np.array([x_hat[:,2], x_hat[:,3]])
    G2_ = G1_ + (S2_-G1_)*2
    ax.plot(*G2_, color="tab:orange", linestyle="dashed", alpha=0.5)
    ax.scatter(*G2_[:,0], color="tab:orange")

    ax.set_xlim(min(G2_[0]), max(G2_[0]))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "phase.pdf"), format="pdf")
plt.show()
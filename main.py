import os
import torch
import numpy as np
import joblib
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from systems import *
from net import Net
import util as u
from ipydex import activate_ips_on_exception
activate_ips_on_exception()

add_path = None
######################################################
s = 5
if s == 1:
    system = UndampedHarmonicOscillator()
    # L = np.array([40, 600, 4000, 10000])
    L = u.get_coefs(np.ones(system.N) * -10)
    x0 = np.array([0.5, 0.5])
    z_hat0 = np.zeros(system.N)
if s == 2:
    system = DuffingOscillator()
    L = u.get_coefs(np.ones(system.N) * -10)
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
    add_path = f"N{system.N}"


t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 50000)
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

z_nom = system.get_q_func()(*x_solution)

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
plt.savefig(os.path.join(folder_path, "x.pdf"), format="pdf")

# plot z
fig, ax = plt.subplots(system.N, 2, sharex=True, figsize=(12,2*system.N))
for i in range(system.N):
    ax[i, 0].plot(t, z_nom[i, 0, :], label=f"$z_{i}$ Nom")
    ax[i, 0].plot(t, z_hat_solution[i, :], label=f"$\hat z_{i}$ Observer", linestyle='dashed')
    ax[i, 0].set_ylabel(f'$z_{i}$')
    ax[i, 0].set_ylim((min(z_nom[i, 0, :])-1), max(z_nom[i, 0, :])+1)
    ax[i, 0].legend()
    ax[i, 0].grid()
ax[-1, 0].set_xlabel('Time (s)')
for i in range(system.N):
    ax[i, 1].plot(t, np.abs(z_nom[i, 0, :]-z_hat_solution[i, :]), label=f"$z_{i}$ Error")
    ax[i, 1].set_ylabel(f'$z_{i}$')
    ax[i, 1].legend()
    ax[i, 1].set_yscale("log")
    ax[i, 1].grid()
ax[-1, 1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "z.pdf"), format="pdf")


if system.n == 2:
    fig = plt.figure()
    plt.plot(x_solution[0, :], x_solution[1, :], label="System")
    plt.plot(x_hat[:, 0], x_hat[:, 1], label="Observer", linestyle="dashed")
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
    ax.set_xlim((min(x_solution[0, :])-1), max(x_solution[0, :])+1)
    ax.set_ylim((min(x_solution[1, :])-1), max(x_solution[1, :])+1)
    ax.set_zlim((min(x_solution[2, :])-1), max(x_solution[2, :])+1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.title("Phase Diagramm")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "phase.pdf"), format="pdf")
plt.show()
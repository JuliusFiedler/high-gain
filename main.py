import os
import torch
import numpy as np
import joblib
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from systems import *
from net import Net
from util import *


######################################################
s = 1
if s == 1:
    system = UndampedHarmonicOscillator()
    # L = np.array([40, 600, 4000, 10000])
    L = get_coefs(np.ones(system.N) * -10)
    x0 = np.array([0.5, 0.5])
    z_hat0 = np.zeros(system.N)
if s == 2:
    system = DuffingOscillator()
    L = get_coefs(np.ones(system.N) * -10)
    x0 = np.array([0.8, 0])
    z_hat0 = np.zeros(system.N)
if s == 3:
    system = VanderPol()
    L = get_coefs(np.ones(system.N) * -200)
    x0 = np.array([-1, 0])
    z_hat0 = np.zeros(system.N)
if s == 4:
    system = Lorenz()
    L = get_coefs(np.ones(system.N) * -200)
    x0 = np.array([0, 1, 10])
    z_hat0 = np.zeros(system.N)


t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
######################################################


model_path = os.path.join("models", system.name, "model_state_dict.pth")
scaler_in = joblib.load(os.path.join("models", system.name, 'scaler_in.pkl'))
scaler_lab = joblib.load(os.path.join("models", system.name, 'scaler_lab.pkl'))

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

# plot x
plt.figure(figsize=(12,6))
for i in range(system.n):
    plt.subplot(system.n, 1, i+1)
    plt.plot(t, x_solution[i, :], label=f"$x_{i}$ System")
    plt.plot(t, x_hat[:, i], label=f"$\hat x_{i}$ Observer", linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel(f'$x_{i}$')
    plt.legend()
# plot z
plt.figure(figsize=(12,6))
for i in range(system.N):
    plt.subplot(system.N, 1, i+1)
    plt.plot(t, z_nom[i, 0, :], label=f"$z_{i}$ Nom")
    plt.plot(t, z_hat_solution[i, :], label=f"$\hat z_{i}$ Observer", linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel(f'$z_{i}$')
    plt.legend()

if system.n == 2:
    plt.figure()
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
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.title("Phase Diagramm")

plt.show()
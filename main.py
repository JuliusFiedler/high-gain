import os
import torch
import numpy as np
import joblib
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import adolc
import time
from ipydex import activate_ips_on_exception, IPS
from matplotlib.animation import FuncAnimation

from systems import *
from net import Net, NetQ
import util as u
activate_ips_on_exception()

add_path = None
noise = False
######################################################
s = 9
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
    L = u.get_coefs(np.ones(system.N) * -50)
    x0 = np.array([0.5, 0, 0, 0])
    z_hat0 = np.zeros(system.N)
    add_path = f"measure_x1_N4_sep"
    system.separate = True

if s == 7:
    system = InvPendulum2()
    L = u.get_coefs(np.ones(system.N) * -200)
    phi = np.pi/2 +0.1
    x0 = np.array([np.sin(phi), -np.cos(phi), 0])

    z_hat0 = np.zeros(system.N)
    add_path = f"measure_x3_N4"
    # noise = True

if s == 8:
    system = DoublePendulum2()
    poles = [-100, -100]
    L = [u.get_coefs(np.ones(N) * pole) for N, pole in zip(system.N, poles)]
    x0 = system.p_x(2,0,0,0).T[0]
    z_hat0 = np.zeros(np.sum(system.N))
    # add_path = f"scale_z_a"
    add_path = f"measure_['phi1fot, phi2dot']_N[4, 4]_sep"
    # add_path = f"N7"

if s == 9:
    system = MagneticPendulum()
    poles = [-100, -100]
    L = [u.get_coefs(np.ones(N) * pole) for N, pole in zip(system.N, poles)]
    # x0 = np.array([0.9, 0.1, 0, 0])
    x0 = np.array([1.5, 1.5, 0, 0])
    z_hat0 = np.zeros(np.sum(system.N))
    # add_path = f"measure_x1_N5"
    # add_path = f"measure_x1*cos(0.0833333333333333*pi) + x2*sin(0.0833333333333333*pi)_N4"
    add_path = f"p1.0_measure_['x1dot', 'x2dot']_N[5, 5]_sep"
    # add_path = "p1.0_measure_['x1dot', 'x2dot']_N[5, 5]_rand_mag_pos_sep"
    system.separate = True


# IPS()
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 100*t_span[1])
######################################################
lenN = len(system.N)

folder_path = os.path.join("models", system.name)
if add_path:
    folder_path = os.path.join(folder_path, add_path)
model_path = os.path.join(folder_path, "model_state_dict.pth")

if system.separate:
    model_alpha = Net(dim_in=np.sum(system.N), dim_out=lenN)
    model_alpha.load_state_dict(torch.load(os.path.join(folder_path, "al_model_state_dict.pth")))
    # model_q = Net(n=system.n-1, N=system.N)
    model_q = NetQ(dim_in=np.sum(system.N), dim_out=system.n)
    model_q.load_state_dict(torch.load(os.path.join(folder_path, "q_model_state_dict.pth")))
    scaler_in_al = joblib.load(os.path.join(folder_path, 'al_scaler_in.pkl'))
    scaler_lab_al = joblib.load(os.path.join(folder_path, 'al_scaler_lab.pkl'))
    scaler_in_q = joblib.load(os.path.join(folder_path, 'q_scaler_in.pkl'))
    scaler_lab_q = joblib.load(os.path.join(folder_path, 'q_scaler_lab.pkl'))
else:
    model_alpha = Net(dim_in=np.sum(system.N), dim_out=lenN+system.n)
    model_alpha.load_state_dict(torch.load(model_path))
    model_q = model_alpha
    scaler_in_al = scaler_in_q = joblib.load(os.path.join(folder_path, 'scaler_in.pkl'))
    scaler_lab_al = scaler_lab_q = joblib.load(os.path.join(folder_path, 'scaler_lab.pkl'))
# IPS()
# Observability Canonical Form
A_list = []
b_list = []
for N in system.N:
    A_list.append(np.eye(N, k=1))
    b = np.zeros(N)
    b[-1] = 1
    b_list.append(b)
# IPS()
log_scaler = u.LogScaler()

def system_with_observer_rhs(t, state):
    assert len(state) == system.n + np.sum(system.N), "Dimension Error"
    x = state[:system.n]
    z_hat = state[system.n:]
    # t1 = time.time()
    if system.log:
        try:
            z_scale = log_scaler.scale_down(z_hat)
            z_hat_normalized = scaler_in_al.transform([z_scale])[0]
        except:
            IPS()
    else:
        z_hat_normalized = scaler_in_al.transform([z_hat])[0]
    # t2 = time.time()
    z_tensor = torch.tensor(z_hat_normalized, dtype=torch.float32).unsqueeze(0)

    # output neural net
    with torch.no_grad():
        x_hat_normalized = model_alpha(z_tensor).numpy()[0]
    # t3 = time.time()
    alpha_hat = scaler_lab_al.inverse_transform([x_hat_normalized])[0][-lenN:]
    # t4 = time.time()
    if system.log:
        alpha_hat = log_scaler.scale_up(alpha_hat)

    if system.alpha_limit:
        alpha_hat = np.sign(alpha_hat) * min(np.abs(alpha_hat), system.alpha_limit)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # alpha_hat = np.sign(alpha_hat) * min(np.abs(alpha_hat), 100)
    # t5 = time.time()
    # print(np.round([t2-t1, t3-t2, t4-t3, t5-t4], 4))

    # original system
    dxdt = system.rhs(t, x)
    y = system.get_output(x)
    if noise:
        y += np.random.normal(loc=0, scale=1e-7)

    derivatives = dxdt
    idx = 0
    # observer, einzelne Integratorketten
    for i in range(lenN):
        dz_hatdt = A_list[i] @ z_hat[idx:idx+system.N[i]] + b_list[i] * alpha_hat[i] - L[i] * (z_hat[idx] - y[i])
        idx += system.N[i]
        derivatives = np.concatenate((derivatives, dz_hatdt))
    return derivatives

# simulation


w0 = np.concatenate((x0, z_hat0))
print("start simulation")
sol = solve_ivp(system_with_observer_rhs, t_span, w0, t_eval=t_eval)#, atol=1e-7, rtol=1e-7)

t = sol.t
x_solution = sol.y[:system.n, :]
z_hat_solution = sol.y[system.n:, :]

# calc nominal z values
# z_nom = system.get_q_func()(*x_solution)
for i in range(lenN):
    def outputwrapper(x):
        return system.get_output(x)[i]
    u.prepare_adolc(system, outputwrapper, Tape_F=2*i, Tape_H=2*i+1)

lie_derivs = []

for k, p0 in enumerate(x_solution.T):
    z = []
    for j in range(lenN):
        z.extend(adolc.lie_scalarc(2*j, 2*j+1, p0, system.N[j]-1))
    lie_derivs.append(z)

z_nom = np.array(lie_derivs).T

# reconstruct x from observer state
print("reconstruct x")
# IPS()
z_hat_solution_normalized = scaler_in_q.transform(z_hat_solution.T)
with torch.no_grad():
    x_hat_normalized = model_q(torch.from_numpy(z_hat_solution_normalized).float()).numpy()
x_hat = scaler_lab_q.inverse_transform(x_hat_normalized)

print("plotting")
# plt.rcParams.update({
#     "text.usetex": True,
# })
# cm = 1/2.54

# IPS()
folder_path = os.path.join(folder_path, f"aw_{x0}__poles_{poles}")
os.makedirs(folder_path, exist_ok=True)
# plot x
fig, ax = plt.subplots(system.n, 2, sharex=True, figsize=(12,2*system.n))
for i in range(system.n):
    ax[i, 0].plot(t, x_solution[i, :], label=f"$x_{i+1}$ System", color="tab:blue")
    ax[i, 0].plot(t, x_hat[:, i], label=f"$\hat x_{i+1}$ Beobachter", linestyle='dashed', color="tab:orange")
    ax[i, 0].set_ylabel(f'$x_{i+1}$')
    ax[i, 0].set_ylim((min(x_solution[i, :])-1), max(x_solution[i, :])+1)
    ax[i, 0].legend()
    ax[i, 0].grid()
ax[-1, 0].set_xlabel('Zeit in s')
for i in range(system.n):
    ax[i, 1].plot(t, np.abs(x_solution[i, :]-x_hat[:, i]), label=f"Fehler $x_{i+1}$")
    ax[i, 1].set_ylabel(f'Fehler $x_{i+1}$')
    ax[i, 1].legend()
    ax[i, 1].set_yscale("log")
    ax[i, 1].grid()
ax[-1, 1].set_xlabel('Zeit in s')
# fig.suptitle(f"observer in original coordinates\nMeasuring {system.h_symb}")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "x.pdf"), format="pdf", bbox_inches='tight')

# plot x big
# fig, ax = plt.subplots(int((system.n+1)/2), 2, sharex=True, figsize=(8,4))
# for i in range(system.n):
#     binary = np.binary_repr(i, width=2)
#     idx = (int(binary[0]), int(binary[1]))
#     ax[idx].plot(t, x_solution[i, :], label=f"$x_{i+1}$ system", color="tab:blue")
#     ax[idx].plot(t, x_hat[:, i], label=f"$\hat x_{i+1}$ observer", linestyle='dashed', color="tab:orange")
#     ax[idx].set_ylabel(f'$x_{i+1}$')
#     ax[idx].set_ylim((min(x_solution[i, :])-0.5), max(x_solution[i, :])+0.5)
#     ax[idx].legend(loc="upper right")
#     ax[idx].grid()
#     ax[idx].scatter(0, x_solution[i, 0], color="tab:blue")
#     ax[idx].scatter(0, x_hat[0, i], color="tab:orange")
# ax[-1, 0].set_xlabel('Zeit in s')
# if system.n %2 == 1:
#     for i in range(system.n):
#         ax[-1, 1].plot(t, np.abs(x_solution[i, :]-x_hat[:, i]), label=f"$\Delta x_{i+1}$", alpha=0.8)
#         ax[-1, 1].set_ylabel(f'Fehler')
#         ax[-1, 1].legend(loc="upper left")
#         ax[-1, 1].set_yscale("log")
#         ax[-1, 1].grid()
# ax[-1, 1].set_xlabel('Zeit in s')
# # ax[1,1].set_visible(False)
# plt.tight_layout()
# # fig.suptitle(f"observer in original coordinates\nMeasuring {system.h_symb}")
# plt.savefig(os.path.join(folder_path, "x_big.pdf"), format="pdf")

# plot z
print("Warning: Watch out for notation: z_21 vs z_6")
fig, ax = plt.subplots(sum(system.N), 2, sharex=True, figsize=(12,2*sum(system.N)))
for i in range(sum(system.N)):
    ax[i, 0].plot(t, z_nom[i, :], label="$z_{"+str(i+1)+"}$ System", color="tab:blue")
    ax[i, 0].plot(t, z_hat_solution[i, :], label="$\hat z_{"+str(i+1)+"}$ Beobachter", linestyle='dashed', color="tab:orange")
    ax[i, 0].set_ylabel("$z_{"+str(i+1)+"}$")
    ax[i, 0].set_ylim((min(z_nom[i, :])-1), max(z_nom[i, :])+1)
    ax[i, 0].legend()
    ax[i, 0].grid()
ax[-1, 0].set_xlabel('Zeit in s')
for i in range(sum(system.N)):
    ax[i, 1].plot(t, np.abs(z_nom[i, :]-z_hat_solution[i, :]), label="Fehler $z_{"+str(i+1)+"}$")
    ax[i, 1].set_ylabel("Fehler $z_{"+str(i+1)+"}$")
    ax[i, 1].legend()
    ax[i, 1].set_yscale("log")
    ax[i, 1].grid()
ax[-1, 1].set_xlabel('Zeit in s')
# fig.suptitle(f"embedded observer\nMeasuring {system.h_symb}")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "z.pdf"), format="pdf", bbox_inches='tight')

# plot z double
if system.N[0] == system.N[1]:#todo this is not sufficient
    N = system.N[0]
    fig, ax = plt.subplots(int(sum(system.N)/2), 2, sharex=True, figsize=(12,2*sum(system.N)/2))
    for i in range(N):
        ax[i, 0].plot(t, z_nom[i, :], label=f"$z_{{1,{i+1}}}$ System", color="tab:blue")
        ax[i, 0].plot(t, z_nom[i+N, :], label=f"$z_{{2,{i+1}}}$ System", color="blue")
        ax[i, 0].plot(t, z_hat_solution[i, :], label=f"$\hat z_{{1,{i+1}}}$ Beobachter", linestyle='dashed', color="tab:orange")
        ax[i, 0].plot(t, z_hat_solution[i+N, :], label=f"$\hat z_{{2,{i+1}}}$ Beobachter", linestyle='dashed', color="gold")
        ax[i, 0].set_ylabel(f"$z_{{i,{i+1}}}$")
        ax[i, 0].set_ylim((np.min([z_nom[i, :], z_nom[i+N, :]])-1), np.max([z_nom[i, :], z_nom[i+N, :]])+1)
        ax[i, 0].legend()
        ax[i, 0].grid()
    ax[-1, 0].set_xlabel('Zeit in s')
    for i in range(N):
        ax[i, 1].plot(t, np.abs(z_nom[i, :]-z_hat_solution[i, :]), label=f"Fehler $z_{{1,{i+1}}}$")
        ax[i, 1].plot(t, np.abs(z_nom[i+N, :]-z_hat_solution[i+N, :]), label=f"Fehler $z_{{2,{i+1}}}$")
        ax[i, 1].set_ylabel(f"Fehler $z_{{i,{i+1}}}$")
        ax[i, 1].legend()
        ax[i, 1].set_yscale("log")
        ax[i, 1].grid()
    ax[-1, 1].set_xlabel('Zeit in s')
    # fig.suptitle(f"embedded observer\nMeasuring {system.h_symb}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "z_double.pdf"), format="pdf", bbox_inches='tight')

# plot z bigger
# fig, ax = plt.subplots(int((system.N+1)/2), 2, sharex=True, figsize=(8,4))
# plt.rcParams.update({"font.size":10})
# for i in range(system.N):
#     binary = np.binary_repr(i, width=2)
#     idx = (int(binary[0]), int(binary[1]))
#     ax[idx].plot(t, z_nom[i, :], label=f"$z_{i+1}$ nominal", color="tab:blue")
#     ax[idx].plot(t, z_hat_solution[i, :], label=f"$\hat z_{i+1}$ observer", linestyle='dashed', color="tab:orange")
#     ax[idx].set_ylabel(f'$z_{i+1}$')
#     ax[idx].set_ylim((min(z_nom[i, :])-0.5), max(z_nom[i, :])+0.5)
#     ax[idx].legend(loc="upper right")
#     ax[idx].grid()
#     ax[idx].scatter(0, z_nom[i, 0], color="tab:blue")
#     ax[idx].scatter(0, z_hat_solution[i, 0], color="tab:orange")
# ax[-1, 0].set_xlabel('Zeit in s')
# if system.N %2 == 1:
#     for i in range(system.n):
#         ax[-1, 1].plot(t, np.abs(z_nom[i, :]-z_hat_solution[i, :]), label=f"$\Delta z_{i+1}$", alpha=0.8)
#         ax[-1, 1].set_ylabel(f'Fehler')
#         ax[-1, 1].legend(loc="upper left")
#         ax[-1, 1].set_yscale("log")
#         ax[-1, 1].grid()
#         ax[-1, 1]
# ax[-1, 1].set_xlabel('Zeit in s')
# plt.tight_layout()
# # plt.show()
# plt.savefig(os.path.join(folder_path, "z_big.pdf"), format="pdf")


if system.n == 2:
    fig = plt.figure()
    plt.plot(x_solution[0, :], x_solution[1, :], label="System", color="tab:blue")
    plt.plot(x_hat[:, 0], x_hat[:, 1], label="Beobachter", linestyle="dashed", color="tab:orange")
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
    ax.plot(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], label="Beobachter", linestyle="dashed")
    ax.scatter(x_solution[0, 0], x_solution[1, 0], x_solution[2, 0], color="tab:blue")
    ax.scatter(x_hat[0, 0], x_hat[0, 1], x_hat[0, 2], color="tab:orange")
    ax.set_xlim((min(x_solution[0, :])-1), max(x_solution[0, :])+1)
    ax.set_ylim((min(x_solution[1, :])-1), max(x_solution[1, :])+1)
    ax.set_zlim((min(x_solution[2, :])-1), max(x_solution[2, :])+1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "phase_big.pdf"), format="pdf", bbox_inches='tight')
    plt.title("Phase Diagramm")

elif system.name == "DoublePendulum2":
    fig = plt.figure()
    # inner arm G1
    ax = fig.add_subplot(111)
    # system
    ax.plot(x_solution[0]*2, x_solution[1]*2, color="tab:blue", label="System")
    ax.scatter(x_solution[0,0]*2, x_solution[1,0]*2, color="tab:blue") # TODO *2 = *l/s1
    # observer
    ax.plot(x_hat[:,0]*2, x_hat[:,1]*2, color="tab:orange", label="Beobachter", alpha=0.5)
    ax.scatter(x_hat[0,0]*2, x_hat[1,0]*2, color="tab:orange") # TODO *2 = *l/s1

    # outer arm G2
    # system
    G1_ = np.array([x_solution[0]*2, x_solution[1]*2])
    S2_ = np.array([x_solution[2], x_solution[3]])
    G2_ = G1_ + (S2_-G1_)*2
    ax.plot(*G2_, color="tab:blue", linestyle="dashed")
    ax.scatter(*G2_[:,0], color="tab:blue")
    ax.set_xlim(min(G2_[0]), max(G2_[0]))
    # Obeserver
    G1_ = np.array([x_hat[:,0]*2, x_hat[:,1]*2])
    S2_ = np.array([x_hat[:,2], x_hat[:,3]])
    G2_ = G1_ + (S2_-G1_)*2
    ax.plot(*G2_, color="tab:orange", linestyle="dashed", alpha=0.5)
    ax.scatter(*G2_[:,0], color="tab:orange")

    plt.ylim(-1, 0.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
elif system.name == "MagneticPendulum":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # system
    ax.plot(x_solution[0], x_solution[1], color="tab:blue", label="System")
    ax.scatter(x_solution[0,0], x_solution[1,0], color="tab:blue")
    # observer
    ax.plot(x_hat[:,0], x_hat[:,1], color="tab:orange", label="Beobachter", alpha=0.5)
    ax.scatter(x_hat[0,0], x_hat[0,1], color="tab:orange")

    # magnets
    ax.scatter(*system.magnet_positions.T, color="red", label="Magnet")
    plt.ylim(-3, 3)
    plt.xlim(-3, 3)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "phase.pdf"), format="pdf", bbox_inches='tight')
plt.show()

# animation
confirm = input("animate? [y]/n\n")
if confirm == "" or confirm == "y":
    print("animation")
    import functools as ft
    if system.name == "MagneticPendulum":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        solutions = [x_solution, x_hat.T]
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        line_sys, = ax.plot([], [], lw = 3, color="tab:blue", label="System")
        line_obs, = ax.plot([], [], lw = 2, color="tab:orange", label="Beobachter", alpha=0.5)
        plt.legend()
        def init(solutions):
            ax.scatter(solutions[0][0][0], solutions[0][1][0])
            ax.scatter(solutions[1][0][0], solutions[1][1][0])
            ax.grid()
            line_sys.set_data([],[])
            ax.scatter(*system.magnet_positions.T, color= "red")
            return [line_sys, line_obs]

        def animate(frame, solutions):
            assert frame < solutions[0].shape[1]
            line_sys.set_data(solutions[0][0, :frame], solutions[0][1, :frame])
            line_obs.set_data(solutions[1][0, :frame], solutions[1][1, :frame])
            return [line_sys, line_obs]

        anim = FuncAnimation(
            fig,
            ft.partial(animate, solutions=solutions),
            init_func = ft.partial(init, solutions=solutions),
            frames = x_solution.shape[1],
            interval = 1,
            blit = True
            )
        anim.save(os.path.join(folder_path, "anim.mp4"), writer = 'ffmpeg', fps = 30)
    plt.show()
# IPS()
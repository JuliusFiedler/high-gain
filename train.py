import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ipydex import IPS, activate_ips_on_exception
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib
import adolc
from systems import *
from net import Net, NetQ
import time
import util as u

activate_ips_on_exception()


def get_lie_derivs(system, output, points, d):

    Tape_F, Tape_H = u.prepare_adolc(system, output)
    lie_derivs = []

    for i, x0 in enumerate(points):
        lie = adolc.lie_scalarc(Tape_F, Tape_H, x0, d)
        if system.alpha_limit:
            if np.abs(lie[-1]) > system.alpha_limit:
                lie[-1] = np.sign(lie[-1]) * system.alpha_limit
        lie_derivs.append(lie)

    lie_derivs = np.array(lie_derivs)

    return lie_derivs

def train(system: System, train: str, num_epoch=100, noise=False):
    lenN = len(system.N) # TODO this will raise errors for old systems

    if system.name == "MagneticPendulum":
        system.add_path = f"p{system.charges[0]}_" + system.add_path
    assert train in ["all", "q", "alpha"]
    if train != "all":
        system.add_path += "_sep"
    if noise:
        system.add_path += "_noise"
    seed = 1

    np.random.seed(seed)
    torch.manual_seed(seed)


    # choose sample coords in x
    if hasattr(system, "get_x_data"):
        print("using trajectory data")
        points = system.get_x_data().T
    else:
        print("using grid")
        num_interpolation_points = 20

        limits = system.get_approx_region()
        mesh_index = []
        for i in range(len(limits)):
            mesh_index.append(np.linspace(*limits[i], num_interpolation_points))
        meshgrid = np.meshgrid(*mesh_index) # shape [(NIP, NIP, ..), (NIP, NIP, ..)]
        points = np.vstack([x.ravel() for x in meshgrid]).T

    # SISO
    # TODO make this obsolete
    if not hasattr(system.N, "__len__"):
        assert hasattr(system.get_output(np.ones(system.n)), "__len__") == False, "Check system MIMO vs SISO. Conflict in N and output."

        lie_derivs = get_lie_derivs(system, system.get_output, points, system.N)

        inputs = lie_derivs[:,:-1].T
        if train == "all":
            labels = np.empty((system.n+1, points.shape[0])) # shape (n+1, NIP**n)
            labels[:-1, :] = points.T
            labels[-1, :] = lie_derivs[:, -1]

        elif train == "q":
            labels = points.T
        elif train == "alpha":
            labels = np.empty((1, points.shape[0])) # shape (n+1, NIP**n)
            labels[-1, :] = lie_derivs[:, -1]
    # MIMO
    else:

        lie_N_minus_1 = np.empty((points.shape[0], np.sum(system.N))) # shape (number of points, sum(N_i))
        alphas = np.empty((points.shape[0], lenN))
        idx = 0
        for i, N in enumerate(system.N):
            def outputwrapper(x):
                return system.get_output(x)[i]
            lie = get_lie_derivs(system, outputwrapper, points, N)
            lie_N_minus_1[:, idx:idx+N] = lie[:,:-1] # fill array left to right blockwise
            alphas[:, i] = lie[:,-1]
            idx += N
        alphas = np.array(alphas)

        inputs = lie_N_minus_1.T
        if train == "all":
            labels = np.empty((system.n+lenN, points.shape[0])) # shape (n+len(N), NIP**n)
            labels[:-lenN, :] = points.T
            labels[-lenN:, :] = alphas.T

        elif train == "q":
            labels = points.T
        elif train == "alpha":
            labels = alphas.T

    if noise:
        inputs = inputs + np.random.normal(0, 1e-3, size=inputs.shape)
        labels = labels + np.random.normal(0, 1e-3, size=labels.shape)
    # IPS()
    if system.log == True:
        print("using log scaler")
        log_scaler = u.LogScaler()
        inputs = log_scaler.scale_down(inputs)
        labels[-1] = log_scaler.scale_down(labels[-1])
    # IPS()
    # scale data
    scaler_in = MinMaxScaler(feature_range=(-1, 1))
    scaler_lab = MinMaxScaler(feature_range=(-1, 1))
    inputs_normalized = scaler_in.fit_transform(inputs.T)
    labels_normalized = scaler_lab.fit_transform(labels.T)

    # type casting
    inputs_normalized = torch.from_numpy(inputs_normalized).float()
    labels_normalized = torch.from_numpy(labels_normalized).float()

    dataset = TensorDataset(inputs_normalized, labels_normalized)
    batch_size = 50
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if train == "all":
        net = Net(dim_in=np.sum(system.N), dim_out=system.n+lenN)
    elif train == "q":
        net = NetQ(dim_in=np.sum(system.N), dim_out=system.n)
    elif train == "alpha":
        net = Net(dim_in=np.sum(system.N), dim_out=lenN)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print("start Training")
    try:
        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {avg_loss:.6f}')
    except KeyboardInterrupt:
        IPS()

    print('Training finished')
    if train == "all":
        note = ""
    elif train == "q":
        note = "q_"
    elif train == "alpha":
        note = "al_"
    folder_path = os.path.join("models", system.name, system.add_path)
    path = os.path.join(folder_path, note + "model_state_dict.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)

    joblib.dump(scaler_in, os.path.join("models", system.name, system.add_path, note + 'scaler_in.pkl'))
    joblib.dump(scaler_lab, os.path.join("models", system.name, system.add_path, note + 'scaler_lab.pkl'))
    # IPS()
    with open(os.path.join(folder_path, "notes.txt"), "at") as f:
        f.write(train + " last loss: " + str(round(avg_loss, 7)) + "\n")

    if train != "q":
        gamma = get_lipschitz_const(net)
        with open(os.path.join(folder_path, "notes.txt"), "at") as f:
            f.write("gamma: " + str(np.round(gamma, 7)) + "\n")
    # IPS()

def get_lipschitz_const(net):
    for i in range(len(system.N)):
        def outputwrapper(x):
            return system.get_output(x)[i]
        u.prepare_adolc(system, outputwrapper, Tape_F=2*i, Tape_H=2*i+1)

    gammas = []
    for k, N in enumerate(system.N):

        num_samples = 20000
        gamma = 0

        if hasattr(system, "get_x_data"):
            x_data = system.get_x_data()
            minmax = np.array([np.min(x_data, axis=1), np.max(x_data, axis=1)])
        else:
            minmax = np.array(system.get_approx_region()).T

        for i in range(num_samples):
            if i % 1000 == 0:
                print(i)
            xa = np.random.uniform(*minmax)
            xb = np.random.uniform(*minmax)
            za = []
            zb = []
            for j in range(len(system.N)):
                # za = q(*xa).T
                za.extend(adolc.lie_scalarc(2*j, 2*j+1, xa, N-1))

                # zb = q(*xb).T
                zb.extend(adolc.lie_scalarc(2*j, 2*j+1, xb, N-1))
            za = np.array(za)
            zb = np.array(zb)
            alphaa = net(torch.tensor(za, dtype=torch.float32)).detach().numpy()[-(len(system.N)-k)]
            alphab = net(torch.tensor(zb, dtype=torch.float32)).detach().numpy()[-(len(system.N)-k)]

            # gamma = delta alpha / delta z
            gamma = max(gamma, np.abs((alphaa - alphab) / np.linalg.norm(za-zb)))
        print("Lipschitz constant approx.:", gamma)
        gammas.append(gamma)
    return gammas

################################################################################
# system = UndampedHarmonicOscillator()
# system = DuffingOscillator()
# system = VanderPol()
# system = Lorenz()
# system = Roessler()
system = DoublePendulum()
# system = DoublePendulum2()
# system = InvPendulum2()
# system = MagneticPendulum()
################################################################################


system.add_path = f"measure_{system.h_symb}_N{system.N}"

sep = True
noise = False
num_epoch = 100
if sep:
    print("Training alpha")
    train(system, train="alpha", num_epoch=num_epoch, noise=noise)
    system.add_path = f"measure_{system.h_symb}_N{system.N}"
    print("Training q^-1")
    train(system, train="q", num_epoch=num_epoch, noise=noise)
else:
    train(system, train="all", num_epoch=num_epoch, noise=noise)

# net = Net(n=system.n, N=system.N)
# net.load_state_dict(torch.load(os.path.join("models", system.name, 'model_state_dict.pth')))
# get_lipschitz_const(net)

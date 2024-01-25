import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ipydex import IPS
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

from systems import *
from net import Net

def train(system: System):
    seed = 1

    np.random.seed(seed)
    torch.manual_seed(seed)

    num_interpolation_points = 100
    
    # choose sample coords in x
    limits = system.get_approx_region()
    mesh_index = []
    for i in range(len(limits)):
        mesh_index.append(np.linspace(*limits[i], num_interpolation_points))
    meshgrid = np.meshgrid(*mesh_index) # shape [(NIP, NIP, ..), (NIP, NIP, ..)]
    
    # calculate alpha(x)
    alpha = system.get_alpha_of_x(meshgrid) # shape (NIP, NIP, ..)

    # calculate z(x)
    q_func = system.get_q_func()  
    z_data = q_func(*meshgrid)[:,0] # shape (N, NIP, NIP, ...)
    
    # shape data
    inputs = z_data.reshape(system.N, num_interpolation_points**system.n) # shape (N, NIP**n)
    labels = np.empty((system.n+1, num_interpolation_points**system.n)) # shape (n+1, NIP**n)
    labels[:-1, :] = [m.reshape(num_interpolation_points**system.n) for m in meshgrid]
    labels[-1, :] = alpha.reshape(num_interpolation_points**system.n)

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

    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/100], Loss: {avg_loss:.6f}')

    print('Training finished')
    path = os.path.join("models", system.name, "model_state_dict.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
    
    joblib.dump(scaler_in, os.path.join("models", system.name, 'scaler_in.pkl'))
    joblib.dump(scaler_lab, os.path.join("models", system.name, 'scaler_lab.pkl'))
    # IPS()
    
    get_lipschitz_const(net)

def get_lipschitz_const(net):
    q = system.get_q_func()
    # TODO AD
    num_samples = 20000
    gamma = 0
    for i in range(num_samples):
        xa = np.random.uniform(*np.array(system.get_approx_region()).T)
        za = q(*xa).T
        alphaa = net(torch.tensor(za, dtype=torch.float32)).detach().numpy()[0, -1]

        xb = np.random.uniform(*np.array(system.get_approx_region()).T)
        zb = q(*xb).T
        alphab = net(torch.tensor(zb, dtype=torch.float32)).detach().numpy()[0, -1]

        # gamma = delta alpha / delta z
        gamma = max(gamma, np.abs((alphaa - alphab) / np.linalg.norm(za-zb)))
    print("Lipschitz constant approx.:", gamma)


system = UndampedHarmonicOscillator()
# train(system)
net = Net()
net.load_state_dict(torch.load(os.path.join("models", system.name, 'model_state_dict.pth')))
get_lipschitz_const(net)

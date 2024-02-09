import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ipydex import IPS
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib
import adolc
from systems import *
from net import Net

def test(system: System):
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
    alpha_func = system.get_alpha_of_x_func()
    alpha = alpha_func(*meshgrid) # shape (NIP, NIP, ..)

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


    d = 5
    # x, y, z = meshgrid
    # points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    points = np.vstack([x.ravel() for x in meshgrid]).T
    for i, x0 in enumerate(points):
        lie = adolc.lie_scalarc(Tape_F, Tape_H, x0, d)
        if i % 1000 == 0:
            print(i)
    IPS()

test(Roessler())
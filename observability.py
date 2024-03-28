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

    rhs = system.rhs
    output = system.get_output
    print("Output", system.h_symb)


    def output(state):
        return state[1]

    Tape_F = 0
    Tape_H = 1

    n = system.n
    adolc.trace_on(Tape_F)
    af = [adolc.adouble() for _ in range(n)]
    for i in range(n):
        af[i].declareIndependent()
    vf = rhs(0, af)
    for a in vf:
        a.declareDependent()
    adolc.trace_off()


    adolc.trace_on(Tape_H)
    ah = [adolc.adouble() for _ in range(n)]
    for i in range(n):
        ah[i].declareIndependent()
    vh = output(ah)
    vh.declareDependent()
    adolc.trace_off()

    np.random.seed(2)
    d = 3
    if system.name == "DoublePendulum2":
        phi0 = (np.random.random(size=10)-0.5)*2
        phi1 = np.zeros(10)
        w0 = np.zeros(10)
        w1 = np.zeros(10)
        w0[-1] = 1
        w1[-2] = 1
        w0[-3] = -1
        w1[-3] = -1
        x0_list = np.array([phi0, phi1, w0, w1])
        x0_list = np.concatenate((x0_list, np.array([
            [0,0,      1 ],
            [0,np.pi,  0 ],
            [0,0,  0.1 ],
            [0,0, 0.1]
        ])), axis=1)
        x0_list = np.concatenate((x0_list, np.array([
            (np.random.random(size=10)-0.5)*np.pi,
            (np.random.random(size=10)-0.5)*np.pi,
            (np.random.random(size=10)-0.5)*2,
            (np.random.random(size=10)-0.5)*2
        ])), axis=1)
        p0_list = system.p_x(*x0_list).T[:,0,:]
    elif system.name == "DoublePendulum":
        p0_list = [[-0.1,0,0,0],
                   [0.05,0,0,0.1],
                   [0.13,0,0.01,0],
                   [0.08, 0.1, 0.1, 0.01],
                   [0.0, 0., 0., 0.01],
                   [0.0, 0., 0.1, 0.0],
                   [0.08, 0.1, 0.1, 0.01],
                   ]
    elif system.name == "MagneticPendulum":
        p0_list = (np.random.random((10,4))-0.5)*4
    else:
        phi0 = (np.random.random(size=10)-0.5)*2*np.pi
        w0 = (np.random.random(size=10)-0.5)*5
        w0[-1] = 0
        x0_list = np.array([np.sin(phi0), -np.cos(phi0), w0])
        p0_list = np.concatenate((x0_list.T, np.array([
            [1,0,      0 ],
            [0, -1,  0 ],
            [0,-1,  0.1 ],
            [0,1, 0.1],
            [0,1, 0]
        ])), axis=0)


    for p0 in p0_list:
        lie = adolc.lie_gradientc(Tape_F, Tape_H, p0, d)
        rank = np.linalg.matrix_rank(lie)
        print(p0, ": Rank", rank)
    # IPS()
    IPS()

test(MagneticPendulum())
# test(DoublePendulum2())
# test(InvPendulum2())

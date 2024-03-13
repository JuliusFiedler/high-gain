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
    # def rhs(t, state):
    #     x1, x2, x3, x4 = state
    #     return np.array([x3, x4, np.sin(x1), x2*x1**(-1)])

    def output(state):
        return state[0]

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


    d = 5
    x0 = np.ones(n)
    lie = adolc.lie_scalarc(Tape_F, Tape_H, x0, d)
    IPS()


test(DoublePendulum())

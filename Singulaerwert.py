import numpy as np
from scipy.linalg import svd, inv
from ipydex import IPS
import symbtools as st
from systems import *


def test_poles(system, poles):

    l = np.flip(np.polynomial.polynomial.polyfromroots(poles)[:-1]).reshape(system.N,1)

    A = np.zeros((system.N, system.N))
    for i, a in enumerate(A):
        if i+1 < len(a):
            A[i,i+1] = 1

    C = np.zeros((1, system.N))
    C[0,0] = 1
    B = np.zeros((system.N, 1))
    B[-1,0] = 1



    def compute_G(s):

        G = inv(np.eye(system.N) * s -(A-l@C))@B
        return G


    omega = np.linspace(0, 1000, 100000)

    max_singular_value = 0
    for i, w in enumerate(omega):
        if i % 10000 == 0:
            print(i)
        G_jw = compute_G(1j * w)

        infnorm = np.linalg.norm(G_jw, ord=2)
        max_singular_value = max(max_singular_value, infnorm)


    print("Maximum Singular Value (H∞-norm):", max_singular_value)


system = MagneticPendulum()
# system = DoublePendulum()

poles = np.ones(system.N) * -15
l = input("enter lipschitz konstant")
test_poles(system, poles)
print(1/float(l))
IPS()
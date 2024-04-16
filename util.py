import sympy as sp
import numpy as np
import adolc

def get_coefs(poles:list):
    return np.flip(np.polynomial.polynomial.polyfromroots(poles)[:-1])

def get_percentage_over_abs(data, limit):
    total = len(data)
    assert total > 1
    count = len(np.where(np.abs(data) > limit)[0])
    return count, total, f"{count/total*100}%"

def prepare_adolc(system, output, Tape_F=0, Tape_H=1):
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
    vh = output(ah)
    vh.declareDependent()
    adolc.trace_off()
    return Tape_F, Tape_H

def trafo_x_to_z(system, x0):
    for i in range(len(system.N)):
        def outputwrapper(x):
            return system.get_output(x)[i]
        prepare_adolc(system, outputwrapper, Tape_F=2*i, Tape_H=2*i+1)

    z = []
    for j in range(len(system.N)):
        z.extend(adolc.lie_scalarc(2*j, 2*j+1, x0, system.N[j]-1))
    return z

def relative_error(truth, measurement):
    assert truth.shape == measurement.shape
    return np.abs((truth-measurement)/truth)

def get_distance(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[0])**2)


class LogScaler():
    def __init__(self, x1=10) -> None:
        self.x1 = x1
        self.a = self.x1
        self.b = self.x1 - self.x1 * np.log(self.x1)

    def scale_down(self, x):
        x = np.array(x)
        return np.where(np.abs(x) >=self.x1, np.sign(x) * (self.a*np.log(np.abs(x)) + self.b), x)

    def scale_up(self, x):
        return np.where(np.abs(x) >= self.x1, np.sign(x) * np.exp((np.abs(x)-self.b)/self.a), x)
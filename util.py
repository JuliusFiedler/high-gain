import sympy as sp
import numpy as np

def get_coefs(poles:list):
    return np.flip(np.polynomial.polynomial.polyfromroots(poles)[:-1])

def get_percentage_over_abs(data, limit):
    total = len(data)
    assert total > 1
    count = len(np.where(np.abs(data) > limit)[0])
    return count, total, f"{count/total*100}%"

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